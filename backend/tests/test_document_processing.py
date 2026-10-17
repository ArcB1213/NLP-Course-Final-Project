import fitz
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api import documents
from app.config import Settings
from app.core.schemas import RawPage
from app.db.database import Base, get_db
from app.db.models import Document
from app.document.ingestion import IngestionService
from app.document.loaders import PDFLoader
from app.document.processing_report import read_processing_report, save_processing_report
from app.document.splitter import ChineseTextSplitter


def test_pdf_removes_repeated_margin_text_but_keeps_body_text(tmp_path):
    path = tmp_path / "slides.pdf"
    with fitz.open() as document:
        for index in range(4):
            page = document.new_page(width=600, height=600)
            page.insert_text((40, 30), "Navigation", fontsize=12)
            page.insert_text((40, 100), f"Section {index + 1}", fontsize=20)
            page.insert_text((40, 160), "Navigation is also a body concept.", fontsize=12)
        document.save(path)

    pages = PDFLoader().load(str(path))
    assert len(pages) == 4
    assert pages[0].text.startswith("Section 1")
    assert "Navigation is also a body concept." in pages[0].text
    assert pages[0].metadata["heading"] == "Section 1"
    assert pages[0].page == 1


def test_pdf_native_table_keeps_rows_and_does_not_duplicate_cells(tmp_path):
    path = tmp_path / "table.pdf"
    with fitz.open() as document:
        page = document.new_page(width=600, height=600)
        for x in [50, 180, 310]:
            page.draw_line((x, 150), (x, 270))
        for y in [150, 190, 230, 270]:
            page.draw_line((50, y), (310, y))
        for index, (name, value) in enumerate([("Item", "Cost"), ("Alpha", "8.22"), ("Beta", "5.69")]):
            page.insert_text((60, 175 + index * 40), name)
            page.insert_text((190, 175 + index * 40), value)
        document.save(path)

    page = PDFLoader().load(str(path))[0]
    assert "| Item | Cost |" in page.text
    assert "| Alpha | 8.22 |" in page.text
    assert page.text.count("8.22") == 1
    assert any(block["type"] == "table" for block in page.metadata["blocks"])


def test_pdf_image_content_is_flagged_and_footer_logo_is_ignored(tmp_path):
    path = tmp_path / "image.pdf"
    image = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100), False)
    image.clear_with(255)
    with fitz.open() as document:
        page = document.new_page(width=600, height=600)
        page.insert_image(fitz.Rect(50, 150, 400, 400), stream=image.tobytes("png"))
        page.insert_image(fitz.Rect(0, 550, 80, 600), stream=image.tobytes("png"))
        document.save(path)

    page = PDFLoader().load(str(path))[0]
    assert page.text == ""
    assert page.metadata["warnings"] == ["visual_content_requires_recognition", "no_native_text"]
    assert len(page.metadata["image_regions"]) == 1


def test_split_table_repeats_header_and_keeps_rows_intact():
    rows = [f"| item-{index} | value-{index} |" for index in range(12)]
    table = "| Item | Value |\n| --- | --- |\n" + "\n".join(rows)
    chunks = ChineseTextSplitter(chunk_size=100, chunk_overlap=20)._split_text("Before.\n\n" + table + "\n\nAfter.")
    assert chunks[0] == "Before."
    assert chunks[-1] == "After."
    table_chunks = chunks[1:-1]
    assert all(chunk.startswith("| Item | Value |\n| --- | --- |\n") for chunk in table_chunks)
    assert all(sum(row in chunk.splitlines() for chunk in table_chunks) == 1 for row in rows)
    assert all(len(chunk) <= 100 for chunk in table_chunks)


def test_split_display_formula_keeps_expression_and_boundaries():
    formula = "$$\nP(S) = \\pi_{t_1} \\prod_{k=1}^{p-1} a_{t_k,t_{k+1}}\n$$"
    chunks = ChineseTextSplitter(chunk_size=40, chunk_overlap=10)._split_text("Definition.\n\n" + formula + "\n\nConditions.")
    assert chunks == ["Definition.", formula, "Conditions."]


def test_splitter_keeps_page_and_quality_warning_on_every_chunk():
    page = RawPage(text="First sentence.\n\nSecond sentence.", page=7,
                   metadata={"heading": "Example", "warnings": ["formula_layout_requires_recognition"]})
    chunks = ChineseTextSplitter(chunk_size=18, chunk_overlap=0).split_pages([page], "doc", "slides.pdf")
    assert len(chunks) == 2
    assert all(chunk.page == 7 and chunk.heading == "Example" for chunk in chunks)
    assert all("公式排列可能损坏" in chunk.text for chunk in chunks)


class FakeRetriever:
    def __init__(self):
        self.rebuilt = 0

    def rebuild(self, db):
        self.rebuilt += 1


@pytest.mark.parametrize("has_text", [True, False])
def test_ingestion_saves_quality_report_and_does_not_index_empty_pdf(tmp_path, has_text):
    path = tmp_path / "upload.pdf"
    image = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100), False)
    image.clear_with(255)
    with fitz.open() as document:
        page = document.new_page(width=600, height=600)
        if has_text:
            page.insert_text((40, 100), "Figure example", fontsize=20)
        page.insert_image(fitz.Rect(50, 150, 400, 400), stream=image.tobytes("png"))
        document.save(path)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    retriever = FakeRetriever()
    service = IngestionService(Settings(_env_file=None, pdf_parser="native"), retriever)
    with Session(engine) as db:
        if has_text:
            result = service.ingest_file(db, document_id="doc", file_path=path, filename="upload.pdf", file_type="pdf")
            assert result.status == "indexed_partial"
            assert result.chunk_count > 0
            assert retriever.rebuilt == 1
        else:
            with pytest.raises(ValueError, match="未提取到可索引内容"):
                service.ingest_file(db, document_id="doc", file_path=path, filename="upload.pdf", file_type="pdf")
            assert db.get(Document, "doc").status == "failed"
            assert retriever.rebuilt == 0
    report = read_processing_report(path)
    assert report["page_count"] == 1
    assert report["warnings"]
    assert report["pages"][0]["metadata"]["image_regions"]


def test_document_api_exposes_quality_and_supports_legacy_documents(tmp_path):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    path = tmp_path / "parsed.pdf"
    save_processing_report(path, [RawPage(text="Title", page=2,
                                          metadata={"warnings": ["visual_content_requires_recognition"]})])
    with Session(engine) as db:
        db.add_all([
            Document(id="new", filename="new.pdf", file_type="pdf", file_path=str(path), status="indexed_partial"),
            Document(id="old", filename="old.pdf", file_type="pdf", file_path=str(tmp_path / "old.pdf")),
        ])
        db.commit()
    app = FastAPI()
    app.include_router(documents.router)

    def session():
        with Session(engine) as db:
            yield db

    app.dependency_overrides[get_db] = session
    with TestClient(app) as client:
        response = client.get("/api/documents")
        assert response.status_code == 200
        items = {item["id"]: item for item in response.json()}
        assert items["new"]["processing_warnings"][0].startswith("第 2 页")
        assert items["old"]["processing_warnings"] == []
        assert client.get("/api/documents/new/processing").json()["pages"][0]["page"] == 2
        assert client.get("/api/documents/old/processing").status_code == 404
        assert client.get("/api/documents/missing/processing").status_code == 404
