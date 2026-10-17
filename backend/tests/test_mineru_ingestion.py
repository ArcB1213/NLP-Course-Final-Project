import json
import base64
from pathlib import Path
import subprocess
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api import documents
from app.config import Settings
from app.core.schemas import RawPage
from app.db.database import Base, get_db
from app.db.models import Chunk, Document
from app.dependencies import get_ingestion_service
from app.document import ingestion, mineru_parser
from app.document.ingestion import IngestionService
from app.document.splitter import ChineseTextSplitter


def test_mineru_subprocess_configuration_and_cache(tmp_path, monkeypatch):
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"test pdf bytes")
    python = tmp_path / "python.exe"
    python.touch()
    settings = Settings(_env_file=None, mineru_python=python, mineru_cache_dir=tmp_path / "cache")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        output = Path(command[-1])
        (output / "pages-text-v2.json").write_text(json.dumps([{"text": "识别内容", "page": 3, "metadata": {}}]), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mineru_parser.subprocess, "run", run)
    first = mineru_parser.load_mineru_pdf(str(source), settings)
    second = mineru_parser.load_mineru_pdf(str(source), settings)
    assert first == second and first[0].page == 3
    assert len(calls) == 1
    assert calls[0][1]["env"]["MINERU_MODEL_VLM_ENGINE"] == "llama-cpp"
    assert calls[0][1]["env"]["MINERU_MODEL_SOURCE"] == "local"
    assert calls[0][1]["env"]["MINERU_MODEL_VLM_SERVER_URL"] == ""
    assert calls[0][1]["timeout"] == settings.mineru_timeout
    source.write_bytes(b"changed PDF")
    mineru_parser.load_mineru_pdf(str(source), settings)
    assert len(calls) == 2


def test_mineru_failure_is_visible_and_not_cached(tmp_path, monkeypatch):
    source = tmp_path / "sample.pdf"
    source.touch()
    python = tmp_path / "python.exe"
    python.touch()
    settings = Settings(_env_file=None, mineru_python=python, mineru_cache_dir=tmp_path / "cache")
    monkeypatch.setattr(mineru_parser.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1))
    with pytest.raises(ValueError, match="MinerU 解析失败"):
        mineru_parser.load_mineru_pdf(str(source), settings)
    assert not list(settings.mineru_cache_dir.rglob("pages-text-v2.json"))


def test_reparse_replaces_stale_chunks_and_preserves_document_on_parse_failure(tmp_path, monkeypatch):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    retriever = SimpleNamespace(rebuild=lambda db: None)
    service = IngestionService(Settings(_env_file=None), retriever)
    pages = [RawPage(text="旧内容一。\n\n旧内容二。", page=2)]
    monkeypatch.setattr(ingestion, "get_loader", lambda kind, settings: SimpleNamespace(load=lambda path: pages))
    with Session(engine) as db:
        doc = service.ingest_file(db, document_id="doc", file_path=tmp_path / "source.pdf", filename="source.pdf", file_type="pdf")
        # Seed an additional stale chunk which must disappear on replacement.
        db.add(Chunk(id="stale", document_id="doc", chunk_index=99, text="stale", source_file="source.pdf"))
        db.commit()
        created = doc.created_at
        pages[:] = [RawPage(text="新表格内容", page=56, metadata={"warnings": ["recognized_math_table_requires_review"]})]
        service.reparse_document(db, document_id="doc")
        chunks = list(db.scalars(select(Chunk)).all())
        assert len(chunks) == 1 and chunks[0].page == 56
        assert "新表格内容" in chunks[0].text and "旧内容" not in chunks[0].text
        assert doc.created_at == created and doc.status == "indexed_partial" and doc.chunk_count == 1

        def fail(path):
            raise ValueError("parse failed")
        monkeypatch.setattr(ingestion, "get_loader", lambda *a: SimpleNamespace(load=fail))
        with pytest.raises(ValueError, match="parse failed"):
            service.reparse_document(db, document_id="doc")
        assert db.get(Chunk, chunks[0].id).text == chunks[0].text
        assert db.get(Document, "doc").chunk_count == 1


def test_html_table_is_not_split_inside_cells():
    table = '<table><tr><td rowspan="2">' + "cell" * 100 + '</td><td>2</td></tr></table>'
    assert ChineseTextSplitter(chunk_size=30)._split_text("Before\n\n" + table + "\n\nAfter") == ["Before", table, "After"]


def test_upload_and_reparse_route_use_configured_loader(tmp_path, monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    settings = Settings(_env_file=None, upload_dir=tmp_path)
    settings.upload_dir.mkdir(exist_ok=True)
    service = IngestionService(settings, SimpleNamespace(rebuild=lambda db: None))
    calls = []
    def loader(kind, config):
        calls.append((kind, config.pdf_parser))
        return SimpleNamespace(load=lambda path: [RawPage(text="MinerU upload test", page=1)])
    monkeypatch.setattr(ingestion, "get_loader", loader)
    monkeypatch.setattr(documents, "get_settings", lambda: settings)
    app = FastAPI()
    app.include_router(documents.router)
    def session():
        with Session(engine) as db:
            yield db
    app.dependency_overrides[get_db] = session
    app.dependency_overrides[get_ingestion_service] = lambda: service
    with TestClient(app) as client:
        response = client.post("/api/documents/upload", files={"files": ("sample.pdf", b"pdf", "application/pdf")})
        assert response.status_code == 200
        doc_id = response.json()["documents"][0]["id"]
        assert client.post(f"/api/documents/{doc_id}/reparse").status_code == 200
        assert client.post("/api/documents/missing/reparse").status_code == 404
    assert calls == [("pdf", "mineru"), ("pdf", "mineru")]


@pytest.mark.parametrize("image_field", ["image_base64", "image_path"])
def test_real_mineru_renderer_excludes_images_but_preserves_text_and_math(tmp_path, image_field):
    python = Settings(_env_file=None).mineru_python
    if not python.is_file():
        pytest.skip("Requires the isolated MinerU installation; no model inference is performed")
    import fitz
    image = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 2, 2), False)
    image.clear_with(255)
    payload = base64.b64encode(image.tobytes("png")).decode("ascii")
    image_value = "data:image/png;base64," + payload if image_field == "image_base64" else "images/figure.png"
    fixture = {"schema": "docvortex.middle", "schema_version": "2.0", "metadata": {
        "file_suffix": "pdf", "producer": {"name": "mineru", "version": "4.0.12"}},
        "pages": [{"page_idx": 0, "blocks": [
            {"type": "text", "index": 0, "content": [{"type": "text", "content": "正文 QLoRA"}]},
            {"type": "equation", "index": 1, "content": "x_{i+1}=y_i"},
            {"type": "image", "index": 2, "content": [{"type": "image_body", "index": 2, image_field: image_value, "content": ""}]},
        ]}], "is_full_document": True}
    for index, block in enumerate(fixture["pages"][0]["blocks"]):
        block["bbox"] = [0.1, index * 0.3, 0.9, index * 0.3 + 0.2]
        if block["type"] == "image":
            block["content"][0]["bbox"] = block["bbox"]
    (tmp_path / "middle_json.json").write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
    # A control run must reproduce the unsafe default rendering; otherwise the
    # fixture could pass without ever exercising an embedded image.
    control = subprocess.run([str(python), "-c", "import sys; from pathlib import Path; from mineru import ParseResult; text=ParseResult.from_json(Path(sys.argv[1]).read_text(encoding='utf-8')).markdown(); assert sys.argv[2] in text", str(tmp_path / "middle_json.json"), image_value], capture_output=True, text=True, encoding="utf-8")
    assert control.returncode == 0, control.stderr
    worker = Path(__file__).resolve().parents[1] / "mineru_worker.py"
    run = subprocess.run([str(python), str(worker), str(tmp_path / "unused.pdf"), str(tmp_path), "--render-only"], capture_output=True, text=True, encoding="utf-8")
    assert run.returncode == 0, run.stderr
    page = json.loads((tmp_path / "pages-text-v2.json").read_text(encoding="utf-8"))[0]
    assert "正文 QLoRA" in page["text"] and "x_{i+1}=y_i" in page["text"]
    assert "data:image" not in page["text"] and payload not in page["text"]
    assert "![" not in page["text"] and "images/figure" not in page["text"]
    assert "image_semantics_not_extracted" in page["metadata"]["warnings"]
