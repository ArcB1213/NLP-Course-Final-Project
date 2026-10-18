from pathlib import Path
from threading import Lock

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import crud
from app.db.models import Chunk, Document
from app.document.loaders import get_loader
from app.document.processing_report import save_processing_report
from app.document.splitter import ChineseTextSplitter
from app.retrieval.retriever import HybridRetriever


_ingestion_lock = Lock()


class IngestionService:
    def __init__(self, settings: Settings, retriever: HybridRetriever):
        self.settings = settings
        self.retriever = retriever

    def ingest_file(self, db: Session, *, document_id: str, file_path: Path, filename: str, file_type: str):
        crud.create_document(
            db,
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            file_path=str(file_path),
            title=Path(filename).stem,
        )

        try:
            return self.reparse_document(db, document_id=document_id)
        except Exception:
            db.rollback()
            crud.update_document_status(db, document_id, "failed", 0)
            raise

    def reparse_document(self, db: Session, *, document_id: str, rebuild: bool = True):
        with _ingestion_lock:
            return self._reparse_document(db, document_id=document_id, rebuild=rebuild)

    def _reparse_document(self, db: Session, *, document_id: str, rebuild: bool):
        document = db.get(Document, document_id)
        if document is None:
            raise ValueError("Document not found")
        file_path = Path(document.file_path)
        pages = get_loader(document.file_type, self.settings).load(str(file_path))
        failed_titles = [page.page for page in pages
                         if page.metadata.get("title_resolution", {}).get("status") == "failed"]
        if document.chunk_count and failed_titles:
            raise ValueError(f"第 {failed_titles} 页标题处理失败，已保留原有分块和解析报告；请检查标题诊断后重试。")
        report = save_processing_report(file_path, pages) if document.file_type == "pdf" else None
        splitter = ChineseTextSplitter(self.settings.chunk_size, self.settings.chunk_overlap)
        chunks = splitter.split_pages(pages, document_id=document_id, source_file=document.filename)
        if not chunks:
            raise ValueError("未提取到可索引内容；扫描或图片资料需要先进行视觉识别或 OCR 处理。")
        try:
            db.execute(delete(Chunk).where(Chunk.document_id == document_id))
            crud.create_chunks(db, chunks, commit=False)
            document.chunk_count = len(chunks)
            document.status = "indexed_partial" if report and report["warnings"] else "indexed"
            db.commit()
        except Exception:
            db.rollback()
            raise
        if rebuild:
            self.retriever.rebuild(db)
        return document
