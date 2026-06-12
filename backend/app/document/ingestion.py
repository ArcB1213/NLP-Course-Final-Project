from pathlib import Path

from sqlalchemy.orm import Session

from app.config import Settings
from app.db import crud
from app.document.loaders import get_loader
from app.document.splitter import ChineseTextSplitter
from app.retrieval.retriever import HybridRetriever


class IngestionService:
    def __init__(self, settings: Settings, retriever: HybridRetriever):
        self.settings = settings
        self.retriever = retriever

    def ingest_file(self, db: Session, *, document_id: str, file_path: Path, filename: str, file_type: str):
        document = crud.create_document(
            db,
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            file_path=str(file_path),
            title=Path(filename).stem,
        )

        try:
            loader = get_loader(file_type)
            pages = loader.load(str(file_path))
            splitter = ChineseTextSplitter(
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            chunks = splitter.split_pages(pages, document_id=document_id, source_file=filename)
            crud.create_chunks(db, chunks)
            crud.update_document_status(db, document_id, "indexed", len(chunks))
            self.retriever.rebuild(db)
        except Exception:
            crud.update_document_status(db, document_id, "failed", 0)
            raise

        db.refresh(document)
        return document
