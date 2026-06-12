from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.schemas import ChunkCreate
from app.db.models import Chunk, Document


def create_document(
    db: Session,
    *,
    document_id: str,
    filename: str,
    file_type: str,
    file_path: str,
    title: str | None = None,
) -> Document:
    document = Document(
        id=document_id,
        filename=filename,
        file_type=file_type,
        file_path=file_path,
        title=title,
        status="uploaded",
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


def update_document_status(db: Session, document_id: str, status: str, chunk_count: int | None = None) -> None:
    document = db.get(Document, document_id)
    if document is None:
        return
    document.status = status
    if chunk_count is not None:
        document.chunk_count = chunk_count
    db.commit()


def list_documents(db: Session) -> list[Document]:
    return list(db.scalars(select(Document).order_by(Document.created_at.desc())).all())


def delete_document(db: Session, document_id: str) -> bool:
    document = db.get(Document, document_id)
    if document is None:
        return False
    db.delete(document)
    db.commit()
    return True


def create_chunks(db: Session, chunks: list[ChunkCreate]) -> list[Chunk]:
    records = [
        Chunk(
            id=f"{chunk.document_id}_{chunk.chunk_index}",
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            source_file=chunk.source_file,
            page=chunk.page,
            heading=chunk.heading,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
        )
        for chunk in chunks
    ]
    db.add_all(records)
    db.commit()
    return records


def list_chunks(db: Session) -> list[Chunk]:
    return list(db.scalars(select(Chunk).order_by(Chunk.document_id, Chunk.chunk_index)).all())


def get_chunks_by_ids(db: Session, chunk_ids: list[str]) -> list[Chunk]:
    if not chunk_ids:
        return []
    records = list(db.scalars(select(Chunk).where(Chunk.id.in_(chunk_ids))).all())
    by_id = {chunk.id: chunk for chunk in records}
    return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

