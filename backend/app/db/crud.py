from datetime import datetime
from uuid import uuid4

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core.schemas import ChunkCreate
from app.db.models import ChatMessage, Chunk, ConversationSession, ConversationSummary, Document, MemoryRecord


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
    db.execute(delete(Chunk).where(Chunk.document_id == document_id))
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
    return list(
        db.scalars(
            select(Chunk)
            .join(Document, Chunk.document_id == Document.id)
            .order_by(Chunk.document_id, Chunk.chunk_index)
        ).all()
    )


def get_chunks_by_ids(db: Session, chunk_ids: list[str]) -> list[Chunk]:
    if not chunk_ids:
        return []
    records = list(
        db.scalars(
            select(Chunk)
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.id.in_(chunk_ids))
        ).all()
    )
    by_id = {chunk.id: chunk for chunk in records}
    return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]


def delete_orphan_chunks(db: Session) -> int:
    document_ids = select(Document.id)
    result = db.execute(delete(Chunk).where(Chunk.document_id.not_in(document_ids)))
    db.commit()
    return result.rowcount or 0


def get_or_create_session(db: Session, session_id: str | None = None) -> ConversationSession:
    if session_id:
        session = db.get(ConversationSession, session_id)
        if session is not None:
            return session

    session = ConversationSession(id=session_id or uuid4().hex)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def touch_session(db: Session, session_id: str) -> None:
    session = db.get(ConversationSession, session_id)
    if session is None:
        return
    session.updated_at = datetime.utcnow()
    db.commit()


def create_chat_message(
    db: Session,
    *,
    session_id: str,
    role: str,
    content: str,
    task_type: str | None = None,
) -> ChatMessage:
    message = ChatMessage(
        id=uuid4().hex,
        session_id=session_id,
        role=role,
        content=content,
        task_type=task_type,
    )
    db.add(message)
    session = db.get(ConversationSession, session_id)
    if session is not None:
        session.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(message)
    return message


def list_recent_messages(db: Session, session_id: str, limit: int) -> list[ChatMessage]:
    rows = list(
        db.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        ).all()
    )
    return list(reversed(rows))


def count_session_messages(db: Session, session_id: str) -> int:
    return int(
        db.scalar(select(func.count()).select_from(ChatMessage).where(ChatMessage.session_id == session_id))
        or 0
    )


def get_session_messages_after(
    db: Session,
    session_id: str,
    last_message_id: str | None,
    limit: int = 24,
) -> list[ChatMessage]:
    statement = select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at)
    if last_message_id:
        last_message = db.get(ChatMessage, last_message_id)
        if last_message is not None:
            statement = statement.where(ChatMessage.created_at > last_message.created_at)
    return list(db.scalars(statement.limit(limit)).all())


def get_conversation_summary(db: Session, session_id: str) -> ConversationSummary | None:
    return db.get(ConversationSummary, session_id)


def upsert_conversation_summary(
    db: Session,
    *,
    session_id: str,
    content: str,
    last_message_id: str | None,
) -> ConversationSummary:
    summary = db.get(ConversationSummary, session_id)
    if summary is None:
        summary = ConversationSummary(
            session_id=session_id,
            content=content,
            last_message_id=last_message_id,
        )
        db.add(summary)
    else:
        summary.content = content
        summary.last_message_id = last_message_id
        summary.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(summary)
    return summary


def list_memory_records(db: Session) -> list[MemoryRecord]:
    return list(db.scalars(select(MemoryRecord).order_by(MemoryRecord.updated_at.desc())).all())


def create_memory_record(
    db: Session,
    *,
    kind: str,
    content: str,
    importance: int,
    source_session_id: str | None,
    source_message_id: str | None,
) -> MemoryRecord:
    record = MemoryRecord(
        id=uuid4().hex,
        kind=kind,
        content=content,
        importance=max(1, min(int(importance), 5)),
        source_session_id=source_session_id,
        source_message_id=source_message_id,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def find_memory_by_content(db: Session, content: str) -> MemoryRecord | None:
    text = content.strip()
    if not text:
        return None
    return db.scalar(select(MemoryRecord).where(MemoryRecord.content == text))


def get_memory_records_by_ids(db: Session, memory_ids: list[str]) -> list[MemoryRecord]:
    if not memory_ids:
        return []
    records = list(db.scalars(select(MemoryRecord).where(MemoryRecord.id.in_(memory_ids))).all())
    by_id = {record.id: record for record in records}
    return [by_id[memory_id] for memory_id in memory_ids if memory_id in by_id]


def mark_memories_used(db: Session, memory_ids: list[str]) -> None:
    if not memory_ids:
        return
    now = datetime.utcnow()
    for record in get_memory_records_by_ids(db, memory_ids):
        record.last_used_at = now
    db.commit()


def delete_memory_record(db: Session, memory_id: str) -> bool:
    record = db.get(MemoryRecord, memory_id)
    if record is None:
        return False
    db.delete(record)
    db.commit()
    return True


def delete_all_memory_records(db: Session) -> int:
    result = db.execute(delete(MemoryRecord))
    db.commit()
    return result.rowcount or 0


def delete_session(db: Session, session_id: str) -> bool:
    session = db.get(ConversationSession, session_id)
    if session is None:
        return False
    db.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))
    db.execute(delete(ConversationSummary).where(ConversationSummary.session_id == session_id))
    db.delete(session)
    db.commit()
    return True
