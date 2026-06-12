from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.schemas import DocumentRead, UploadResponse
from app.db import crud
from app.db.database import get_db
from app.dependencies import get_ingestion_service, get_vector_retriever
from app.document.ingestion import IngestionService
from app.utils.file_utils import get_file_type, new_document_id, save_upload_file


router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: Annotated[list[UploadFile], File(description="Upload PDF, Markdown, or TXT files")],
    db: Session = Depends(get_db),
    ingestion: IngestionService = Depends(get_ingestion_service),
) -> UploadResponse:
    settings = get_settings()
    documents = []
    for upload in files:
        filename = upload.filename or ""
        try:
            file_type = get_file_type(filename)
            document_id = new_document_id()
            file_path = await save_upload_file(upload, settings.upload_dir, document_id)
            document = ingestion.ingest_file(
                db,
                document_id=document_id,
                file_path=file_path,
                filename=filename,
                file_type=file_type,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        documents.append(document)
    return UploadResponse(documents=documents)


@router.get("", response_model=list[DocumentRead])
def list_documents(db: Session = Depends(get_db)) -> list[DocumentRead]:
    return crud.list_documents(db)


@router.delete("/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    deleted = crud.delete_document(db, document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    get_vector_retriever().rebuild(db)
    return {"status": "deleted"}
