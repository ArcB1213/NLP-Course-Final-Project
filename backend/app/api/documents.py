from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.core.schemas import DocumentRead, UploadResponse
from app.db import crud
from app.db.database import get_db
from app.db.models import Document
from app.dependencies import get_ingestion_service, get_vector_retriever
from app.document.ingestion import IngestionService
from app.document.processing_report import read_processing_report
from app.utils.file_utils import get_file_type, new_document_id, save_upload_file


router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: Annotated[list[UploadFile], File(description="Upload PDF, Markdown, or TXT files")],
    db: Session = Depends(get_db),
    ingestion: IngestionService = Depends(get_ingestion_service),
) -> UploadResponse:
    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    documents = []
    for upload in files:
        filename = upload.filename or ""
        try:
            file_type = get_file_type(filename)
            if upload.size is not None and upload.size > max_bytes:
                size_mb = upload.size / 1024 / 1024
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"{filename}: {size_mb:.1f} MB exceeds the "
                        f"{settings.max_upload_size_mb} MB upload limit"
                    ),
                )
            document_id = new_document_id()
            file_path = await save_upload_file(upload, settings.upload_dir, document_id)
            document = await run_in_threadpool(ingestion.ingest_file,
                db,
                document_id=document_id,
                file_path=file_path,
                filename=filename,
                file_type=file_type,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        documents.append(_document_read(document))
    return UploadResponse(documents=documents)


@router.post("/{document_id}/reparse", response_model=DocumentRead)
def reparse_document(document_id: str, db: Session = Depends(get_db),
                     ingestion: IngestionService = Depends(get_ingestion_service)) -> DocumentRead:
    if db.get(Document, document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        return _document_read(ingestion.reparse_document(db, document_id=document_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("", response_model=list[DocumentRead])
def list_documents(db: Session = Depends(get_db)) -> list[DocumentRead]:
    return [_document_read(document) for document in crud.list_documents(db)]


@router.get("/{document_id}/processing")
def document_processing(document_id: str, db: Session = Depends(get_db)) -> dict:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    report = read_processing_report(document.file_path)
    if report is None:
        raise HTTPException(status_code=404, detail="Processing report not found; re-upload the PDF to generate it")
    return report


def _document_read(document) -> DocumentRead:
    response = DocumentRead.model_validate(document)
    report = read_processing_report(document.file_path)
    if report:
        response.processing_warnings = report["warnings"]
    return response


@router.delete("/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    deleted = crud.delete_document(db, document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    get_vector_retriever().rebuild(db)
    return {"status": "deleted"}
