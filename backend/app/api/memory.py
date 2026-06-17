from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.schemas import ClearResponse, MemoryListResponse
from app.db import crud
from app.db.database import get_db
from app.dependencies import get_memory_service


router = APIRouter(prefix="/api", tags=["memory"])


@router.get("/memory", response_model=MemoryListResponse)
def list_memory(db: Session = Depends(get_db)) -> MemoryListResponse:
    return MemoryListResponse(memories=crud.list_memory_records(db))


@router.delete("/memory/{memory_id}", response_model=ClearResponse)
def delete_memory(memory_id: str, db: Session = Depends(get_db)) -> ClearResponse:
    deleted = crud.delete_memory_record(db, memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="memory not found")
    get_memory_service().rebuild_memory_index(db)
    return ClearResponse(deleted=1)


@router.delete("/memory", response_model=ClearResponse)
def clear_memory(db: Session = Depends(get_db)) -> ClearResponse:
    deleted = crud.delete_all_memory_records(db)
    get_memory_service().rebuild_memory_index(db)
    return ClearResponse(deleted=deleted)


@router.delete("/sessions/{session_id}", response_model=ClearResponse)
def clear_session(session_id: str, db: Session = Depends(get_db)) -> ClearResponse:
    deleted = crud.delete_session(db, session_id)
    return ClearResponse(deleted=1 if deleted else 0)
