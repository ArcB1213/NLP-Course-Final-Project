from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.schemas import SearchRequest, SearchResponse
from app.db.database import get_db
from app.dependencies import get_vector_retriever


router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    results = get_vector_retriever().retrieve(db, request.query, request.top_k)
    return SearchResponse(results=results)

