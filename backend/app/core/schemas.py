from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


TaskTypeValue = Literal["auto", "qa", "summary", "quiz", "grade"]
ConfidenceValue = Literal["high", "medium", "low"]


class RawPage(BaseModel):
    text: str
    page: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkCreate(BaseModel):
    document_id: str
    chunk_index: int
    text: str
    source_file: str
    page: int | None = None
    heading: str | None = None
    start_char: int | None = None
    end_char: int | None = None


class DocumentRead(BaseModel):
    id: str
    filename: str
    file_type: str
    status: str
    chunk_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class UploadResponse(BaseModel):
    documents: list[DocumentRead]


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    source_file: str
    page: int | None = None
    heading: str | None = None
    vector_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    final_score: float


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResponse(BaseModel):
    results: list[RetrievedChunk]


class ChatRequest(BaseModel):
    query: str
    task_type: TaskTypeValue | None = "qa"
    use_pro_model: bool = False
    top_k: int = 5
    extra_context: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    task_type: str
    answer: str
    sources: list[RetrievedChunk]
    confidence: ConfidenceValue
    message: str | None = None

