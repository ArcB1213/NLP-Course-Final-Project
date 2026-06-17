from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


TaskTypeValue = Literal["auto", "qa", "summary", "quiz", "grade"]
ConfidenceValue = Literal["high", "medium", "low"]
RetrievalProfileValue = Literal["default", "qa", "summary"]


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
    profile: str = "default"


class SearchResponse(BaseModel):
    results: list[RetrievedChunk]


class ChatRequest(BaseModel):
    query: str
    task_type: TaskTypeValue | None = "qa"
    use_pro_model: bool = False
    top_k: int | None = None
    session_id: str | None = None
    extra_context: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    task_type: str
    answer: str
    sources: list[RetrievedChunk]
    confidence: ConfidenceValue
    message: str | None = None
    session_id: str | None = None
    agent_trace: "AgentTrace | None" = None


class MemoryRecordRead(BaseModel):
    id: str
    kind: str
    content: str
    importance: int
    source_session_id: str | None = None
    created_at: datetime
    updated_at: datetime
    last_used_at: datetime | None = None

    model_config = {"from_attributes": True}


class MemoryListResponse(BaseModel):
    memories: list[MemoryRecordRead]


class ClearResponse(BaseModel):
    deleted: int


class AgentDecision(BaseModel):
    task_type: Literal["qa", "summary", "quiz", "grade"]
    retrieval_profile: RetrievalProfileValue = "qa"
    rewritten_query: str | None = None
    needs_pro_model: bool = False
    confidence: ConfidenceValue = "medium"
    reason: str | None = None


class AgentSubtask(BaseModel):
    task_type: Literal["qa", "summary", "quiz", "grade"]
    query: str
    retrieval_profile: RetrievalProfileValue = "qa"
    rewritten_query: str | None = None
    reason: str | None = None


class EvidenceDecision(BaseModel):
    is_sufficient: bool
    reason: str
    suggested_queries: list[str] = Field(default_factory=list)
    suggested_profile: RetrievalProfileValue | None = None
    should_refuse: bool = False


class SummaryCoverageDecision(BaseModel):
    is_sufficient: bool
    reason: str
    covered_topics: list[str] = Field(default_factory=list)
    missing_topics: list[str] = Field(default_factory=list)
    next_queries: list[str] = Field(default_factory=list)
    should_refuse: bool = False


class AgentStep(BaseModel):
    step_type: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "failed"] = "ok"
    message: str | None = None


class AgentTrace(BaseModel):
    steps: list[AgentStep] = Field(default_factory=list)


class AgentState(BaseModel):
    original_query: str
    task_type: str
    retrieval_profile: RetrievalProfileValue
    use_pro_model: bool
    subtasks: list[AgentSubtask] = Field(default_factory=list)
    final_sources: list[RetrievedChunk] = Field(default_factory=list)
    trace: AgentTrace = Field(default_factory=AgentTrace)
