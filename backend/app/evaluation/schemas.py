from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Label = Literal["irrelevant", "partial", "sufficient", "uncertain"]


class DatasetRequest(BaseModel):
    name: str = Field(default="课程检索评测", min_length=1, max_length=100)
    count: int = Field(default=20, ge=1, le=200)
    top_k: int = Field(default=5, ge=1, le=20)
    profile: Literal["qa", "summary", "default"] = "qa"
    seed: int = Field(default=42, ge=0, le=2147483647)
    judge_passes: int = Field(default=2, ge=1, le=2)
    generation_model: str = Field(min_length=1, max_length=200)
    judge_model: str = Field(min_length=1, max_length=200)
    document_ids: list[str] = Field(default_factory=list)


class AppendRequest(BaseModel):
    count: int = Field(ge=1, le=200, strict=True)


class Quote(BaseModel):
    chunk_id: str
    quote: str = Field(min_length=1)


class GeneratedQuestion(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    key_points: list[str] = Field(min_length=1, max_length=10)
    evidence: list[Quote] = Field(min_length=1, max_length=8)


class Judgment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: Label
    reason: str = Field(min_length=1, max_length=3000)
    evidence: list[Quote] = Field(default_factory=list, max_length=20)


class ChunkJudgment(Judgment):
    chunk_id: str


class Assessment(BaseModel):
    chunks: list[ChunkJudgment]
    overall: Judgment


class Review(BaseModel):
    status: Literal["accepted", "corrected", "skipped", "excluded"]
    overall: Label | None = None
    chunks: dict[str, Label] = Field(default_factory=dict)
    note: str = Field(default="", max_length=3000)
