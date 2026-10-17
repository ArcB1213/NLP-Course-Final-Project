from functools import lru_cache
from typing import Literal
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "CourseRAGAgent"
    app_env: str = "dev"

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_pro_model: str = "deepseek-v4-pro"
    deepseek_thinking: bool = False

    eval_generation_model: str = "deepseek-v4-flash"
    eval_judge_model: str = "deepseek-v4-pro"
    eval_generation_thinking: bool = False
    eval_judge_thinking: bool = False
    eval_generation_base_url: str = ""
    eval_judge_base_url: str = ""
    eval_generation_api_key: str = ""
    eval_judge_api_key: str = ""
    eval_dir: Path = Path("./data/evaluations")

    embedding_model: str = "BAAI/bge-base-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    upload_dir: Path = Path("./data/uploads")
    index_dir: Path = Path("./data/indexes")
    memory_index_dir: Path = Path("./data/memory_indexes")
    sqlite_path: Path = Path("./data/sqlite/app.db")

    chunk_size: int = 500
    pdf_parser: Literal["mineru", "native"] = "mineru"
    mineru_python: Path = Path(__file__).resolve().parents[2] / "data/tools/mineru/venv/Scripts/python.exe"
    mineru_home: Path = Path(__file__).resolve().parents[2] / "data/tools/mineru"
    mineru_cache_dir: Path = Path(__file__).resolve().parents[2] / "data/parsed/mineru-upload"
    mineru_timeout: int = 1800
    chunk_overlap: int = 80
    vector_top_k: int = 10
    bm25_top_k: int = 10
    final_top_k: int = 5
    summary_final_top_k: int = 10
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    enable_reranker: bool = True

    request_timeout: float = 60.0
    max_upload_size_mb: int = 100
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    memory_enabled: bool = True
    memory_recent_turns: int = 8
    memory_summary_threshold: int = 12
    memory_recall_top_k: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def sqlite_url(self) -> str:
        return f"sqlite:///{self.sqlite_path.as_posix()}"

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.memory_index_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
