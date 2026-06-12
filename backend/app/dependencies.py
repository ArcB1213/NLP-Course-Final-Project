from functools import lru_cache

from app.config import get_settings
from app.core.llm_client import DeepSeekClient
from app.document.ingestion import IngestionService
from app.retrieval.bm25_store import BM25Store
from app.retrieval.embedder import Embedder
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import HybridRetriever
from app.retrieval.vector_store import FaissVectorStore
from app.tools.qa_tool import QATool


@lru_cache
def get_embedder() -> Embedder:
    return Embedder(get_settings().embedding_model)


@lru_cache
def get_vector_store() -> FaissVectorStore:
    return FaissVectorStore(get_settings().index_dir)


@lru_cache
def get_bm25_store() -> BM25Store:
    return BM25Store(get_settings().index_dir)


@lru_cache
def get_reranker() -> Reranker | None:
    settings = get_settings()
    if not settings.enable_reranker:
        return None
    return Reranker(settings.reranker_model)


@lru_cache
def get_vector_retriever() -> HybridRetriever:
    settings = get_settings()
    return HybridRetriever(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        bm25_store=get_bm25_store(),
        reranker=get_reranker(),
        vector_top_k=settings.vector_top_k,
        bm25_top_k=settings.bm25_top_k,
        vector_weight=settings.vector_weight,
        bm25_weight=settings.bm25_weight,
        enable_reranker=settings.enable_reranker,
    )


@lru_cache
def get_llm_client() -> DeepSeekClient:
    settings = get_settings()
    return DeepSeekClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        default_model=settings.deepseek_model,
        timeout=settings.request_timeout,
    )


def get_ingestion_service() -> IngestionService:
    return IngestionService(get_settings(), get_vector_retriever())


def get_qa_tool() -> QATool:
    return QATool(get_vector_retriever(), get_llm_client(), get_settings())
