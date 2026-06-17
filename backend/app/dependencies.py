from functools import lru_cache

from app.config import get_settings
from app.core.agent_service import AgentService
from app.core.llm_client import DeepSeekClient
from app.core.memory_service import MemoryService
from app.core.router_agent import RouterAgent
from app.document.ingestion import IngestionService
from app.retrieval.bm25_store import BM25Store
from app.retrieval.embedder import Embedder
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import HybridRetriever
from app.retrieval.vector_store import FaissVectorStore
from app.tools.grading_tool import GradingTool
from app.tools.qa_tool import QATool
from app.tools.quiz_tool import QuizTool
from app.tools.summary_tool import SummaryTool


@lru_cache
def get_embedder() -> Embedder:
    return Embedder(get_settings().embedding_model)


@lru_cache
def get_vector_store() -> FaissVectorStore:
    return FaissVectorStore(get_settings().index_dir)


@lru_cache
def get_memory_vector_store() -> FaissVectorStore:
    return FaissVectorStore(get_settings().memory_index_dir)


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


@lru_cache
def get_memory_service() -> MemoryService:
    return MemoryService(
        settings=get_settings(),
        llm=get_llm_client(),
        embedder=get_embedder(),
        vector_store=get_memory_vector_store(),
    )


def get_ingestion_service() -> IngestionService:
    return IngestionService(get_settings(), get_vector_retriever())


def get_qa_tool() -> QATool:
    return QATool(get_vector_retriever(), get_llm_client(), get_settings())


def get_summary_tool() -> SummaryTool:
    return SummaryTool(get_vector_retriever(), get_llm_client(), get_settings())


def get_quiz_tool() -> QuizTool:
    return QuizTool(get_vector_retriever(), get_llm_client(), get_settings())


def get_grading_tool() -> GradingTool:
    return GradingTool(get_vector_retriever(), get_llm_client(), get_settings())


def get_agent_service() -> AgentService:
    return AgentService(
        router=RouterAgent(get_llm_client()),
        qa_tool=get_qa_tool(),
        summary_tool=get_summary_tool(),
        quiz_tool=get_quiz_tool(),
        grading_tool=get_grading_tool(),
        memory_service=get_memory_service(),
    )
