from sqlalchemy.orm import Session

from app.core.schemas import RetrievedChunk
from app.db import crud
from app.db.models import Chunk
from app.retrieval.bm25_store import BM25Store
from app.retrieval.embedder import Embedder
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store import FaissVectorStore


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: FaissVectorStore,
        bm25_store: BM25Store,
        reranker: Reranker | None = None,
        vector_top_k: int = 10,
        bm25_top_k: int = 10,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        enable_reranker: bool = True,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.reranker = reranker
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.enable_reranker = enable_reranker

    def rebuild(self, db: Session) -> None:
        chunks = crud.list_chunks(db)
        if not chunks:
            self.vector_store.build([], None)
            self.bm25_store.build([], [])
            return
        embeddings = self.embedder.encode_texts([chunk.text for chunk in chunks])
        self.vector_store.build([chunk.id for chunk in chunks], embeddings)
        self.bm25_store.build([chunk.id for chunk in chunks], [chunk.text for chunk in chunks])

    def retrieve(self, db: Session, query: str, top_k: int) -> list[RetrievedChunk]:
        self._ensure_indexes(db)
        query_embedding = self.embedder.encode_query(query)
        vector_results = self.vector_store.search(query_embedding, self.vector_top_k)
        bm25_results = self.bm25_store.search(query, self.bm25_top_k)
        merged_scores = self._merge_scores(vector_results, bm25_results)
        chunk_ids = list(merged_scores.keys())
        chunks = crud.get_chunks_by_ids(db, chunk_ids)
        by_id = {chunk.id: chunk for chunk in chunks}

        results: list[RetrievedChunk] = []
        for chunk_id, scores in merged_scores.items():
            chunk = by_id.get(chunk_id)
            if chunk is None:
                continue
            results.append(self._to_retrieved_chunk(chunk, scores))

        ranked = sorted(results, key=lambda item: item.final_score, reverse=True)
        if self.enable_reranker and self.reranker is not None:
            return self.reranker.rerank(query, ranked, top_k)
        return ranked[:top_k]

    def _ensure_indexes(self, db: Session) -> None:
        chunks = crud.list_chunks(db)
        if not chunks:
            return
        if not self.bm25_store.path.exists():
            self.bm25_store.build([chunk.id for chunk in chunks], [chunk.text for chunk in chunks])

    def _merge_scores(
        self,
        vector_results: list[tuple[str, float]],
        bm25_results: list[tuple[str, float]],
    ) -> dict[str, dict[str, float | None]]:
        vector_norm = _normalize_scores(vector_results)
        bm25_norm = _normalize_scores(bm25_results)
        chunk_ids = list(dict.fromkeys([chunk_id for chunk_id, _ in vector_results + bm25_results]))

        merged: dict[str, dict[str, float | None]] = {}
        for chunk_id in chunk_ids:
            vector_score = next((score for cid, score in vector_results if cid == chunk_id), None)
            bm25_score = next((score for cid, score in bm25_results if cid == chunk_id), None)
            final_score = (
                self.vector_weight * vector_norm.get(chunk_id, 0.0)
                + self.bm25_weight * bm25_norm.get(chunk_id, 0.0)
            )
            merged[chunk_id] = {
                "vector_score": vector_score,
                "bm25_score": bm25_score,
                "final_score": final_score,
            }
        return merged

    def _to_retrieved_chunk(self, chunk: Chunk, scores: dict[str, float | None]) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            text=chunk.text,
            source_file=chunk.source_file,
            page=chunk.page,
            heading=chunk.heading,
            vector_score=scores["vector_score"],
            bm25_score=scores["bm25_score"],
            final_score=scores["final_score"] or 0.0,
        )


def _normalize_scores(results: list[tuple[str, float]]) -> dict[str, float]:
    if not results:
        return {}
    values = [score for _, score in results]
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return {chunk_id: 1.0 for chunk_id, _ in results}
    return {chunk_id: (score - min_score) / (max_score - min_score) for chunk_id, score in results}


def estimate_confidence(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "low"
    top_score = max(chunk.final_score for chunk in chunks)
    if top_score >= 0.75:
        return "high"
    if top_score >= 0.45:
        return "medium"
    return "low"
