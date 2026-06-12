import logging

from sentence_transformers import CrossEncoder

from app.core.schemas import RetrievedChunk


logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model: CrossEncoder | None = None
        self._disabled = False

    def _ensure_model(self) -> CrossEncoder | None:
        if self._disabled:
            return None
        if self._model is None:
            try:
                self._model = CrossEncoder(self.model_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Reranker model %s unavailable, fall back to hybrid score: %s",
                    self.model_name,
                    exc,
                )
                self._disabled = True
                return None
        return self._model

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []

        model = self._ensure_model()
        if model is None:
            return chunks[:top_k]

        try:
            pairs = [(query, chunk.text) for chunk in chunks]
            scores = model.predict(pairs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reranker.predict failed, fall back to hybrid score: %s", exc)
            self._disabled = True
            return chunks[:top_k]

        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)

        ranked = sorted(chunks, key=lambda chunk: chunk.rerank_score or 0.0, reverse=True)
        return ranked[:top_k]
