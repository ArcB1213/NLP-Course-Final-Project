from sentence_transformers import CrossEncoder

from app.core.schemas import RetrievedChunk


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)

        ranked = sorted(chunks, key=lambda chunk: chunk.rerank_score or 0.0, reverse=True)
        return ranked[:top_k]
