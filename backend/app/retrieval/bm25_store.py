import pickle
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi


def tokenize_zh(text: str) -> list[str]:
    return [token.strip() for token in jieba.cut(text) if token.strip()]


class BM25Store:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.path = index_dir / "bm25.pkl"
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []
        self.corpus_tokens: list[list[str]] = []

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        self.chunk_ids = list(chunk_ids)
        self.corpus_tokens = [tokenize_zh(text) for text in texts]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None
        self.save()

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        self.load()
        if self.bm25 is None or not self.chunk_ids:
            return []

        scores = self.bm25.get_scores(tokenize_zh(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        results: list[tuple[str, float]] = []
        for index, score in ranked[:top_k]:
            if score <= 0:
                continue
            results.append((self.chunk_ids[index], float(score)))
        return results

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunk_ids": self.chunk_ids,
            "corpus_tokens": self.corpus_tokens,
        }
        with self.path.open("wb") as file:
            pickle.dump(payload, file)

    def load(self) -> None:
        if self.bm25 is not None or not self.path.exists():
            return
        with self.path.open("rb") as file:
            payload = pickle.load(file)
        self.chunk_ids = payload.get("chunk_ids", [])
        self.corpus_tokens = payload.get("corpus_tokens", [])
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

