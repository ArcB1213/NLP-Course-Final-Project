import json
from pathlib import Path

import faiss
import numpy as np


class FaissVectorStore:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.index_path = index_dir / "faiss.index"
        self.mapping_path = index_dir / "faiss_mapping.json"
        self.index: faiss.Index | None = None
        self.id_mapping: list[str] = []

    def build(self, chunk_ids: list[str], embeddings: np.ndarray | None) -> None:
        if embeddings is None or embeddings.size == 0:
            self.index = None
            self.id_mapping = []
            self.save()
            return
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        self.index = index
        self.id_mapping = list(chunk_ids)
        self.save()

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        self.load()
        if self.index is None or self.index.ntotal == 0:
            return []
        query = query_embedding.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))
        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.id_mapping):
                continue
            results.append((self.id_mapping[idx], float(score)))
        return results

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        elif self.index_path.exists():
            self.index_path.unlink()
        self.mapping_path.write_text(json.dumps(self.id_mapping, ensure_ascii=False), encoding="utf-8")

    def load(self) -> None:
        if self.index_path.exists() and self.mapping_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.id_mapping = json.loads(self.mapping_path.read_text(encoding="utf-8"))
