import numpy as np
from sentence_transformers import SentenceTransformer


BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

    def encode_query(self, query: str) -> np.ndarray:
        embedding = self.encode_texts([f"{BGE_QUERY_PREFIX}{query}"])
        return embedding[0]

