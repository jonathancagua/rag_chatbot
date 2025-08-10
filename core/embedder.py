from __future__ import annotations
from typing import List
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(texts)
        return [v.tolist() for v in vecs]
