from __future__ import annotations
from typing import List, Dict, Any
import uuid
from pinecone import Pinecone, ServerlessSpec


class PineconeStore:
    def __init__(self, api_key: str, cloud: str, region: str, index_name: str):
        self.api_key = api_key
        self.cloud = cloud
        self.region = region
        self.index_name = index_name
        self.pc = Pinecone(api_key=self.api_key)
        self._index = None

    def _index_exists(self, name: str) -> bool:
        try:
            listings = self.pc.list_indexes()
            # Compat: may be an object with .names() or a list of dicts/objects
            if hasattr(listings, "names") and callable(listings.names):
                return name in listings.names()
            names = []
            for it in listings:
                if isinstance(it, dict) and "name" in it:
                    names.append(it["name"])
                elif hasattr(it, "name"):
                    names.append(it.name)
            return name in names
        except Exception:
            return False

    def ensure_index(self, dimension: int, metric: str = "cosine"):
        # Serverless: no pods. Define cloud/region.
        if not self._index_exists(self.index_name):
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
        self._index = self.pc.Index(self.index_name)

    @property
    def index(self):
        if self._index is None:
            self._index = self.pc.Index(self.index_name)
        return self._index

    def upsert_chunks(self, chunks: List[str], base_meta: Dict[str, Any], embedder) -> int:
        if not chunks:
            return 0
        embs = embedder.embed_many(chunks)
        doc_id = base_meta.get("doc_id") or str(uuid.uuid4())
        to_upsert = []
        for i, (chunk, vec) in enumerate(zip(chunks, embs)):
            meta = dict(base_meta or {})
            meta.update({"texto": chunk, "chunk_id": i})
            to_upsert.append((f"{doc_id}_{i:04d}", vec, meta))
        self.index.upsert(vectors=to_upsert)
        return len(to_upsert)

    def query(self, query_text: str, embedder, top_k: int = 4, metadata_filter: Dict[str, Any] | None = None):
        qvec = embedder.embed(query_text)
        res = self.index.query(
            vector=qvec,
            top_k=top_k,
            include_metadata=True,
            filter=metadata_filter,
        )
        # res may be dict-like or an object with .matches
        matches = res.get("matches") if isinstance(res, dict) else getattr(res, "matches", [])
        docs = []
        for m in matches:
            md = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
            docs.append({
                "id": m.get("id") if isinstance(m, dict) else getattr(m, "id", None),
                "score": m.get("score") if isinstance(m, dict) else getattr(m, "score", None),
                "texto": md.get("texto", ""),
                "metadata": md,
            })
        return docs
