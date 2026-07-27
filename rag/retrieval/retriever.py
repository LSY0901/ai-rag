"""混合检索：embed query → store.hybrid_search（dense+sparse, RRF 融合）。"""
from rag.models import SearchHit


class HybridRetriever:
    def __init__(self, embedder, store, fetch_k: int = 20):
        self.embedder = embedder
        self.store = store
        self.fetch_k = fetch_k

    def retrieve(self, query: str, top_k: int) -> list[SearchHit]:
        emb = self.embedder.encode([query])
        return self.store.hybrid_search(
            dense=emb.dense[0],
            sparse=emb.sparse[0],
            limit=max(self.fetch_k, top_k),
        )
