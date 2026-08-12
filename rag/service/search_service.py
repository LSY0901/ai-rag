"""业务编排层：api 唯一依赖。编排 parse→chunk→embed→insert 和 retrieve→rerank。"""
from rag.models import SearchHit


class SearchService:
    def __init__(self, parser, embedder, store, reranker, retriever=None):
        self.parser = parser
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.retriever = retriever

    def ingest(self, path: str, filename: str) -> int:
        chunks = self.parser.parse_and_chunk(path, source=filename)
        if not chunks:
            return 0
        emb = self.embedder.encode([c.content for c in chunks])
        for c, dense, sparse in zip(chunks, emb.dense, emb.sparse):
            c.dense = dense
            c.sparse = sparse
        self.store.delete_by_source(filename)
        self.store.insert(chunks)
        return len(chunks)

    def search(self, query: str, top_k: int, score_threshold: float = 0.0) -> list[SearchHit]:
        if self.retriever is None:
            from rag.retrieval.retriever import HybridRetriever

            self.retriever = HybridRetriever(self.embedder, self.store)
        candidates = self.retriever.retrieve(query, top_k)
        hits = self.reranker.rerank(query, candidates, top_k)
        if score_threshold > 0.0:
            hits = [h for h in hits if h.score >= score_threshold]
        return hits
