from unittest.mock import MagicMock

from rag.models import SearchHit
from rag.retrieval.retriever import HybridRetriever


def test_retrieve_calls_store_and_returns_hits():
    embedder = MagicMock()
    embedder.encode.return_value = MagicMock(
        dense=[[0.1] * 1024], sparse=[{"5": 0.3}]
    )
    store = MagicMock()
    store.hybrid_search.return_value = [
        SearchHit(content="c", source="s.pdf", score=0.8)
    ]
    r = HybridRetriever(embedder=embedder, store=store, fetch_k=10)
    hits = r.retrieve(query="问题", top_k=5)
    assert len(hits) == 1
    assert hits[0].content == "c"
    store.hybrid_search.assert_called_once()
    _, kwargs = store.hybrid_search.call_args
    assert kwargs["limit"] == 10
