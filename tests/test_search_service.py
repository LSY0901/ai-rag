from unittest.mock import MagicMock

from rag.models import Chunk, SearchHit
from rag.service.search_service import SearchService


def test_ingest_chains_parse_embed_store():
    parser = MagicMock()
    embedder = MagicMock()
    store = MagicMock()
    svc = SearchService(
        parser=parser, embedder=embedder, store=store, reranker=MagicMock()
    )

    parser.parse_and_chunk.return_value = [
        Chunk(content="c1", source="d.pdf"),
        Chunk(content="c2", source="d.pdf"),
    ]
    embedder.encode.return_value = MagicMock(
        dense=[[0.1] * 1024, [0.2] * 1024],
        sparse=[{"5": 0.3}, {"9": 0.1}],
    )

    n = svc.ingest(path="/fake/d.pdf", filename="d.pdf")
    assert n == 2
    store.delete_by_source.assert_called_once_with("d.pdf")
    store.insert.assert_called_once()
    inserted = store.insert.call_args[0][0]
    assert inserted[0].dense is not None
    assert inserted[0].sparse is not None


def test_search_chains_retrieve_rerank():
    retriever = MagicMock()
    reranker = MagicMock()
    reranker.rerank.return_value = [
        SearchHit(content="答案", source="d.pdf", score=0.9)
    ]
    svc = SearchService(
        parser=MagicMock(),
        embedder=MagicMock(),
        store=MagicMock(),
        reranker=reranker,
        retriever=retriever,
    )
    hits = svc.search(query="问题", top_k=5)
    assert hits[0].content == "答案"
    reranker.rerank.assert_called_once()
