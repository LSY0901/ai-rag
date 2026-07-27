from unittest.mock import MagicMock, patch

from rag.models import Chunk
from rag.storage.milvus_store import MilvusStore


def _chunk(content="c", source="s.pdf", dense=None, sparse=None):
    return Chunk(
        content=content,
        source=source,
        dense=dense or [0.1] * 1024,
        sparse=sparse or {"5": 0.3},
    )


@patch("rag.storage.milvus_store.MilvusClient")
def test_ensure_collection_creates_and_loads(mock_client_cls):
    store = MilvusStore(
        uri="http://localhost:19530", collection="rag_docs", dense_dim=1024
    )
    store.client = MagicMock()
    store.client.has_collection.return_value = False
    store.ensure_collection()
    store.client.create_collection.assert_called_once()
    store.client.load_collection.assert_called_once()


@patch("rag.storage.milvus_store.MilvusClient")
def test_insert_maps_chunks_to_rows(mock_client_cls):
    store = MilvusStore(
        uri="http://localhost:19530", collection="rag_docs", dense_dim=1024
    )
    store.client = MagicMock()
    store.insert([_chunk(), _chunk(content="d")])
    args, kwargs = store.client.insert.call_args
    rows = kwargs["data"]
    assert len(rows) == 2
    assert set(rows[0].keys()) == {"content", "source", "dense", "sparse"}


@patch("rag.storage.milvus_store.MilvusClient")
def test_delete_by_source(mock_client_cls):
    store = MilvusStore(
        uri="http://localhost:19530", collection="rag_docs", dense_dim=1024
    )
    store.client = MagicMock()
    store.delete_by_source("doc.pdf")
    store.client.delete.assert_called_once()
    args, kwargs = store.client.delete.call_args
    assert "source" in kwargs["filter"]


@patch("rag.storage.milvus_store.MilvusClient")
def test_hybrid_search_returns_chunks(mock_client_cls):
    store = MilvusStore(
        uri="http://localhost:19530", collection="rag_docs", dense_dim=1024
    )
    store.client = MagicMock()
    store.client.hybrid_search.return_value = [
        [
            {
                "id": 1,
                "entity": {"content": "命中", "source": "doc.pdf"},
                "distance": 0.9,
            }
        ]
    ]
    hits = store.hybrid_search(dense=[0.1] * 1024, sparse={"5": 0.3}, limit=3)
    assert len(hits) == 1
    assert hits[0].content == "命中"
    assert hits[0].source == "doc.pdf"
