from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.embedding_app import app
from rag.embedding.bge_embedder import EmbeddingResult

client = TestClient(app)


def test_embedding_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


@patch("api.embedding_app.get_embedder")
def test_create_embeddings_single_string(mock_get_embedder):
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = EmbeddingResult(
        dense=[[0.1, 0.2, 0.3]],
        sparse=[{"word": 0.5}],
    )
    mock_get_embedder.return_value = mock_embedder

    res = client.post(
        "/v1/embeddings",
        json={"model": "bge-m3", "input": "test query"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert data["data"][0]["index"] == 0


@patch("api.embedding_app.get_embedder")
def test_create_embeddings_batch_list(mock_get_embedder):
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = EmbeddingResult(
        dense=[[0.1, 0.2], [0.3, 0.4]],
        sparse=[{}, {}],
    )
    mock_get_embedder.return_value = mock_embedder

    res = client.post(
        "/embeddings",
        json={"model": "bge-m3", "input": ["query 1", "query 2"]},
    )
    assert res.status_code == 200
    data = res.json()
    assert len(data["data"]) == 2
    assert data["data"][0]["index"] == 0
    assert data["data"][1]["index"] == 1
