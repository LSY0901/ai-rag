from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.rerank_app import app
from rag.models import SearchHit

client = TestClient(app)


def test_rerank_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


@patch("api.rerank_app.get_reranker")
def test_rerank_documents_string_list(mock_get_reranker):
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        SearchHit(content="doc2", source="1", score=0.9),
        SearchHit(content="doc1", source="0", score=0.3),
    ]
    mock_get_reranker.return_value = mock_reranker

    res = client.post(
        "/rerank",
        json={
            "model": "bge-reranker-v2-m3",
            "query": "apple",
            "documents": ["doc1", "doc2"],
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert len(data["results"]) == 2
    assert data["results"][0]["index"] == 1
    assert data["results"][0]["relevance_score"] == 0.9
    assert data["results"][0]["document"]["text"] == "doc2"
    assert data["results"][1]["index"] == 0


@patch("api.rerank_app.get_reranker")
def test_rerank_documents_dict_list(mock_get_reranker):
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        SearchHit(content="banana", source="0", score=0.85),
    ]
    mock_get_reranker.return_value = mock_reranker

    res = client.post(
        "/v1/rerank",
        json={
            "model": "bge-reranker-v2-m3",
            "query": "fruit",
            "documents": [{"text": "banana"}],
            "top_n": 1,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["index"] == 0
    assert data["results"][0]["relevance_score"] == 0.85
    assert data["results"][0]["document"]["text"] == "banana"
