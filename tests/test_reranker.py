from unittest.mock import MagicMock, patch

from rag.models import SearchHit
from rag.retrieval.reranker import Reranker


@patch("rag.retrieval.reranker.FlagReranker")
def test_rerank_reorders_by_score(mock_cls):
    r = Reranker(model_path="/fake/reranker")
    r._model = MagicMock()
    r._model.compute_score.return_value = [0.1, 0.9]
    candidates = [
        SearchHit(content="低相关", source="a.pdf", score=0.5),
        SearchHit(content="高相关", source="b.pdf", score=0.5),
    ]
    out = r.rerank(query="问题", candidates=candidates, top_k=2)
    assert len(out) == 2
    assert out[0].content == "高相关"
    assert out[0].score >= out[1].score


@patch("rag.retrieval.reranker.FlagReranker")
def test_rerank_respects_top_k(mock_cls):
    r = Reranker(model_path="/fake/reranker")
    r._model = MagicMock()
    r._model.compute_score.return_value = [0.9, 0.1, 0.5]
    candidates = [
        SearchHit(content=f"c{i}", source=f"{i}.pdf", score=0.0) for i in range(3)
    ]
    out = r.rerank(query="q", candidates=candidates, top_k=2)
    assert len(out) == 2


@patch("rag.retrieval.reranker.FlagReranker")
def test_rerank_normalizes_scores_to_unit_interval(mock_cls):
    r = Reranker(model_path="/fake/reranker")
    r._model = MagicMock()
    r._model.compute_score.return_value = [0.8, 0.2]
    candidates = [
        SearchHit(content="a", source="a.pdf", score=0.0),
        SearchHit(content="b", source="b.pdf", score=0.0),
    ]
    r.rerank(query="q", candidates=candidates, top_k=2)
    _, kwargs = r._model.compute_score.call_args
    assert kwargs.get("normalize") is True
