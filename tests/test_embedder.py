from unittest.mock import MagicMock, patch

from rag.embedding.bge_embedder import BGEM3Embedder, EmbeddingResult


@patch("rag.embedding.bge_embedder.BGEM3FlagModel")
def test_encode_returns_dense_and_sparse(mock_model_cls):
    embedder = BGEM3Embedder(model_path="/fake/bge-m3")
    embedder._model = MagicMock()
    embedder._model.encode.return_value = {
        "dense_vecs": [[0.1] * 1024, [0.2] * 1024],
        "lexical_weights": [{"5": 0.3}, {"9": 0.1}],
    }
    result = embedder.encode(["文本一", "文本二"])
    assert isinstance(result, EmbeddingResult)
    assert len(result.dense) == 2
    assert len(result.dense[0]) == 1024
    assert result.sparse[0] == {"5": 0.3}


@patch("rag.embedding.bge_embedder.BGEM3FlagModel")
def test_lazy_load_called_once(mock_model_cls):
    embedder = BGEM3Embedder(model_path="/fake/bge-m3")
    embedder.encode(["x"])
    embedder.encode(["y"])
    assert mock_model_cls.call_count == 1
