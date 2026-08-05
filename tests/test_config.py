from rag.config import Settings


def test_default_values():
    s = Settings()
    assert s.milvus_host == "localhost"
    assert s.milvus_port == 19530
    assert s.milvus_collection == "rag_docs"
    assert s.dense_dim == 1024
    assert s.embedding_model_path.endswith("bge-m3")
    assert s.reranker_model_path.endswith("bge-reranker-v2-m3")
    assert s.chunk_max_tokens == 450
    assert s.upload_dir == "data/uploads"


def test_env_override(monkeypatch):
    monkeypatch.setenv("MILVUS_PORT", "19531")
    monkeypatch.setenv("MILVUS_COLLECTION", "custom_docs")
    s = Settings()
    assert s.milvus_port == 19531
    assert s.milvus_collection == "custom_docs"
