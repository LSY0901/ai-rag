from rag.models import (
    Chunk,
    IngestResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)


def test_chunk_creation():
    c = Chunk(content="一段文本", source="doc.pdf")
    assert c.content == "一段文本"
    assert c.source == "doc.pdf"
    assert c.dense is None
    assert c.sparse is None


def test_chunk_with_vectors():
    c = Chunk(content="x", source="d.pdf", dense=[0.1] * 1024, sparse={"5": 0.3})
    assert len(c.dense) == 1024
    assert c.sparse == {"5": 0.3}


def test_search_request_defaults():
    req = SearchRequest(query="问题")
    assert req.top_k == 5


def test_search_hit_and_response():
    hit = SearchHit(content="c", source="s.pdf", score=0.9)
    resp = SearchResponse(results=[hit])
    assert resp.results[0].score == 0.9


def test_ingest_response():
    r = IngestResponse(filename="f.pdf", chunks=3)
    assert r.chunks == 3
