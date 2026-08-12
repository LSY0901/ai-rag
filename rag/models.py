"""全链路共享的 Pydantic 数据契约。"""
from typing import Optional

from pydantic import BaseModel


class Chunk(BaseModel):
    """文档分块。dense/sparse 在向量化后填入。"""

    content: str
    source: str
    dense: Optional[list[float]] = None
    sparse: Optional[dict[str, float]] = None


class SearchHit(BaseModel):
    content: str
    source: str
    score: float


# --- API 请求/响应 ---


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    score_threshold: float = 0.0


class SearchResponse(BaseModel):
    results: list[SearchHit]


class IngestResponse(BaseModel):
    filename: str
    chunks: int


class HealthResponse(BaseModel):
    status: str
    collection: str
    count: int


# --- Embedding API 请求/响应 (Spring AI / OpenAI 兼容) ---


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = "bge-m3"
    encoding_format: Optional[str] = "float"


class EmbeddingDataItem(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingDataItem]
    model: str = "bge-m3"
    usage: EmbeddingUsage = EmbeddingUsage()


# --- Rerank API 请求/响应 (Spring AI / TEI / Cohere 兼容) ---


class RerankRequest(BaseModel):
    model: str = "bge-reranker-v2-m3"
    query: str
    documents: list[str | dict]
    top_n: Optional[int] = None


class RerankResultItem(BaseModel):
    index: int
    relevance_score: float
    score: float
    document: Optional[dict] = None


class RerankResponse(BaseModel):
    model: str = "bge-reranker-v2-m3"
    results: list[RerankResultItem]

