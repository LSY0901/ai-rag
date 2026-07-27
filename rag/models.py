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


class SearchResponse(BaseModel):
    results: list[SearchHit]


class IngestResponse(BaseModel):
    filename: str
    chunks: int


class HealthResponse(BaseModel):
    status: str
    collection: str
    count: int
