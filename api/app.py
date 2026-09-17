"""FastAPI：/health, /ingest, /search。供 Spring AI 调用。"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from api.dependencies import get_search_service, get_store, warmup
from api.embedding_app import create_embeddings
from api.rerank_app import rerank
from rag.exceptions import EmbedError, ParseError, StoreError
from rag.models import (
    EmbeddingResponse,
    HealthResponse,
    IngestResponse,
    RerankResponse,
    SearchRequest,
    SearchResponse,
)

@asynccontextmanager
async def _lifespan(app: FastAPI):
    warmup()  # ponytail: 启动时多几十秒，首查省十几秒；常驻服务只付一次
    yield


app = FastAPI(title="RAG Retrieval Service", lifespan=_lifespan)


# 注册单独的 embedding 与 rerank 路由，使主应用 8000 端口同样支持直接调用
app.post("/v1/embeddings", response_model=EmbeddingResponse)(create_embeddings)
app.post("/embeddings", response_model=EmbeddingResponse)(create_embeddings)
app.post("/api/embeddings", response_model=EmbeddingResponse)(create_embeddings)
app.post("/rerank", response_model=RerankResponse)(rerank)
app.post("/v1/rerank", response_model=RerankResponse)(rerank)
app.post("/api/v1/rerank", response_model=RerankResponse)(rerank)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    store = get_store()
    try:
        count = store.count()
    except StoreError:
        count = -1
    return HealthResponse(status="ok", collection=store.collection, count=count)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    strategy: str = Form("hybrid")
) -> IngestResponse:
    os.makedirs("data/uploads", exist_ok=True)
    dest = os.path.join("data/uploads", file.filename)
    with open(dest, "wb") as f:
        f.write(await file.read())
    try:
        n = get_search_service().ingest(path=dest, filename=file.filename, strategy=strategy)
    except ParseError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except (StoreError, EmbedError) as e:
        raise HTTPException(status_code=503, detail=str(e))
    return IngestResponse(filename=file.filename, chunks=n)


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    try:
        hits = get_search_service().search(
            query=req.query, top_k=req.top_k, score_threshold=req.score_threshold
        )
    except (StoreError, EmbedError) as e:
        raise HTTPException(status_code=503, detail=str(e))
    return SearchResponse(results=hits)

