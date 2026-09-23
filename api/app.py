"""FastAPI：/health, /ingest, /search, /files。供 Spring AI 调用。"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api.dependencies import get_search_service, get_store, warmup
from api.embedding_app import create_embeddings
from api.rerank_app import rerank
from rag.chunking.parser import ALLOWED_EXTS
from rag.config import settings
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
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=422, detail=f"不支持的文件类型：{ext}，仅支持 pdf/docx/html")
    os.makedirs(settings.upload_dir, exist_ok=True)
    dest = os.path.join(settings.upload_dir, file.filename)
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
    base = (settings.public_base_url or "").rstrip("/")
    if base:
        for h in hits:
            if h.image_path:
                fname = h.image_path.replace("\\", "/").split("/")[-1]
                h.image_path = f"{base}/files/images/{fname}"
    return SearchResponse(results=hits)


@app.get("/files/{filename}")
def download_file(filename: str):
    """原文件下载（保真载荷之外的原字节通道）。"""
    return _serve_file("", filename)


@app.get("/files/images/{filename}")
def download_image(filename: str):
    """抽出的图片原图下载，供前端按 SearchHit.image_path 渲染。"""
    return _serve_file("images", filename)


def _serve_file(subdir: str, filename: str):
    safe = os.path.basename(filename or "")
    if not safe or safe != filename:
        raise HTTPException(status_code=422, detail="非法文件名")
    dest = os.path.join(settings.upload_dir, subdir, safe) if subdir else os.path.join(settings.upload_dir, safe)
    if not os.path.isfile(dest):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(dest, filename=safe)

