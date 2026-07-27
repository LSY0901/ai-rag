"""FastAPI：/health, /ingest, /search。供 Spring AI 调用。"""
import os

from fastapi import FastAPI, File, HTTPException, UploadFile

from api.dependencies import get_search_service, get_store
from rag.exceptions import EmbedError, ParseError, StoreError
from rag.models import HealthResponse, IngestResponse, SearchRequest, SearchResponse

app = FastAPI(title="RAG Retrieval Service")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    store = get_store()
    try:
        count = store.count()
    except StoreError:
        count = -1
    return HealthResponse(status="ok", collection=store.collection, count=count)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    os.makedirs("data/uploads", exist_ok=True)
    dest = os.path.join("data/uploads", file.filename)
    with open(dest, "wb") as f:
        f.write(await file.read())
    try:
        n = get_search_service().ingest(path=dest, filename=file.filename)
    except ParseError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except (StoreError, EmbedError) as e:
        raise HTTPException(status_code=503, detail=str(e))
    return IngestResponse(filename=file.filename, chunks=n)


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    try:
        hits = get_search_service().search(query=req.query, top_k=req.top_k)
    except (StoreError, EmbedError) as e:
        raise HTTPException(status_code=503, detail=str(e))
    return SearchResponse(results=hits)
