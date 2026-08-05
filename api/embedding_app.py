"""FastAPI Embedding 服务：运行在端口 8082，供 Spring AI 访问。

支持标准的 POST /v1/embeddings、/embeddings、/api/embeddings 接口。
"""
from fastapi import FastAPI, HTTPException

from api.dependencies import get_embedder
from rag.exceptions import EmbedError
from rag.models import (
    EmbeddingDataItem,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)

app = FastAPI(title="Embedding Service (Spring AI Compatible)")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "embedding"}


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
@app.post("/embeddings", response_model=EmbeddingResponse)
@app.post("/api/embeddings", response_model=EmbeddingResponse)
def create_embeddings(req: EmbeddingRequest) -> EmbeddingResponse:
    texts = [req.input] if isinstance(req.input, str) else req.input
    if not texts:
        return EmbeddingResponse(
            data=[],
            model=req.model,
            usage=EmbeddingUsage(prompt_tokens=0, total_tokens=0),
        )

    try:
        embedder = get_embedder()
        res = embedder.encode(texts)
    except EmbedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    data_items = [
        EmbeddingDataItem(object="embedding", embedding=vec, index=idx)
        for idx, vec in enumerate(res.dense)
    ]

    total_length = sum(len(t) for t in texts)
    return EmbeddingResponse(
        data=data_items,
        model=req.model,
        usage=EmbeddingUsage(prompt_tokens=total_length, total_tokens=total_length),
    )
