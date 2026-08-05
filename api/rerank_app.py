"""FastAPI Rerank 服务：运行在端口 8083，供 Spring AI 访问。

支持标准的 POST /rerank、/v1/rerank、/api/v1/rerank 接口。
"""
from fastapi import FastAPI, HTTPException

from api.dependencies import get_reranker
from rag.exceptions import EmbedError
from rag.models import (
    RerankRequest,
    RerankResponse,
    RerankResultItem,
    SearchHit,
)

app = FastAPI(title="Rerank Service (Spring AI Compatible)")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "rerank"}


@app.post("/rerank", response_model=RerankResponse)
@app.post("/v1/rerank", response_model=RerankResponse)
@app.post("/api/v1/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    if not req.documents:
        return RerankResponse(model=req.model, results=[])

    candidates: list[SearchHit] = []
    doc_payloads: list[dict | str] = []

    for idx, doc in enumerate(req.documents):
        doc_payloads.append(doc)
        if isinstance(doc, dict):
            text = str(doc.get("text") or doc.get("content") or doc)
        else:
            text = str(doc)
        candidates.append(SearchHit(content=text, source=str(idx), score=0.0))

    top_k = req.top_n if req.top_n is not None and req.top_n > 0 else len(candidates)

    try:
        reranker = get_reranker()
        reranked_hits = reranker.rerank(
            query=req.query, candidates=candidates, top_k=top_k
        )
    except EmbedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    results = []
    for hit in reranked_hits:
        orig_idx = int(hit.source)
        orig_doc = doc_payloads[orig_idx]
        doc_dict = {"text": orig_doc} if isinstance(orig_doc, str) else orig_doc
        results.append(
            RerankResultItem(
                index=orig_idx,
                relevance_score=hit.score,
                score=hit.score,
                document=doc_dict,
            )
        )

    return RerankResponse(model=req.model, results=results)
