"""BGE-Reranker-v2-m3 重排。"""
from FlagEmbedding import FlagReranker

from rag.exceptions import EmbedError
from rag.models import SearchHit


class Reranker:
    def __init__(self, model_path: str, use_fp16: bool = False):
        self.model_path = model_path
        self.use_fp16 = use_fp16
        self._model: FlagReranker | None = None

    def _ensure_model(self) -> None:
        if self._model is None:
            self._model = FlagReranker(self.model_path, use_fp16=self.use_fp16)

    def rerank(
        self, query: str, candidates: list[SearchHit], top_k: int
    ) -> list[SearchHit]:
        if not candidates:
            return []
        self._ensure_model()
        pairs = [[query, c.content] for c in candidates]
        try:
            # normalize=True 走 sigmoid，分数落在 (0,1)，与 score_threshold/测试的约定一致；
            # 默认返回的是 raw logit（可为负、无界），阈值根本没法设。
            scores = self._model.compute_score(pairs, normalize=True)
        except Exception as e:
            raise EmbedError(f"rerank failed: {e}") from e
        if isinstance(scores, (int, float)):
            scores = [float(scores)]
        scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [
            SearchHit(
                content=c.content,
                source=c.source,
                score=float(s),
                page_no=c.page_no,
                headings=c.headings,
                chunk_index=c.chunk_index,
                block_type=c.block_type,
                image_path=c.image_path,
                ocr_text=c.ocr_text,
                vlm_caption=c.vlm_caption,
                vlm_status=c.vlm_status,
            )
            for c, s in scored[:top_k]
        ]
