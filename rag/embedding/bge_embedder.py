"""BGE-M3 向量化：一次 encode 同时产出 dense + sparse。"""
from dataclasses import dataclass

from FlagEmbedding import BGEM3FlagModel

from rag.exceptions import EmbedError


@dataclass
class EmbeddingResult:
    dense: list[list[float]]
    sparse: list[dict[str, float]]


class BGEM3Embedder:
    def __init__(self, model_path: str, use_fp16: bool = False):
        self.model_path = model_path
        self.use_fp16 = use_fp16
        self._model: BGEM3FlagModel | None = None

    def _ensure_model(self) -> None:
        if self._model is None:
            self._model = BGEM3FlagModel(self.model_path, use_fp16=self.use_fp16)

    def encode(self, texts: list[str]) -> EmbeddingResult:
        """一次调用产出 dense + sparse。"""
        if not texts:
            return EmbeddingResult(dense=[], sparse=[])
        self._ensure_model()
        try:
            out = self._model.encode(
                texts,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
        except Exception as e:
            raise EmbedError(f"BGE-M3 encode failed: {e}") from e
        dense = [
            v.tolist() if hasattr(v, "tolist") else list(v) for v in out["dense_vecs"]
        ]
        sparse = [dict(w) for w in out["lexical_weights"]]
        return EmbeddingResult(dense=dense, sparse=sparse)
