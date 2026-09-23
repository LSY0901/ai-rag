"""图片双通道：裁片 OCR（RapidOCR-torch）+ VLM 描述（本地 SmolVLM / API）。

全部惰性加载、全部 try/except：单图失败返回空串 + failed，不阻塞整份摄入。
transformers/rapidocr 只在函数内 import，保持 FastAPI 秒开与测试轻量。
"""
import base64
import urllib.request

VLM_PROMPT = "用中文描述这张图片，并逐字转录其中的所有文字；没有文字则只描述。"

_ocr_engine = None
_vlm_pipe = None


def ocr_image(path: str) -> str:
    """裁片 OCR，失败返回空串。"""
    global _ocr_engine
    try:
        if _ocr_engine is None:
            from rapidocr import EngineType, RapidOCR

            _ocr_engine = RapidOCR(
                params={
                    "Det.engine_type": EngineType.TORCH,
                    "Cls.engine_type": EngineType.TORCH,
                    "Rec.engine_type": EngineType.TORCH,
                }
            )
        out = _ocr_engine(path)
        texts = getattr(out, "txts", None) or []
        return "\n".join(t for t in texts if t and t.strip()).strip()
    except Exception:
        return ""


def describe_image(
    path: str, model_id: str, endpoint: str = "", enabled: bool = True
) -> tuple[str, str]:
    """VLM 描述，返回 (caption, status)。status: ok | failed | skipped。"""
    if not enabled:
        return "", "skipped"
    try:
        if endpoint:
            return _describe_via_api(path, endpoint), "ok"
        return _describe_local(path, model_id), "ok"
    except Exception:
        return "", "failed"


def _describe_via_api(path: str, endpoint: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    import json

    req = urllib.request.Request(
        endpoint,
        data=json.dumps({"image": b64, "prompt": VLM_PROMPT}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode())
    if isinstance(body, dict):
        for key in ("caption", "description", "text", "result"):
            if body.get(key):
                return str(body[key]).strip()
    return str(body).strip()


def _describe_local(path: str, model_id: str) -> str:
    global _vlm_pipe
    if _vlm_pipe is None:
        import torch
        from transformers import pipeline

        _vlm_pipe = pipeline(
            "image-to-text",
            model=model_id,
            torch_dtype=torch.float32,
            device=0 if torch.cuda.is_available() else -1,
        )
    out = _vlm_pipe(path, prompt=VLM_PROMPT, generate_kwargs={"max_new_tokens": 256})
    if isinstance(out, list) and out:
        first = out[0]
        if isinstance(first, dict):
            text = first.get("generated_text") or first.get("text") or ""
            return str(text).strip()
    return str(out).strip()
