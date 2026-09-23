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
            from rapidocr import EngineType, LangRec, ModelType, OCRVersion, RapidOCR

            # torch 后端只支持 PP-OCRv4/v5，不显式 pin 默认 v6 直接 ValueError。
            # 与 Docling 页级 OCR 同一组 v4 中文 mobile 权重，不新增依赖。
            _ocr_engine = RapidOCR(
                params={
                    "Det.engine_type": EngineType.TORCH,
                    "Det.ocr_version": OCRVersion.PPOCRV4,
                    "Det.lang_type": LangRec.CH,
                    "Det.model_type": ModelType.MOBILE,
                    "Cls.engine_type": EngineType.TORCH,
                    "Rec.engine_type": EngineType.TORCH,
                    "Rec.ocr_version": OCRVersion.PPOCRV4,
                    "Rec.lang_type": LangRec.CH,
                    "Rec.model_type": ModelType.MOBILE,
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

        # image-to-text 传 prompt 在 idefics3 上 shape mismatch，
        # 4.49 起 SmolVLM 走 image-text-to-text + chat 格式。
        _vlm_pipe = pipeline(
            "image-text-to-text",
            model=model_id,
            torch_dtype=torch.float32,
            device=0 if torch.cuda.is_available() else -1,
        )
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": VLM_PROMPT}],
        }
    ]
    out = _vlm_pipe([path], messages, generate_kwargs={"max_new_tokens": 256})
    if isinstance(out, list) and out:
        first = out[0]
        if isinstance(first, dict):
            gen = first.get("generated_text")
            if isinstance(gen, list):
                # 全量 chat 返回，取最后一条 assistant
                for m in reversed(gen):
                    if isinstance(m, dict) and m.get("role") == "assistant":
                        return _message_text(m.get("content")).strip()
                return ""
            if isinstance(gen, str):
                return gen.strip()
            return str(first.get("text") or "").strip()
    return str(out).strip()


def _message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(p.get("text", ""))
            for p in content
            if isinstance(p, dict) and p.get("text")
        )
    return str(content or "")
