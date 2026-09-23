# ADR 0002 — 图片 OCR+VLM 双通道，集合升 rag_blocks_v3

## 背景
扫描 PDF/docx 占 90%，图片文字（OCR）与图片含义（VLM）都要可检索、可展示。
Docling 管线 OCR 只落页级文本，图片裁片文字与图片描述需要块级双通道。

## 决定
1. VLM 默认本地 `SmolVLM-256M-Instruct`（`transformers` image-to-text，float32 CPU，
   惰性加载）；`VLM_ENDPOINT` 非空则 POST `{"image": b64, "prompt"}` 走 API。
   单图失败记 `vlm_status=failed`，内容回退纯 OCR。
2. 图片裁片 OCR 用已装的 `rapidocr==3.9.2` torch 后端（onnxruntime 缺装，
   torch 可用，PP-OCRv6 中文），与 Docling 页级 OCR 同引擎族，不新增依赖。
3. `Chunk/SearchHit`/Milvus 加 `ocr_text/ vlm_caption/ vlm_status` 三列，
   `content = caption + OCR文字 + 图片描述` 拼串供 BGE-M3 向量化。
   集合升 `rag_blocks_v3`（v2 不动，切流后删）。
4. `ocr_enabled` 默认翻 `True`（`force_backend_text=False`）；Docling 无 OCR 引擎时
   自动跳过不崩。`ocr_options.lang` 不显式传，用 Docling 默认（中文优先），
   避免语言码在 rapidocr/easyocr 间口径不一导致整份 convert 失败。

备选（Docling 内联 `do_picture_description`）省代码，但 VLM 失败会整体
`ParseError` 丢整份文件，违背“单图降级不阻塞”要求，故自建块级调用。

## 后果
- 首次 VLM 推理下载约 500MB 权重；ingest 慢 5-10 倍，接受。
- `transformers==4.49.0` 不动，SmolVLM 经 `AutoModelForImageTextToText` 加载已验证存在。
