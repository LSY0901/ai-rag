# CONTEXT — ai-rag 领域语言

## Block（块）
最小检索单元。一份 `Source` 文件切出多个 `Block`，每个 `Block` 独占 Milvus 一行。
`block_type` 三选一：`text`（正文段落）、`table`（表格转 markdown）、`image`（图片双通道 + 原图指针）。

## Source（来源）
上传文件名（如 `合同.pdf`）。摄入去重键：`ingest` 先 `delete_by_source(source)` 再 `insert`。
原文件字节落盘 `data/uploads/`，可经 `GET /files/{source}` 下载。

## 保真存储
两层含义同时成立才算保真：
1. 原文件字节落盘可下载（`data/uploads/` 原文件，`data/uploads/images/` 抽出的图片）；
2. 检索返回结构不丢（正文/表格返 markdown，图片返 `ocr_text + vlm_caption + image_path`）。
Milvus 只存指针和可检索文本，不存大二进制。

## OCR文本
图片/扫描页里抠出的字。页级 OCR（Docling 管线，RapidOCR-torch）进 `text` 块；
图片裁片 OCR（同引擎单图调用）进 `image` 块的 `ocr_text`。失败留空，不阻塞摄入。

## VLM描述
图片讲的是什么，一句话 caption。默认本地 `SmolVLM-256M`，`VLM_ENDPOINT` 非空则走 API。
单图超时/爆显存时该图记 `vlm_status=failed` 降级为纯 OCR，不阻塞整份文件。

## 保真载荷
召回时前端能还原排版的东西 = `content(markdown) + block_type + image_path + headings[] + page_no + chunk_index`。
不是原 PDF 二进制，也不是整页截图。

## 非目标（二期）
`xlsx/pptx` 解析、CLIP 多模态向量列 `image_dense`、图搜图。
