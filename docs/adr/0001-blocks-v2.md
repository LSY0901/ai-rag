# ADR 0001 — 新建 rag_blocks_v2 集合承载 text/table/image

## 背景
老 `rag_docs` 只有文本字段（`content/source/dense/sparse/page_no/headings/chunk_index`），
且 `content VARCHAR 8192`、`do_table_structure=False`、`do_ocr=False`，表格结构与图片全部丢失。

## 决定
新建集合 `rag_blocks_v2`，新增 `block_type VARCHAR 32` + `image_path VARCHAR 1024`，
`PdfPipelineOptions` 改为 `do_table_structure=True` + `generate_picture_images=True`，
OCR 用 flag 控制默认关闭。老 `rag_docs` 数据不动，切流后删除。

备选（重建 `rag_docs` 原地加列）更快但线上会断，且 Milvus 不支持给已有集合加非动态字段，
必须重建；双集合平滑是唯一不丢服务的做法。

## 后果
- 必须全量重摄入一次；老数据可读，新数据只进 v2。
- 图片向量（CLIP）预留但本次不加列：首版图片只靠 caption 走 BGE-M3，命中率不够时再加
  `image_dense` 列 + 图片模型，那是 ADR 0002 的事。
