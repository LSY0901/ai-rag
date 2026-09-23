"""多格式解析 + 结构化分块。

pdf/docx/html 本地文件经 Docling 转统一文档模型；HybridChunker 做结构分块，
表格/Image 另起 Block 保真：表格存 markdown，图片落盘 + OCR/VLM 双通道。
页级 OCR 走 Docling 管线（do_ocr 默认开）；单图失败降级不阻塞摄入。
overlap 由 DoclingParser 在 chunker 输出后做尾部拼接实现
（docling-core 2.84.0 的 HybridChunker 本身不支持 overlap）。
"""
import os

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)

from rag.exceptions import ParseError
from rag.models import Chunk
from rag.vision.describer import describe_image, ocr_image

ALLOWED_EXTS = {".pdf", ".docx", ".html", ".htm"}


class DoclingParser:
    def __init__(
        self,
        model_path: str,
        max_tokens: int,
        overlap_tokens: int = 0,
        do_ocr: bool = True,
        image_dir: str = "data/uploads/images",
        vlm_enabled: bool = True,
        vlm_endpoint: str = "",
        vlm_model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
    ):
        # OcrAutoOptions 在 macOS 会因缺 ocrmac/onnxruntime/easyocr 静默跳过，
        # 这里显式指定 rapidocr torch 后端（PP-OCRv4 中文，modelscope 自动下载约 40MB）。
        # rapidocr_params 必须 pin v4：无 artifacts_path 时 rapidocr 自行解析模型，
        # 默认 v6 在 torch 上直接 ValueError（与 describer.py 同一组权重）。
        from rapidocr import LangRec, ModelType, OCRVersion

        pipeline_options = PdfPipelineOptions(
            do_ocr=do_ocr,
            do_table_structure=True,
            generate_picture_images=True,
            force_backend_text=not do_ocr,
            ocr_options=RapidOcrOptions(
                lang=["chinese"],
                backend="torch",
                rapidocr_params={
                    "Det.ocr_version": OCRVersion.PPOCRV4,
                    "Det.lang_type": LangRec.CH,
                    "Det.model_type": ModelType.MOBILE,
                    "Rec.ocr_version": OCRVersion.PPOCRV4,
                    "Rec.lang_type": LangRec.CH,
                    "Rec.model_type": ModelType.MOBILE,
                },
            ),
        )
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        # 用 tokenizer 对象传参，避免 HybridChunker(tokenizer=str, max_tokens=...)
        # 这种已废弃写法（会触发 DeprecationWarning）。
        hf_tokenizer = HuggingFaceTokenizer.from_pretrained(
            model_name=model_path,
            max_tokens=max_tokens,
        )
        self._chunker = HybridChunker(tokenizer=hf_tokenizer)
        self._structural_chunker = HierarchicalChunker()
        # 保留底层 transformers tokenizer 供 overlap 计算用。
        self._hf_tokenizer = hf_tokenizer.tokenizer
        self._overlap_tokens = overlap_tokens
        self._image_dir = image_dir
        self._vlm_enabled = vlm_enabled
        self._vlm_endpoint = vlm_endpoint
        self._vlm_model_id = vlm_model_id

    def parse_and_chunk(self, path: str, source: str, strategy: str = "hybrid") -> list[Chunk]:
        ext = os.path.splitext(path)[1].lower()
        if ext not in ALLOWED_EXTS:
            raise ParseError(f"不支持的文件类型 [{source}]：{ext}，仅支持 pdf/docx/html")
        try:
            result = self._converter.convert(path)
            dl_doc = result.document
            if strategy == "structural":
                raw_chunks = [
                    c for c in self._structural_chunker.chunk(dl_doc) if c.text and c.text.strip()
                ]
            else:
                raw_chunks = [
                    c for c in self._chunker.chunk(dl_doc) if c.text and c.text.strip()
                ]
        except Exception as e:
            raise ParseError(f"解析/分块失败 [{source}]: {e}") from e

        chunks = []
        for i, c in enumerate(raw_chunks):
            content = c.text
            # 从上一个块尾部取 overlap_tokens 个 token 的文本拼到当前块前面，
            # 缓解句子被切在两块边界、上下文断裂的问题。
            if self._overlap_tokens > 0 and i > 0:
                tail = self._tail_overlap(raw_chunks[i - 1].text)
                if tail:
                    content = tail + "\n" + content
            chunks.append(
                Chunk(
                    content=content,
                    source=source,
                    page_no=_page_no(c),
                    headings=list(c.meta.headings or []),
                    chunk_index=i,
                    block_type="table" if _looks_like_table(content) else "text",
                )
            )
        chunks.extend(self._table_blocks(dl_doc, source, start=len(chunks), text_chunks=chunks))
        chunks.extend(self._image_blocks(dl_doc, source, start=len(chunks), text_chunks=chunks))
        return chunks

    @staticmethod
    def _inherit_headings(text_chunks: list[Chunk] | None, page_no: int | None) -> list[str]:
        """表/图块继承同页最近正文块的 headings；无同页则取最后一个非空。"""
        if not text_chunks:
            return []
        for c in reversed(text_chunks):
            if page_no is not None and c.page_no == page_no and c.headings:
                return list(c.headings)
        for c in reversed(text_chunks):
            if c.headings:
                return list(c.headings)
        return []

    def _table_blocks(
        self, dl_doc, source: str, start: int, text_chunks: list[Chunk] | None = None
    ) -> list[Chunk]:
        """表格保真：每张表一张 Block，content 为 markdown（含章节信息靠 headings 页级块互补）。"""
        blocks: list[Chunk] = []
        try:
            tables = list(getattr(dl_doc, "tables", []) or [])
        except TypeError:
            return []
        for j, t in enumerate(tables):
            try:
                md = t.export_to_markdown(doc=dl_doc)
            except Exception:
                continue
            if not isinstance(md, str) or not md.strip():
                continue
            page = None
            try:
                if getattr(t, "prov", None):
                    page = t.prov[0].page_no
            except (AttributeError, IndexError):
                pass
            blocks.append(
                Chunk(
                    content=md.strip(),
                    source=source,
                    page_no=page,
                    headings=self._inherit_headings(text_chunks, page),
                    chunk_index=start + j,
                    block_type="table",
                )
            )
        return blocks

    def _image_blocks(
        self, dl_doc, source: str, start: int, text_chunks: list[Chunk] | None = None
    ) -> list[Chunk]:
        """图片保真：原图落盘，Milvus 存指针 + OCR文字 + VLM描述拼串。"""
        blocks: list[Chunk] = []
        try:
            pictures = list(getattr(dl_doc, "pictures", []) or [])
        except TypeError:
            return []
        if not pictures:
            return []
        os.makedirs(self._image_dir, exist_ok=True)
        for j, pic in enumerate(pictures):
            page = None
            try:
                if getattr(pic, "prov", None):
                    page = pic.prov[0].page_no
            except (AttributeError, IndexError):
                pass
            try:
                caption = pic.caption_text(doc=dl_doc)
            except Exception:
                caption = ""
            if not isinstance(caption, str) or caption.startswith("<"):
                caption = ""
            caption = caption.strip()
            fname = f"{os.path.splitext(source)[0]}_p{page or 0}_{j}.png"
            fname = fname.replace("/", "_").replace("\\", "_")
            rel_path = os.path.join(self._image_dir, fname)
            image_path = None
            try:
                img = pic.get_image(doc=dl_doc)
                if img is not None:
                    img.save(rel_path)
                    image_path = rel_path
            except Exception:
                image_path = None
            # docx 内嵌图可能无像素：落盘失败仍保留文本块，VLM 记 skipped。
            if image_path is not None:
                ocr_text = ocr_image(image_path)
                vlm_caption, vlm_status = describe_image(
                    image_path,
                    model_id=self._vlm_model_id,
                    endpoint=self._vlm_endpoint,
                    enabled=self._vlm_enabled,
                )
            else:
                ocr_text, vlm_caption, vlm_status = "", "", "skipped"
            parts = [p for p in (caption, ocr_text, vlm_caption) if p]
            content = "\n".join(parts) or f"[图片] {source} p{page or 0}"
            blocks.append(
                Chunk(
                    content=content,
                    source=source,
                    page_no=page,
                    headings=self._inherit_headings(text_chunks, page),
                    chunk_index=start + j,
                    block_type="image",
                    image_path=image_path,
                    ocr_text=ocr_text,
                    vlm_caption=vlm_caption,
                    vlm_status=vlm_status,
                )
            )
        return blocks

    def _tail_overlap(self, prev_text: str) -> str:
        """取上一块尾部 overlap_tokens 个 token 解码回文本。"""
        ids = self._hf_tokenizer.encode(prev_text, add_special_tokens=False)
        tail_ids = ids[-self._overlap_tokens:]
        return self._hf_tokenizer.decode(tail_ids).strip()


def _looks_like_table(content: str) -> bool:
    """HybridChunker 已把表格转成 markdown，这里只打标不另起块。"""
    return "|" in content and "---" in content


def _page_no(chunk) -> int | None:
    """块内首个 doc_item 的起始页；跨页块取起始页，取不到返回 None。"""
    try:
        items = chunk.meta.doc_items or []
        if items and items[0].prov:
            return items[0].prov[0].page_no
    except (AttributeError, IndexError):
        pass
    return None
