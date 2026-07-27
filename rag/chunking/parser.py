"""PDF 解析 + 结构化分块。

HybridChunker 用 docling 文档结构分块（标题/段落感知），并控制 token 上限。
overlap 由 DoclingParser 在 chunker 输出后做尾部拼接实现
（docling-core 2.84.0 的 HybridChunker 本身不支持 overlap）。
"""
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)

from rag.exceptions import ParseError
from rag.models import Chunk


class DoclingParser:
    def __init__(
        self,
        model_path: str,
        max_tokens: int,
        overlap_tokens: int = 0,
    ):
        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=False,
            force_backend_text=True,
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
        # 保留底层 transformers tokenizer 供 overlap 计算用。
        self._hf_tokenizer = hf_tokenizer.tokenizer
        self._overlap_tokens = overlap_tokens

    def parse_and_chunk(self, path: str, source: str) -> list[Chunk]:
        try:
            result = self._converter.convert(path)
            dl_doc = result.document
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
            chunks.append(Chunk(content=content, source=source))
        return chunks

    def _tail_overlap(self, prev_text: str) -> str:
        """取上一块尾部 overlap_tokens 个 token 解码回文本。"""
        ids = self._hf_tokenizer.encode(prev_text, add_special_tokens=False)
        tail_ids = ids[-self._overlap_tokens:]
        return self._hf_tokenizer.decode(tail_ids).strip()
