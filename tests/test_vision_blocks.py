from unittest.mock import MagicMock, patch

from rag.chunking.parser import DoclingParser
from rag.models import Chunk
from rag.vision import describer as vd


def _parser(**kw):
    with (
        patch("rag.chunking.parser.DocumentConverter"),
        patch("rag.chunking.parser.HybridChunker"),
        patch("rag.chunking.parser.HuggingFaceTokenizer"),
    ):
        return DoclingParser(model_path="/fake/bge-m3", max_tokens=512, **kw)


def test_unsupported_ext_rejected():
    p = _parser()
    try:
        p.parse_and_chunk("/fake/x.xlsx", source="x.xlsx")
        assert False, "应抛 ParseError"
    except Exception as e:
        assert "不支持的文件类型" in str(e)


def test_image_block_dual_channel(tmp_path):
    p = _parser()
    pic = MagicMock()
    pic.prov = []
    pic.caption_text.return_value = "图1 架构"
    img = MagicMock()
    pic.get_image.return_value = img
    dl_doc = MagicMock()
    dl_doc.tables = []
    dl_doc.pictures = [pic]
    with (
        patch("rag.chunking.parser.ocr_image", return_value="转录文字") as m_ocr,
        patch(
            "rag.chunking.parser.describe_image", return_value=("VLM描述", "ok")
        ) as m_vlm,
    ):
        p._image_dir = str(tmp_path)
        blocks = p._image_blocks(dl_doc, source="doc.pdf", start=3)
    assert len(blocks) == 1
    b = blocks[0]
    assert b.block_type == "image" and b.chunk_index == 3
    assert b.ocr_text == "转录文字" and b.vlm_caption == "VLM描述"
    assert "图1 架构" in b.content and "转录文字" in b.content
    img.save.assert_called_once()
    m_ocr.assert_called_once()
    m_vlm.assert_called_once()


def test_image_block_vlm_failed_degrades():
    p = _parser()
    pic = MagicMock()
    pic.prov = []
    pic.caption_text.return_value = ""
    pic.get_image.return_value = None  # docx 内嵌图无像素
    dl_doc = MagicMock()
    dl_doc.tables = []
    dl_doc.pictures = [pic]
    blocks = p._image_blocks(dl_doc, source="doc.docx", start=0)
    assert len(blocks) == 1
    assert blocks[0].vlm_status == "skipped"
    assert blocks[0].image_path is None
    assert "[图片]" in blocks[0].content


def test_image_block_inherits_same_page_headings(tmp_path):
    p = _parser()
    pic = MagicMock()
    pic.prov = []
    pic.caption_text.return_value = ""
    pic.get_image.return_value = MagicMock()
    dl_doc = MagicMock()
    dl_doc.tables = []
    dl_doc.pictures = [pic]
    texts = [
        Chunk(content="a", source="d.pdf", page_no=1, headings=["第一章"]),
        Chunk(content="b", source="d.pdf", page_no=2, headings=["第二章", "2.1"]),
    ]
    with (
        patch("rag.chunking.parser.ocr_image", return_value=""),
        patch("rag.chunking.parser.describe_image", return_value=("cap", "ok")),
    ):
        p._image_dir = str(tmp_path)
        blocks = p._image_blocks(dl_doc, source="d.pdf", start=0, text_chunks=texts)
    assert blocks[0].headings == ["第二章", "2.1"]  # page 无归属 → 最后一个非空


def test_ocr_image_failure_returns_empty():
    with patch("rapidocr.RapidOCR", side_effect=RuntimeError("no engine")):
        vd._ocr_engine = None
        assert vd.ocr_image("/nonexistent.png") == ""
        vd._ocr_engine = None


def test_describe_disabled_skipped():
    assert vd.describe_image("/x.png", model_id="m", enabled=False) == ("", "skipped")


def test_describe_failure_returns_failed():
    with patch("transformers.pipeline", side_effect=RuntimeError("no model")):
        vd._vlm_pipe = None
        assert vd.describe_image("/x.png", model_id="m") == ("", "failed")
        vd._vlm_pipe = None
