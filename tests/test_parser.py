from unittest.mock import MagicMock, patch

from rag.chunking.parser import DoclingParser


@patch("rag.chunking.parser.HuggingFaceTokenizer")
@patch("rag.chunking.parser.HybridChunker")
@patch("rag.chunking.parser.DocumentConverter")
def test_parse_and_chunk(mock_conv_cls, mock_chunker_cls, mock_hf_tok_cls):
    converter = MagicMock()
    mock_conv_cls.return_value = converter
    fake_doc = MagicMock(name="DoclingDocument")
    converter.convert.return_value.document = fake_doc

    chunker = MagicMock()
    mock_chunker_cls.return_value = chunker
    c1, c2 = MagicMock(), MagicMock()
    c1.text = "第一段"
    c2.text = "第二段"
    chunker.chunk.return_value = iter([c1, c2])

    parser = DoclingParser(model_path="/fake/bge-m3", max_tokens=512)
    chunks = parser.parse_and_chunk("/fake/doc.pdf", source="doc.pdf")

    assert len(chunks) == 2
    assert chunks[0].content == "第一段"
    assert chunks[0].source == "doc.pdf"
    assert chunks[0].dense is None


@patch("rag.chunking.parser.HuggingFaceTokenizer")
@patch("rag.chunking.parser.HybridChunker")
@patch("rag.chunking.parser.DocumentConverter")
def test_parse_error_wrapped(mock_conv_cls, mock_chunker_cls, mock_hf_tok_cls):
    from rag.exceptions import ParseError

    converter = MagicMock()
    mock_conv_cls.return_value = converter
    converter.convert.side_effect = Exception("boom")

    parser = DoclingParser(model_path="/fake/bge-m3", max_tokens=512)
    try:
        parser.parse_and_chunk("/fake/x.pdf", source="x.pdf")
        assert False, "应抛 ParseError"
    except ParseError:
        pass

