"""诊断脚本：只跑切分，不入库，打印每个 chunk 的统计和边界。
用法: ./venv/bin/python scripts/dump_chunks.py data/uploads/智能制造发展前景.pdf
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunking.parser import DoclingParser
from rag.config import settings


def char_len(s: str) -> int:
    return len(s.replace("\n", "").replace(" ", ""))


def main() -> None:
    path = sys.argv[1]
    parser = DoclingParser(
        model_path=settings.embedding_model_path,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )

    chunks = parser.parse_and_chunk(path=path, source=Path(path).name)

    lens = [char_len(c.content) for c in chunks]
    print(f"总块数: {len(chunks)}")
    if lens:
        print(
            f"每块字符数: min={min(lens)} max={max(lens)} "
            f"avg={sum(lens)//len(lens)}"
        )
    print("=" * 70)

    for i, c in enumerate(chunks):
        head = c.content[:60].replace("\n", "⏎")
        tail = c.content[-60:].replace("\n", "⏎")
        print(f"[{i:02d}] ({char_len(c.content)} 字) {head}")
        print(f"      ...尾: ...{tail}")
        # 边界硬切粗判：尾或首是否被句中标点截断
        last_char = c.content.strip()[-1:]
        first_char = c.content.strip()[:1]
        bad_end = last_char not in "。！？；!?;…\"'」』）)"
        bad_start = first_char not in "（(\"「『『"
        flag = ""
        if bad_end and i < len(chunks) - 1:
            flag += " ⚠️尾非句末"
        print(f"      首字符={first_char!r} 末字符={last_char!r}{flag}")
        print("-" * 70)


if __name__ == "__main__":
    main()
