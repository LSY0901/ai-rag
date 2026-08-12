"""命令行入库：python scripts/ingest_cli.py <pdf_path>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.dependencies import get_search_service

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/ingest_cli.py <pdf_path> [strategy]")
        print("strategy 可选: hybrid (默认), structural")
        sys.exit(1)
    path = sys.argv[1]
    filename = path.rsplit("/", 1)[-1]
    strategy = sys.argv[2] if len(sys.argv) > 2 else "hybrid"
    n = get_search_service().ingest(path=path, filename=filename, strategy=strategy)
    print(f"已入库 {filename} (策略: {strategy}): {n} 个 chunk")
