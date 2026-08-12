"""命令行检索：python scripts/search_cli.py <query> [top_k]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.dependencies import get_search_service

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/search_cli.py <query> [top_k] [threshold]")
        sys.exit(1)
    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    for h in get_search_service().search(query, top_k, threshold):
        print(f"[{h.score:.3f}] {h.source}: {h.content[:100]}")
