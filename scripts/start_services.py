"""启动本地模型服务 (Embedding: 8082, Rerank: 8083, RAG Main: 8000)。

用法:
    ./venv/bin/python scripts/start_services.py --all
    ./venv/bin/python scripts/start_services.py --embedding-port 8082 --rerank-port 8083
"""
import argparse
import os
import signal
import sys
from multiprocessing import Process

import uvicorn

# 确保项目根目录在 Python 模块搜索路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def run_embedding(host: str, port: int):
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    uvicorn.run("api.embedding_app:app", host=host, port=port, log_level="info")


def run_rerank(host: str, port: int):
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    uvicorn.run("api.rerank_app:app", host=host, port=port, log_level="info")


def run_rag(host: str, port: int):
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    uvicorn.run("api.app:app", host=host, port=port, log_level="info")



def main():
    parser = argparse.ArgumentParser(description="Start Local RAG Model & API Services")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--embedding-port", type=int, default=8082, help="Embedding service port")
    parser.add_argument("--rerank-port", type=int, default=8083, help="Rerank service port")
    parser.add_argument("--rag-port", type=int, default=8000, help="Main RAG service port")
    parser.add_argument("--embedding-only", action="store_true", help="Start only embedding service")
    parser.add_argument("--rerank-only", action="store_true", help="Start only rerank service")
    parser.add_argument("--all", action="store_true", help="Start all services (Embedding + Rerank + RAG Main)")

    args = parser.parse_args()

    processes: list[Process] = []

    if args.embedding_only:
        run_embedding(args.host, args.embedding_port)
        return

    if args.rerank_only:
        run_rerank(args.host, args.rerank_port)
        return

    # 默认启动 Embedding (8082) 与 Rerank (8083)
    p_embed = Process(target=run_embedding, args=(args.host, args.embedding_port))
    p_rerank = Process(target=run_rerank, args=(args.host, args.rerank_port))
    processes.extend([p_embed, p_rerank])

    if args.all:
        p_rag = Process(target=run_rag, args=(args.host, args.rag_port))
        processes.append(p_rag)

    def signal_handler(sig, frame):
        print("\nStopping services...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"Starting Embedding service on http://{args.host}:{args.embedding_port}")
    print(f"Starting Rerank service on http://{args.host}:{args.rerank_port}")
    if args.all:
        print(f"Starting Main RAG service on http://{args.host}:{args.rag_port}")

    for p in processes:
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
