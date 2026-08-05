# AGENTS.md

Workspace instructions for ZCode agents working in `/Users/leo/ai-rag`.

## Project Purpose

Personal RAG (Retrieval-Augmented Generation) experimentation sandbox in Python. The project
integrates embedding, vector database, and reranker into a local RAG retrieval service callable
by Spring AI via FastAPI.

## Major Layout

```
ai-rag/
├── rag/                  # 主包（7 层单向依赖）
│   ├── config.py         # pydantic-settings 配置
│   ├── models.py         # Pydantic 数据契约
│   ├── exceptions.py     # 领域异常
│   ├── storage/          # MilvusStore（MilvusClient 新 API）
│   ├── chunking/         # DoclingParser + HybridChunker
│   ├── embedding/        # BGEM3Embedder（dense + sparse）
│   ├── retrieval/        # HybridRetriever + Reranker
│   └── service/          # SearchService 业务编排
├── api/                  # FastAPI 路由（/health, /ingest, /search）
├── tests/                # pytest 单测（可 mock）
├── scripts/              # 命令行入库/检索 CLI
├── test/                 # 旧探针脚本（保持不动）
├── data/uploads/         # PDF 上传落盘目录
└── venv/                 # 本地虚拟env（Python 3.11，勿编辑）
```

## Environment & Dependencies

Run everything via the local venv interpreter:

```bash
./venv/bin/python scripts/ingest_cli.py data/uploads/sample.pdf
./venv/bin/python scripts/search_cli.py "查询内容" 3
./venv/bin/uvicorn api.app:app --port 8000
./venv/bin/uvicorn api.embedding_app:app --port 8082
./venv/bin/uvicorn api.rerank_app:app --port 8083
./venv/bin/python scripts/start_services.py --all
./venv/bin/pytest
```

Key pinned versions (`requirements.txt`):
- `pymilvus` 3.0.0 — MilvusClient 新 API
- `FlagEmbedding` 1.4.0 — BGE-M3 + reranker
- `transformers` 4.49.0 — **必须此版本**（Docling 可导入 + BGE-M3 兼容）
- `docling` 2.107.0 / `docling-core` 2.84.0
- `torch` 2.12.1, `numpy` 2.4.6
- `fastapi`, `uvicorn`, `python-multipart`, `pytest`

Copy `.env.example` to `.env` to override defaults (Milvus host/port, model paths, etc.).

## External Services & Resources

- **Milvus** must be running at `localhost:19530` (server 2.5.5+ for sparse/hybrid_search).
- **Local model weights** under `/Users/leo/ai/models/`:
  - `bge-m3` — 1024-dim dense + sparse vectors
  - `bge-reranker-v2-m3` — cross-encoder reranker
  - Paths are configurable via `.env`; do not refactor to relative paths without confirming.

## Conventions

- Collection schema: `id` (auto), `content` (VARCHAR 8192), `source` (VARCHAR 512),
  `dense` (FLOAT_VECTOR 1024), `sparse` (SPARSE_FLOAT_VECTOR).
- Index: dense `AUTOINDEX` + `COSINE`; sparse `SPARSE_INVERTED_INDEX` + `IP`.
- Hybrid search uses RRF fusion (k=60).
- `use_fp16=False` for both BGE models.
- Python 3.11, formatter is **Black**.

## Gotchas

- Old `test/` scripts use deprecated `connections.connect` API; new code uses `MilvusClient`.
- `test_docs` collection is dropped by `test/test_milvus_insert.py` — do not reuse that name.
- Collections must be `load()`ed before search (handled by `MilvusStore.ensure_collection`).
- Models are lazy-loaded on first ingest/search to avoid slow FastAPI startup.
- `HybridChunker` uses `max_tokens` (not `chunk_max_tokens`) with docling-core 2.84.0.
