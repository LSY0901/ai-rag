# ai-rag Agent Rules

Python 3.11 + FastAPI + BGE-M3 + Milvus 2.5 的本地 RAG 微服务沙箱，为 Spring AI 等外部 LLM 应用提供检索增强生成能力。

本文件是跨工具 Agent 入口，只放长期有效、代码里不容易直接推断、猜错会影响结果的规则。

## Tech Stack

- **Python 3.11** + FastAPI + uvicorn
- **向量数据库**: Milvus 2.5.5+ (`localhost:19530`)，使用 `MilvusClient` 新 API（pymilvus 3.0.0）
- **Embedding**: BGE-M3 (dense 1024-dim + sparse)，本地权重 `/Users/leo/ai/models/bge-m3`
- **Reranker**: bge-reranker-v2-m3 (cross-encoder)，本地权重 `/Users/leo/ai/models/bge-reranker-v2-m3`
- **文档解析**: Docling 2.107.0 + docling-core 2.84.0 (HybridChunker + HierarchicalChunker)
- **数据契约**: Pydantic 2.13.4 + pydantic-settings 2.14.2
- **服务端口**: 8000 (RAG主服务), 8082 (Embedding API), 8083 (Rerank API)

## Commands

```bash
# 启动全部服务 (8000+8082+8083)
./venv/bin/python scripts/start_services.py --all

# 仅启动 Embedding + Rerank (供 Spring AI 外部调用)
./venv/bin/python scripts/start_services.py

# 单文档摄入 (hybrid/structural)
./venv/bin/python scripts/ingest_cli.py data/uploads/sample.pdf [hybrid|structural]

# CLI 搜索
./venv/bin/python scripts/search_cli.py "查询内容" 5 0.3

# 诊断 PDF 分块边界
./venv/bin/python scripts/dump_chunks.py data/uploads/sample.pdf

# 运行单元测试 (100% mocked, <5s)
./venv/bin/pytest
```

## Project Structure

```
rag/                          # 核心领域包 (7层单向依赖)
├── config.py                 # Layer 1: pydantic-settings 配置 (读取 .env)
├── exceptions.py             # Layer 1: 领域异常 (RagError, StoreError, ParseError, EmbedError)
├── models.py                 # Layer 1: Pydantic 数据契约 (Chunk, SearchHit, API schemas)
├── storage/milvus_store.py   # Layer 2: MilvusClient CRUD + hybrid_search
├── chunking/parser.py        # Layer 3: Docling PDF 解析 + 分块
├── embedding/bge_embedder.py # Layer 4: BGE-M3 dense + sparse 向量化
├── retrieval/retriever.py    # Layer 5: HybridRetriever (RRF fusion)
├── retrieval/reranker.py     # Layer 5: bge-reranker-v2-m3 cross-encoder
└── service/search_service.py # Layer 6: 摄入 & 搜索编排
api/                          # Layer 7: FastAPI 端点 + 依赖注入
├── dependencies.py           # @lru_cache 单例提供者 (模型惰性加载)
├── app.py                    # 主 RAG API (port 8000)
├── embedding_app.py          # Embedding API (port 8082, Spring AI 兼容)
└── rerank_app.py             # Rerank API (port 8083, Cohere/TEI 兼容)
scripts/                      # CLI 工具 & 服务编排
tests/                        # 隔离单元测试 (100% mocked)
test/                         # 历史探测脚本 (勿运行)
```

## Architecture

- **严格单向依赖**: `api/` → `service/` → `retrieval/` → `embedding/` → `chunking/` → `storage/` → `config/models/exceptions`
- **Pydantic 契约传递**: 层间数据交换必须使用 `rag/models.py` 中的 Pydantic 模型，禁止裸 dict
- **领域异常封装**: 底层库异常 (`pymilvus`, `torch`, `docling`) 必须捕获并包装为领域异常 (`StoreError`, `EmbedError`, `ParseError`)，API 层统一翻译为 HTTP 422/503
- **Idempotent 摄入**: `ingest` 先 `delete_by_source(filename)` 再 `insert(chunks)`，防止重复写入
- **惰性加载**: 模型权重通过 `@lru_cache` 在首次请求时惰性加载，保持 FastAPI 启动秒开

## Critical Invariants

- **Milvus API**: 只用 `MilvusClient`，禁止 `connections.connect` / `Collection` / `utility`（已废弃）
- **向量维度**: dense 1024-dim COSINE，sparse SPARSE_INVERTED_INDEX IP，`RRFRanker(k=60)` 硬编码
- **分块重叠**: `docling-core 2.84.0` 的 `HybridChunker` 不支持原生 overlap，`DoclingParser` 通过 tail-to-head 手动拼接实现
- **混合检索**: `MilvusClient.hybrid_search` 双路 ANN + 服务端 RRF 融合，`fetch_k=20` 过采样供 reranker
- **Apple Silicon**: 两个模型必须 `use_fp16=False`，FP16 在 macOS MPS/CPU 会导致 NaN 或崩溃

## Config (.env)

```ini
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=rag_docs
EMBEDDING_MODEL_PATH=/Users/leo/ai/models/bge-m3
RERANKER_MODEL_PATH=/Users/leo/ai/models/bge-reranker-v2-m3
CHUNK_MAX_TOKENS=450
CHUNK_OVERLAP_TOKENS=60
UPLOAD_DIR=data/uploads
```

## Testing

- 框架: pytest + `@patch` / `MagicMock` / `monkeypatch.setenv`
- 约束: `tests/` 下全部 mocked，禁止连接真实 Milvus 或加载真实模型权重，全量 <5s
- 命名: `test_<行为描述>`，snake_case
- 覆盖: 配置、模型、存储、解析、向量化、检索、重排、Embedding API、Rerank API 共 20 个测试

## Pinned Versions (严格锁定)

- `transformers==4.49.0` — 升降级破坏 Docling 与 FlagEmbedding 兼容性
- `pymilvus==3.0.0` — MilvusClient API 必需
- `FlagEmbedding==1.4.0` — BGE-M3 + reranker
- `docling==2.107.0` & `docling-core==2.84.0`
- `torch==2.12.1`, `numpy==2.4.6`
- `pydantic==2.13.4`, `pydantic-settings==2.14.2`

## Never Do

- 不要 `from pymilvus import connections, Collection, utility, FieldSchema` — 已废弃，只用 `MilvusClient`
- 不要 `python ...` 或 `pytest ...` — 必须用 `./venv/bin/python` 或 `./venv/bin/pytest`
- 不要 `pip install ...` — 未经明确指示不得修改依赖
- 不要把 `transformers` 升级或降级 — `4.49.0` 是 Docling + FlagEmbedding 兼容性的唯一版本
- 不要 `use_fp16=True` — macOS MPS/CPU 下导致数值不稳定
- 不要在 `test/` 目录下运行脚本 — 历史探测代码，使用废弃 API，会 drop/recreate `test_docs` 集合
- 不要向 `storage/`、`chunking/`、`retrieval/` 层反向导入 `service/` 或 `api/` — 破坏单向依赖
- 不要裸 `dict` 传递层间数据 — 必须使用 `rag/models.py` 中的 Pydantic 模型
- 不要静默 catch 吞掉异常 — 底层异常必须包装为领域异常 (`StoreError`/`ParseError`/`EmbedError`) 重新抛出
- 不要在 `HybridChunker(tokenizer=...)` 传入字符串 — 必须传入 `HuggingFaceTokenizer` 实例对象
