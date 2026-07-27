# RAG 检索服务设计规格

- **日期**: 2026-06-29
- **范围**: 实现 9 步 RAG 全链路（Milvus 封装 → Docling 解析 → 分块 → BGE-M3 向量化 → 入库 → 混合检索 → 重排 → 检索服务 → FastAPI）
- **状态**: 待审查

## 1. 目标与范围

把当前 `test/*.py` 里的零散探针脚本，整合成一个可被 Spring AI 调用的本地 RAG 检索服务：

- **入库**：上传 PDF → 解析 → 分块 → 向量化 → 写入 Milvus
- **检索**：查询文本 → 向量化 → 混合检索 → 重排 → 返回 top-k

**非目标（YAGNI）**：多租户、鉴权、向量库迁移抽象层、增量更新、批量任务队列。所有这些都是本地实验不需要的。

## 2. 关键技术决策（已与用户确认）

| 维度 | 决策 | 理由 |
|---|---|---|
| 混合检索 | BGE-M3 一次 encode 同时产出 dense(1024) + sparse | 同一模型同时负责语义与词面匹配，一致性最好；无需外部 BM25 库 |
| Milvus 客户端 | `MilvusClient`（pymilvus 3.x 新 API） | `connections.connect`/`Collection`/`utility` 在 3.1 会被移除；`test/*.py` 保持不动 |
| 输入来源 | FastAPI multipart upload → `data/uploads/` | 贴近 Spring AI 实际调用方式 |
| 分块策略 | docling-core `HybridChunker` | 基于文档结构分块，天然带上下文，且能控制 token 上限（BGE-M3 上限 8192） |

**已验证的环境约束**：
- Milvus 服务端 `2.5.5`，支持 sparse 向量与 `hybrid_search` ✅
- Docling 2.107.0、pydantic-settings、FastAPI 技术栈均已安装 ✅
- BGE-M3 sparse 输出格式确认：`lexical_weights` = `{token_id(str): weight(float)}`

## 3. 架构与分层

按依赖方向自底向上分层，每层单向依赖下层，不反向调用：

```
api/        FastAPI 路由层      ← Step 9   只做请求/响应转换 + 调 service
service/    业务编排层          ← Step 8   SearchService：编排 retrieve+rerank
retrieval/  检索/重排层         ← Step 6,7 HybridRetriever + Reranker
embedding/  向量化层            ← Step 4   BGEM3Embedder：dense+sparse 一次产出
chunking/   解析/分块层         ← Step 2,3 DoclingParser + HybridChunker 封装
storage/    存储层              ← Step 1   MilvusStore：MilvusClient 封装 (Collection/索引/CRUD)
models/     数据契约            ← 全程     Pydantic：Chunk, Document, 请求/响应
config/     配置层              ← 全程     pydantic-settings：路径/模型/Milvus 连接
```

**边界规则**：
- `api/` 不直接碰 `storage/`/`embedding/`，只通过 `service/`。
- `storage/` 不感知 embedding 维度细节——维度由 `config` 提供，`storage` 只接收已算好的向量。
- 各层都用 `models/` 里的 Pydantic 对象传递，不用裸 dict。

## 4. 各模块设计

### 4.1 config/（Step 0 基础）

`Settings`（pydantic-settings，从 `.env` 读，有合理默认值）：

| 字段 | 默认值 |
|---|---|
| `milvus_host` / `milvus_port` | `localhost` / `19530` |
| `milvus_collection` | `rag_docs` |
| `dense_dim` | `1024` |
| `embedding_model_path` | `/Users/leo/ai/models/bge-m3` |
| `reranker_model_path` | `/Users/leo/ai/models/bge-reranker-v2-m3` |
| `upload_dir` | `data/uploads` |
| `chunk_max_tokens` | `1024`（适配 BGE-M3） |

### 4.2 storage/ — MilvusStore（Step 1）

`MilvusClient` 封装，单例。

**Collection schema**（字段）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | INT64, auto_id, primary | 主键 |
| `content` | VARCHAR(8192) | chunk 文本 |
| `source` | VARCHAR(512) | 来源文件名 |
| `dense` | FLOAT_VECTOR(1024) | BGE-M3 dense |
| `sparse` | SPARSE_FLOAT_VECTOR | BGE-M3 sparse |

**索引**：dense 用 `AUTOINDEX` + `COSINE`；sparse 用 `SPARSE_INVERTED_INDEX` + `IP`。

**对外方法**：
- `ensure_collection()` — 幂等建表 + 建索引 + load
- `insert(chunks: list[Chunk])` — 批量插入
- `hybrid_search(dense, sparse, top_k)` — 并行两路 anns_search → RRF 融合
- `delete_by_source(source)` — 按文件名清理（重新入库时用）

### 4.3 chunking/ — DoclingParser + Chunker（Step 2, 3）

- `DoclingParser.parse(path) -> Document`：用 `DocumentConverter`，返回 markdown 文本 + Docling 文档对象（供 chunker 用结构信息）。
- `Chunker.chunk(document, max_tokens) -> list[Chunk]`：用 `docling_core.transforms.chunker.hybrid_chunker.HybridChunker`（注意：此 docling-core 2.84.0 版本路径是 `.chunker.`，不是文档常见的 `.transforms.hybrid_chunker.`），每个 chunk 带 `source` 元数据。`HybridChunker` 需要 tokenizer，用其内置 Huggingface tokenizer（`chunk_max_tokens` 模型上限为 8192）。

### 4.4 embedding/ — BGEM3Embedder（Step 4）

- 单例，懒加载模型（避免 FastAPI 启动卡住）。
- `encode(texts) -> EmbeddingResult`：一次调用同时返回 `dense(list[list[float]])` + `sparse(list[dict[str,float]])`。
- 内部调用 `model.encode(..., return_dense=True, return_sparse=True, return_colbert_vecs=False)`。

### 4.5 retrieval/（Step 6, 7）

- `HybridRetriever.retrieve(query, top_k)`：对 query 调 embedder → 调 store.hybrid_search → 返回候选 Chunk 列表。
- `Reranker.rerank(query, candidates, top_k)`：bge-reranker-v2-m3 的 `compute_score([query, chunk])` → 按分数重排取 top-k。

### 4.6 service/ — SearchService（Step 8）

编排层，`api/` 唯一入口：
- `ingest(file_path, filename)`：parse → chunk → embed → store.insert
- `search(query, top_k)`：retrieve → rerank → 返回带分数的结果

### 4.7 api/ — FastAPI（Step 9）

| 路由 | 方法 | 入参 | 出参 |
|---|---|---|---|
| `/health` | GET | — | `{status, collection, count}` |
| `/ingest` | POST | multipart `file` | `{filename, chunks, doc_id}` |
| `/search` | POST | `{query, top_k}` | `{results: [{content, source, score}]}` |

**Spring AI 对接约定**：响应字段用 snake_case（与 Spring AI 的默认 JSON 映射一致）；`/search` 的 score 是 rerank 后的归一化分数。

## 5. 数据流

**入库**：
```
POST /ingest (file)
  → 存到 data/uploads/<filename>
  → DoclingParser.parse → Document
  → Chunker.chunk → [Chunk{content, source}]
  → BGEM3Embedder.encode → 每个 Chunk 填上 dense + sparse
  → MilvusStore.insert
  → {filename, chunks: N}
```

**检索**：
```
POST /search {query, top_k}
  → BGEM3Embedder.encode(query) → dense_q, sparse_q
  → MilvusStore.hybrid_search(dense_q, sparse_q, top_k*N)  # 多召回
  → Reranker.rerank(query, candidates, top_k) → top-k
  → {results: [{content, source, score}]}
```

## 6. 错误处理

各层抛领域异常（`MilvusStoreError`、`ParseError`、`EmbedError`），`api/` 层统一捕获转 HTTP：
- 解析失败 → 422
- Milvus 连接失败 → 503
- 其余 → 500，返回简短错误信息（不泄露 traceback）

## 7. 配置与运行

```bash
# 写 .env（可选，所有项都有默认值）
MILVUS_HOST=localhost
MILVUS_PORT=19530

# 启动
./venv/bin/uvicorn api.app:app --reload --port 8000
```

启动时**懒加载**模型（首次 ingest/search 才加载 BGE-M3 + reranker），避免冷启动卡死。

## 8. 依赖新增（已核实，venv 当前缺失）

实现前必须安装：

```bash
./venv/bin/pip install fastapi uvicorn python-multipart
```

已核实 `fastapi`、`uvicorn`、`multipart`、`starlette` 在 venv 中**全部缺失**。已就绪：`pydantic-settings`、`docling`(2.107.0)、`docling-core`(2.84.0)、`pymilvus`(3.0.0)、`FlagEmbedding`(1.4.0)。

## 9. 不做的事（YAGNI）

- 不做鉴权/CORS（本地服务，Spring AI 同机调用）
- 不做异步任务队列（PDF 不大，同步处理够用）
- 不做 ORM→MilvusClient 的旧脚本迁移
- 不做多 collection / 多租户
- 不做增量更新/版本管理
```
