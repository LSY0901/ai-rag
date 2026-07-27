# RAG 检索服务实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 把 `test/*.py` 里的零散探针脚本，整合成可被 Spring AI 调用的本地 RAG 检索服务（PDF 上传 → 解析 → 分块 → 向量化 → 入库 → 混合检索 → 重排 → FastAPI 接口）。

**架构：** 7 层单向依赖：config→storage→chunking→embedding→retrieval→service→api。每层只依赖下层，用 Pydantic 模型传递数据，`MilvusClient` 新 API + BGE-M3 一次 encode 产出 dense+sparse 做原生 hybrid search。

**技术栈：** Python 3.11、pymilvus 3.0.0（MilvusClient API，服务端 2.5.5）、FlagEmbedding 1.4.0（BGE-M3 + bge-reranker-v2-m3）、docling 2.107.0、docling-core 2.84.0（HybridChunker）、FastAPI、uvicorn、pytest。

**规格文档：** `docs/superpowers/specs/2026-06-29-rag-pipeline-design.md`

---

## 文件结构

```
ai-rag/
├── rag/                          # 主包
│   ├── __init__.py
│   ├── config.py                 # Settings：pydantic-settings，从 .env 读
│   ├── models.py                 # Pydantic 数据契约：Chunk, SearchHit, 请求/响应
│   ├── storage/
│   │   ├── __init__.py
│   │   └── milvus_store.py       # MilvusStore：MilvusClient 封装（Collection/索引/CRUD/hybrid_search）
│   ├── chunking/
│   │   ├── __init__.py
│   │   └── parser.py             # DoclingParser：PDF→DoclingDocument；Chunker：→list[Chunk]
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── bge_embedder.py       # BGEM3Embedder：一次 encode 出 dense+sparse
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py          # HybridRetriever：embed query → store.hybrid_search
│   │   └── reranker.py           # Reranker：bge-reranker-v2-m3 重排
│   ├── service/
│   │   ├── __init__.py
│   │   └── search_service.py     # SearchService：编排 ingest/search
│   └── exceptions.py             # 领域异常：ParseError, EmbedError, StoreError
├── api/
│   ├── __init__.py
│   └── app.py                    # FastAPI：/health, /ingest, /search
├── tests/                        # pytest 单测（可 mock）
│   ├── __init__.py
│   ├── conftest.py               # 共享 fixture
│   ├── test_config.py
│   ├── test_models.py
│   ├── test_milvus_store.py
│   ├── test_embedder.py
│   ├── test_parser.py
│   ├── test_retriever.py
│   ├── test_reranker.py
│   └── test_search_service.py
├── data/
│   └── uploads/                  # 上传的 PDF 落盘目录（gitignore）
├── scripts/
│   ├── ingest_cli.py             # 命令行入库（集成验证）
│   └── search_cli.py             # 命令行检索（集成验证）
├── .env.example                  # 配置模板
├── .gitignore
├── requirements.txt              # 锁定依赖（新建）
├── pytest.ini                    # pytest 配置
└── AGENTS.md                     # 已存在，更新
```

**职责边界：** 每个文件单一职责。`api/` 不直接碰 `storage/embedding`，只经 `service/`。`storage/` 不感知 embedding 维度细节，维度来自 `config`。

---

## Task 0：前置依赖与项目骨架

**文件：**
- 创建：`.gitignore`、`requirements.txt`、`.env.example`、`pytest.ini`
- 创建：`rag/__init__.py`、`rag/config.py`、`rag/models.py`、`rag/exceptions.py`（及各子包 `__init__.py`）

**说明：** 本任务建立可运行的地基。`transformers==4.49.0` 已在规格验证阶段安装并双重验证通过（Docling 可导入 + BGE-M3 可 encode）。`requirements.txt` 把它和缺失的 FastAPI 全家桶固定下来。

- [ ] **步骤 1：初始化 git 仓库 + .gitignore**

创建 `.gitignore`：

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/

# 虚拟环境（本地）
venv/

# 运行时数据
data/uploads/*
!data/uploads/.gitkeep

# 环境
.env

# IDE
.idea/

# macOS
.DS_Store
```

运行：

```bash
cd /Users/leo/ai-rag
git init
mkdir -p data/uploads && touch data/uploads/.gitkeep
git add .gitignore data/uploads/.gitkeep
git commit -m "chore: init repo with gitignore"
```

- [ ] **步骤 2：创建 requirements.txt**

创建 `requirements.txt`（固定所有运行依赖；transformers 4.49.0 是 Docling 可导入与 BGE-M3 兼容的已验证版本）：

```text
pymilvus==3.0.0
FlagEmbedding==1.4.0
transformers==4.49.0
docling==2.107.0
docling-core==2.84.0
torch==2.12.1
numpy==2.4.6
pydantic==2.13.4
pydantic-settings==2.14.2
fastapi
uvicorn
python-multipart
pytest
```

- [ ] **步骤 3：安装缺失的 FastAPI 全家桶 + pytest**

运行：

```bash
./venv/bin/pip install fastapi uvicorn python-multipart pytest
```

预期：安装成功，`fastapi`、`uvicorn`、`multipart`、`starlette`、`pytest` 都能 import。

- [ ] **步骤 4：创建 .env.example 和 pytest.ini**

创建 `.env.example`：

```bash
# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=rag_docs

# 模型
EMBEDDING_MODEL_PATH=/Users/leo/ai/models/bge-m3
RERANKER_MODEL_PATH=/Users/leo/ai/models/bge-reranker-v2-m3

# 分块与上传
CHUNK_MAX_TOKENS=1024
UPLOAD_DIR=data/uploads
```

创建 `pytest.ini`：

```ini
[pytest]
testpaths = tests
pythonpath = .
addopts = -q
```

- [ ] **步骤 5：创建包骨架**

创建空 `__init__.py`：`rag/__init__.py`、`rag/storage/__init__.py`、`rag/chunking/__init__.py`、`rag/embedding/__init__.py`、`rag/retrieval/__init__.py`、`rag/service/__init__.py`、`api/__init__.py`、`tests/__init__.py`。

运行验证骨架可导入：

```bash
./venv/bin/python -c "import rag, rag.storage, rag.chunking, rag.embedding, rag.retrieval, rag.service, api, tests; print('skeleton OK')"
```

预期：`skeleton OK`。

- [ ] **步骤 6：Commit 骨架**

```bash
git add requirements.txt .env.example pytest.ini rag/ api/ tests/
git commit -m "chore: project skeleton, requirements, pytest config"
```

---

## Task 1：config 层

**文件：**
- 创建：`rag/config.py`
- 测试：`tests/test_config.py`

- [ ] **步骤 1：编写失败的测试**

创建 `tests/test_config.py`：

```python
import os
from rag.config import Settings


def test_default_values():
    s = Settings()
    assert s.milvus_host == "localhost"
    assert s.milvus_port == 19530
    assert s.milvus_collection == "rag_docs"
    assert s.dense_dim == 1024
    assert s.embedding_model_path.endswith("bge-m3")
    assert s.reranker_model_path.endswith("bge-reranker-v2-m3")
    assert s.chunk_max_tokens == 1024
    assert s.upload_dir == "data/uploads"


def test_env_override(monkeypatch):
    monkeypatch.setenv("MILVUS_PORT", "19531")
    monkeypatch.setenv("MILVUS_COLLECTION", "custom_docs")
    s = Settings()
    assert s.milvus_port == 19531
    assert s.milvus_collection == "custom_docs"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./venv/bin/pytest tests/test_config.py -v`
预期：FAIL，报错 `ModuleNotFoundError: No module named 'rag.config'`。

- [ ] **步骤 3：编写实现**

创建 `rag/config.py`：

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """运行配置。从 .env 读，所有字段都有默认值。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "rag_docs"

    # 向量维度（BGE-M3 dense 固定 1024）
    dense_dim: int = 1024

    # 模型权重路径（仓库外）
    embedding_model_path: str = "/Users/leo/ai/models/bge-m3"
    reranker_model_path: str = "/Users/leo/ai/models/bge-reranker-v2-m3"

    # 分块
    chunk_max_tokens: int = 1024

    # 上传
    upload_dir: str = "data/uploads"

    @property
    def milvus_uri(self) -> str:
        return f"http://{self.milvus_host}:{self.milvus_port}"


settings = Settings()
```

- [ ] **步骤 4：运行测试验证通过**

运行：`./venv/bin/pytest tests/test_config.py -v`
预期：PASS（2 passed）。

- [ ] **步骤 5：Commit**

```bash
git add rag/config.py tests/test_config.py
git commit -m "feat(config): add Settings with env-driven config"
```

---

## Task 2：models 层（数据契约）

**文件：**
- 创建：`rag/models.py`
- 测试：`tests/test_models.py`

- [ ] **步骤 1：编写失败的测试**

创建 `tests/test_models.py`：

```python
from rag.models import Chunk, SearchHit, IngestResponse, SearchRequest, SearchResponse


def test_chunk_creation():
    c = Chunk(content="一段文本", source="doc.pdf")
    assert c.content == "一段文本"
    assert c.source == "doc.pdf"
    assert c.dense is None
    assert c.sparse is None


def test_chunk_with_vectors():
    c = Chunk(content="x", source="d.pdf", dense=[0.1] * 1024, sparse={"5": 0.3})
    assert len(c.dense) == 1024
    assert c.sparse == {"5": 0.3}


def test_search_request_defaults():
    req = SearchRequest(query="问题")
    assert req.top_k == 5


def test_search_hit_and_response():
    hit = SearchHit(content="c", source="s.pdf", score=0.9)
    resp = SearchResponse(results=[hit])
    assert resp.results[0].score == 0.9


def test_ingest_response():
    r = IngestResponse(filename="f.pdf", chunks=3)
    assert r.chunks == 3
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./venv/bin/pytest tests/test_models.py -v`
预期：FAIL，`ModuleNotFoundError: No module named 'rag.models'`。

- [ ] **步骤 3：编写实现**

创建 `rag/models.py`：

```python
"""全链路共享的 Pydantic 数据契约。"""
from typing import Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """文档分块。dense/sparse 在向量化后填入。"""
    content: str
    source: str
    dense: Optional[list[float]] = None
    sparse: Optional[dict[str, float]] = None


class SearchHit(BaseModel):
    content: str
    source: str
    score: float


# --- API 请求/响应 ---

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResponse(BaseModel):
    results: list[SearchHit]


class IngestResponse(BaseModel):
    filename: str
    chunks: int


class HealthResponse(BaseModel):
    status: str
    collection: str
    count: int
```

- [ ] **步骤 4：运行测试验证通过**

运行：`./venv/bin/pytest tests/test_models.py -v`
预期：PASS（5 passed）。

- [ ] **步骤 5：Commit**

```bash
git add rag/models.py tests/test_models.py
git commit -m "feat(models): add Chunk, SearchHit, request/response contracts"
```

---

## Task 3：exceptions 层

**文件：**
- 创建：`rag/exceptions.py`

无独立测试（纯异常类，由调用方测试覆盖）。

- [ ] **步骤 1：编写实现**

创建 `rag/exceptions.py`：

```python
"""领域异常。api 层统一捕获转 HTTP 状态码。"""


class RagError(Exception):
    """所有领域异常的基类。"""


class StoreError(RagError):
    """Milvus 存储层错误（连接、建表、读写）。"""


class ParseError(RagError):
    """文档解析/分块错误。"""


class EmbedError(RagError):
    """向量化错误。"""
```

- [ ] **步骤 2：验证可导入**

运行：`./venv/bin/python -c "from rag.exceptions import RagError, StoreError, ParseError, EmbedError; print('OK')"`
预期：`OK`。

- [ ] **步骤 3：Commit**

```bash
git add rag/exceptions.py
git commit -m "feat(exceptions): add domain exceptions"
```

---

## Task 4：storage 层 — MilvusStore（Step 1）

**文件：**
- 创建：`rag/storage/milvus_store.py`
- 测试：`tests/test_milvus_store.py`

**已验证的 API（写计划前用 inspect 确认）：**
- `MilvusClient(uri=...)` 构造
- `create_collection(collection_name, schema=..., ...)` + `MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)`
- `FieldSchema` / `DataType.SPARSE_FLOAT_VECTOR` / `add_field(...)`
- `prepare_index_params()` → `index_params.add_index(field_name, index_type, metric_type=...)`
- `insert(collection_name, data=[{...}, {...}])` 接受 list[dict]
- `hybrid_search(collection_name, reqs=[AnnSearchRequest(...)], ranker=RRFRanker(k=60), limit=N, output_fields=[...])`
- `AnnSearchRequest(data=dense, anns_field="dense", param={"metric_type":"COSINE"}, limit=N)`
- `delete(collection_name, filter='source == "x"')`

- [ ] **步骤 1：编写失败的测试（mock MilvusClient）**

创建 `tests/test_milvus_store.py`：

```python
from unittest.mock import MagicMock, patch

from rag.models import Chunk
from rag.storage.milvus_store import MilvusStore


def _chunk(content="c", source="s.pdf", dense=None, sparse=None):
    return Chunk(content=content, source=source,
                 dense=dense or [0.1] * 1024, sparse=sparse or {"5": 0.3})


@patch("rag.storage.milvus_store.MilvusClient")
def test_ensure_collection_creates_and_loads(mock_client_cls):
    store = MilvusStore(uri="http://localhost:19530", collection="rag_docs", dense_dim=1024)
    store.client = MagicMock()
    store.ensure_collection()
    store.client.create_collection.assert_called_once()
    store.client.create_index.assert_called_once()
    store.client.load_collection.assert_called_once()


@patch("rag.storage.milvus_store.MilvusClient")
def test_insert_maps_chunks_to_rows(mock_client_cls):
    store = MilvusStore(uri="http://localhost:19530", collection="rag_docs", dense_dim=1024)
    store.client = MagicMock()
    store.insert([_chunk(), _chunk(content="d")])
    args, kwargs = store.client.insert.call_args
    rows = kwargs["data"]
    assert len(rows) == 2
    assert set(rows[0].keys()) == {"content", "source", "dense", "sparse"}


@patch("rag.storage.milvus_store.MilvusClient")
def test_delete_by_source(mock_client_cls):
    store = MilvusStore(uri="http://localhost:19530", collection="rag_docs", dense_dim=1024)
    store.client = MagicMock()
    store.delete_by_source("doc.pdf")
    store.client.delete.assert_called_once()
    args, kwargs = store.client.delete.call_args
    assert 'source' in kwargs["filter"]


@patch("rag.storage.milvus_store.MilvusClient")
def test_hybrid_search_returns_chunks(mock_client_cls):
    store = MilvusStore(uri="http://localhost:19530", collection="rag_docs", dense_dim=1024)
    store.client = MagicMock()
    store.client.hybrid_search.return_value = [[
        {"id": 1, "entity": {"content": "命中", "source": "doc.pdf"}, "distance": 0.9}
    ]]
    hits = store.hybrid_search(dense=[0.1] * 1024, sparse={"5": 0.3}, limit=3)
    assert len(hits) == 1
    assert hits[0].content == "命中"
    assert hits[0].source == "doc.pdf"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./venv/bin/pytest tests/test_milvus_store.py -v`
预期：FAIL，`ModuleNotFoundError: No module named 'rag.storage.milvus_store'`。

- [ ] **步骤 3：编写实现**

创建 `rag/storage/milvus_store.py`：

```python
"""MilvusClient 封装：Collection、双索引、CRUD、hybrid_search。

用 pymilvus 3.x 的 MilvusClient 新 API（旧的 connections/Collection/utility 已 deprecated）。
"""
from pymilvus import (
    AnnSearchRequest,
    DataType,
    FieldSchema,
    MilvusClient,
    RRFRanker,
)

from rag.exceptions import StoreError
from rag.models import Chunk, SearchHit


class MilvusStore:
    def __init__(self, uri: str, collection: str, dense_dim: int):
        self.uri = uri
        self.collection = collection
        self.dense_dim = dense_dim
        self.client = MilvusClient(uri=uri)

    def ensure_collection(self) -> None:
        """幂等建表 + 双索引 + load。"""
        if self.client.has_collection(self.collection):
            self.client.load_collection(self.collection)
            return

        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("content", DataType.VARCHAR, max_length=8192)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.dense_dim)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(
            field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="IP"
        )

        self.client.create_collection(
            collection_name=self.collection, schema=schema, index_params=index_params
        )
        self.client.load_collection(self.collection)

    def insert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        rows = [
            {"content": c.content, "source": c.source, "dense": c.dense, "sparse": c.sparse}
            for c in chunks
        ]
        self.client.insert(collection_name=self.collection, data=rows)

    def delete_by_source(self, source: str) -> None:
        self.client.delete(
            collection_name=self.collection, filter=f'source == "{source}"'
        )

    def hybrid_search(self, dense: list[float], sparse: dict[str, float], limit: int) -> list[SearchHit]:
        """并行 dense + sparse 两路 anns_search，RRF 融合。"""
        dense_req = AnnSearchRequest(
            data=[dense], anns_field="dense", param={"metric_type": "COSINE"}, limit=limit
        )
        sparse_req = AnnSearchRequest(
            data=[sparse], anns_field="sparse", param={"metric_type": "IP"}, limit=limit
        )
        results = self.client.hybrid_search(
            collection_name=self.collection,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=60),
            limit=limit,
            output_fields=["content", "source"],
        )
        hits: list[SearchHit] = []
        for r in results[0]:
            entity = r.get("entity", {})
            hits.append(
                SearchHit(
                    content=entity.get("content", ""),
                    source=entity.get("source", ""),
                    score=float(r.get("distance", 0.0)),
                )
            )
        return hits

    def count(self) -> int:
        stats = self.client.get_collection_stats(self.collection)
        return int(stats.get("row_count", 0))
```

- [ ] **步骤 4：运行测试验证通过**

运行：`./venv/bin/pytest tests/test_milvus_store.py -v`
预期：PASS（4 passed）。注意 mock 掉 `MilvusClient`，不连真实 Milvus。

- [ ] **步骤 5：手动集成验证（连真实 Milvus）**

运行（先确保 Milvus 在 localhost:19530 运行）：

```bash
./venv/bin/python -c "
from rag.storage.milvus_store import MilvusStore
from rag.models import Chunk
s = MilvusStore(uri='http://localhost:19530', collection='rag_docs', dense_dim=1024)
s.ensure_collection()
s.insert([Chunk(content='集成测试文档', source='integ.pdf', dense=[0.1]*1024, sparse={'5':0.3})])
print('count after insert:', s.count())
hits = s.hybrid_search(dense=[0.1]*1024, sparse={'5':0.3}, limit=3)
print('search hits:', [(h.content, round(h.score,3)) for h in hits])
s.delete_by_source('integ.pdf')
print('集成 OK')
"
```

预期：建表成功、count≥1、能搜到"集成测试文档"、删除后 OK。若失败，根据真实 Milvus 报错调整 schema/索引参数。

- [ ] **步骤 6：Commit**

```bash
git add rag/storage/milvus_store.py tests/test_milvus_store.py
git commit -m "feat(storage): MilvusStore with hybrid search and dual index"
```

---

## Task 5：embedding 层 — BGEM3Embedder（Step 4）

**文件：**
- 创建：`rag/embedding/bge_embedder.py`
- 测试：`tests/test_embedder.py`

**已验证：** BGE-M3 在 transformers 4.49.0 下 `encode(return_dense=True, return_sparse=True)` 返回 `dense_vecs`(N,1024) + `lexical_weights`([{token_id:str: weight}]，单条 sparse key 数正常)。

- [ ] **步骤 1：编写失败的测试（mock 模型）**

创建 `tests/test_embedder.py`：

```python
from unittest.mock import MagicMock, patch

from rag.embedding.bge_embedder import BGEM3Embedder, EmbeddingResult


@patch("rag.embedding.bge_embedder.BGEM3FlagModel")
def test_encode_returns_dense_and_sparse(mock_model_cls):
    embedder = BGEM3Embedder(model_path="/fake/bge-m3")
    embedder._model = MagicMock()
    embedder._model.encode.return_value = {
        "dense_vecs": [[0.1] * 1024, [0.2] * 1024],
        "lexical_weights": [{"5": 0.3}, {"9": 0.1}],
    }
    result = embedder.encode(["文本一", "文本二"])
    assert isinstance(result, EmbeddingResult)
    assert len(result.dense) == 2
    assert len(result.dense[0]) == 1024
    assert result.sparse[0] == {"5": 0.3}


@patch("rag.embedding.bge_embedder.BGEM3FlagModel")
def test_lazy_load_called_once(mock_model_cls):
    embedder = BGEM3Embedder(model_path="/fake/bge-m3")
    embedder.encode(["x"])
    embedder.encode(["y"])
    assert mock_model_cls.call_count == 1
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./venv/bin/pytest tests/test_embedder.py -v`
预期：FAIL，`ModuleNotFoundError`。

- [ ] **步骤 3：编写实现**

创建 `rag/embedding/bge_embedder.py`：

```python
"""BGE-M3 向量化：一次 encode 同时产出 dense + sparse。"""
from dataclasses import dataclass

from FlagEmbedding import BGEM3FlagModel

from rag.exceptions import EmbedError


@dataclass
class EmbeddingResult:
    dense: list[list[float]]
    sparse: list[dict[str, float]]


class BGEM3Embedder:
    def __init__(self, model_path: str, use_fp16: bool = False):
        self.model_path = model_path
        self.use_fp16 = use_fp16
        self._model: BGEM3FlagModel | None = None

    def _ensure_model(self) -> None:
        if self._model is None:
            self._model = BGEM3FlagModel(self.model_path, use_fp16=self.use_fp16)

    def encode(self, texts: list[str]) -> EmbeddingResult:
        """一次调用产出 dense + sparse。"""
        if not texts:
            return EmbeddingResult(dense=[], sparse=[])
        self._ensure_model()
        try:
            out = self._model.encode(
                texts, return_dense=True, return_sparse=True, return_colbert_vecs=False
            )
        except Exception as e:
            raise EmbedError(f"BGE-M3 encode failed: {e}") from e
        dense = [v.tolist() if hasattr(v, "tolist") else list(v) for v in out["dense_vecs"]]
        sparse = [dict(w) for w in out["lexical_weights"]]
        return EmbeddingResult(dense=dense, sparse=sparse)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`./venv/bin/pytest tests/test_embedder.py -v`
预期：PASS（2 passed）。

- [ ] **步骤 5：手动集成验证（加载真实模型）**

运行：

```bash
./venv/bin/python -c "
from rag.embedding.bge_embedder import BGEM3Embedder
e = BGEM3Embedder('/Users/leo/ai/models/bge-m3')
r = e.encode(['订单状态查询', '今天天气如何'])
print('dense:', len(r.dense), len(r.dense[0]))
print('sparse keys:', [len(s) for s in r.sparse])
print('集成 OK')
"
```

预期：dense 2×1024，sparse 两条各若干 key。

- [ ] **步骤 6：Commit**

```bash
git add rag/embedding/bge_embedder.py tests/test_embedder.py
git commit -m "feat(embedding): BGEM3Embedder producing dense+sparse"
```

---

## Task 6：chunking 层 — DoclingParser + Chunker（Step 2, 3）

**文件：**
- 创建：`rag/chunking/parser.py`
- 测试：`tests/test_parser.py`

**已验证：**
- `DocumentConverter`（`from docling.document_converter import DocumentConverter`）在 transformers 4.49.0 可导入。
- `HybridChunker` 路径是 `docling_core.transforms.chunker.hybrid_chunker.HybridChunker`（不是常见的 `.transforms.hybrid_chunker.`）。
- `HybridChunker.chunk(dl_doc)` 返回迭代器，BaseChunk 有 `text` 和 `meta` 字段。
- `DocumentConverter().convert(path).document` 得到 DoclingDocument。

- [ ] **步骤 1：编写失败的测试（mock DocumentConverter/HybridChunker）**

创建 `tests/test_parser.py`：

```python
from unittest.mock import MagicMock, patch

from rag.chunking.parser import DoclingParser


@patch("rag.chunking.parser.HybridChunker")
@patch("rag.chunking.parser.DocumentConverter")
def test_parse_and_chunk(mock_conv_cls, mock_chunker_cls):
    # DocumentConverter().convert(path).document 返回一个假 DoclingDocument
    converter = MagicMock()
    mock_conv_cls.return_value = converter
    fake_doc = MagicMock(name="DoclingDocument")
    converter.convert.return_value.document = fake_doc

    # HybridChunker().chunk 返回带 text 的 BaseChunk
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
    assert chunks[0].dense is None  # 尚未向量化


@patch("rag.chunking.parser.DocumentConverter", side_effect=Exception("boom"))
def test_parse_error_wrapped(_):
    from rag.exceptions import ParseError
    parser = DoclingParser(model_path="/fake/bge-m3", max_tokens=512)
    try:
        parser.parse_and_chunk("/fake/x.pdf", source="x.pdf")
        assert False, "应抛 ParseError"
    except ParseError:
        pass
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./venv/bin/pytest tests/test_parser.py -v`
预期：FAIL，`ModuleNotFoundError`。

- [ ] **步骤 3：编写实现**

创建 `rag/chunking/parser.py`：

```python
"""PDF 解析 + 结构化分块。

HybridChunker 用 docling 文档结构分块（标题/段落感知），并控制 token 上限。
注意：HybridChunker 需要 tokenizer；用 Huggingface tokenizer，model_path 传 bge-m3 即可。
"""
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

from rag.exceptions import ParseError
from rag.models import Chunk


class DoclingParser:
    def __init__(self, model_path: str, max_tokens: int):
        # HybridChunker 用 tokenizer_name 控制 chunk token 上限
        self._converter = DocumentConverter()
        self._chunker = HybridChunker(tokenizer=model_path, chunk_max_tokens=max_tokens)

    def parse_and_chunk(self, path: str, source: str) -> list[Chunk]:
        try:
            result = self._converter.convert(path)
            dl_doc = result.document
            chunks = [
                Chunk(content=c.text, source=source)
                for c in self._chunker.chunk(dl_doc)
                if c.text and c.text.strip()
            ]
        except Exception as e:
            raise ParseError(f"解析/分块失败 [{source}]: {e}") from e
        return chunks
```

- [ ] **步骤 4：运行测试验证通过**

运行：`./venv/bin/pytest tests/test_parser.py -v`
预期：PASS（2 passed）。

- [ ] **步骤 5：手动集成验证（解析真实 PDF）**

如果没有现成 PDF，先生成一个最小 PDF 放入 `data/uploads/`：

```bash
./venv/bin/python -c "
from reportlab.lib.pages import letter
from reportlab.pdfgen import canvas
" 2>/dev/null || echo "（reportlab 未装，请手动放一个测试 PDF 到 data/uploads/test.pdf）"
```

如果 reportlab 不可用，跳过生成，直接放任意真实 PDF 到 `data/uploads/sample.pdf`，然后：

```bash
./venv/bin/python -c "
from rag.chunking.parser import DoclingParser
p = DoclingParser(model_path='/Users/leo/ai/models/bge-m3', max_tokens=512)
chunks = p.parse_and_chunk('data/uploads/sample.pdf', source='sample.pdf')
print('chunk count:', len(chunks))
print('first chunk:', (chunks[0].content[:80] if chunks else 'NONE'))
print('集成 OK')
" 2>&1 | tail -5
```

预期：解析成功，产出若干 chunk，第一个 chunk 有内容。

- [ ] **步骤 6：Commit**

```bash
git add rag/chunking/parser.py tests/test_parser.py
git commit -m "feat(chunking): DoclingParser with HybridChunker"
```

---

## Task 7：retrieval 层 — HybridRetriever + Reranker（Step 6, 7）

**文件：**
- 创建：`rag/retrieval/retriever.py`、`rag/retrieval/reranker.py`
- 测试：`tests/test_retriever.py`、`tests/test_reranker.py`

- [ ] **步骤 1：编写 retriever 失败测试**

创建 `tests/test_retriever.py`：

```python
from unittest.mock import MagicMock

from rag.models import Chunk, SearchHit
from rag.retrieval.retriever import HybridRetriever


def test_retrieve_calls_store_and_returns_hits():
    embedder = MagicMock()
    embedder.encode.return_value = MagicMock(dense=[[0.1] * 1024], sparse=[{"5": 0.3}])
    store = MagicMock()
    store.hybrid_search.return_value = [SearchHit(content="c", source="s.pdf", score=0.8)]
    r = HybridRetriever(embedder=embedder, store=store, fetch_k=10)
    hits = r.retrieve(query="问题", top_k=5)
    assert len(hits) == 1
    assert hits[0].content == "c"
    # 用 fetch_k（> top_k）召回，交给 reranker 再裁剪
    store.hybrid_search.assert_called_once()
    _, kwargs = store.hybrid_search.call_args
    assert kwargs["limit"] == 10
```

- [ ] **步骤 2：编写 reranker 失败测试**

创建 `tests/test_reranker.py`：

```python
from unittest.mock import MagicMock, patch

from rag.models import SearchHit
from rag.retrieval.reranker import Reranker


@patch("rag.retrieval.reranker.FlagReranker")
def test_rerank_reorders_by_score(mock_cls):
    r = Reranker(model_path="/fake/reranker")
    r._model = MagicMock()
    # 第一条 query-candidate 分数低，第二条高 → 重排后第二条在前
    r._model.compute_score.return_value = [0.1, 0.9]
    candidates = [
        SearchHit(content="低相关", source="a.pdf", score=0.5),
        SearchHit(content="高相关", source="b.pdf", score=0.5),
    ]
    out = r.rerank(query="问题", candidates=candidates, top_k=2)
    assert len(out) == 2
    assert out[0].content == "高相关"  # 分数高的排前
    assert out[0].score >= out[1].score


@patch("rag.retrieval.reranker.FlagReranker")
def test_rerank_respects_top_k(mock_cls):
    r = Reranker(model_path="/fake/reranker")
    r._model = MagicMock()
    r._model.compute_score.return_value = [0.9, 0.1, 0.5]
    candidates = [SearchHit(content=f"c{i}", source=f"{i}.pdf", score=0.0) for i in range(3)]
    out = r.rerank(query="q", candidates=candidates, top_k=2)
    assert len(out) == 2
```

- [ ] **步骤 3：运行测试验证失败**

运行：`./venv/bin/pytest tests/test_retriever.py tests/test_reranker.py -v`
预期：FAIL，`ModuleNotFoundError`。

- [ ] **步骤 4：编写 HybridRetriever 实现**

创建 `rag/retrieval/retriever.py`：

```python
"""混合检索：embed query → store.hybrid_search（dense+sparse, RRF 融合）。"""
from rag.models import SearchHit


class HybridRetriever:
    def __init__(self, embedder, store, fetch_k: int = 20):
        self.embedder = embedder
        self.store = store
        self.fetch_k = fetch_k  # 多召回，交给 reranker 裁剪

    def retrieve(self, query: str, top_k: int) -> list[SearchHit]:
        emb = self.embedder.encode([query])
        return self.store.hybrid_search(
            dense=emb.dense[0], sparse=emb.sparse[0], limit=max(self.fetch_k, top_k)
        )
```

- [ ] **步骤 5：编写 Reranker 实现**

创建 `rag/retrieval/reranker.py`：

```python
"""BGE-Reranker-v2-m3 重排。"""
from FlagEmbedding import FlagReranker

from rag.exceptions import EmbedError
from rag.models import SearchHit


class Reranker:
    def __init__(self, model_path: str, use_fp16: bool = False):
        self.model_path = model_path
        self.use_fp16 = use_fp16
        self._model: FlagReranker | None = None

    def _ensure_model(self) -> None:
        if self._model is None:
            self._model = FlagReranker(self.model_path, use_fp16=self.use_fp16)

    def rerank(self, query: str, candidates: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not candidates:
            return []
        self._ensure_model()
        pairs = [[query, c.content] for c in candidates]
        try:
            scores = self._model.compute_score(pairs)
        except Exception as e:
            raise EmbedError(f"rerank failed: {e}") from e
        # compute_score 单条返回 float，多条返回 list
        if isinstance(scores, (int, float)):
            scores = [float(scores)]
        scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [
            SearchHit(content=c.content, source=c.source, score=float(s))
            for c, s in scored[:top_k]
        ]
```

- [ ] **步骤 6：运行测试验证通过**

运行：`./venv/bin/pytest tests/test_retriever.py tests/test_reranker.py -v`
预期：PASS（3 passed）。

- [ ] **步骤 7：手动集成验证（真实 reranker）**

运行：

```bash
./venv/bin/python -c "
from rag.retrieval.reranker import Reranker
from rag.models import SearchHit
r = Reranker('/Users/leo/ai/models/bge-reranker-v2-m3')
cs = [SearchHit(content='订单状态查询方法', source='a.pdf', score=0),
      SearchHit(content='今天天气很好', source='b.pdf', score=0)]
out = r.rerank('如何查询订单状态', cs, top_k=2)
print([(h.content, round(h.score,3)) for h in out])
print('集成 OK')
"
```

预期：相关项（订单状态）排在前且分数更高。

- [ ] **步骤 8：Commit**

```bash
git add rag/retrieval/ tests/test_retriever.py tests/test_reranker.py
git commit -m "feat(retrieval): HybridRetriever and Reranker"
```

---

## Task 8：service 层 — SearchService（Step 8）

**文件：**
- 创建：`rag/service/search_service.py`
- 测试：`tests/test_search_service.py`

- [ ] **步骤 1：编写失败的测试**

创建 `tests/test_search_service.py`：

```python
from unittest.mock import MagicMock

from rag.models import SearchHit
from rag.service.search_service import SearchService


def test_ingest_chains_parse_embed_store():
    parser = MagicMock()
    parser.parse_and_chunk.return_value = []  # 实际由 embed/store 处理，见下
    embedder = MagicMock()
    store = MagicMock()
    svc = SearchService(parser=parser, embedder=embedder, store=store, reranker=MagicMock())

    # 重新设定 parser 返回带内容但无向量的 chunk
    from rag.models import Chunk
    parser.parse_and_chunk.return_value = [Chunk(content="c1", source="d.pdf"),
                                           Chunk(content="c2", source="d.pdf")]
    embedder.encode.return_value = MagicMock(dense=[[0.1]*1024, [0.2]*1024],
                                             sparse=[{"5": 0.3}, {"9": 0.1}])

    n = svc.ingest(path="/fake/d.pdf", filename="d.pdf")
    assert n == 2
    store.delete_by_source.assert_called_once_with("d.pdf")
    store.insert.assert_called_once()
    inserted = store.insert.call_args[0][0]
    assert inserted[0].dense is not None  # 已向量化
    assert inserted[0].sparse is not None


def test_search_chains_retrieve_rerank():
    retriever = MagicMock()
    reranker = MagicMock()
    reranker.rerank.return_value = [SearchHit(content="答案", source="d.pdf", score=0.9)]
    svc = SearchService(parser=MagicMock(), embedder=MagicMock(), store=MagicMock(),
                        reranker=reranker, retriever=retriever)
    hits = svc.search(query="问题", top_k=5)
    assert hits[0].content == "答案"
    reranker.rerank.assert_called_once()
```

- [ ] **步骤 2：运行测试验证失败**

运行：`./venv/bin/pytest tests/test_search_service.py -v`
预期：FAIL，`ModuleNotFoundError`。

- [ ] **步骤 3：编写实现**

创建 `rag/service/search_service.py`：

```python
"""业务编排层：api 唯一依赖。编排 parse→chunk→embed→insert 和 retrieve→rerank。"""
from rag.models import SearchHit


class SearchService:
    def __init__(self, parser, embedder, store, reranker, retriever=None):
        self.parser = parser
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.retriever = retriever  # 可选；若为 None 在 search 时懒构造

    def ingest(self, path: str, filename: str) -> int:
        chunks = self.parser.parse_and_chunk(path, source=filename)
        if not chunks:
            return 0
        emb = self.embedder.encode([c.content for c in chunks])
        for c, dense, sparse in zip(chunks, emb.dense, emb.sparse):
            c.dense = dense
            c.sparse = sparse
        # 同名文件先清理再写入，避免重复
        self.store.delete_by_source(filename)
        self.store.insert(chunks)
        return len(chunks)

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        if self.retriever is None:
            from rag.retrieval.retriever import HybridRetriever
            self.retriever = HybridRetriever(self.embedder, self.store)
        candidates = self.retriever.retrieve(query, top_k)
        return self.reranker.rerank(query, candidates, top_k)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`./venv/bin/pytest tests/test_search_service.py -v`
预期：PASS（2 passed）。

- [ ] **步骤 5：Commit**

```bash
git add rag/service/search_service.py tests/test_search_service.py
git commit -m "feat(service): SearchService orchestration"
```

---

## Task 9：api 层 — FastAPI（Step 9）

**文件：**
- 创建：`api/app.py`
- 依赖注入：`api/dependencies.py`

- [ ] **步骤 1：编写实现（FastAPI + 懒加载单例依赖）**

创建 `api/dependencies.py`（懒加载，避免启动即加载模型卡死）：

```python
"""应用级单例，懒加载（首次 ingest/search 才加载 BGE-M3/reranker）。"""
from functools import lru_cache

from rag.chunking.parser import DoclingParser
from rag.config import settings
from rag.embedding.bge_embedder import BGEM3Embedder
from rag.retrieval.reranker import Reranker
from rag.retrieval.retriever import HybridRetriever
from rag.service.search_service import SearchService
from rag.storage.milvus_store import MilvusStore


@lru_cache
def get_store() -> MilvusStore:
    store = MilvusStore(
        uri=settings.milvus_uri,
        collection=settings.milvus_collection,
        dense_dim=settings.dense_dim,
    )
    store.ensure_collection()
    return store


@lru_cache
def get_embedder() -> BGEM3Embedder:
    return BGEM3Embedder(settings.embedding_model_path)


@lru_cache
def get_parser() -> DoclingParser:
    return DoclingParser(model_path=settings.embedding_model_path, max_tokens=settings.chunk_max_tokens)


@lru_cache
def get_reranker() -> Reranker:
    return Reranker(settings.reranker_model_path)


@lru_cache
def get_search_service() -> SearchService:
    embedder = get_embedder()
    store = get_store()
    return SearchService(
        parser=get_parser(),
        embedder=embedder,
        store=store,
        reranker=get_reranker(),
        retriever=HybridRetriever(embedder, store),
    )
```

创建 `api/app.py`：

```python
"""FastAPI：/health, /ingest, /search。供 Spring AI 调用。"""
import os

from fastapi import FastAPI, File, HTTPException, UploadFile

from api.dependencies import get_search_service, get_store
from rag.exceptions import EmbedError, ParseError, StoreError
from rag.models import HealthResponse, IngestResponse, SearchRequest, SearchResponse

app = FastAPI(title="RAG Retrieval Service")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    store = get_store()
    try:
        count = store.count()
    except StoreError:
        count = -1
    return HealthResponse(status="ok", collection=store.collection, count=count)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    os.makedirs("data/uploads", exist_ok=True)
    dest = os.path.join("data/uploads", file.filename)
    with open(dest, "wb") as f:
        f.write(await file.read())
    try:
        n = get_search_service().ingest(path=dest, filename=file.filename)
    except ParseError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except (StoreError, EmbedError) as e:
        raise HTTPException(status_code=503, detail=str(e))
    return IngestResponse(filename=file.filename, chunks=n)


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    try:
        hits = get_search_service().search(query=req.query, top_k=req.top_k)
    except (StoreError, EmbedError) as e:
        raise HTTPException(status_code=503, detail=str(e))
    return SearchResponse(results=hits)
```

- [ ] **步骤 2：验证 app 可启动导入**

运行：

```bash
./venv/bin/python -c "from api.app import app; print([r.path for r in app.routes])"
```

预期：输出包含 `/health`、`/ingest`、`/search`。

- [ ] **步骤 3：启动服务并端到端验证**

启动（后台）：

```bash
./venv/bin/uvicorn api.app:app --port 8000 &
sleep 3
```

健康检查：

```bash
curl -s http://localhost:8000/health
```

预期：`{"status":"ok","collection":"rag_docs","count":...}`

- [ ] **步骤 4：端到端检索验证（需已 ingest 过文档）**

```bash
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"订单状态","top_k":3}'
```

预期：返回 `{"results":[{"content":...,"source":...,"score":...}]}`。

- [ ] **步骤 5：关闭服务 + Commit**

```bash
kill %1 2>/dev/null
git add api/
git commit -m "feat(api): FastAPI /health /ingest /search endpoints"
```

---

## Task 10：端到端集成验证 + 文档收尾

**文件：**
- 创建：`scripts/ingest_cli.py`、`scripts/search_cli.py`
- 修改：`AGENTS.md`（补充新结构与运行命令）

- [ ] **步骤 1：编写 ingest_cli**

创建 `scripts/ingest_cli.py`：

```python
"""命令行入库：python scripts/ingest_cli.py <pdf_path>"""
import sys

from api.dependencies import get_search_service

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python scripts/ingest_cli.py <pdf_path>")
        sys.exit(1)
    path = sys.argv[1]
    filename = path.rsplit("/", 1)[-1]
    n = get_search_service().ingest(path=path, filename=filename)
    print(f"已入库 {filename}: {n} 个 chunk")
```

- [ ] **步骤 2：编写 search_cli**

创建 `scripts/search_cli.py`：

```python
"""命令行检索：python scripts/search_cli.py <query> [top_k]"""
import sys

from api.dependencies import get_search_service

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/search_cli.py <query> [top_k]")
        sys.exit(1)
    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    for h in get_search_service().search(query, top_k):
        print(f"[{h.score:.3f}] {h.source}: {h.content[:100]}")
```

- [ ] **步骤 3：完整端到端跑通**

准备一个真实 PDF（放 `data/uploads/sample.pdf`），依次跑：

```bash
./venv/bin/python scripts/ingest_cli.py data/uploads/sample.pdf
./venv/bin/python scripts/search_cli.py "查询内容" 3
```

预期：入库报 chunk 数；检索返回带分数的命中片段。

- [ ] **步骤 4：跑全量单测**

运行：

```bash
./venv/bin/pytest -v
```

预期：所有单测 PASS（约 16-18 个）。

- [ ] **步骤 5：更新 AGENTS.md**

在 `AGENTS.md` 中补充：包结构（`rag/` 7 层 + `api/`）、运行命令（`uvicorn api.app:app --port 8000`、`scripts/*_cli.py`）、测试命令（`./venv/bin/pytest`）、transformers 4.49.0 约束说明。

- [ ] **步骤 6：Commit 收尾**

```bash
git add scripts/ AGENTS.md
git commit -m "feat: e2e CLI scripts and docs"
```

---

## 自检

**1. 规格覆盖度：**

| 规格章节 | 覆盖任务 |
|---|---|
| Step 1 Milvus 基础封装 | Task 4 |
| Step 2 Docling Parser | Task 6 |
| Step 3 Chunk | Task 6 |
| Step 4 BGE-M3 Embedding | Task 5 |
| Step 5 入库 | Task 8（service.ingest 编排） |
| Step 6 Hybrid Retrieval | Task 4（hybrid_search）+ Task 7（HybridRetriever） |
| Step 7 Reranker | Task 7 |
| Step 8 Search Service | Task 8 |
| Step 9 FastAPI | Task 9 |
| config / models / exceptions | Task 1/2/3 |
| 依赖安装 + transformers 约束 | Task 0 |

全部 9 步 + 基础设施都有对应任务。✅

**2. 占位符扫描：** 无 TODO/待定/「类似上文」。每个代码步骤都含完整代码块。✅

**3. 类型一致性：**
- `Chunk`（content/source/dense/sparse）在 Task 2 定义，Task 4/5/6/8 使用一致。✅
- `SearchHit`（content/source/score）Task 2 定义，Task 4/7/8 使用一致。✅
- `MilvusStore.hybrid_search(dense, sparse, limit)` Task 4 定义，Task 7 `HybridRetriever.retrieve` 调用签名一致。✅
- `BGEM3Embedder.encode(texts)->EmbeddingResult` Task 5 定义，Task 7/8 调用一致（`.dense`/`.sparse` 属性）。✅
- `Reranker.rerank(query, candidates, top_k)` Task 7 定义，Task 8 调用一致。✅
- `SearchService.ingest(path, filename)` / `.search(query, top_k)` Task 8 定义，Task 9 调用一致。✅

**注意点（非阻塞，实现时留意）：**
- Task 6 的 `HybridChunker(tokenizer=..., chunk_max_tokens=...)` 参数名需在集成验证时按实际 docling-core 2.84.0 的签名确认（inspect 显示其 `__init__` 是 pydantic `**data`，字段名以实测为准）。如字段名不符，用实测正确的字段名。
- Task 4 的 `get_collection_stats` 返回结构以实际 Milvus 2.5.5 为准；`count` 实现做了 `.get("row_count",0)` 容错。
- 两个集成验证步骤（Task 6 解析真实 PDF、Task 10 端到端）依赖用户提供/放置一个真实 PDF 文件。
