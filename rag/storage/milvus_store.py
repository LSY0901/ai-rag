"""MilvusClient 封装：Collection、双索引、CRUD、hybrid_search。

用 pymilvus 3.x 的 MilvusClient 新 API（旧的 connections/Collection/utility 已 deprecated）。
"""
from pymilvus import (
    AnnSearchRequest,
    DataType,
    MilvusClient,
    RRFRanker,
)

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
        schema.add_field("page_no", DataType.INT64)  # -1 表示未知
        schema.add_field("headings", DataType.VARCHAR, max_length=2048)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("block_type", DataType.VARCHAR, max_length=32)
        schema.add_field("image_path", DataType.VARCHAR, max_length=1024)
        schema.add_field("ocr_text", DataType.VARCHAR, max_length=8192)
        schema.add_field("vlm_caption", DataType.VARCHAR, max_length=4096)
        schema.add_field("vlm_status", DataType.VARCHAR, max_length=32)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense", index_type="AUTOINDEX", metric_type="COSINE"
        )
        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        self.client.create_collection(
            collection_name=self.collection, schema=schema, index_params=index_params
        )
        self.client.load_collection(self.collection)

    def insert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        rows = [
            {
                "content": c.content,
                "source": c.source,
                "dense": c.dense,
                "sparse": c.sparse,
                "page_no": c.page_no if c.page_no is not None else -1,
                "headings": " / ".join(c.headings),
                "chunk_index": c.chunk_index,
                "block_type": c.block_type or "text",
                "image_path": c.image_path or "",
                "ocr_text": c.ocr_text or "",
                "vlm_caption": c.vlm_caption or "",
                "vlm_status": c.vlm_status or "ok",
            }
            for c in chunks
        ]
        self.client.insert(collection_name=self.collection, data=rows)
        self.client.flush(self.collection)

    def delete_by_source(self, source: str) -> None:
        self.client.delete(
            collection_name=self.collection, filter=f'source == "{source}"'
        )

    def hybrid_search(
        self, dense: list[float], sparse: dict[str, float], limit: int
    ) -> list[SearchHit]:
        """并行 dense + sparse 两路 anns_search，RRF 融合。"""
        dense_req = AnnSearchRequest(
            data=[dense],
            anns_field="dense",
            param={"metric_type": "COSINE"},
            limit=limit,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse],
            anns_field="sparse",
            param={"metric_type": "IP"},
            limit=limit,
        )
        results = self.client.hybrid_search(
            collection_name=self.collection,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=60),
            limit=limit,
            output_fields=["content", "source", "page_no", "headings", "chunk_index", "block_type", "image_path", "ocr_text", "vlm_caption", "vlm_status"],
        )
        hits: list[SearchHit] = []
        for r in results[0]:
            entity = r.get("entity", {})
            page_no = entity.get("page_no", -1)
            headings = entity.get("headings", "")
            hits.append(
                SearchHit(
                    content=entity.get("content", ""),
                    source=entity.get("source", ""),
                    score=float(r.get("distance", 0.0)),
                    page_no=None if page_no == -1 else int(page_no),
                    headings=[h for h in headings.split(" / ") if h],
                    chunk_index=int(entity.get("chunk_index", 0)),
                    block_type=str(entity.get("block_type", "text") or "text"),
                    image_path=str(entity.get("image_path", "") or "") or None,
                    ocr_text=str(entity.get("ocr_text", "") or ""),
                    vlm_caption=str(entity.get("vlm_caption", "") or ""),
                    vlm_status=str(entity.get("vlm_status", "ok") or "ok"),
                )
            )
        return hits

    def count(self) -> int:
        stats = self.client.get_collection_stats(self.collection)
        return int(stats.get("row_count", 0))
