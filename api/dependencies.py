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
    return DoclingParser(
        model_path=settings.embedding_model_path,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )


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
