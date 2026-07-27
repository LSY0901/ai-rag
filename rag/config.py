from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """运行配置。从 .env 读，所有字段都有默认值。"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

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
    chunk_max_tokens: int = 450
    chunk_overlap_tokens: int = 60

    # 上传
    upload_dir: str = "data/uploads"

    @property
    def milvus_uri(self) -> str:
        return f"http://{self.milvus_host}:{self.milvus_port}"


settings = Settings()
