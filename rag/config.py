from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """运行配置。从 .env 读，所有字段都有默认值。"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "rag_blocks_v3"

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

    # 解析：扫描件占 90%，OCR 默认开；Docling 无引擎时自动跳过不崩
    ocr_enabled: bool = True

    # VLM 图片描述：默认本地 SmolVLM；endpoint 非空则走 API
    vlm_enabled: bool = True
    vlm_model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    vlm_endpoint: str = ""

    # 对外暴露的图片 URL 前缀，为空则 image_path 保持相对路径
    # 例：http://rag-host:8000 → /search 直接返 http://rag-host:8000/files/images/x.png
    public_base_url: str = ""

    @property
    def milvus_uri(self) -> str:
        return f"http://{self.milvus_host}:{self.milvus_port}"


settings = Settings()
