"""领域异常。api 层统一捕获转 HTTP 状态码。"""


class RagError(Exception):
    """所有领域异常的基类。"""


class StoreError(RagError):
    """Milvus 存储层错误（连接、建表、读写）。"""


class ParseError(RagError):
    """文档解析/分块错误。"""


class EmbedError(RagError):
    """向量化错误。"""
