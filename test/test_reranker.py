# test_reranker.py

from FlagEmbedding import FlagReranker

reranker = FlagReranker(
    "/Users/leo/ai/models/bge-reranker-v2-m3",
    use_fp16=False
)

pairs = [
    ["如何查询订单状态", "订单状态查询方法"],
    ["如何查询订单状态", "天气很好"]
]

scores = reranker.compute_score(pairs)

print(scores)
