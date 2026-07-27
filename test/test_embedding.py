# test_embedding.py

from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel(
    "/Users/leo/ai/models/bge-m3",
    use_fp16=False
)

result = model.encode(
    ["这是一个测试文档"]
)

print(result["dense_vecs"].shape)
print(len(result["dense_vecs"][0]))
