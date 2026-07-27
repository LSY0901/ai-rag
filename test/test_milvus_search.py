# test_milvus_search.py

from pymilvus import *

connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

collection = Collection("test_docs")

collection.load()

results = collection.search(
    data=[[0.1] * 1024],
    anns_field="vector",
    param={"metric_type": "COSINE"},
    limit=3,
    output_fields=["content"]
)

for hit in results[0]:
    print(hit.entity.get("content"))
