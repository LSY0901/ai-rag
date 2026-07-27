# test_milvus_insert.py

from pymilvus import *

connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

collection_name = "test_docs"

if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

fields = [
    FieldSchema(
        name="id",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=True
    ),
    FieldSchema(
        name="content",
        dtype=DataType.VARCHAR,
        max_length=1000
    ),
    FieldSchema(
        name="vector",
        dtype=DataType.FLOAT_VECTOR,
        dim=1024
    )
]

schema = CollectionSchema(fields)

collection = Collection(
    name=collection_name,
    schema=schema
)

collection.insert([
    ["测试文档"],
    [[0.1] * 1024]
])

collection.flush()

print("insert success")
