# create_index.py

from pymilvus import *

connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

collection = Collection("test_docs")

collection.create_index(
    field_name="vector",
    index_params={
        "index_type": "AUTOINDEX",
        "metric_type": "COSINE"
    }
)

collection.load()

print("index created")
