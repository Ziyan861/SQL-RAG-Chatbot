from kafka import KafkaConsumer
import json
import uuid
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from langchain_community.llms import Ollama
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Connect to Qdrant
qdrant = QdrantClient(host="localhost", port=6333)

# Create collection if it doesn't exist
collection_name = "rag_kafka_docs"
if not qdrant.collection_exists(collection_name):
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

# Embedding model
embed_model = OllamaEmbeddings(model="nomic-embed-text")

# Kafka consumer
consumer = KafkaConsumer(
    "rag-topic",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="latest",
    group_id="rag-consumer"
)

print("Listening for incoming documents from Kafka...")

# Consume loop
for message in consumer:
    doc = message.value
    content = doc["content"]
    print(f"\n Received: {content}")

    vector = embed_model.embed_query(content)

    qdrant.upsert(
    collection_name="rag_kafka_docs",
    points=[{
        "id": doc["id"],
        "vector": vector,
        "payload": {
            "page_content": content  
        }
    }]
)

