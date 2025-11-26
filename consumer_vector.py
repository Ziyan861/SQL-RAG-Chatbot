from kafka import KafkaConsumer
import json
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

# Initialize Qdrant client
qdrant = QdrantClient(host="localhost", port=6333)

# Create collection if not exists
try:
    qdrant.get_collection("products_rag")
except Exception:
    qdrant.recreate_collection(
        collection_name="products_rag",
        vectors_config={"size": 768, "distance": "Cosine"}  # Adjust 'size' to match embedding dimension
    )

# Initialize embedding model
embed_model = OllamaEmbeddings(model="nomic-embed-text")

# Kafka consumer
consumer = KafkaConsumer(
    "mysql-server.inventory.products",
    bootstrap_servers="localhost:9092",
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("Listening for product changes from MySQL via Kafka...")

for message in consumer:
    try:
        value = message.value
        after = value.get("payload", {}).get("after", {})
        
        if not after:
            continue  # skip tombstone or empty messages

        # Safely extract and stringify fields
        product_id = int(after.get("id", 0))
        name = str(after.get("name", ""))
        description = str(after.get("description", ""))
        quantity = str(after.get("quantity", ""))
        price = str(after.get("price", ""))
        last_updated = str(after.get("last_updated", ""))

        # Construct a clean string
        doc_text = f"Product: {name}. Description: {description}. Quantity: {quantity}. Price: {price}. Last Updated: {last_updated}"

        # Embed
        vector = embed_model.embed_query(doc_text)

        # Upsert to Qdrant
        qdrant.upsert(
            collection_name="products_rag",
            points=[
                PointStruct(
                    id=product_id,
                    vector=vector,
                    payload={"content": doc_text}
                )
            ]
        )

        print(f"Indexed: {name}")

    except Exception as e:
        print(f"Error processing message: {e}")

