import json
from kafka import KafkaConsumer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
import uuid

# Configs
QDRANT_COLLECTION = "full_pg_dump"
KAFKA_BOOTSTRAP = "localhost:9092"
KAFKA_TOPICS_PREFIX = "pgserver.public."

# Initialize Qdrant and model
qdrant = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Consumer for all topics starting with pgserver.public.
consumer = KafkaConsumer(
    bootstrap_servers=KAFKA_BOOTSTRAP,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="latest",
    group_id="pg_cdc_group"
)

# Subscribe to all relevant topics dynamically
from kafka.admin import KafkaAdminClient
admin = KafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP)
all_topics = admin.list_topics()
watched_topics = [t for t in all_topics if t.startswith(KAFKA_TOPICS_PREFIX)]
consumer.subscribe(watched_topics)
print("Subscribed to topics:", watched_topics)

# Listen for changes
print("Listening for changes...")

for message in consumer:
    payload = message.value
    if "payload" not in payload or not payload["payload"]:
        continue

    after = payload["payload"].get("after")
    if not after:
        continue  # skip deletes or tombstones

    table_name = message.topic.split(".")[-1]
    row_data = after

    # Convert row into text
    text = " ".join(str(v) for v in row_data.values() if v is not None)

    # Create vector
    try:
        vector = model.encode(text).tolist()
    except Exception as e:
        print("Vectorization failed:", e)
        continue

    # Upsert into Qdrant
    qdrant.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "table": table_name,
                    "row": row_data
                }
            )
        ]
    )
    print(f"Synced change from table `{table_name}` to Qdrant.")

