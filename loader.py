import psycopg2
import json
import uuid
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# --- Configuration ---
DB_CONFIG = {
    "dbname": "idris_db",
    "user": "postgres",
    "password": "",#add pass here
    "host": "localhost",
    "port": "5432"
}

QDRANT_COLLECTION = "full_pg_dump"
VECTOR_SIZE = 384  # For 'all-MiniLM-L6-v2'

# --- Setup ---
model = SentenceTransformer("all-MiniLM-L6-v2")
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

qdrant = QdrantClient("http://localhost:6333")

# --- Create or recreate collection ---
if qdrant.collection_exists(QDRANT_COLLECTION):
    qdrant.delete_collection(collection_name=QDRANT_COLLECTION)

qdrant.create_collection(
    collection_name=QDRANT_COLLECTION,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
)

# --- Get all tables in 'public' schema ---
cur.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
""")
tables = cur.fetchall()

all_points = []

for (table,) in tables:
    print(f" Reading table: {table}")

    try:
        # Get column names
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = %s", (table,))
        columns = [col[0] for col in cur.fetchall()]

        cur.execute(f'SELECT * FROM "{table}"')  # Use quotes to handle CamelCase or reserved keywords
        rows = cur.fetchall()
    except Exception as e:
        print(f" Skipping {table}: {e}")
        continue

    for row in rows:
        row_dict = {}
        text_chunks = []

        for col, val in zip(columns, row):
            if isinstance(val, memoryview):
                row_dict[col] = "<binary>"
            else:
                row_dict[col] = val
                if val is not None:
                    text_chunks.append(str(val))

        text = " ".join(text_chunks)

        try:
            embedding = model.encode(text).tolist()
        except Exception as e:
            print(f" Encoding failed on row: {e}")
            continue

        all_points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "table": table,
                "row": row_dict
            }
        ))

print(f"Uploading {len(all_points)} records to Qdrant...")

# --- Batch upload to Qdrant ---
BATCH_SIZE = 100
for i in tqdm(range(0, len(all_points), BATCH_SIZE)):
    qdrant.upsert(
        collection_name=QDRANT_COLLECTION,
        points=all_points[i:i + BATCH_SIZE]
    )

print(" Done: All tables and rows indexed in Qdrant.")

