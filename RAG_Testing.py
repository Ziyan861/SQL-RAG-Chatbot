#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd



# In[ ]:





# In[7]:


url = "https://people.sc.fsu.edu/~jburkardt/data/csv/cities.csv"

df = pd.read_csv(url, skipinitialspace=True)
df.columns = df.columns.str.strip().str.replace('"', '')
df.head()


# In[8]:


def dms_to_decimal(degrees, minutes, seconds, direction):
    decimal = degrees + minutes / 60 + seconds / 3600
    if direction in ["S", "W"]:
        decimal *= -1
    return decimal

df["Latitude"] = df.apply(
    lambda row: dms_to_decimal(row["LatD"], row["LatM"], row["LatS"], row["NS"]),
    axis=1
)

df["Longitude"] = df.apply(
    lambda row: dms_to_decimal(row["LonD"], row["LonM"], row["LonS"], row["EW"]),
    axis=1
)


# In[9]:


texts = []
for row in df.itertuples():
    city = row.City
    state = row.State
    lat = round(row.Latitude, 4)
    lon = round(row.Longitude, 4)
    sentence = f"{city} is a city in {state}. It is located at latitude {lat} and longitude {lon}."
    texts.append(sentence)

texts[:5]


# In[10]:


from langchain.embeddings import OllamaEmbeddings

embed_model = OllamaEmbeddings(model="nomic-embed-text") 
embeddings = embed_model.embed_documents(texts)


# In[6]:


from langchain.vectorstores import Qdrant

qdrant = QdrantClient(url="http://localhost:6333")


qdrant.recreate_collection(
    collection_name="us_cities",
    vectors_config=VectorParams(
        size=len(embeddings[0]),   # vector size from embedding
        distance=Distance.COSINE   # for semantic similarity
    )
)


qdrant.upsert(
    collection_name="us_cities",
    points=[
        PointStruct(id=i, vector=embeddings[i], payload={"page_content": texts[i]})
        for i in range(len(texts))
    ]
)


# In[11]:


from langchain.vectorstores import Qdrant as QdrantVectorStore
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from qdrant_client.models import VectorParams, Distance

vector_store = QdrantVectorStore(
    client=qdrant,
    collection_name="us_cities",
    embeddings=embed_model
)

retriever = vector_store.as_retriever()

llm = Ollama(model="deepseek-r1:1.5b")


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)


# In[ ]:





# In[8]:


query = "Which cities in the dataset are located in WA?"
response = qa_chain(query)

print("Answer:", response['result'])

# Optional: Check sources
for doc in response['source_documents']:
    print("\nSource:", doc.page_content)


# In[10]:


from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

# Get collection status
status = client.get_collection("us_cities")
print(status)


# In[ ]:





# In[24]:


from qdrant_client.http.models import OptimizersConfigDiff

qdrant._client.update_collection(
    collection_name="us_cities",
    optimizer_config=OptimizersConfigDiff(
    indexing_threshold=1)
)



# In[26]:


query = "Which cities are located above 47 degrees latitude?"
response = qa_chain(query)
print("Answer:", response['result'])
for doc in response['source_documents']:
    print("Source:", doc.page_content)


# In[28]:


filter = rest.Filter(
    must=[
        rest.FieldCondition(key="state", match=rest.MatchValue(value="WA"))
    ]
)


# In[13]:


qdrant.get_collections()


# In[ ]:


# QDRANT HTTP API (Direct Upsert) - Simple + works with current code
from qdrant_client import QdrantClient
from langchain.embeddings import OllamaEmbeddings

qdrant = QdrantClient(url="http://localhost:6333")
embed_model = OllamaEmbeddings(model="nomic-embed-text")

def update_qdrant(doc_id, content, metadata={}):
    vector = embed_model.embed_query(content)
    qdrant.upsert(
        collection_name="us_cities",
        points=[{
            "id": doc_id,
            "vector": vector,
            "payload": {"content": content, **metadata},
        }]
    )

import uuid

update_qdrant(doc_id=str(uuid.uuid4()), content="San Francisco is in California.", metadata={"state": "CA"})



# In[ ]:


# Option 2: Using Kafka to dynamically stream documents into Qdrant

from kafka import KafkaConsumer
import json
from langchain.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from uuid import uuid4

# Set up Ollama embeddings and Qdrant HTTP client
embed_model = OllamaEmbeddings(model="nomic-embed-text")
qdrant = QdrantClient(host="localhost", port=6333)

# Create the consumer (assumes Kafka is already running and topic exists)
consumer = KafkaConsumer(
    'city_updates',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    group_id='city-rag-group',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

print("Listening to Kafka topic: city_updates")

for message in consumer:
    data = message.value
    city_text = f"{data['City']} is a city in {data['State']}. It is located at latitude {data['Latitude']} and longitude {data['Longitude']}."

    # Embed and upsert into Qdrant
    embedding = embed_model.embed_query(city_text)
    qdrant.upsert(
        collection_name="us_cities",
        points=[
            PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={"text": city_text, "state": data['State']}
            )
        ]
    )
    print(f" Inserted: {city_text}")




# In[ ]:


# Option 3: Vanna for SQL-to-LLM Retrieval Augmented Querying

# pip install vanna[duckdb] vanna-llm-openai
from vanna.openai import OpenAI_Chat
from vanna.duckdb import DuckDB

# Setup Vanna for DuckDB and OpenAI-compatible model (adjust if local model)
vn = OpenAI_Chat(model="gpt-3.5-turbo")  # Or replace with LangChain wrapper if needed
vn_duckdb = DuckDB()

# Load data into DuckDB (you can use your cities CSV)
vn_duckdb.load_file("cities.csv")

# Train Vanna on schema and example SQL (optional for better performance)
vn.train(["SELECT * FROM cities LIMIT 5;", "SELECT City FROM cities WHERE State = 'WA';"])

# Ask a question and get SQL and answer
question = "Which cities are located in Washington?"
sql_query = vn.generate_sql(question)
print("Generated SQL:", sql_query)

answer = vn.run_sql(sql_query)
print("Answer:", answer)

