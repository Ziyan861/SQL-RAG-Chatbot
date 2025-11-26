import json
import re
import sys
import contextlib
from io import StringIO

import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams
from langchain_huggingface import HuggingFaceEmbeddings
import plotly.io as pio

from sqlalchemy import text
from transformers import AutoTokenizer, AutoModelForCausalLM

from vanna.base import VannaBase
from vanna.vannadb.vannadb_vector import VannaDB_VectorStore


pio.renderers.default = "browser"

#Gemini Setup 
genai.configure(api_key="")#enter your key here
def gemini_invoke(prompt: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Gemini Error] {e}"

#  Semantic Embeddings Setup ===
qdrant = QdrantClient(host="localhost", port=6333)
collection = "full_pg_dump"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def semantic_search(question: str) -> str:
    query_vector = embedding_model.embed_query(question)
    results = qdrant.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=5,
        search_params=SearchParams(hnsw_ef=128)
    )
    context = ""
    for point in results:
        payload = getattr(point, "payload", {})
        if isinstance(payload, dict):
            table = payload.get("table", "unknown")
            row = payload.get("row", {})
            context += f"Table: {table}\n{json.dumps(row, indent=2)}\n\n"
    return context.strip()

# === Vanna Offline Backend via Hugging Face ===
class HfLocalLLM(VannaBase):
    def __init__(self, model_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)

    def submit_prompt(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class LocalVanna(VannaDB_VectorStore, HfLocalLLM):
    def __init__(self, model_dir: str):
        VannaDB_VectorStore.__init__(self, config={"embedding_model": "sentence-transformers/all-MiniLM-L6-v2"})
        HfLocalLLM.__init__(self, model_dir)

vn = LocalVanna(model_dir="/home/hdoop/Downloads/pip_sql_gguf")#update path

vn.connect_to_postgres(
    host="localhost", port=5432, dbname="idris_db", user="postgres", password="whatever787"
)
vn.allow_llm_to_see_data = True

# === Suppress verbose output ===
@contextlib.contextmanager
def suppress_stdout():
    buffer = StringIO()
    old = sys.stdout
    sys.stdout = buffer
    try:
        yield
    finally:
        sys.stdout = old

# === Response Formatting Helpers ===
def format_response(result):
    if isinstance(result, list) and result:
        return "\n".join([", ".join([f"{k}: {v}" for k, v in row.items()]) for row in result])
    return str(result)

def generate_natural_response(question: str, sql_result):
    if isinstance(sql_result, list) and len(sql_result) == 1:
        row = sql_result[0]
        if len(row) == 1:
            key, val = next(iter(row.items()))
            return f"{key}: {val}"
        return ", ".join(f"{k}: {v}" for k, v in row.items())
    elif isinstance(sql_result, list):
        formatted = format_response(sql_result)
        prompt = f"Convert this data to a conversational response for '{question}':\n{formatted}\n\nMake it sound natural."
        return gemini_invoke(prompt)
    return str(sql_result)

# === SQL Execution Logic ===
def is_structured_query(question: str) -> bool:
    q = question.lower()
    if any(x in q for x in ["what is this", "explain", "describe"]):
        return False
    return any(x in q for x in ["mrp", "price", "count", "sum", "how many", "average", "manufacturer", "book", "which", "stock"])

def execute_sql_query(question: str) -> str:
    try:
        with suppress_stdout():
            sql = vn.generate_sql(question)
            result = vn.run_sql(sql)
        return generate_natural_response(question, result)
    except Exception as e:
        return f"Error processing SQL: {e}"

# === Main Interactive Loop ===
if __name__ == "__main__":
    print("Offline Vanna Assistant ready. Type 'exit' to quit.\n")
    while True:
        question = input("\n> ").strip()
        if question.lower() == "exit":
            break
        if re.search(r"what (is|does) this (system|assistant)", question, re.I):
            print("This assistant uses offline Vanna (Hugging Face LLM) for SQL generation, and Gemini API only for conversational formatting.")
        elif is_structured_query(question):
            print(execute_sql_query(question))
        else:
            ctx = semantic_search(question)
            prompt = f"{question}\n\nRelevant context:\n{ctx}" if ctx else question
            print(gemini_invoke(prompt))

