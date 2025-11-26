import json
import re
import google.generativeai as genai
from sqlalchemy import create_engine, text
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams
from langchain_huggingface import HuggingFaceEmbeddings
from vanna.remote import VannaDefault
import plotly.io as pio
import streamlit as st
import sys
import contextlib
from io import StringIO
import pandas as pd  # Added for DataFrame handling

# === Plotly Setup ===
pio.renderers.default = "browser"

# === Gemini Setup ===
genai.configure(api_key="")#enter your api key here!

def gemini_invoke(prompt: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Gemini Error] {e}"

# === Qdrant Setup ===
qdrant = QdrantClient(host="localhost", port=6333)
collection = "full_pg_dump"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Vanna Setup ===
vn = VannaDefault(model="md59", api_key="")#enter your api key here
vn.connect_to_postgres(
    host="localhost",
    port=5432,
    dbname="idris_db",
    user="postgres",
    password=""#add your password here
)
vn.allow_llm_to_see_data = True

# === Suppress Vanna Logs from UI ===
@contextlib.contextmanager
def capture_stdout():
    buffer = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer
    try:
        yield buffer
    finally:
        sys.stdout = old_stdout

# === Semantic Search ===
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
            row_str = "\n".join([f"- {k.replace('_', ' ').title()}: {v}" 
                                 for k, v in row.items() 
                                 if str(v).lower() not in ["not available", "na", "n/a"]])
            if row_str:
                context += f"**Table: {table}**\n{row_str}\n\n"
    return context.strip()

# === Structured Query Detection ===
def is_structured_query(question: str) -> bool:
    question = question.lower()
    if any(kw in question for kw in ["what is this", "about this", "explain", "describe"]):
        return False
    return any(kw in question for kw in [
        "mrp", "price", "value", "count", "sum", "most", "average", "manufacturer",
        "author", "book", "order", "list", "publisher", "how many", "which",
        "what is", "stock", "total", "cheapest", "expensive", "between", "group by"
    ])

# === Format SQL Result Nicely for Laymen ===
def format_result_nicely(result):
    if not result:
        return "No data found."
    keys_of_interest = ["book_name", "name", "title", "author", "manufacturer", 
                        "mrp", "price", "stock", "quantity", "bundle", "cheapest"]
    lines = []
    for row in result:
        filtered_row = {k: v for k, v in row.items() 
                        if any(key in k.lower() for key in keys_of_interest) 
                        and str(v).lower() not in ["not available", "na", "n/a"]}
        if not filtered_row:
            continue
        line = ", ".join([f"**{k.replace('_', ' ').title()}**: {v}" for k, v in filtered_row.items()])
        lines.append(f"• {line}")
    return "\n".join(lines) if lines else "No relevant information found."

# === Generate Response for SQL ===
def generate_natural_response(question: str, sql_result) -> str:
    if isinstance(sql_result, pd.DataFrame):
        sql_result = sql_result.to_dict('records')
    
    if not sql_result:
        return "I couldn't find any relevant data."
    
    if isinstance(sql_result, list):
        if len(sql_result) == 1:
            result_data = sql_result[0]
            if len(result_data) == 1:
                key, value = next(iter(result_data.items()))
                return f"The {key.replace('_', ' ')} is {value}."
            else:
                formatted = format_result_nicely(sql_result)
                return f"Here's what I found:\n{formatted}"
        else:
            formatted = format_result_nicely(sql_result)
            prompt = f"Rephrase this SQL result for the question '{question}':\n{formatted}\n\nMake it friendly and readable."
            return gemini_invoke(prompt)
    return str(sql_result)

# === SQL Execution Handler ===
def execute_sql_query(question: str):
    try:
        with capture_stdout() as output:
            sql_query = vn.generate_sql(question)
            sql_result = vn.run_sql(sql_query)
            internal_logs = output.getvalue()
            with open("vanna_logs.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"\n\nPrompt: {question}\n{internal_logs}")
        return generate_natural_response(question, sql_result)
    except Exception as e:
        return f"Sorry, I couldn't process that query: {str(e)}"

# === Streamlit App UI ===
st.markdown("""
<style>
/* Reset base layout */
html, body, .stApp {
    margin: 0 !important;
    padding: 0 !important;
    height: 100% !important;
    background-color: white !important;
    background-image: none !important;
    background-attachment: fixed !important;
    overflow-x: hidden;
}

/* Strip default spacing on Streamlit containers */
[data-testid="stAppViewContainer"],
[data-testid="stVerticalBlock"],
.main .block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

/* Remove internal flex spacing that causes vertical white space */
[data-testid="stVerticalBlock"] > div:first-child {
    padding-top: 0 !important;
}
[data-testid="stVerticalBlock"] > div:last-child {
    padding-bottom: 0 !important;
}

/* Chat input spacing and style */
.stChatInputContainer {
    background-color: white !important;
    padding: 0.5rem 1rem !important;
    border-top: none !important;
}

/* Chat message area */
.stChatMessage {
    border-radius: 18px !important;
    padding: 1rem 1.5rem !important;
    margin: 0.75rem 0 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
    border: 1px solid #e0e0e0 !important;
}

/* Sidebar */
.css-1d391kg, .css-1lcbmhc {
    background-color: rgba(255, 255, 255, 0.7) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important;
    margin: 1rem !important;
    padding: 1.5rem !important;
    border: 1px solid #e0e0e0 !important;
}

/* Buttons */
.stButton > button {
    background-color: #1976d2 !important;
    color: white !important;
    border: none !important;
    border-radius: 24px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    background-color: #1565c0 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(25, 118, 210, 0.3) !important;
}

/* Headers */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #0d47a1 !important;
    font-weight: 700 !important;
    text-shadow: 1px 1px 2px rgba(255,255,255,0.7);
}

/* General text */
.stMarkdown p, .stMarkdown li {
    color: #212121 !important;
}

/* Hide footer */
footer, #MainMenu {
    display: none !important;
}

/* Hide labels on input */
.stTextInput label {
    display: none !important;
}

/* Input field style */
.stTextInput > div > div {
    background-color: #ffffff !important;
    border-radius: 24px !important;
    padding: 0.5rem 1rem !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1) !important;
}
</style>
""", unsafe_allow_html=True)

# === Main UI ===
def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #1976d2, #0d47a1); 
                    padding: 1.5rem; 
                    border-radius: 12px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                    margin-bottom: 2rem;'>
            <h1 style='color: white; margin: 0;'>📚 Bookstore Assistant</h1>
            <p style='color: #e3f2fd; margin: 0.5rem 0 0; font-size: 1.1rem;'>
                Ask about products, prices, inventory, and orders in natural language!
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.header("ℹ️ How to use")
        st.markdown("""
        **Structured Queries:**
        - "What is the MRP of item 173?"
        - "Which item has the highest price?"
        - "How many books by Oxford?"
        
        **General Questions:**
        - "Tell me about this database"
        - "What products do you have?"
        """)
        
        st.header("🔧 Connection Status")
        st.success("✅ Gemini AI")
        st.success("✅ Qdrant")
        st.success("✅ Vanna DB")

    st.subheader(" Chat with your Database")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Type your question here...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if re.search(r"what (is|does) (this|the) (chatbot|system|app)", prompt, re.I):
            response = "I'm your friendly Bookstore Assistant! Ask me anything about book prices, stock, orders, or publishers."
        elif is_structured_query(prompt):
            response = execute_sql_query(prompt)
        else:
            context = semantic_search(prompt)
            full_prompt = f"{prompt}\n\nRelevant context:\n{context}" if context else prompt
            response = gemini_invoke(full_prompt)

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

if __name__ == "__main__":
      main()

