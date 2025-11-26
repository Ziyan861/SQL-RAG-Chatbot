from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

# Initialize Qdrant client
qdrant_client = QdrantClient(host="localhost", port=6333)
collection_name = "products_rag"

# Initialize embedding model
embed_model = OllamaEmbeddings(model="nomic-embed-text")

# Connect to vector store (set the correct key used in payload)
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
    embeddings=embed_model,
    content_payload_key="content"
)

# Setup retriever
retriever = vector_store.as_retriever()

# Setup LLM
llm = OllamaLLM(model="deepseek-r1:1.5b")

# Define QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Helper: safely get documents with valid content
def safe_get_docs(query):
    docs = retriever.invoke(query)
    return [doc for doc in docs if isinstance(doc.page_content, str) and doc.page_content.strip()]

# Ask loop
while True:
    query = input("\nAsk about any product (or type 'exit'): ").strip()
    if query.lower() == "exit":
        break

    try:
        docs = safe_get_docs(query)
        if not docs:
            print("No relevant documents found.")
            continue

        context = "\n".join(doc.page_content for doc in docs)
        answer = llm.invoke(f"{query}\n\nContext:\n{context}")
        print("\nAnswer:", answer)

    except Exception as e:
        print("Error:", e)

