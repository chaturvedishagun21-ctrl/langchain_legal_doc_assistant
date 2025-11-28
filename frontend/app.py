import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
import os

load_dotenv()

# --- Your SAME CONFIG from chat.py ---
COLLECTION = "law_docs"
QDRANT_URL = "http://localhost:6333"

openai_client = OpenAI()

# Embeddings (same as chat.py)
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Load vector store (same)
vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url=QDRANT_URL,
    collection_name=COLLECTION
)

# --------------------------
# STREAMLIT UI
# --------------------------

st.set_page_config(page_title="Legal PDF RAG", layout="wide")
st.title("📘 Legal PDF Question Answering App")

st.write("This app uses logic for answering questions from PDFs stored in Qdrant.")

# st.info("To index new PDFs, run: `python index.py` before using the app.")

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Show messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("citations"):
            st.json(msg["citations"])

# Input box
user_query = st.chat_input("Ask a question...")

if user_query:
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # --------------------------
    # YOUR EXACT chat.py LOGIC STARTS HERE
    # --------------------------

    # Retrieve chunks
    with st.spinner("Searching relevant chunks..."):
        results = vector_db.similarity_search(user_query, k=5)

    # Build context + citations (same as chat.py)
    context = ""
    citations = []

    for res in results:
        txt = res.page_content
        page = res.metadata.get("page_label", "N/A")
        file = res.metadata.get("file_name", "unknown")

        context += f"\nPage {page} — {file}\n{txt}\n"

        citations.append({
            "page": page,
            "file": file,
            "snippet": txt[:150] + "..."
        })

    # SYSTEM PROMPT (same as chat.py)
    SYSTEM_PROMPT = f"""
You are a helpful assistant who ONLY answers using the provided context.

Rules:
- Answer ONLY from the text below.
- Mention page numbers when supporting statements.
- If answer not found in context, say: "Not found in the provided documents."
- No hallucinations.

CONTEXT:
{context}
"""

    # Call LLM (same as chat.py)
    with st.spinner("Generating answer..."):
        response = openai_client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ]
        )

    answer = response.choices[0].message.content

    # --------------------------
    # YOUR CHAT.PY LOGIC ENDS HERE
    # --------------------------

    # Show assistant message
    with st.chat_message("assistant"):
        st.write(answer)
        st.caption("📚 Citations Used:")
        st.json(citations)

    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
        "citations": citations
    })
