# chat.py
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()

COLLECTION = "law_docs"
QDRANT_URL = "http://localhost:6333"

openai_client = OpenAI()

# Embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Load vector store
vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url=QDRANT_URL,
    collection_name=COLLECTION
)

# Ask user
user_query = input("\n🔍 Ask something: ")

# Retrieve chunks
results = vector_db.similarity_search(user_query, k=5)

# Format context
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

# System prompt
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

response = openai_client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
)

print("\n🤖 Answer:\n")
print(response.choices[0].message.content)

print("\n📚 Citations Used:")
for c in citations:
    print(f"• {c['file']} — Page {c['page']} — Snippet: {c['snippet']}")
