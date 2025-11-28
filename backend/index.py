# index.py
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

# -------- CONFIG --------
PDF_DIR = Path(__file__).parent / "pdfs"     # store PDFs here
COLLECTION_NAME = "law_docs"
QDRANT_URL = "http://localhost:6333"

# Create dir if needed
PDF_DIR.mkdir(exist_ok=True)

# Qdrant client
client = QdrantClient(url=QDRANT_URL)

# Embeddings model
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def process_pdf(pdf_file: Path):
    print(f"\n📄 Loading {pdf_file.name} ...")
    loader = PyPDFLoader(str(pdf_file))
    docs = loader.load()

    # Split into chunks
    chunks = text_splitter.split_documents(docs)
    print(f"   → Chunks created: {len(chunks)}")

    # Add filename metadata
    for c in chunks:
        c.metadata["file_name"] = pdf_file.name

    return chunks


def index_all():
    all_chunks = []

    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found in /pdfs folder.")
        return

    for pdf in pdf_files:
        all_chunks.extend(process_pdf(pdf))

    print("\n📦 Indexing into Qdrant...")

    QdrantVectorStore.from_documents(
        documents=all_chunks,
        embedding=embedding_model,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME
    )

    print("\n✅ Indexing complete!")
    print(f"👍 Total chunks stored: {len(all_chunks)}")


if __name__ == "__main__":
    index_all()
