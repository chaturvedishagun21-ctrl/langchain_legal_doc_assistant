📘 Legal Document Question Answering (RAG) – Streamlit + Qdrant + OpenAI

A Retrieval-Augmented Generation (RAG) system that lets users upload legal PDFs, index them into Qdrant vector database, and ask legal questions through a Streamlit chat interface.

The assistant answers only from the uploaded documents, provides page-number citations, and avoids hallucinations.

This project uses:

Python

Streamlit (frontend UI)

LangChain + OpenAI embeddings

Qdrant (vector search)

PDF loaders + text splitters

GPT-5 for final answers

🚀 Features
📤 Upload PDFs

Add one or multiple PDF files to the /pdfs folder.

🔍 Automatic Indexing

Run index.py to extract text, chunk documents, embed using OpenAI, and store vectors in Qdrant.

💬 Ask Questions

Use the Streamlit UI to query your documents using natural language.

📑 Accurate Citations

Every answer includes:

Page number

File name

Snippets from the source pages

🧠 Fully Local/Cloud Hybrid

Index PDFs locally

Store vectors in local Qdrant or Qdrant Cloud

Query through deployed Streamlit app

🧱 Portable + Easy to Deploy

No complex backend — just Streamlit + Qdrant.
