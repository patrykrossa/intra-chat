# IntraChat — Secure Internal Document Q&A Assistant

**IntraChat** is a private, enterprise-ready RAG (Retrieval-Augmented Generation) application that allows employees to ask natural language questions based on internal documents.  
It retrieves relevant fragments from your document base and generates precise, context-aware answers using a local language model — with full data control.

## Key Features

- Upload and parse internal documents (.pdf, .txt)
- Generate natural language answers using a local LLM
- Expose `/ask` endpoint for querying model

## Tech Stack

- **Backend**: Python + FastAPI
- **Vector Search**: FAISS
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Language Model**: `flan-t5-base`
- **Frontend**: (To be implemented)
- **Document Parsing**: PyMuPDF (`fitz`)
