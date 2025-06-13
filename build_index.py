from app.document_loader import load_all_documents
from app.embeddings import embed_chunks
from app.vector_store import save_faiss_index
import faiss


def build_index():
    print("[1] Loading documents...")
    chunks = load_all_documents()
    texts = [chunk["text"] for chunk in chunks]

    print(f"[2] Creating embeddings of ({len(texts)} chunks)...")
    vectors = embed_chunks(texts)

    print("[3] Building FAISS index...")
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)  # Can use also IndexIVFFlat or IndexHNSW
    index.add(vectors)

    print(f"[4] Saving index and sources ({len(chunks)} chunks)...")
    save_faiss_index(index, chunks)

    print("[✓] Done: FAISS index saved.")


if __name__ == "__main__":
    build_index()
