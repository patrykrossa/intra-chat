import faiss
import numpy as np
import pickle
import os

INDEX_PATH = "data/faiss_index.index"
DOCS_PATH = "data/chunk_metadata.pkl"


def save_faiss_index(index: faiss.Index, documents: list[dict]):
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)


def load_faiss_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_PATH):
        raise FileNotFoundError("FAISS index or metadata not found.")

    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "rb") as f:
        documents = pickle.load(f)
    return index, documents


def search_similar_chunks(
    query_vector: np.ndarray, index: faiss.Index, documents: list[dict], top_k=3
):
    D, I = index.search(query_vector.reshape(1, -1), top_k)
    return [documents[i] for i in I[0]]
