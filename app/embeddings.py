from sentence_transformers import SentenceTransformer
import numpy as np

# (can change to other, e.g.. all-mpnet-base)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_text(text: str) -> np.ndarray:
    return model.encode(text, convert_to_numpy=True)


def embed_chunks(chunks: list[str]) -> np.ndarray:
    return model.encode(chunks, convert_to_numpy=True)
