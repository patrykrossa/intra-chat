import os
import fitz
from typing import List
import re

CHUNK_SIZE = 500
DOCS_DIR = "data/documents"


def clean_text(text: str) -> str:
    text = re.sub(r"\\n|\\t|\\r", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text_to_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    words = text.split()
    return [
        " ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)
    ]


def extract_from_pdf(path: str) -> List[dict]:
    doc = fitz.open(path)
    chunks = []
    for page_num, page in enumerate(doc, start=1):
        text = clean_text(page.get_text())
        for chunk in split_text_to_chunks(text):
            chunks.append(
                {
                    "text": chunk,
                    "source": f"{os.path.basename(path)} (page {page_num})",
                }
            )
    return chunks


def extract_from_txt(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = clean_text(f.read())
    return [
        {"text": chunk, "source": os.path.basename(path)}
        for chunk in split_text_to_chunks(text)
    ]


def load_all_documents(directory: str = DOCS_DIR) -> List[dict]:
    all_chunks = []
    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)
        if filename.endswith(".pdf"):
            all_chunks.extend(extract_from_pdf(full_path))
        elif filename.endswith(".txt"):
            all_chunks.extend(extract_from_txt(full_path))
    return all_chunks
