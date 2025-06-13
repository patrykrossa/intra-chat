from app.embeddings import embed_text
from app.vector_store import load_faiss_index, search_similar_chunks
from transformers import pipeline

# generator = pipeline(
#     "text-generation", model="tiiuae/falcon-7b-instruct", max_new_tokens=512
# )
generator = pipeline("text2text-generation", model="google/flan-t5-base")

index, documents = load_faiss_index()


def answer_question(query: str):
    query_vector = embed_text(query)
    print("Text embedded")

    top_chunks = search_similar_chunks(query_vector, index, documents, top_k=3)
    print("Chunks found")

    context = "\n".join([chunk["text"] for chunk in top_chunks])
    print(f"Context: ", context)
    prompt = f"Based on documents below answer the question, if you do not know the answer, just answer 'I do not know that yet.'. Documents:\n\n{context}\n\Question: {query}\nOdpowiedź:"

    # answer = generator(prompt)[0]["generated_text"].split("Odpowiedź:")[-1].strip()
    answer = (
        generator(prompt, max_new_tokens=512)[0]["generated_text"]
        .split("Odpowiedź:")[-1]
        .strip()
    )
    print("Answer generated")

    return {"answer": answer, "sources": [chunk["source"] for chunk in top_chunks]}
