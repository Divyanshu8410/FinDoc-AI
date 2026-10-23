
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.chunking import split_into_chunks


def search_chunks(text, question, top_k=3):
    chunks = split_into_chunks(text)
    if not chunks:
        return []

    if top_k <= 0:
        return []
    model = SentenceTransformer("all-MiniLM-L6-v2")

    if not chunks:
        return []

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    chunk_embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    scores = cosine_similarity(
        question_embedding,
        chunk_embeddings
    )[0]

    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indices:
        results.append({
            "chunk": chunks[index],
            "score": float(scores[index])
        })

    return results
