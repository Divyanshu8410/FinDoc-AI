
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")


def compare_sentences(sentences):
    if len(sentences) < 2:
        raise ValueError("Provide at least two sentences.")

    embeddings = model.encode(
        sentences,
        normalize_embeddings=True
    )

    scores = cosine_similarity(embeddings)

    return scores


if __name__ == "__main__":
    sentences = [
        "Company revenue increased.",
        "The business earned more money.",
        "The weather is sunny today."
    ]

    scores = compare_sentences(sentences)

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            print(f"\nSentence A: {sentences[i]}")
            print(f"Sentence B: {sentences[j]}")
            print(f"Similarity: {scores[i][j]:.3f}")
