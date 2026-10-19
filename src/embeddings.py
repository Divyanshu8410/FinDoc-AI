
from sentence_transformers import SentenceTransformer

# Load a pretrained NLP model
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(texts):
    """
    Convert a list of text chunks into numerical vectors.
    """
    if not texts:
        return model.encode([])

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings


if __name__ == "__main__":
    sentences = [
        "Company revenue increased.",
        "The business earned more money.",
        "The weather is sunny today."
    ]

    vectors = generate_embeddings(sentences)

    print("Number of sentences:", len(sentences))
    print("Embedding shape:", vectors.shape)
    print("First vector (first 10 values):", vectors[0][:10])
