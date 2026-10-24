
import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.chunking import split_into_chunks


_model = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75


def get_model():
    """Load the embedding model once per Python process."""
    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    return _model


def _cache_key(text):
    """Create a cache key based on document and chunking settings."""
    content = (
        f"{MODEL_NAME}|{CHUNK_SIZE}|{CHUNK_OVERLAP}|{text}"
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def prepare_document(text):
    """Split text and reuse cached embeddings when available."""
    chunks = split_into_chunks(
        text,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    if not chunks:
        return [], None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(text)

    embeddings_path = CACHE_DIR / f"{key}.npy"
    chunks_path = CACHE_DIR / f"{key}.json"

    if embeddings_path.exists() and chunks_path.exists():
        try:
            with chunks_path.open("r", encoding="utf-8") as file:
                cached_chunks = json.load(file)

            embeddings = np.load(embeddings_path, allow_pickle=False)
            print("Cached chunks:", len(cached_chunks))
            print("Current chunks:", len(chunks))
            print("Cached embeddings:", len(embeddings))
            print("Chunks match:", cached_chunks == chunks)
            if (
                cached_chunks == chunks
                and len(embeddings) == len(chunks)
            ):
                print("Loaded document embeddings from cache.")
                return chunks, embeddings
            print("Cache exists, but validation failed.")

        except (OSError, ValueError, json.JSONDecodeError):
            print("Cache is invalid; regenerating embeddings.")

    model = get_model()

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Save temporary files first, then replace cache files.
    temp_embeddings = embeddings_path.with_suffix(".tmp.npy")
    temp_chunks = chunks_path.with_suffix(".tmp.json")

    np.save(temp_embeddings, embeddings)

    with temp_chunks.open("w", encoding="utf-8") as file:
        json.dump(chunks, file)

    temp_embeddings.replace(embeddings_path)
    temp_chunks.replace(chunks_path)

    print("Generated and cached document embeddings.")

    return chunks, embeddings


def search_prepared_document(question, chunks, embeddings, top_k=3):
    """Search an already embedded document."""
    if not chunks or embeddings is None or top_k <= 0:
        return []

    model = get_model()

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        question_embedding,
        embeddings
    )[0]

    ranked_indices = scores.argsort()[::-1][:top_k]

    return [
        {
            "chunk": chunks[index],
            "score": float(scores[index])
        }
        for index in ranked_indices
    ]


def search_chunks(text, question, top_k=3):
    """Convenience function for one-off searches."""
    chunks, embeddings = prepare_document(text)

    return search_prepared_document(
        question,
        chunks,
        embeddings,
        top_k
    )
