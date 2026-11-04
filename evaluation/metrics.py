"""Ranking metrics for FinDoc AI document retrieval."""


def precision_at_k(retrieved_ids, relevant_ids, k=10):
    """Fraction of unique retrieved chunks in top-k that are relevant."""
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    retrieved = list(dict.fromkeys(retrieved_ids))[:k]
    relevant = set(relevant_ids)

    if not retrieved:
        return 0.0

    return sum(chunk_id in relevant for chunk_id in retrieved) / len(retrieved)


def recall_at_k(retrieved_ids, relevant_ids, k=10):
    """Fraction of known relevant chunks found in the top-k results."""
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    retrieved = set(list(dict.fromkeys(retrieved_ids))[:k])
    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    return len(retrieved & relevant) / len(relevant)


def mean_reciprocal_rank(retrieved_ids, relevant_ids, k=10):
    """Reciprocal rank of the first relevant result, or zero."""
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    relevant = set(relevant_ids)

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0
