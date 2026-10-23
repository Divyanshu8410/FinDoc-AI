
import numpy as np
from unittest.mock import patch

from src import search


def test_search_returns_ranked_results():
    text = " ".join(f"word{i}" for i in range(10))

    # Exactly 2 chunks: each contains 6 words, with 2 words overlap.
    with patch(
        "src.search.split_into_chunks",
        return_value=["weather is sunny today", "company revenue increased"]
    ):
        fake_model = patch("src.search.SentenceTransformer").start()

        try:
            instance = fake_model.return_value
            instance.encode.side_effect = [
                np.array([[1.0, 0.0]]),  # Question embedding
                np.array([
                    [0.0, 1.0],  # Weather chunk: low similarity
                    [0.8, 0.6]   # Revenue chunk: high similarity
                ])
            ]

            results = search.search_chunks(
                text,
                "What happened to company revenue?",
                top_k=1
            )

            assert len(results) == 1
            assert results[0]["chunk"] == "company revenue increased"
            assert results[0]["score"] > 0.7
        finally:
            fake_model.stop()
