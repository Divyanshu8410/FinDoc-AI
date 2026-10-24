
import numpy as np
from unittest.mock import patch

from src import search


def test_search_returns_ranked_results(tmp_path):
    text = " ".join(f"word{i}" for i in range(10))

    # Use a temporary cache so real project cache cannot affect the test.
    with (
        patch.object(search, "CACHE_DIR", tmp_path),
        patch(
            "src.search.split_into_chunks",
            return_value=[
                "weather is sunny today",
                "company revenue increased"
            ]
        ),
        patch("src.search.get_model") as mock_get_model
    ):
        instance = mock_get_model.return_value

        # prepare_document embeds chunks first; search embeds the question next.
        instance.encode.side_effect = [
            np.array([
                [0.0, 1.0],  # Weather passage
                [1.0, 0.0]   # Revenue passage
            ]),
            np.array([[0.8, 0.6]])  # Question embedding
        ]

        results = search.search_chunks(
            text,
            "What happened to company revenue?",
            top_k=1
        )

        assert len(results) == 1
        assert results[0]["chunk"] == "company revenue increased"
        assert results[0]["score"] > 0.7
