
from unittest.mock import patch

import numpy as np

from src.search import prepare_document


def test_empty_document_returns_no_chunks_or_embeddings():
    chunks, embeddings = prepare_document("")

    assert chunks == []
    assert embeddings is None

def test_cached_embeddings_are_reused(tmp_path):
    import json
    import numpy as np

    from src import search

    text = "Revenue increased during the financial year."

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    key = search._cache_key(text)
    embeddings_path = cache_dir / f"{key}.npy"
    chunks_path = cache_dir / f"{key}.json"

    chunks = search.split_into_chunks(
        text,
        chunk_size=search.CHUNK_SIZE,
        overlap=search.CHUNK_OVERLAP
    )

    fake_embeddings = np.array([[0.1, 0.2, 0.3]])

    np.save(embeddings_path, fake_embeddings)

    with chunks_path.open("w", encoding="utf-8") as file:
        json.dump(chunks, file)

    with (
        patch.object(search, "CACHE_DIR", cache_dir),
        patch.object(search, "get_model") as mock_get_model
    ):
        actual_chunks, actual_embeddings = search.prepare_document(text)

    assert actual_chunks == chunks
    np.testing.assert_array_equal(actual_embeddings, fake_embeddings)
    mock_get_model.assert_not_called()

def test_changed_document_uses_new_cache_key():
    from src.search import _cache_key

    text_v1 = "Revenue increased during the financial year."
    text_v2 = "Revenue declined during the financial year."

    assert _cache_key(text_v1) != _cache_key(text_v2)
