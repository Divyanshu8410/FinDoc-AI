
import pytest

from src.chunking import split_into_chunks


def test_empty_text_returns_no_chunks():
    assert split_into_chunks("") == []


def test_short_text_returns_one_chunk():
    text = "Revenue increased during the year."

    chunks = split_into_chunks(text, chunk_size=10, overlap=2)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunks_respect_maximum_size():
    text = " ".join(f"word{i}" for i in range(1200))

    chunks = split_into_chunks(
        text,
        chunk_size=500,
        overlap=75
    )

    assert len(chunks) == 3
    assert all(len(chunk.split()) <= 500 for chunk in chunks)


def test_adjacent_chunks_have_expected_overlap():
    text = " ".join(f"word{i}" for i in range(12))

    chunks = split_into_chunks(
        text,
        chunk_size=5,
        overlap=2
    )

    first_words = chunks[0].split()
    second_words = chunks[1].split()

    assert first_words[-2:] == second_words[:2]


def test_invalid_chunk_size_raises_error():
    with pytest.raises(ValueError):
        split_into_chunks("Some financial text", chunk_size=0)


def test_invalid_overlap_raises_error():
    with pytest.raises(ValueError):
        split_into_chunks("Some financial text", chunk_size=5, overlap=5)
