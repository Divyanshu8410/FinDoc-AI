
from src.chunking import split_into_chunks


def test_chunking_returns_empty_for_empty_document():
    assert split_into_chunks("") == []


def test_chunking_preserves_searchable_content():
    text = (
        "Company revenue increased during the year. "
        "Operating costs also increased due to material prices."
    )

    chunks = split_into_chunks(text, chunk_size=20, overlap=3)

    combined_text = " ".join(chunks)

    assert "revenue increased" in combined_text
    assert "Operating costs" in combined_text


def test_chunking_does_not_duplicate_words_within_a_chunk():
    text = " ".join(f"word{i}" for i in range(10))

    chunks = split_into_chunks(text, chunk_size=5, overlap=2)

    assert len(chunks) == 3
    assert len(chunks[0].split()) == 5
    assert len(chunks[1].split()) == 5
    assert len(chunks[2].split()) == 4
