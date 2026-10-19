from src.chunking import split_into_chunks

sample_text = " ".join(
    f"word{i}" for i in range(1200)
)

chunks = split_into_chunks(sample_text)

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}: {len(chunk.split())} words")