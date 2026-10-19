
from src.search import search_chunks

sample_text = """
Revenue increased significantly during the year.
The company expanded its electric vehicle business.
Operating costs increased due to higher material prices.
The company invested in new manufacturing facilities.
"""

question = "Why did the company's operating costs increase?"

results = search_chunks(sample_text, question, top_k=2)

for result in results:
    print("\nSimilarity:", round(result["score"], 3))
    print("Passage:", result["chunk"])
