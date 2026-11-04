
import json
from pathlib import Path

import fitz

from src.search import prepare_document, search_prepared_document
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
)


BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = BASE_DIR / "questions.json"
RELEVANCE_FILE = BASE_DIR / "relevance.json"

PDF_PATH = Path(r"C:\Users\varsh\Downloads\Tata Motors report.pdf")
TOP_K = 10


def load_pdf_text():
    """Extract selectable text from every PDF page."""
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    pages = []

    with fitz.open(PDF_PATH) as pdf:
        for page in pdf:
            page_text = page.get_text("text").strip()

            if page_text:
                pages.append(page_text)

    return "\n\n".join(pages)


def load_json_file(file_path):
    """Load and validate that a JSON file contains a list."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"{file_path.name} must contain a JSON list.")

    return data


def load_questions():
    """Load and validate evaluation questions."""
    questions = load_json_file(QUESTIONS_FILE)

    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {index} must be an object.")

        question = item.get("question")
        keywords = item.get("expected_keywords")

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Question {index} is invalid.")

        if (
            not isinstance(keywords, list)
            or not keywords
            or not all(
                isinstance(keyword, str) and keyword.strip()
                for keyword in keywords
            )
        ):
            raise ValueError(
                f"Question {index} needs valid expected keywords."
            )

    return questions


def load_relevance():
    """Load manually verified relevant chunk IDs."""
    relevance_data = load_json_file(RELEVANCE_FILE)

    relevance = {}

    for index, item in enumerate(relevance_data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Relevance entry {index} must be an object.")

        question = item.get("question")
        relevant_chunks = item.get("relevant_chunks")

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"Relevance entry {index} has an invalid question."
            )

        if (
            not isinstance(relevant_chunks, list)
            or not all(
                isinstance(chunk_id, int) and chunk_id >= 0
                for chunk_id in relevant_chunks
            )
        ):
            raise ValueError(
                f"Relevance entry {index} must contain valid chunk IDs."
            )

        if question in relevance:
            raise ValueError(f"Duplicate relevance question: {question}")

        relevance[question] = relevant_chunks

    return relevance


def evaluate_search():
    """Evaluate keyword coverage and information retrieval metrics."""
    try:
        questions = load_questions()
        relevance = load_relevance()

        if not questions:
            print("No evaluation questions found.")
            return

        print(f"Loading PDF: {PDF_PATH.name}")

        document_text = load_pdf_text()

        print(f"Extracted characters: {len(document_text):,}")

        if not document_text.strip():
            print("No selectable text found. The PDF may require OCR.")
            return

        print("\nPreparing document chunks and embeddings...")

        chunks, embeddings = prepare_document(document_text)

        print(f"Prepared chunks: {len(chunks)}")

        if not chunks or embeddings is None:
            print("No searchable chunks were generated.")
            return

        keyword_passed = 0
        precision_scores = []
        recall_scores = []
        mrr_scores = []

        total = len(questions)

        for index, item in enumerate(questions, start=1):
            question = item["question"]
            expected_keywords = item["expected_keywords"]

            print(f"\n{'=' * 70}")
            print(f"Question {index}/{total}: {question}")

            results = search_prepared_document(
                question,
                chunks,
                embeddings,
                top_k=TOP_K,
            )

            print(f"\nTop {len(results)} retrieved passages:")

            for rank, result in enumerate(results, start=1):
                print(f"\nRank: {rank}")
                print(f"Chunk ID: {result['chunk_id']}")
                print(f"Similarity: {result['score']:.3f}")
                print("-" * 50)
                print(result["chunk"][:500])

            # Existing keyword coverage evaluation
            retrieved_text = " ".join(
                result["chunk"].lower()
                for result in results
            )

            matched_keywords = [
                keyword
                for keyword in expected_keywords
                if keyword.lower() in retrieved_text
            ]

            keyword_success = (
                len(matched_keywords) == len(expected_keywords)
            )

            if keyword_success:
                keyword_passed += 1

            print(f"\nExpected keywords: {expected_keywords}")
            print(f"Matched keywords: {matched_keywords}")
            print(f"Keyword result: {'PASS' if keyword_success else 'FAIL'}")

            # Ranking metrics using manually verified relevance labels
            relevant_ids = relevance.get(question, [])

            if relevant_ids:
                retrieved_ids = [
                    result["chunk_id"] for result in results
                ]

                precision = precision_at_k(
                    retrieved_ids, relevant_ids, TOP_K
                )
                recall = recall_at_k(
                    retrieved_ids, relevant_ids, TOP_K
                )
                mrr = mean_reciprocal_rank(
                    retrieved_ids, relevant_ids, TOP_K
                )

                precision_scores.append(precision)
                recall_scores.append(recall)
                mrr_scores.append(mrr)

                print("\nRetrieval metrics:")
                print(f"Precision@{TOP_K}: {precision:.3f}")
                print(f"Recall@{TOP_K}:    {recall:.3f}")
                print(f"MRR@{TOP_K}:       {mrr:.3f}")

            else:
                print(
                    "\nRetrieval metrics skipped: "
                    "no verified relevant chunk IDs for this question."
                )

        # Final summary
        print(f"\n{'=' * 70}")
        print("FINAL EVALUATION SUMMARY")
        print(f"{'=' * 70}")

        keyword_score = keyword_passed / total * 100

        print(
            f"Keyword coverage: {keyword_passed}/{total} "
            f"({keyword_score:.1f}%)"
        )

        evaluated = len(precision_scores)

        if evaluated:
            print(f"\nQuestions with relevance labels: {evaluated}")

            print(
                f"Mean Precision@{TOP_K}: "
                f"{sum(precision_scores) / evaluated:.3f}"
            )
            print(
                f"Mean Recall@{TOP_K}: "
                f"{sum(recall_scores) / evaluated:.3f}"
            )
            print(
                f"Mean MRR@{TOP_K}: "
                f"{sum(mrr_scores) / evaluated:.3f}"
            )
        else:
            print("\nNo verified relevance labels available.")
            print("Precision, Recall and MRR could not be calculated.")

        print(
            "\nNote: Ranking metrics are meaningful only when "
            "relevant chunk IDs have been manually verified."
        )

    except (
        FileNotFoundError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"Evaluation setup error: {error}")

    except Exception as error:
        print(f"Evaluation failed: {type(error).__name__}: {error}")


if __name__ == "__main__":
    evaluate_search()
