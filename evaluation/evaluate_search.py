
import json
from pathlib import Path

import fitz

from src.search import prepare_document, search_prepared_document


BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = BASE_DIR / "questions.json"

PDF_PATH = Path(r"C:\Users\varsh\Downloads\Tata Motors report.pdf")
TOP_K = 10


def load_pdf_text():
    """Extract selectable text from every page of the PDF."""
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    pages = []

    with fitz.open(PDF_PATH) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text("text").strip()

            if page_text:
                pages.append(page_text)

    return "\n\n".join(pages)


def load_questions():
    """Load evaluation questions from the JSON file."""
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Questions file not found: {QUESTIONS_FILE}"
        )

    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError("questions.json must contain a JSON list.")

    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {index} must be a JSON object.")

        if not item.get("question") or not isinstance(
            item.get("question"), str
        ):
            raise ValueError(f"Question {index} is missing its question.")

        keywords = item.get("expected_keywords")

        if not isinstance(keywords, list) or not keywords:
            raise ValueError(
                f"Question {index} needs a non-empty expected_keywords list."
            )

    return questions


def evaluate_search():
    """Evaluate retrieval against the configured PDF and questions."""
    try:
        questions = load_questions()

        if not questions:
            print("No evaluation questions found.")
            return

        print(f"Loading PDF: {PDF_PATH.name}")

        document_text = load_pdf_text()

        print(f"Extracted characters: {len(document_text):,}")

        if not document_text.strip():
            print("No selectable text found in the PDF.")
            print("The document may require OCR.")
            return

        print("\nPreparing document chunks and embeddings...")

        chunks, embeddings = prepare_document(document_text)

        print(f"Prepared chunks: {len(chunks)}")

        if not chunks or embeddings is None:
            print("No searchable chunks were generated.")
            return

        passed = 0
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
                top_k=TOP_K
            )

            print(f"\nTop {len(results)} retrieved passages:")

            for rank, result in enumerate(results, start=1):
                print(
                    f"\n{rank}. Similarity: {result['score']:.3f}"
                )
                print(result["chunk"][:500])

            retrieved_text = " ".join(
                result["chunk"].lower() for result in results
            )

            matched_keywords = [
                keyword
                for keyword in expected_keywords
                if keyword.lower() in retrieved_text
            ]

            success = len(matched_keywords) == len(expected_keywords)

            if success:
                passed += 1

            print(f"\nExpected keywords: {expected_keywords}")
            print(f"Matched keywords: {matched_keywords}")
            print(f"Result: {'PASS' if success else 'FAIL'}")

        score = passed / total * 100

        print(f"\n{'=' * 70}")
        print(f"Evaluation score: {passed}/{total} ({score:.1f}%)")
        print("Metric: all expected keywords found in the top results.")
        print("Note: This is keyword coverage, not semantic accuracy.")

    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(f"Evaluation setup error: {error}")

    except Exception as error:
        print(f"Evaluation failed: {type(error).__name__}: {error}")


if __name__ == "__main__":
    evaluate_search()
