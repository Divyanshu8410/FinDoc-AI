# FinDoc AI — Financial Document Intelligence Platform

FinDoc AI is an NLP-powered application that extracts text from financial PDF reports and retrieves relevant passages using semantic search. It uses sentence embeddings and cosine similarity to help users explore lengthy financial documents efficiently.

## Features

- **PDF Text Extraction:** Extract selectable text from PDF financial reports using PyMuPDF.
- **Text Chunking:** Split documents into overlapping word-based chunks.
- **Semantic Search:** Retrieve relevant passages using Sentence Transformers embeddings.
- **Similarity Ranking:** Rank passages using cosine similarity.
- **Page-Aware Processing:** Preserve PDF page numbers in the Streamlit application.
- **Persistent Embedding Cache:** Reuse document embeddings across evaluation runs when the document and chunking configuration are unchanged.
- **Automated Tests:** Validate chunking behavior and search ranking with pytest.
- **Search Evaluation:** Evaluate retrieved passages against predefined expected keywords.

## Tech Stack

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- all-MiniLM-L6-v2
- NumPy
- scikit-learn
- pytest

## Project Structure

```text
FinDoc-AI/
├── src/
│   ├── chunking.py
│   ├── embeddings.py
│   ├── similarity.py
│   └── search.py
├── evaluation/
│   ├── questions.json
│   └── evaluate_search.py
├── tests/
│   ├── test_chunking_unit.py
│   ├── test_search_unit.py
│   └── test_search_behavior.py
├── app.py
├── test_chunking.py
├── test_search.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Divyanshu8410/FinDoc-AI.git
cd FinDoc-AI
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Upload a selectable-text PDF report and ask a question about its contents.

## Run Tests

```bash
python -m pytest tests/test_chunking_unit.py tests/test_search_unit.py tests/test_search_behavior.py -v
```

## Evaluate Search

The evaluation script currently uses a locally configured PDF path. Update `PDF_PATH` in `evaluation/evaluate_search.py` to point to your own PDF before running:

```bash
python -m evaluation.evaluate_search
```

The evaluation script reports keyword coverage in the top retrieved passages. This metric is a basic retrieval check, not a measure of factual correctness or semantic accuracy.

## Current Limitations

- Scanned PDFs may require OCR.
- Retrieval quality depends on chunking, document content, and the embedding model.
- The current evaluation dataset is small and needs manually verified relevance labels.
- The evaluation script uses a local PDF path that must be configured by each user.
- The application retrieves passages; it does not yet generate grounded answers with citations.
- Persistent caching is implemented for document embeddings used by the search module.

## Future Improvements

- Add a configurable PDF upload workflow for evaluation.
- Measure Recall@K, Precision@K, and Mean Reciprocal Rank on a verified evaluation dataset.
- Add automated tests for cache hits, cache invalidation, and PDF processing.
- Improve retrieval with page-level citations and hybrid keyword-semantic search.
- Add grounded question answering with source references.

## License

Add a license file before distributing the project for reuse.
