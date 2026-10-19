
import hashlib

import fitz
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.chunking import split_into_chunks


st.set_page_config(
    page_title="FinDoc AI",
    page_icon="📄",
    layout="wide"
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
TOP_K = 3


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_pages(pdf_bytes):
    """Extract text while preserving original PDF page numbers."""
    pages = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


def build_page_chunks(pages):
    """Create chunks separately for each page."""
    chunks = []

    for page in pages:
        if not page["text"]:
            continue

        page_chunks = split_into_chunks(
            page["text"],
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP
        )

        for chunk in page_chunks:
            if chunk.strip():
                chunks.append({
                    "page": page["page"],
                    "text": chunk
                })

    return chunks


@st.cache_data(show_spinner=False)
def embed_chunks(chunk_texts):
    """Generate and cache embeddings for document chunks."""
    model = load_model()

    return model.encode(
        list(chunk_texts),
        normalize_embeddings=True,
        show_progress_bar=False
    )


def search_document(question, chunks, embeddings, top_k=TOP_K):
    """Return the most semantically relevant document chunks."""
    model = load_model()

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )[0]

    # Both question and chunk vectors are normalized.
    # Their dot product gives cosine similarity.
    scores = np.asarray(embeddings) @ question_embedding

    top_indices = np.argsort(scores)[::-1][:min(top_k, len(chunks))]

    results = []

    for index in top_indices:
        results.append({
            "page": chunks[index]["page"],
            "text": chunks[index]["text"],
            "score": float(scores[index])
        })

    return results


st.title("📄 FinDoc AI")
st.subheader("Financial Document Intelligence Platform")

st.write(
    "Upload a financial report, extract its text, "
    "and search for relevant information using NLP."
)

uploaded_file = st.file_uploader(
    "Upload financial report",
    type=["pdf"]
)

if uploaded_file is not None:
    try:
        pdf_bytes = uploaded_file.getvalue()

        if not pdf_bytes:
            st.error("The uploaded PDF is empty.")
            st.stop()

        document_id = hashlib.sha256(pdf_bytes).hexdigest()

        # Reprocess only when a different PDF is uploaded.
        if st.session_state.get("document_id") != document_id:
            with st.spinner("Extracting text from PDF..."):
                pages = extract_pages(pdf_bytes)

            extracted_text = "\n\n".join(
                f"--- Page {page['page']} ---\n{page['text']}"
                for page in pages
            )

            st.session_state["document_id"] = document_id
            st.session_state["pages"] = pages
            st.session_state["extracted_text"] = extracted_text
            st.session_state["chunks"] = build_page_chunks(pages)
            st.session_state.pop("chunk_embeddings", None)

        pages = st.session_state["pages"]
        extracted_text = st.session_state["extracted_text"]
        chunks = st.session_state["chunks"]

        st.success("PDF uploaded successfully!")

        col1, col2 = st.columns(2)
        col1.metric("Total pages", len(pages))
        col2.metric("Text chunks", len(chunks))

        if extracted_text.strip():
            st.download_button(
                "Download extracted text",
                data=extracted_text,
                file_name="extracted_report.txt",
                mime="text/plain"
            )

            with st.expander("Preview extracted PDF text"):
                st.text_area(
                    "Extracted text (preview)",
                    value=extracted_text[:10000],
                    height=300,
                    disabled=True
                )
                if len(extracted_text) > 10000:
                    st.caption(
                        "Preview limited to the first 10,000 characters. "
                        "Download the text file to view everything."
                    )
        else:
            st.warning(
                "No selectable text was found. "
                "This PDF may be scanned and require OCR."
            )

        st.divider()
        st.header("🔎 Semantic Search")

        if not chunks:
            st.warning("No text chunks are available to search.")
        else:
            question = st.text_input(
                "Ask a question about this financial report",
                placeholder="e.g. What are the company's major risks?"
            )

            if st.button("Search report", type="primary"):
                if not question.strip():
                    st.warning("Please enter a question.")
                else:
                    try:
                        if "chunk_embeddings" not in st.session_state:
                            with st.spinner(
                                "Generating document embeddings..."
                            ):
                                chunk_texts = tuple(
                                    chunk["text"] for chunk in chunks
                                )
                                st.session_state["chunk_embeddings"] = (
                                    embed_chunks(chunk_texts)
                                )

                        with st.spinner("Searching relevant passages..."):
                            results = search_document(
                                question.strip(),
                                chunks,
                                st.session_state["chunk_embeddings"]
                            )

                        st.subheader("Relevant passages")

                        for rank, result in enumerate(results, start=1):
                            with st.container(border=True):
                                st.markdown(
                                    f"**Result {rank} · PDF page "
                                    f"{result['page']}**"
                                )
                                st.caption(
                                    "Cosine similarity: "
                                    f"{result['score']:.3f}"
                                )
                                st.write(result["text"])

                        st.info(
                            "These are retrieved passages, not AI-generated "
                            "answers. Verify financial claims against the "
                            "original report. Similarity scores are not "
                            "probabilities of correctness."
                        )

                    except Exception as error:
                        st.error(f"Search failed: {error}")

    except Exception as error:
        st.error(f"Could not process the PDF: {error}")
