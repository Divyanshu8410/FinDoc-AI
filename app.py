
import streamlit as st
import fitz  # PyMuPDF

st.set_page_config(
    page_title="FinDoc AI",
    page_icon="📄",
    layout="wide"
)

st.title("📄 FinDoc AI")
st.subheader("Financial Document Intelligence Platform")

st.write(
    "Upload a financial report PDF to extract and inspect its text."
)

uploaded_file = st.file_uploader(
    "Upload financial report",
    type=["pdf"]
)

if uploaded_file is not None:
    try:
        pdf_bytes = uploaded_file.getvalue()

        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            st.success("PDF uploaded successfully!")

            st.write(f"Total pages: {len(pdf)}")

            extracted_pages = []

            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text")
                extracted_pages.append(
                    f"--- Page {page_number} ---\n{text}"
                )

            extracted_text = "\n\n".join(extracted_pages)

            if extracted_text.strip():
                st.subheader("Extracted Document Text")
                st.text_area(
                    "PDF content",
                    value=extracted_text,
                    height=400
                )

                st.download_button(
                    label="Download extracted text",
                    data=extracted_text,
                    file_name="extracted_report.txt",
                    mime="text/plain"
                )
            else:
                st.warning(
                    "No selectable text found. "
                    "This PDF may be scanned and require OCR."
                )

    except Exception as error:
        st.error(f"Could not read PDF: {error}")
