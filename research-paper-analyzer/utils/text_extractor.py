import PyPDF2
import pdfplumber
import io
from typing import Union
import logging

logger = logging.getLogger(__name__)


def extract_text_from_file(uploaded_file) -> str:
    """Extract text from uploaded PDF or text file."""

    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        return str(uploaded_file.read(), "utf-8")
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.type}")


def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF using multiple methods with fallback.
    
    Reads the file bytes once upfront to avoid file-pointer issues
    when falling back between extraction methods.
    """

    # BUG FIX: Read bytes once and reuse, avoids file-pointer exhaustion
    pdf_file.seek(0)
    file_bytes = pdf_file.read()

    # Method 1: Try pdfplumber (better for complex layouts)
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            if text.strip():
                return text
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed: {e}")

    # Method 2: Fallback to PyPDF2
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if text.strip():
            return text
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed: {e}")

    raise ValueError("Could not extract text from PDF. File may be corrupted or image-based.")