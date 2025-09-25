import PyPDF2
from typing import Union


def extract_text(file: Union[bytes, object]) -> str:
    """
    Extract text from an uploaded file-like object.
    Supports PDF and plain text uploads from Streamlit's uploader.
    """
    if hasattr(file, "type") and file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    # Fallback: assume text bytes or file-like
    try:
        content = file.read()
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="ignore")
        return str(content)
    except Exception:
        return "" 