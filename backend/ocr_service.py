"""
CogniFuse — OCR Service
Handles text extraction from images (EasyOCR) and PDFs (PyPDF2).
"""

import io


def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract text from an image using EasyOCR.
    Returns plain text string.
    """
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)
        # Tuned parameters for handwritten text
        results = reader.readtext(
            image_bytes,
            paragraph=True,           # Group text into paragraphs
            x_ths=1.0,                # Allow wider horizontal spacing 
            y_ths=0.5,                # Allow slight vertical misalignment
            contrast_ths=0.1,         # Sensitive to faint strokes
            adjust_contrast=0.5,
            text_threshold=0.6        # Filter out random noise
        )
        text_parts = [result[1] for result in results]
        return "\\n".join(text_parts)
    except Exception as e:
        print(f"[OCR] Error extracting text from image: {e}")
        return ""


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF using PyPDF2.
    Returns plain text string.
    """
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts)
    except Exception as e:
        print(f"[OCR] Error extracting text from PDF: {e}")
        return ""
