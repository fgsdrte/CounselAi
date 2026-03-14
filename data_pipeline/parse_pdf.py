"""
parse_pdf.py
============
Extracts clean text from PDF files (scanned or digital).
Uses pdfplumber for digital PDFs and pytesseract for scanned/image-based PDFs.

Usage:
    from data_pipeline.parse_pdf import parse_pdf
    pages = parse_pdf("path/to/file.pdf")
"""

import logging
from pathlib import Path
from typing import Optional

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

logger = logging.getLogger(__name__)


def _is_scanned_page(page) -> bool:
    """Return True if a pdfplumber page has no extractable text (likely scanned)."""
    text = page.extract_text()
    return not text or len(text.strip()) < 20


def _ocr_page(pil_image: Image.Image) -> str:
    """Run Tesseract OCR on a PIL image and return extracted text."""
    return pytesseract.image_to_string(pil_image, lang="eng")


def parse_pdf(file_path: str, ocr_fallback: bool = True) -> list[dict]:
    """
    Parse a PDF file and return a list of page dicts.

    Args:
        file_path: Path to the PDF file.
        ocr_fallback: If True, run OCR on scanned pages.

    Returns:
        List of dicts, one per page:
        [
            {
                "page_number": 1,
                "text": "...",
                "source": "digital" | "ocr",
                "file": "filename.pdf"
            },
            ...
        ]
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    results = []
    filename = path.name

    logger.info(f"Parsing PDF: {filename}")

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"  Total pages: {total_pages}")

        # For OCR fallback, convert all pages to images once
        pil_images: Optional[list] = None
        if ocr_fallback:
            try:
                pil_images = convert_from_path(file_path, dpi=300)
            except Exception as e:
                logger.warning(f"  Could not convert PDF to images for OCR: {e}")

        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            text = page.extract_text() or ""

            source = "digital"

            if _is_scanned_page(page) and ocr_fallback and pil_images:
                logger.info(f"  Page {page_num}: scanned → running OCR")
                try:
                    text = _ocr_page(pil_images[i])
                    source = "ocr"
                except Exception as e:
                    logger.warning(f"  OCR failed on page {page_num}: {e}")
                    text = ""

            results.append({
                "page_number": page_num,
                "text": text.strip(),
                "source": source,
                "file": filename,
            })

    logger.info(f"  Parsed {len(results)} pages from {filename}")
    return results


if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python parse_pdf.py <path_to_pdf>")
        sys.exit(1)

    pages = parse_pdf(sys.argv[1])
    for p in pages:
        print(f"\n--- Page {p['page_number']} ({p['source']}) ---")
        print(p["text"][:500])