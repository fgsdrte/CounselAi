"""
clean.py
========
Normalises raw text extracted from PDFs and DOCX files.
Handles legal-document quirks: section numbers, citations,
whitespace artifacts, and non-ASCII characters.

Usage:
    from data_pipeline.clean import clean_text, clean_sections
"""

import re
import unicodedata
import logging
from typing import Union

logger = logging.getLogger(__name__)


# ── Regex patterns ────────────────────────────────────────────────────────────

# Collapse 3+ newlines into 2
_EXCESS_NEWLINES = re.compile(r"\n{3,}")

# Collapse multiple spaces/tabs into one space
_EXCESS_SPACES = re.compile(r"[ \t]{2,}")

# Remove form-feed / carriage return
_FORM_FEED = re.compile(r"[\f\r]")

# Remove lone page numbers (e.g. lines that are just "— 12 —" or "Page 3")
_PAGE_MARKERS = re.compile(r"^\s*(page\s+\d+|[-–—]+\s*\d+\s*[-–—]+)\s*$", re.IGNORECASE | re.MULTILINE)

# Remove repeated dashes used as separators
_SEPARATOR_LINES = re.compile(r"^[-=_*]{4,}\s*$", re.MULTILINE)

# Fix broken hyphenation across lines (word-\nword → wordword)
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")

# Remove null bytes and non-printable control characters (except \n \t)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Normalise various quote/dash styles to ASCII
_FANCY_QUOTES = str.maketrans({
    "\u2018": "'", "\u2019": "'",   # curly single quotes
    "\u201c": '"', "\u201d": '"',   # curly double quotes
    "\u2013": "-", "\u2014": "-",   # en-dash, em-dash
    "\u00a0": " ",                  # non-breaking space
    "\u2026": "...",                # ellipsis
})


# ── Core cleaner ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise a raw text string extracted from a legal document.

    Steps applied (in order):
    1. Decode unicode to closest ASCII equivalents
    2. Fix fancy quotes and dashes
    3. Strip control characters
    4. Fix broken hyphenation
    5. Remove page markers and separator lines
    6. Collapse excess whitespace
    7. Strip leading/trailing whitespace

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned, normalised text string.
    """
    if not text:
        return ""

    # 1. Normalise unicode (NFC) then translate fancy characters
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_FANCY_QUOTES)

    # 2. Remove control characters (keep \n and \t)
    text = _CONTROL_CHARS.sub("", text)
    text = _FORM_FEED.sub("\n", text)

    # 3. Fix broken hyphenation
    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    # 4. Remove page markers and separator lines
    text = _PAGE_MARKERS.sub("", text)
    text = _SEPARATOR_LINES.sub("", text)

    # 5. Collapse whitespace
    text = _EXCESS_SPACES.sub(" ", text)
    text = _EXCESS_NEWLINES.sub("\n\n", text)

    # 6. Final strip
    text = text.strip()

    return text


def clean_sections(sections: list[dict]) -> list[dict]:
    """
    Apply clean_text to every section/page dict produced by the parsers.

    Accepts both PDF page dicts and DOCX section dicts.
    Drops any section whose text is empty after cleaning.

    Args:
        sections: List of dicts with at least a "text" key.

    Returns:
        List of cleaned dicts (empty-text sections removed).
    """
    cleaned = []
    dropped = 0

    for item in sections:
        item = item.copy()
        item["text"] = clean_text(item.get("text", ""))

        if len(item["text"]) < 30:
            # Skip near-empty sections (headers, blank pages, etc.)
            dropped += 1
            continue

        cleaned.append(item)

    if dropped:
        logger.info(f"  Dropped {dropped} near-empty sections after cleaning")

    return cleaned


def clean_for_embedding(text: str, max_chars: int = 8000) -> str:
    """
    Final clean pass before sending text to the embedding model.
    Truncates to max_chars to stay within model limits.

    Args:
        text: Already-cleaned text.
        max_chars: Maximum character length (Vertex AI text-embedding-004
                   supports ~8192 tokens ≈ ~8000 chars for English legal text).

    Returns:
        Truncated, stripped text.
    """
    text = clean_text(text)
    if len(text) > max_chars:
        # Truncate at the last sentence boundary before the limit
        truncated = text[:max_chars]
        last_period = truncated.rfind(".")
        if last_period > max_chars * 0.8:
            truncated = truncated[: last_period + 1]
        text = truncated
    return text


if __name__ == "__main__":
    sample = """
    Section\t1.1  –  Introduction
    \u201cThe party of the first part\u201d  shall here-
    inafter be referred to as the \u2018Licensor\u2019.

    Page 1

    ================

    All   rights    reserved.
    """
    print(clean_text(sample))