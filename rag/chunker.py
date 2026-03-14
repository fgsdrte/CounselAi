"""
rag/chunker.py
==============
Splits cleaned legal-document text into overlapping chunks
ready for embedding and storage in Vertex AI Vector Search.

Strategy: Sentence-aware sliding window with overlap.
- Chunk size  ~500 tokens  (~2000 chars) — good balance for legal text
- Overlap     ~100 tokens  (~400 chars)  — ensures no context is lost at edges
- Respects paragraph boundaries where possible

Usage:
    from rag.chunker import chunk_sections
    chunks = chunk_sections(cleaned_sections, source_doc_id="ipc_2023.pdf")
"""

import re
import uuid
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_CHUNK_SIZE = 2000     # characters (~500 tokens for English legal text)
DEFAULT_OVERLAP    = 400      # characters (~100 tokens)
MIN_CHUNK_SIZE     = 200      # discard chunks smaller than this

# Split on sentence endings followed by whitespace
_SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences. Falls back to word-level if no sentences found."""
    sentences = _SENTENCE_SPLIT.split(text)
    if len(sentences) <= 1:
        # No sentence boundaries — split on paragraph breaks instead
        sentences = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [s.strip() for s in sentences if s.strip()]


def _make_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Generate a deterministic, URL-safe chunk ID."""
    return f"{doc_id}__chunk_{chunk_index:04d}"


# ── Main chunker ──────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    metadata: Optional[dict] = None,
) -> list[dict]:
    """
    Split a single block of text into overlapping chunks.

    Args:
        text:       Cleaned text to chunk.
        doc_id:     Identifier for the source document (used in chunk IDs).
        chunk_size: Target chunk size in characters.
        overlap:    Overlap between consecutive chunks in characters.
        metadata:   Extra metadata to include in every chunk dict.

    Returns:
        List of chunk dicts:
        [
            {
                "chunk_id":   "ipc_2023.pdf__chunk_0001",
                "doc_id":     "ipc_2023.pdf",
                "chunk_index": 1,
                "text":       "...",
                "char_count": 487,
                "metadata":   { "file": "...", "section": "...", ... }
            },
            ...
        ]
    """
    if not text.strip():
        return []

    sentences = _split_into_sentences(text)
    chunks = []
    chunk_index = 0
    buffer = []
    buffer_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if buffer_len + sentence_len > chunk_size and buffer:
            # Emit current buffer as a chunk
            chunk_text_str = " ".join(buffer)
            if len(chunk_text_str) >= MIN_CHUNK_SIZE:
                chunks.append({
                    "chunk_id":    _make_chunk_id(doc_id, chunk_index),
                    "doc_id":      doc_id,
                    "chunk_index": chunk_index,
                    "text":        chunk_text_str,
                    "char_count":  len(chunk_text_str),
                    "metadata":    metadata or {},
                })
                chunk_index += 1

            # Backfill overlap: keep last N chars worth of sentences
            overlap_buffer = []
            overlap_len = 0
            for prev_sentence in reversed(buffer):
                if overlap_len + len(prev_sentence) > overlap:
                    break
                overlap_buffer.insert(0, prev_sentence)
                overlap_len += len(prev_sentence)

            buffer = overlap_buffer
            buffer_len = overlap_len

        buffer.append(sentence)
        buffer_len += sentence_len

    # Emit any remaining text
    if buffer:
        chunk_text_str = " ".join(buffer)
        if len(chunk_text_str) >= MIN_CHUNK_SIZE:
            chunks.append({
                "chunk_id":    _make_chunk_id(doc_id, chunk_index),
                "doc_id":      doc_id,
                "chunk_index": chunk_index,
                "text":        chunk_text_str,
                "char_count":  len(chunk_text_str),
                "metadata":    metadata or {},
            })

    return chunks


def chunk_sections(
    sections: list[dict],
    source_doc_id: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict]:
    """
    Chunk all sections/pages from a parsed document.

    Accepts both PDF page dicts and DOCX section dicts (output of parsers).
    Automatically extracts metadata (file, page/section info, heading).

    Args:
        sections:       List of cleaned section/page dicts.
        source_doc_id:  Override doc_id; defaults to the "file" field.
        chunk_size:     Target chunk character size.
        overlap:        Overlap between chunks in characters.

    Returns:
        Flat list of all chunk dicts across all sections.
    """
    all_chunks = []

    for section in sections:
        doc_id = source_doc_id or section.get("file", str(uuid.uuid4()))

        # Build metadata from whatever the section provides
        metadata = {
            "file":    section.get("file", ""),
            "heading": section.get("heading", ""),
        }
        if "page_number" in section:
            metadata["page_number"] = section["page_number"]
            metadata["source_type"] = section.get("source", "digital")  # digital | ocr
        if "section_index" in section:
            metadata["section_index"] = section["section_index"]
            metadata["heading_level"] = section.get("heading_level", 0)

        text = section.get("text", "")
        chunks = chunk_text(text, doc_id, chunk_size, overlap, metadata)
        all_chunks.extend(chunks)

    logger.info(f"  Generated {len(all_chunks)} chunks from {len(sections)} sections")
    return all_chunks


if __name__ == "__main__":
    import json

    sample_sections = [
        {
            "section_index": 0,
            "heading": "Section 1 — Definitions",
            "heading_level": 1,
            "text": (
                "In this Act, unless the context otherwise requires, the following "
                "expressions shall have the meanings hereby assigned to them respectively. "
                "'Abet' — A person abets the doing of a thing who instigates any person to do "
                "that thing. Explanation: Whoever, either prior to or at the time of the "
                "commission of an act, does anything in order to facilitate the commission of "
                "that act, and thereby facilitates the commission thereof, is said to aid the "
                "doing of that act."
            ),
            "file": "ipc_sample.docx",
        }
    ]

    chunks = chunk_sections(sample_sections, source_doc_id="ipc_sample")
    print(json.dumps(chunks, indent=2))