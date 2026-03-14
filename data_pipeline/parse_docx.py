"""
parse_docx.py
=============
Extracts clean text from .docx and .doc Word files.
Preserves paragraph structure and handles tables.

Usage:
    from data_pipeline.parse_docx import parse_docx
    sections = parse_docx("path/to/file.docx")
"""

import logging
from pathlib import Path

import docx
from docx.oxml.ns import qn

logger = logging.getLogger(__name__)


def _extract_table_text(table) -> str:
    """Convert a docx table to readable text rows."""
    rows = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def _get_heading_level(paragraph) -> int:
    """Return heading level (1–6) or 0 if not a heading."""
    style_name = paragraph.style.name.lower()
    if "heading" in style_name:
        for level in range(1, 7):
            if str(level) in style_name:
                return level
        return 1
    return 0


def parse_docx(file_path: str) -> list[dict]:
    """
    Parse a .docx file and return structured sections.

    Args:
        file_path: Path to the .docx file.

    Returns:
        List of section dicts:
        [
            {
                "section_index": 0,
                "heading": "Introduction",
                "heading_level": 1,
                "text": "Full text of the section...",
                "file": "filename.docx"
            },
            ...
        ]
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"DOCX not found: {file_path}")

    filename = path.name
    logger.info(f"Parsing DOCX: {filename}")

    doc = docx.Document(file_path)

    sections = []
    current_heading = "Introduction"
    current_heading_level = 0
    current_text_parts = []
    section_index = 0

    for block in doc.element.body:
        tag = block.tag.split("}")[-1]  # strip namespace

        if tag == "p":
            para = docx.text.paragraph.Paragraph(block, doc)
            text = para.text.strip()
            if not text:
                continue

            level = _get_heading_level(para)

            if level > 0:
                # Save previous section before starting new one
                if current_text_parts:
                    sections.append({
                        "section_index": section_index,
                        "heading": current_heading,
                        "heading_level": current_heading_level,
                        "text": "\n".join(current_text_parts).strip(),
                        "file": filename,
                    })
                    section_index += 1
                    current_text_parts = []

                current_heading = text
                current_heading_level = level
            else:
                current_text_parts.append(text)

        elif tag == "tbl":
            try:
                table = docx.table.Table(block, doc)
                table_text = _extract_table_text(table)
                current_text_parts.append(f"[TABLE]\n{table_text}\n[/TABLE]")
            except Exception as e:
                logger.warning(f"  Could not parse table: {e}")

    # Save the last section
    if current_text_parts:
        sections.append({
            "section_index": section_index,
            "heading": current_heading,
            "heading_level": current_heading_level,
            "text": "\n".join(current_text_parts).strip(),
            "file": filename,
        })

    logger.info(f"  Extracted {len(sections)} sections from {filename}")
    return sections


if __name__ == "__main__":
    import sys
    import logging

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python parse_docx.py <path_to_docx>")
        sys.exit(1)

    sections = parse_docx(sys.argv[1])
    for s in sections:
        print(f"\n--- Section {s['section_index']}: {s['heading']} (H{s['heading_level']}) ---")
        print(s["text"][:500])