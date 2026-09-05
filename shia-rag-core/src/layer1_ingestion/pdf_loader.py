"""
SHIA-RAG Layer 1: PDF Document Loader
======================================
Extracts structural text blocks from PDF documents using PyMuPDF (fitz),
producing Tier 1 DocumentNodes and TextBlocks with reading order, page numbers,
bounding boxes, and inferred section paths.

Reference: Chapter 6.2 (Layer 1), SHIA_RAG_Complete_Report.md
"""

from __future__ import annotations

import hashlib
import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple

import pymupdf

_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.layer0_data_model.schemas import DocumentMimeType, DocumentNode, TextBlock

logger = logging.getLogger(__name__)

# Heading detection patterns
HEADING_NUMBERED_RE = re.compile(r"^\s*(\d+(\.\d+)*)\s+([A-Z][A-Za-z0-9\s\-:]{2,80})")
STANDARD_SECTIONS = {
    "abstract", "introduction", "background", "related work",
    "methodology", "method", "proposed method", "architecture",
    "system design", "experiments", "experimental setup",
    "results", "evaluation", "discussion", "conclusion",
    "future work", "references", "appendix"
}


class PDFLoader:
    """
    Ingests PDF files and decomposes them into structural Tier 1 text blocks.
    """

    def __init__(self, min_block_chars: int = 5):
        self.min_block_chars = min_block_chars

    def load_pdf(self, pdf_path: str | Path) -> Tuple[DocumentNode, List[TextBlock]]:
        """
        Parses a PDF document into a DocumentNode and an ordered list of TextBlocks.

        Args:
            pdf_path: Filepath to the PDF document.

        Returns:
            Tuple of (DocumentNode, List[TextBlock]).
        """
        path = Path(pdf_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        # Compute SHA-256
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        sha256_hash = hasher.hexdigest()

        doc = pymupdf.open(str(path))
        total_pages = len(doc)

        doc_node = DocumentNode(
            filename=path.name,
            mime_type=DocumentMimeType.PDF,
            sha256_hash=sha256_hash,
            total_pages=total_pages,
        )

        blocks: List[TextBlock] = []
        global_reading_order = 1
        current_section = "0.0 Document Overview"

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1
            raw_blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)

            # Sort blocks primarily top-to-bottom, secondarily left-to-right
            sorted_raw = sorted(raw_blocks, key=lambda b: (round(b[1] / 20) * 20, b[0]))

            for b in sorted_raw:
                if b[6] != 0:  # Skip image blocks
                    continue

                text = b[4].strip()
                if len(text) < self.min_block_chars:
                    continue

                # Clean control characters and unusual whitespace
                text = re.sub(r"\s+", " ", text)

                # Check if this block is a section heading
                heading_match = HEADING_NUMBERED_RE.match(text)
                first_word = text.split()[0].strip().lower() if text.split() else ""

                if heading_match and len(text) < 120:
                    sec_num = heading_match.group(1)
                    sec_title = heading_match.group(3).strip()
                    current_section = f"{sec_num} {sec_title}"
                elif text.lower() in STANDARD_SECTIONS and len(text) < 80:
                    current_section = f"SEC-{text.upper()[:4]} {text.title()}"
                elif first_word in STANDARD_SECTIONS and len(text) < 80:
                    current_section = f"SEC-{first_word.upper()[:4]} {text.title()}"

                block = TextBlock(
                    doc_id=doc_node.doc_id,
                    reading_order=global_reading_order,
                    text_content=text,
                    section_path=current_section,
                    char_start=0,
                    char_end=len(text),
                    page_number=page_num,
                    bbox_coords=[float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                )
                blocks.append(block)
                global_reading_order += 1

        doc.close()
        logger.info(f"Loaded '{path.name}': {total_pages} pages, {len(blocks)} text blocks.")
        return doc_node, blocks
