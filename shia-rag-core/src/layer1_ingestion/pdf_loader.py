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
HEADING_CHAPTER_RE = re.compile(r"^\s*(?:Chapter|CHAPTER)\s+([0-9IVXLCDM]+)[\.:\s\-—]*([^\n\r]{0,85})", re.I)
HEADING_NUMBERED_RE = re.compile(r"^\s*([1-9]\d*(?:\.\d+)*|[A-H]\.\d+(?:\.\d+)*)\.?\s+([A-Z\u0370-\u03ff][^\n\r]{1,85})")
HEADING_APPENDIX_RE = re.compile(r"^\s*(?:Appendix\s+([A-H])|([A-H]))[\.:\s]\s+([A-Z\u0370-\u03ff][^\n\r]{1,85})", re.I)
HEADING_ROMAN_RE = re.compile(r"^\s*(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI)\.?\s+([A-Z\u0370-\u03ff][^\n\r]{1,85})")
HEADING_TEXTBOOK_RE = re.compile(r"^\s*(EXERCISE\s+\d+(?:\.\d+)*|TRY THESE|THINK,\s*DISCUSS\s*AND\s*WRITE|WHAT HAVE WE DISCUSSED\??|PROPERTIES OF [A-Z0-9\s\-]+|SOME (?:MORE )?INTERESTING PATTERNS|FINDING (?:THE )?[A-Z\s]+)\b", re.I)

STANDARD_SECTIONS = {
    "abstract", "introduction", "background", "related work",
    "methodology", "proposed method", "our approach", "our scheme",
    "system model", "architecture", "system design", "preliminaries", "preliminary",
    "security analysis", "performance analysis", "experiments",
    "experimental setup", "experimental results", "results and discussion",
    "evaluation", "empirical evaluation", "discussion", "conclusion",
    "conclusions", "conclusions and future work", "future work",
    "references", "appendix", "system overview",
    "theoretical analysis", "ablation study", "problem formulation",
    "threat model", "implementation details", "comparative analysis"
}

WATERMARK_PATTERNS = re.compile(
    r"^\s*(?:Reprint\s+20\d\d(?:-\d\d)?|Rationalised\s+20\d\d(?:-\d\d)?|Draft|Confidential|Page\s+\d+\s+of\s+\d+|\d{1,4})\s*$",
    re.I
)


class PDFLoader:
    """
    Ingests PDF files and decomposes them into structural Tier 1 text blocks.
    Supports academic papers, textbooks, technical reports, and arbitrary unnumbered documents.
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

        # Step 1: Compute modal body font size for dynamic heading scale detection
        font_counts: dict[float, int] = {}
        for p_idx in range(min(total_pages, 8)):
            p_dict = doc[p_idx].get_text("dict")
            for b in p_dict.get("blocks", []):
                if "lines" in b:
                    for line in b["lines"]:
                        for span in line.get("spans", []):
                            sz = round(span.get("size", 10.0), 1)
                            t_len = len(span.get("text", "").strip())
                            if t_len > 0:
                                font_counts[sz] = font_counts.get(sz, 0) + t_len
        body_font_size = 10.0
        if font_counts:
            body_font_size = max(font_counts.items(), key=lambda x: x[1])[0]

        blocks: List[TextBlock] = []
        global_reading_order = 1
        current_section = "0.0 Document Overview"
        is_ref_mode = False

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1
            raw_blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)

            # PyMuPDF naturally analyzes layout columns and assigns block_no (b[5]).
            # We sort by b[5] to respect natural multi-column reading order.
            sorted_raw = sorted(raw_blocks, key=lambda b: b[5])

            for b in sorted_raw:
                if b[6] != 0:  # Skip image blocks
                    continue

                raw_text = b[4].strip()
                if not raw_text or WATERMARK_PATTERNS.match(raw_text):
                    continue

                # Split raw block if an interior line contains a section heading (e.g. 5.5.1 Finding square roots)
                raw_lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
                sub_chunks: List[str] = []
                curr_lines: List[str] = []
                for line in raw_lines:
                    clean_l = re.sub(r"\s+", " ", line).strip()
                    is_heading_cand = (
                        bool(HEADING_NUMBERED_RE.match(clean_l))
                        or bool(HEADING_CHAPTER_RE.match(clean_l))
                        or bool(HEADING_APPENDIX_RE.match(clean_l))
                        or bool(HEADING_ROMAN_RE.match(clean_l))
                        or bool(HEADING_TEXTBOOK_RE.match(clean_l))
                        or (clean_l.lower() in STANDARD_SECTIONS and len(clean_l) < 50)
                    ) and not (
                        clean_l.endswith((".", ",", ";", ":", "=", "<", ">", "+", "-", "(", ")", "/"))
                        or bool(re.search(r"\.\s*\.\s*\.", clean_l))
                        or bool(re.search(r"[←→⇒⇐∪∩∑∫√≠≤≥]|//", clean_l))
                        or len(re.findall(r"\d+\.\d+", clean_l)) >= 2
                        or bool(re.search(r"\d+\s+\d+", clean_l))
                    )
                    if is_heading_cand and curr_lines:
                        sub_chunks.append("\n".join(curr_lines))
                        curr_lines = [line]
                    else:
                        curr_lines.append(line)
                if curr_lines:
                    sub_chunks.append("\n".join(curr_lines))

                for sub_text in sub_chunks:
                    text = sub_text.strip()
                    if len(text) < self.min_block_chars or WATERMARK_PATTERNS.match(text):
                        continue

                    # Clean control characters and unusual whitespace
                    text = re.sub(r"\s+", " ", text)

                    # Check if this block or its first line is a section heading
                    raw_sub_lines = [l.strip() for l in sub_text.split("\n") if l.strip()]
                    first_line = re.sub(r"\s+", " ", raw_sub_lines[0]).strip() if raw_sub_lines else ""
                    clean_heading = text.strip()

                    candidates_to_test = [first_line] if first_line and len(first_line) < len(clean_heading) else []
                    candidates_to_test.append(clean_heading)

                    for cand in candidates_to_test:
                        is_invalid_heading = (
                            cand.endswith((".", ",", ";", ":", "=", "<", ">", "+", "-", "(", ")", "/"))
                            or bool(re.search(r"\.\s*\.\s*\.", cand))
                            or bool(re.search(r"[←→⇒⇐∪∩∑∫√≠≤≥]|//", cand))
                            or len(re.findall(r"\d+\.\d+", cand)) >= 2
                            or bool(re.search(r"\d+\s+\d+", cand))
                            or (bool(re.search(r"\s+\d{1,3}$", cand)) and ("Appendix" in cand or bool(re.match(r"^[A-H]\s+", cand))))
                        )

                        if not is_invalid_heading and len(cand) <= 90:
                            num_m = HEADING_NUMBERED_RE.match(cand)
                            chap_m = HEADING_CHAPTER_RE.match(cand) if not num_m else None
                            app_m = HEADING_APPENDIX_RE.match(cand) if not (num_m or chap_m) else None
                            rom_m = HEADING_ROMAN_RE.match(cand) if not (num_m or chap_m or app_m) else None
                            tb_m = HEADING_TEXTBOOK_RE.match(cand) if not (num_m or chap_m or app_m or rom_m) else None

                            if num_m and not is_ref_mode and not re.search(r"\b(et al|proceedings|arxiv|doi|conference|journal)\b", num_m.group(2), re.I):
                                current_section = f"{num_m.group(1)} {num_m.group(2).strip()}"
                                break
                            elif chap_m:
                                chap_num = chap_m.group(1)
                                chap_title = chap_m.group(2).strip()
                                current_section = f"Chapter {chap_num}: {chap_title}" if chap_title else f"Chapter {chap_num}"
                                break
                            elif app_m:
                                app_letter = app_m.group(1) or app_m.group(2)
                                app_title = app_m.group(3).strip()
                                current_section = f"Appendix {app_letter}: {app_title}"
                                is_ref_mode = False  # Appendices come after references
                                break
                            elif rom_m and not is_ref_mode:
                                current_section = f"{rom_m.group(1)} {rom_m.group(2).strip()}"
                                break
                            elif tb_m:
                                clean_name = cand.rstrip(":").strip().title()
                                current_section = f"SEC-{clean_name.upper()[:4]} {clean_name}"
                                break
                            elif cand.lower() in STANDARD_SECTIONS or (cand.lower().rstrip(":") in STANDARD_SECTIONS and len(cand) < 50):
                                clean_name = cand.rstrip(":").strip().title()
                                current_section = f"SEC-{clean_name.upper()[:4]} {clean_name}"
                                if "reference" in clean_name.lower():
                                    is_ref_mode = True
                                break

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
