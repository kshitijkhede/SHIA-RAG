"""
Module 2 --- Structural parser: reading-order recovery + heading classification.

Reading order uses the column-gutter algorithm (report Algorithm 1): blocks are
split at the page gutter (W/2), full-width spanning blocks are pulled out to the
top or bottom, and each column is sorted top-to-bottom with a vertical tolerance
tau_y that groups blocks on the same visual line.

Heading classification uses section numbering and font size relative to the body
font.

Reference: report Chapter 3 (Reading-Order Recovery; Heading Classification).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

from .schemas import TextBlock

TAU_Y = 20.0                 # pt: vertical band tolerance
SPAN_FRAC = 0.60             # width fraction that counts as full-width spanning
TOP_FRAC = 0.30             # y fraction that counts as a top spanning block
HEADING_FONT_RATIO = 1.25    # font >= 1.25x body -> candidate heading
NUM_RE = re.compile(r"^(\d+(?:\.\d+){0,5})")
_JUNK = re.compile(r"https?://|www\.|@|doi|arxiv|github|\bet al\b|\bpp\.|\bvol\.|proceedings", re.I)
_STD_SECTIONS = {
    "abstract", "introduction", "related work", "background", "methodology",
    "method", "approach", "experiments", "experiment", "results",
    "results and discussion", "discussion", "evaluation", "conclusion",
    "conclusions", "conclusion and future work", "references", "acknowledgments",
    "acknowledgements", "appendix",
}


class StructuralParser:
    def __init__(self, tau_y: float = TAU_Y):
        self.tau_y = tau_y

    # -- Algorithm 1: column-gutter reading order --
    def sort_reading_order(self, blocks: List[TextBlock],
                           page_w: float, page_h: float) -> List[TextBlock]:
        gut = page_w / 2.0
        s_top, s_bot, c1, c2 = [], [], [], []
        for b in blocks:
            spanning = b.bbox.width > SPAN_FRAC * page_w
            if spanning and b.bbox.y0 < TOP_FRAC * page_h:
                b.column = -1; s_top.append(b)
            elif spanning:
                b.column = -1; s_bot.append(b)
            elif b.bbox.x1 <= gut:
                b.column = 0; c1.append(b)
            else:
                b.column = 1; c2.append(b)
        ordered = (self._sort_span(s_top) + self._sort_column(c1)
                   + self._sort_column(c2) + self._sort_span(s_bot))
        for i, b in enumerate(ordered):
            b.reading_order = i
        return ordered

    def _sort_span(self, bucket: List[TextBlock]) -> List[TextBlock]:
        return sorted(bucket, key=lambda b: b.bbox.y0)

    def _sort_column(self, bucket: List[TextBlock]) -> List[TextBlock]:
        bucket.sort(key=lambda b: b.bbox.y0)
        out, band = [], []
        for b in bucket:
            if band and abs(b.bbox.y0 - band[-1].bbox.y0) > self.tau_y:
                band.sort(key=lambda z: z.bbox.x0); out.extend(band); band = []
            band.append(b)
        if band:
            band.sort(key=lambda z: z.bbox.x0); out.extend(band)
        return out

    # -- body font + heading classification --
    def body_font_size(self, blocks: List[TextBlock]) -> float:
        c: Counter = Counter()
        for b in blocks:
            if len(b.text) > 40:
                c[round(b.font_size)] += len(b.text)
        return float(c.most_common(1)[0][0]) if c else 10.0

    def classify_heading(self, block: TextBlock, body: float) -> Tuple[bool, int, str]:
        """Return (is_heading, level, section_number).

        A numbered heading must be a real section number followed by a short
        title (not a 4-digit year, citation index, or a long body sentence).
        """
        t = block.text.strip()
        first_line = t.split("\n", 1)[0].strip()
        if _JUNK.search(first_line):                     # URLs, citations, DOIs
            return False, 0, ""
        # numbered heading: "3.1 Title" with a Title-cased title, heading-sized
        m = NUM_RE.match(first_line)
        if m and len(first_line) < 90:
            num = m.group(1)
            rest = first_line[m.end():].strip(" .:-")
            parts = num.split(".")
            is_year = len(parts) == 1 and len(parts[0]) >= 4          # e.g. 2024
            # a real section title starts with an UPPERCASE letter of any script
            # ("Experiments", "Ψ-RAG"); this rejects equations ("n ( n-1)"),
            # footers ("th Conference") and ids ("v92500") that start lowercase.
            titled = len(rest) >= 2 and rest[:1].isupper()
            few_words = len(first_line.split()) <= 12
            big_enough = block.font_size >= 0.95 * body
            if not is_year and titled and few_words and big_enough:
                return True, min(6, num.count(".") + 1), num
        # standard unnumbered section by name (Abstract, Introduction, ...)
        key = first_line.lower().strip(" .:0123456789")
        if key in _STD_SECTIONS and len(first_line) < 40:
            return True, 1, ""
        # other unnumbered heading: larger font, short, title-like (>=3 chars
        # so decorative single-letter drop-caps are not treated as headings)
        if 3 <= len(first_line) < 70 and len(first_line.split()) <= 10 and re.match(r"[A-Z]", first_line):
            if block.font_size >= 1.5 * body:
                return True, 1, ""
            if block.font_size >= HEADING_FONT_RATIO * body:
                return True, 2, ""
        return False, 0, ""

    def page_dimensions(self, blocks: List[TextBlock]) -> Dict[int, Tuple[float, float]]:
        """Estimate (width, height) per page from block extents."""
        dims: Dict[int, Tuple[float, float]] = {}
        for b in blocks:
            w, h = dims.get(b.page, (0.0, 0.0))
            dims[b.page] = (max(w, b.bbox.x1), max(h, b.bbox.y1))
        return dims
