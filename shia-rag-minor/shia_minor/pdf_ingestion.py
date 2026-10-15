"""
Module 1 --- PDF ingestion.

Extracts Tier 1 text blocks (geometry + font) from a digital PDF. The reference
implementation uses pdfplumber; any extractor that reports per-block coordinates
and font size (for example PyMuPDF) can replace it. Words are grouped into lines,
then consecutive lines with a small vertical gap are merged into paragraph blocks.

Reference: report Chapter 3 (Layout-Aware Ingestion).
"""
from __future__ import annotations

from typing import List
import re
from collections import Counter
import pdfplumber

from .schemas import BoundingBox, TextBlock

_LINE_TOL = 3.0   # pt: words on the same visual line
_PARA_GAP = 6.0   # pt: vertical gap that still counts as one paragraph block
_X_TOL = 1.5      # pt: word-splitting tolerance (recovers spaces on PDFs whose
                  #     font has very narrow space glyphs; default 3 glues words)
# "3.1 Title", "5.Experiments" (no space) or "3.Ψ-RAG" (non-ASCII) all count
_NUM = re.compile(r"^\d+(?:\.\d+){0,5}\.?\s*[^\W\d_]", re.UNICODE)


class PDFIngestionService:
    def extract(self, pdf_path: str) -> List[TextBlock]:
        blocks: List[TextBlock] = []
        bid = 0
        with pdfplumber.open(pdf_path) as pdf:
            for pi, page in enumerate(pdf.pages, start=1):
                for b in self._page_blocks(page):
                    x0 = max(0.0, b["x0"]); y0 = max(0.0, b["top"])
                    x1 = max(x0 + 0.1, b["x1"]); y1 = max(y0 + 0.1, b["bottom"])
                    blocks.append(TextBlock(
                        block_id=bid, page=pi, text=b["text"],
                        bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
                        font_size=max(0.1, b["size"]), is_bold=b["bold"],
                    ))
                    bid += 1
        return blocks

    def _page_blocks(self, page) -> List[dict]:
        # x_tolerance recovers spaces on PDFs whose space glyph is very narrow;
        # without it, whole lines come back as one glued "word".
        words = page.extract_words(extra_attrs=["size", "fontname"],
                                   use_text_flow=False, keep_blank_chars=False,
                                   x_tolerance=_X_TOL)
        if not words:
            return []

        # Column-aware line grouping: group words into visual rows by vertical
        # position, then SPLIT each row at the column gutter so a heading in one
        # column is never concatenated with body text from the other column.
        page_w = float(getattr(page, "width", None) or max(w["x1"] for w in words))
        gutter = page_w / 2.0
        col_gap = max(10.0, 0.03 * page_w)

        words.sort(key=lambda w: (round(w["top"] / _LINE_TOL), w["x0"]))
        rows, cur, cur_top = [], [], None
        for w in words:
            if cur and abs(w["top"] - cur_top) > _LINE_TOL:
                rows.append(cur); cur = []
            if not cur:
                cur_top = w["top"]
            cur.append(w)
        if cur:
            rows.append(cur)

        lines = []
        for row in rows:
            row.sort(key=lambda w: w["x0"])
            seg = [row[0]]
            for prv, w in zip(row, row[1:]):
                straddles = prv["x1"] <= gutter <= w["x0"]
                if (w["x0"] - prv["x1"] > col_gap) or (straddles and w["x0"] - prv["x1"] > 6.0):
                    lines.append(self._mkline(seg)); seg = []
                seg.append(w)
            if seg:
                lines.append(self._mkline(seg))

        # page body size = most common line size weighted by text length
        cnt: Counter = Counter()
        for l in lines:
            if len(l["text"]) > 30:
                cnt[round(l["size"])] += len(l["text"])
        body = cnt.most_common(1)[0][0] if cnt else 10.0

        # merge lines into paragraphs, keeping heading-like lines isolated and
        # never merging across the gutter (same-column guard).
        def col_of(l):
            return 0 if (l["x0"] + l["x1"]) / 2.0 < gutter else 1
        lines.sort(key=lambda l: (col_of(l), l["top"]))
        merged, prev = [], None
        for l in lines:
            txt = l["text"].strip()
            # drop decorative drop-cap / single huge letters (e.g. "M", "X")
            if len(txt) <= 1:
                continue
            heading_like = ((l["size"] >= 1.15 * body or _NUM.match(txt))
                            and len(txt) < 90 and len(txt.split()) <= 12)
            same_col = prev is not None and col_of(l) == col_of(prev)
            if heading_like:
                merged.append(dict(l)); prev = None      # heading stands alone
            elif (prev and same_col and 0 <= (l["top"] - prev["bottom"]) <= _PARA_GAP
                    and abs(l["size"] - prev["size"]) < 0.6):
                prev["bottom"] = l["bottom"]
                prev["x0"] = min(prev["x0"], l["x0"]); prev["x1"] = max(prev["x1"], l["x1"])
                prev["text"] += "\n" + l["text"]
            else:
                prev = dict(l); merged.append(prev)
        return merged

    @staticmethod
    def _mkline(ws) -> dict:
        ws.sort(key=lambda w: w["x0"])
        sizes = [round(float(w.get("size", 10.0)), 1) for w in ws]
        fonts = [str(w.get("fontname", "")) for w in ws]
        return {
            "x0": min(w["x0"] for w in ws), "x1": max(w["x1"] for w in ws),
            "top": min(w["top"] for w in ws), "bottom": max(w["bottom"] for w in ws),
            "text": " ".join(w["text"] for w in ws),
            "size": max(set(sizes), key=sizes.count) if sizes else 10.0,
            "bold": any("bold" in f.lower() for f in fonts),
        }
