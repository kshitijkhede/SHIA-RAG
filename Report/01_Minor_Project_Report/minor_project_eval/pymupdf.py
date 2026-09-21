"""
Minimal PyMuPDF (fitz) compatibility shim backed by pdfplumber.

Implements only what shia-rag-core/src/layer1_ingestion/pdf_loader.py uses:
    pymupdf.open(path) -> _Doc ; len(doc) ; doc[i] -> _Page ; doc.close()
    page.get_text("blocks") -> [(x0,y0,x1,y1,text,block_no,block_type), ...]
    page.get_text("dict")   -> {"blocks":[{"lines":[{"spans":[{"size","text"}]}]}]}

Words are extracted with pdfplumber (bbox + font size), grouped into lines by a
vertical band, then consecutive lines in the same column merged into paragraph
blocks. Coordinates are already top-left origin in pdfplumber (top/bottom), so
no flip is needed. block_type 0 == text.
"""
from __future__ import annotations
import pdfplumber

_LINE_TOL = 3.0     # pt: words on the same visual line
_PARA_GAP = 6.0     # pt: vertical gap that still counts as same paragraph block


def _cluster_page(page):
    words = page.extract_words(extra_attrs=["size"], use_text_flow=False,
                               keep_blank_chars=False)
    if not words:
        return [], []
    W = float(page.width)
    gutter = W / 2.0
    # assign each word to a column: 0=left,1=right,2=spanning(full-width word rare)
    def col(w):
        midx = (w["x0"] + w["x1"]) / 2.0
        return 0 if midx <= gutter else 1
    # group words into lines within each column
    cols = {0: [], 1: []}
    for w in words:
        cols[col(w)].append(w)
    line_records = []  # (col, top, bottom, x0, x1, text, size)
    for c, ws in cols.items():
        ws.sort(key=lambda w: (round(w["top"] / _LINE_TOL), w["x0"]))
        cur = []
        cur_top = None
        for w in ws:
            if cur and abs(w["top"] - cur_top) > _LINE_TOL:
                line_records.append(_mkline(c, cur))
                cur = []
            if not cur:
                cur_top = w["top"]
            cur.append(w)
        if cur:
            line_records.append(_mkline(c, cur))
    return line_records, words


def _mkline(c, ws):
    ws.sort(key=lambda w: w["x0"])
    x0 = min(w["x0"] for w in ws); x1 = max(w["x1"] for w in ws)
    top = min(w["top"] for w in ws); bottom = max(w["bottom"] for w in ws)
    text = " ".join(w["text"] for w in ws)
    sizes = [round(float(w.get("size", 10.0)), 1) for w in ws]
    size = max(set(sizes), key=sizes.count) if sizes else 10.0
    return {"col": c, "top": top, "bottom": bottom, "x0": x0, "x1": x1,
            "text": text, "size": size}


def _lines_to_blocks(lines):
    """Merge consecutive same-column lines with small vertical gaps into blocks."""
    # order: left column top->bottom, then right column top->bottom
    lines = sorted(lines, key=lambda l: (l["col"], l["top"]))
    blocks = []
    cur = None
    for l in lines:
        if cur and l["col"] == cur["col"] and (l["top"] - cur["bottom"]) <= _PARA_GAP \
                and abs(l["size"] - cur["size"]) < 0.6:
            cur["bottom"] = l["bottom"]
            cur["x0"] = min(cur["x0"], l["x0"]); cur["x1"] = max(cur["x1"], l["x1"])
            cur["text"] += "\n" + l["text"]
        else:
            if cur:
                blocks.append(cur)
            cur = dict(l)
    if cur:
        blocks.append(cur)
    return blocks


class _Page:
    def __init__(self, plumber_page):
        self._p = plumber_page
        self.width = float(plumber_page.width)
        self.height = float(plumber_page.height)
        self._lines, _ = _cluster_page(plumber_page)
        self._blocks = _lines_to_blocks(self._lines)

    def get_text(self, kind):
        if kind == "blocks":
            out = []
            for i, b in enumerate(self._blocks):
                out.append((float(b["x0"]), float(b["top"]), float(b["x1"]),
                            float(b["bottom"]), b["text"], i, 0))
            return out
        elif kind == "dict":
            blocks = []
            for b in self._blocks:
                spans = [{"size": b["size"], "text": t} for t in b["text"].split("\n")]
                blocks.append({"lines": [{"spans": [s]} for s in spans]})
            return {"blocks": blocks}
        raise ValueError(f"shim get_text kind not supported: {kind}")


class _Doc:
    def __init__(self, path):
        self._pdf = pdfplumber.open(path)
        self._pages = [_Page(p) for p in self._pdf.pages]

    def __len__(self):
        return len(self._pages)

    def __getitem__(self, i):
        return self._pages[i]

    def close(self):
        try:
            self._pdf.close()
        except Exception:
            pass
        self._pages = []


def open(path):  # noqa: A001
    return _Doc(str(path))
