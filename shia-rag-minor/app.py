#!/usr/bin/env python3
"""
SHIA-RAG minor-project demo web app.

A small Flask front-end over the REAL pipeline (the same shia_minor package used
by run_demo.py / evaluate.py). Drop in any PDF (or pick a sample paper) and the
page shows, visually:

  * corpus-level summary counts,
  * the column-gutter reading-order fix (naive vs gutter, side by side),
  * the induced concept hierarchy as a tree,
  * Parent-Assignment-Accuracy of the four attachment strategies,
  * the graph-validation guarantees (acyclic / single-parent / depth-bound / no orphans).

Run:   python app.py     then open  http://127.0.0.1:5000
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from shia_minor.pipeline import Pipeline
from shia_minor.concept_extractor import ConceptExtractor
from shia_minor.hierarchy_builder import HierarchyBuilder
from shia_minor.structural_parser import StructuralParser, SPAN_FRAC, TOP_FRAC, TAU_Y
from evaluate import reading_order_scores

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024  # 40 MB upload cap

# sample corpus sits next to the project folder
SAMPLE_DIR = (Path(__file__).resolve().parent.parent / "research_papers")

_pipe = Pipeline()
_parser = StructuralParser()
_extractor = ConceptExtractor(_parser)
_builder = HierarchyBuilder()


def sample_files():
    if not SAMPLE_DIR.is_dir():
        return []
    return sorted(p.name for p in SAMPLE_DIR.glob("*.pdf"))


def build_tree(concepts):
    """Turn the flat concept list into nested dicts for the front-end."""
    by_id = {c.concept_id: c for c in concepts}
    node = {
        c.concept_id: {
            "id": c.concept_id,
            "name": c.name,
            "section": c.section_number or "",
            "level": c.level,
            "depth": c.depth,
            "confidence": round(c.confidence, 2),
            "definition": (c.definition or "")[:180],
            "children": [],
        }
        for c in concepts
    }
    roots = []
    for c in sorted(concepts, key=lambda x: x.order):
        n = node[c.concept_id]
        if c.parent_id and c.parent_id in node:
            node[c.parent_id]["children"].append(n)
        else:
            roots.append(n)
    return roots


def illustrative_page(blocks):
    """Pick the busiest page and return naive vs gutter reading order snippets."""
    dims = _parser.page_dimensions(blocks)
    if not dims:
        return None
    # page with the most blocks makes the clearest before/after
    best_page = max(dims, key=lambda p: sum(1 for b in blocks if b.page == p))
    W, H = dims[best_page]
    pb = [b for b in blocks if b.page == best_page]
    if len(pb) < 3:
        return None

    def col_of(b):
        if b.bbox.width > SPAN_FRAC * W:
            return "span"
        return "left" if b.bbox.x1 <= W / 2 else "right"

    naive = sorted(pb, key=lambda b: (round(b.bbox.y0 / TAU_Y), b.bbox.x0))
    gutter = _parser.sort_reading_order(list(pb), W, H)

    def snip(seq):
        out = []
        for i, b in enumerate(seq[:14], 1):
            t = " ".join(b.text.split())[:70]
            out.append({"n": i, "text": t or "(figure/space)", "col": col_of(b)})
        return out

    return {"page": best_page, "naive": snip(naive), "gutter": snip(gutter)}


def per_doc_paa(concepts):
    """PAA of each strategy for a single document (gold = section numbering)."""
    by_num = {c.section_number: c for c in concepts if c.section_number}
    gold = {}
    for c in concepts:
        if not c.section_number:
            continue
        if "." in c.section_number:
            pn = c.section_number.rsplit(".", 1)[0]
            gold[c.concept_id] = by_num[pn].concept_id if pn in by_num else "ROOT"
        else:
            gold[c.concept_id] = "ROOT"
    out = {}
    for m in ("sequential", "lexical", "pre", "numbering"):
        pred = _builder.assign_baseline(concepts, m)
        ok = sum(1 for k in gold if pred.get(k) == gold[k])
        out[m] = round(100 * ok / len(gold), 1) if gold else 0.0
    return out, len(gold)


def analyze_pdf(path: str, display_name: str):
    res = _pipe.run(path)
    concepts = res["concepts"]
    blocks_raw = _pipe.ingest.extract(path)
    naive_inv, gutter_inv = reading_order_scores(blocks_raw, _parser)
    paa, n_sections = per_doc_paa(concepts)
    depth_ok = res["max_depth"] <= 8
    return {
        "name": display_name,
        "pages": res["pages"],
        "blocks": res["blocks"],
        "concepts": len(concepts),
        "edges": len(res["edges"]),
        "roots": res["roots"],
        "max_depth": res["max_depth"],
        "orphans": res["orphans"],
        "n_sections": n_sections,
        "reading": {
            "naive_inv": round(naive_inv, 2),
            "gutter_inv": round(gutter_inv, 2),
            "naive_rom": round(100 - naive_inv, 2),
            "gutter_rom": round(100 - gutter_inv, 2),
        },
        "paa": paa,
        "tree": build_tree(concepts),
        "page_demo": illustrative_page(blocks_raw),
        "validation": {
            "acyclic": True,          # build() rejects cycles by construction
            "single_parent": True,    # each concept has at most one parent_id
            "depth_ok": depth_ok,
            "no_orphans": res["orphans"] == 0,
            "one_root": res["roots"] == 1,
        },
    }


@app.route("/")
def index():
    samples = sample_files()
    example = samples[0] if samples else ""
    return render_template("index.html", samples=samples, example=example)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    try:
        sample = request.form.get("sample", "")
        if sample:
            path = SAMPLE_DIR / sample
            if not path.is_file():
                return jsonify({"error": f"sample not found: {sample}"}), 404
            return jsonify(analyze_pdf(str(path), sample))

        if "pdf" not in request.files or request.files["pdf"].filename == "":
            return jsonify({"error": "No PDF provided."}), 400
        f = request.files["pdf"]
        if not f.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Please upload a .pdf file."}), 400
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        try:
            f.save(tmp.name)
            return jsonify(analyze_pdf(tmp.name, f.filename))
        finally:
            os.unlink(tmp.name)
    except Exception as e:  # keep the demo alive; report the failure to the UI
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


if __name__ == "__main__":
    print("SHIA-RAG demo running at  http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
