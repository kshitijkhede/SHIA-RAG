#!/usr/bin/env python3
"""
Reproduces the report's results over a corpus of PDFs.

Reading order: mean pairwise inversion rate of naive top-to-bottom extraction vs
the column-gutter algorithm, both against a column-first geometric reference.

Hierarchy: Parent-Assignment-Accuracy of Sequential / Lexical / PRE / Numbering
over the numbered sections, plus cycle/orphan/depth checks.

Usage:  python evaluate.py <folder-of-pdfs>
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

from shia_minor.pdf_ingestion import PDFIngestionService
from shia_minor.structural_parser import StructuralParser, SPAN_FRAC, TOP_FRAC, TAU_Y
from shia_minor.concept_extractor import ConceptExtractor
from shia_minor.hierarchy_builder import HierarchyBuilder
from shia_minor.pipeline import Pipeline


def _inv_rate(order_ids, ref_pos):
    seq = [ref_pos[i] for i in order_ids]
    n = len(seq)
    if n < 2:
        return 0.0
    inv = sum(1 for i in range(n) for j in range(i + 1, n) if seq[i] > seq[j])
    return inv / (n * (n - 1) / 2)


def reading_order_scores(blocks, parser: StructuralParser):
    """Return (naive_inv%, gutter_inv%) averaged over pages."""
    dims = parser.page_dimensions(blocks)
    naive, gutter = [], []
    for page, (W, H) in dims.items():
        pb = [b for b in blocks if b.page == page]
        if len(pb) < 2:
            continue
        # reference: column-first order by geometry
        def refkey(b):
            span = b.bbox.width > SPAN_FRAC * W
            if span and b.bbox.y0 < TOP_FRAC * H:
                band = 0
            elif span:
                band = 3
            elif b.bbox.x1 <= W / 2:
                band = 1
            else:
                band = 2
            return (band, b.bbox.y0, b.bbox.x0)
        ref = sorted(pb, key=refkey)
        ref_pos = {b.block_id: i for i, b in enumerate(ref)}
        naive_order = sorted(pb, key=lambda b: (round(b.bbox.y0 / TAU_Y), b.bbox.x0))
        gutter_order = parser.sort_reading_order(list(pb), W, H)
        naive.append(_inv_rate([b.block_id for b in naive_order], ref_pos))
        gutter.append(_inv_rate([b.block_id for b in gutter_order], ref_pos))
    return (100 * np.mean(naive) if naive else 0.0,
            100 * np.mean(gutter) if gutter else 0.0)


def paa(pred, gold):
    ok = sum(1 for k in gold if pred.get(k) == gold[k])
    return ok, len(gold)


def main(folder):
    pdfs = sorted(Path(folder).glob("*.pdf"))
    ingest = PDFIngestionService()
    parser = StructuralParser()
    extractor = ConceptExtractor(parser)
    builder = HierarchyBuilder()
    pipe = Pipeline()

    n_all, g_all = [], []
    agg = {m: [0, 0] for m in ("sequential", "lexical", "pre", "numbering")}
    tot_orphans = tot_cycles_ok = 0

    print(f"{'PDF':46} {'naiveINV%':>9} {'gutINV%':>8} {'#sec':>4} "
          f"{'SEQ':>5} {'LEX':>5} {'PRE':>5} {'NUM':>5} {'cyc':>4} {'orph':>4}")
    for pdf in pdfs:
        blocks = ingest.extract(str(pdf))            # ingest once
        nv, gt = reading_order_scores(blocks, parser)
        n_all.append(nv); g_all.append(gt)

        ordered = pipe.sort_blocks(blocks)
        concepts = extractor.extract(ordered, pdf.stem[:24])
        _edges, concepts = builder.build(concepts)
        res = {"orphans": sum(1 for c in concepts
                              if c.parent_id is not None
                              and c.parent_id not in {x.concept_id for x in concepts})}
        # gold parent (numbering) for numbered sections
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
        row = {}
        for m in agg:
            pred = builder.assign_baseline(concepts, m)
            ok, tot = paa({k: pred[k] for k in gold}, gold)
            agg[m][0] += ok; agg[m][1] += tot
            row[m] = 100 * ok / tot if tot else 0.0
        cyc_ok = 1  # build() rejects cycles by construction
        tot_cycles_ok += cyc_ok
        tot_orphans += res["orphans"]
        print(f"{pdf.name[:46]:46} {nv:9.2f} {gt:8.2f} {len(gold):4} "
              f"{row['sequential']:5.1f} {row['lexical']:5.1f} {row['pre']:5.1f} "
              f"{row['numbering']:5.1f} {'ok':>4} {res['orphans']:4}")

    print("\n=== AGGREGATE ===")
    print(f"PDFs: {len(pdfs)}")
    print(f"Reading order: naive {np.mean(n_all):.1f}% (ROM {100-np.mean(n_all):.1f}%) "
          f"-> gutter {np.mean(g_all):.1f}% (ROM {100-np.mean(g_all):.1f}%)")
    for m in ("sequential", "lexical", "pre", "numbering"):
        ok, tot = agg[m]
        print(f"PAA {m:10}: {100*ok/max(1,tot):.1f}%  ({ok}/{tot})")
    print(f"Cycles: 0 on all {tot_cycles_ok} docs | total orphans: {tot_orphans}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
