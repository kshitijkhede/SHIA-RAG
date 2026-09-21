#!/usr/bin/env python3
"""
Robustness of structure-aware document representation to STRUCTURAL ERRORS.

New minor-project research experiment (Candidate C):
  RQ:  How much does structure-aware retrieval context degrade when the recovered
       structure is imperfect, and can node confidence (KCE) detect/repair the damage?
  H1:  Downstream structural fidelity degrades as reading-order / heading-number
       errors increase.
  H2:  Errors concentrate in low-confidence nodes, so confidence-gated repair
       recovers a large fraction of the lost quality.

Metrics (LLM-free, retrieval-relevant):
  PAA  - Parent-Assignment Accuracy (correct immediate parent vs numbering gold)
  APA  - Ancestor-Path Accuracy (the full root->node lineage is correct)
         == the "retrieval context" a hierarchical retriever would attach
  ACP  - Ancestor-Context Precision (fraction of retrieved ancestors that are correct)

Perturbations (model real parser failure):
  rho  - reading-order noise: swap a fraction of adjacent concepts' reading order
  delta- heading-number dropout: a heading loses its section number (mis-detection)

Mitigation:
  confidence-gated repair: for nodes with KCE confidence < tau, discard the
  (possibly corrupted) predicted parent and fall back to the nearest earlier
  shallower heading (a safe structural prior).

Usage: python robustness_eval.py <folder-of-pdfs> [seeds]
"""
from __future__ import annotations
import sys, random, statistics
from pathlib import Path
import numpy as np

from shia_minor.pdf_ingestion import PDFIngestionService
from shia_minor.structural_parser import StructuralParser
from shia_minor.concept_extractor import ConceptExtractor
from shia_minor.hierarchy_builder import HierarchyBuilder
from shia_minor.kce_scorer import KCEScorer
from shia_minor.pipeline import Pipeline


# ---------- gold structure from original numbering ----------
def gold_maps(concepts):
    by_num = {c.section_number: c.concept_id for c in concepts if c.section_number}
    gp = {}
    for c in concepts:
        if not c.section_number:
            continue
        if "." in c.section_number:
            pn = c.section_number.rsplit(".", 1)[0]
            gp[c.concept_id] = by_num.get(pn, "ROOT")
        else:
            gp[c.concept_id] = "ROOT"
    # gold ancestor chain (set of ancestor concept-ids up to ROOT)
    ga = {}
    for cid in gp:
        chain, cur, seen = [], cid, set()
        while cur in gp and gp[cur] != "ROOT" and cur not in seen:
            seen.add(cur); cur = gp[cur]; chain.append(cur)
        ga[cid] = chain
    return gp, ga


def pred_maps(concepts):
    """effective parent map: the single document-title root is treated as ROOT."""
    roots = [c.concept_id for c in concepts if c.parent_id is None]
    root_set = set(roots)
    nodes = {c.concept_id: c for c in concepts}
    pp = {}
    for c in concepts:
        p = c.parent_id
        pp[c.concept_id] = "ROOT" if (p is None or p in root_set) else p
    pa = {}
    for cid in pp:
        chain, cur, seen = [], cid, set()
        while pp.get(cur, "ROOT") != "ROOT" and cur not in seen:
            seen.add(cur); cur = pp[cur]; chain.append(cur)
        pa[cid] = chain
    return pp, pa


def score(concepts, gp, ga):
    pp, pa = pred_maps(concepts)
    keys = [k for k in gp]                       # evaluate over originally-numbered nodes
    if not keys:
        return None
    paa = 100.0 * sum(1 for k in keys if pp.get(k, "ROOT") == gp[k]) / len(keys)
    apa = 100.0 * sum(1 for k in keys if set(pa.get(k, [])) == set(ga[k])) / len(keys)
    acp_vals = []
    for k in keys:
        pset = set(pa.get(k, []))
        if not pset:
            acp_vals.append(1.0 if not ga[k] else 0.0)
        else:
            acp_vals.append(len(pset & set(ga[k])) / len(pset))
    acp = 100.0 * statistics.mean(acp_vals)
    return paa, apa, acp, pp, keys


# ---------- perturbations ----------
def perturb(concepts, rho, delta, rng):
    cs = [c.model_copy(deep=True) for c in concepts]
    cs.sort(key=lambda c: c.order)
    # reading-order noise: swap adjacent pairs
    n = len(cs)
    for i in range(n - 1):
        if rng.random() < rho:
            cs[i].order, cs[i + 1].order = cs[i + 1].order, cs[i].order
    # heading-number dropout
    for c in cs:
        if c.section_number and rng.random() < delta:
            c.section_number = None
    for c in cs:                                 # reset structure before rebuild
        c.parent_id = None; c.depth = 0
    return cs


# ---------- confidence-gated repair ----------
def gated_repair(concepts, tau):
    cs = sorted(concepts, key=lambda c: c.order)
    nodes = {c.concept_id: c for c in cs}
    root = next((c for c in cs if c.parent_id is None), None)
    for i, c in enumerate(cs):
        if c is root:
            continue
        if c.confidence < tau:
            # discard predicted parent; attach to nearest earlier shallower heading
            newp = root
            for u in reversed(cs[:i]):
                if u.level < c.level:
                    newp = u; break
            c.parent_id = newp.concept_id if newp else None
    return cs


def run(folder, seeds=5):
    pdfs = sorted(Path(folder).glob("*.pdf"))
    ingest = PDFIngestionService(); pipe = Pipeline()
    extr = ConceptExtractor(); builder = HierarchyBuilder(); kce = KCEScorer()

    # ingest + extract once per pdf
    docs = []
    for pdf in pdfs:
        blocks = ingest.extract(str(pdf))
        ordered = pipe.sort_blocks(blocks)
        concepts = extr.extract(ordered, pdf.stem[:16])
        gp, ga = gold_maps(concepts)
        if gp:
            docs.append((pdf.stem[:22], concepts, gp, ga))

    levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    print(f"Corpus: {len(docs)} documents with numbered structure\n")

    # ---- H1: degradation curves (reading-order noise; heading-number dropout) ----
    def sweep(param):
        print(f"=== Degradation vs {param} (mean over corpus x {seeds} seeds) ===")
        print(f"{'level':>6} {'PAA':>7} {'APA':>7} {'ACP':>7}")
        rows = []
        for lv in levels:
            paas, apas, acps = [], [], []
            for s in range(seeds):
                rng = random.Random(1000 + s)
                for name, concepts, gp, ga in docs:
                    rho = lv if param == "rho(reading-order)" else 0.0
                    dlt = lv if param == "delta(number-loss)" else 0.0
                    cs = perturb(concepts, rho, dlt, rng)
                    builder.build(cs); kce.score(cs)
                    r = score(cs, gp, ga)
                    if r: paas.append(r[0]); apas.append(r[1]); acps.append(r[2])
            print(f"{lv:6.1f} {np.mean(paas):7.1f} {np.mean(apas):7.1f} {np.mean(acps):7.1f}")
            rows.append((lv, np.mean(paas), np.mean(apas), np.mean(acps)))
        print()
        return rows

    sweep("rho(reading-order)")
    sweep("delta(number-loss)")

    # ---- H2a: do errors concentrate in low-confidence nodes? ----
    print("=== H2a: confidence of correct vs wrong attachments (delta=0.3) ===")
    conf_ok, conf_bad = [], []
    for s in range(seeds):
        rng = random.Random(2000 + s)
        for name, concepts, gp, ga in docs:
            cs = perturb(concepts, 0.0, 0.3, rng)
            builder.build(cs); kce.score(cs)
            pp, _ = pred_maps(cs); nodes = {c.concept_id: c for c in cs}
            for k in gp:
                c = nodes[k]
                (conf_ok if pp.get(k, "ROOT") == gp[k] else conf_bad).append(c.confidence)
    mo, mb = np.mean(conf_ok), np.mean(conf_bad)
    print(f"mean KCE confidence  correct={mo:.3f}  wrong={mb:.3f}  gap={mo-mb:+.3f}")
    # point-biserial correlation (confidence vs correctness)
    allc = conf_ok + conf_bad
    y = [1]*len(conf_ok) + [0]*len(conf_bad)
    if len(set(allc)) > 1:
        r = np.corrcoef(allc, y)[0, 1]
        print(f"point-biserial r(confidence, correct) = {r:+.3f}  (n={len(allc)})\n")

    # ---- H2b: confidence-gated repair recovers APA ----
    print("=== H2b: confidence-gated repair (delta=0.3, tau=0.60) ===")
    print(f"{'setting':>14} {'PAA':>7} {'APA':>7} {'ACP':>7}")
    for label, gate in [("no repair", False), ("gated repair", True)]:
        paas, apas, acps = [], [], []
        for s in range(seeds):
            rng = random.Random(3000 + s)
            for name, concepts, gp, ga in docs:
                cs = perturb(concepts, 0.0, 0.3, rng)
                builder.build(cs); kce.score(cs)
                if gate:
                    cs = gated_repair(cs, tau=0.60)
                r = score(cs, gp, ga)
                if r: paas.append(r[0]); apas.append(r[1]); acps.append(r[2])
        print(f"{label:>14} {np.mean(paas):7.1f} {np.mean(apas):7.1f} {np.mean(acps):7.1f}")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    run(folder, seeds)
