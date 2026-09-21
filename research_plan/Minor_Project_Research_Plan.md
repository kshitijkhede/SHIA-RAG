# Minor Project — Research Plan (Redesigned)

**Title (working):** *How Robust is Structure-Aware Retrieval? Quantifying and Mitigating the Effect of Document-Structure Errors in RAG*

**Programme:** M.Tech (Information Technology), NITK Surathkal
**Duration:** 10–12 weeks
**Relationship to prior work:** Reuses the author's implemented layout-aware pipeline (reading-order recovery + numbering/PRE hierarchy + graph validation + KCE confidence) as the *enabling instrument*, and reframes it around a genuine research question.

---

## Why this replaces the earlier Minor

The earlier Minor asked *"can we rebuild document structure?"* — an engineering/parsing task, evaluated on proxy metrics against self-derived gold, with no end-task evidence. This redesign keeps all that working code but turns it into a **research question with a hypothesis and measured evidence**: *structure-aware RAG methods (TreeRAG, HiChunk, BookRAG, LAD-RAG) all assume the extracted structure is correct — but what happens when it isn't, and can we detect and repair the damage?* No prior work quantifies this.

---

## A. Problem Statement
Structure-aware retrieval systems attach retrieval context using a document's recovered hierarchy, yet that hierarchy is itself produced by an imperfect parser; the sensitivity of downstream retrieval context to such structural errors, and whether it can be detected and repaired at inference time, is unquantified.

## B. Motivation
Hierarchical/structure-aware RAG is now mainstream (RAPTOR, TreeRAG, HiChunk, BookRAG, LAD-RAG, 2024–2025). All of them *build* structure and *assume it is right*. Real PDFs produce imperfect structure (multi-column mis-reads, missed or garbled heading numbers). If a section is attached to the wrong parent, a hierarchical retriever pulls the **wrong ancestor context**, silently corrupting what the LLM sees. Nobody has measured how fragile this is or how to guard against it. A robustness study is exactly the kind of "does this actually matter, and what do we do about it?" contribution a research guide expects.

## C. Research Gap
1. **No robustness characterisation.** Structure-aware RAG papers report gains on clean structure; none report degradation under realistic structural error.
2. **No error localisation.** None ask *which* nodes fail or whether that failure is *predictable* from a confidence signal.
3. **No inference-time mitigation.** None provide a way to detect and repair low-quality structure without re-parsing.

## D. Research Questions
- **RQ1:** How much does retrieval-context quality degrade as (a) reading-order errors and (b) heading-detection errors increase?
- **RQ2:** Are structural errors predictable from a per-node confidence signal (KCE)?
- **RQ3:** Can a confidence-gated repair step recover retrieval-context quality at inference time, without re-parsing?

## E. Hypotheses
- **H1:** Retrieval-context quality (ancestor-path accuracy) degrades monotonically with structural error rate.
- **H2:** Errors concentrate in low-confidence nodes (confidence is a usable error detector).
- **H3:** Confidence-gated repair recovers a measurable fraction of the lost quality.

## F. Objectives (measurable)
1. Define retrieval-relevant structural-fidelity metrics that need **no LLM**: Parent-Assignment Accuracy (PAA), **Ancestor-Path Accuracy (APA)**, Ancestor-Context Precision (ACP).
2. Build a **controlled structural-perturbation harness** (reading-order noise ρ, heading-number dropout δ).
3. Measure degradation curves (RQ1/H1).
4. Measure the confidence–error relationship (RQ2/H2).
5. Design and evaluate **confidence-gated repair** (RQ3/H3), with an ablation.

## G. Proposed Method
1. **Instrument** (reused): ingest → column-gutter reading order → heading/concept extraction → numbering-primary + PRE hierarchy → graph validation → KCE confidence.
2. **Perturbation model** (new): inject two independent, controllable error sources that mimic real parser failure — ρ (adjacent reading-order swaps) and δ (a heading loses its section number, forcing the semantic PRE fallback).
3. **Retrieval-context metrics** (new): APA measures whether the full root→node ancestor chain — i.e., the context a hierarchical retriever would attach — is correct; ACP measures the fraction of attached ancestor context that is correct.
4. **Confidence-gated repair** (new, the core contribution): for any node whose KCE confidence < τ, discard the (possibly corrupted) predicted parent and fall back to the nearest earlier shallower heading (a safe structural prior). This is an inference-time guard requiring no re-parse.

## H. Novelty (precise)
> Existing structure-aware RAG assumes the recovered hierarchy is correct (assumption Y). Real parsers produce errors (problem Z), which silently corrupt the ancestor context fed to the LLM. We propose (A) a *controlled structural-perturbation + retrieval-context evaluation framework* that quantifies this fragility, and (B) a *confidence-gated repair* mechanism that uses per-node structural confidence to detect and correct low-quality attachments at inference time. Contribution validated by degradation curves, a confidence–error correlation, and a repair ablation.

The novelty is **not** a new parser or a new embedding — it is the first *robustness characterisation of structure-aware retrieval* plus a *confidence-driven inference-time mitigation*.

## I. Architecture / Data Flow
`PDF → ingestion → reading-order recovery → concept extraction → [PERTURBATION INJECTOR (ρ, δ)] → hierarchy induction → graph validation → KCE confidence → [CONFIDENCE-GATED REPAIR (τ)] → metrics (PAA / APA / ACP)`. The two bracketed blocks are the new research components; everything else is the reused instrument.

## J. Algorithms
- **Perturbation:** adjacent-swap reading-order noise at rate ρ; Bernoulli(δ) heading-number dropout.
- **Repair:** for node v with `conf(v) < τ`, set `parent(v) ←` nearest u with `order(u)<order(v)` and `level(u)<level(v)`, else root.
- **Metrics:** APA over gold ancestor sets; point-biserial correlation of confidence vs. correctness.

## K. Implementation (what code exists / is new)
- **Reused (already implemented & tested):** `pdf_ingestion`, `structural_parser`, `concept_extractor`, `hierarchy_builder`, `graph_validator`, `kce_scorer`, `pipeline`.
- **New (implemented this iteration):** `robustness_eval.py` — perturbation harness, APA/ACP metrics, confidence–error analysis, gated-repair ablation. Runs on the 11-paper corpus, 5 seeds, no LLM.

## L. Dataset
Corpus of **11 real research-paper PDFs** (RAG literature: RAPTOR, GraphRAG, TreeRAG, Self-RAG, HiChunk, RETRO, Atlas, T-RAG, REALM, HAT-RAG, RAG). Digital, multi-column academic PDFs — the target document type. Gold structure derived from section numbering. *(Extension: add a public multi-page document-QA set — e.g., MMLongBench-Doc — for the Major.)*

## M. Baselines
- **No-perturbation** clean hierarchy (upper bound).
- **Perturbed, no repair** (shows fragility).
- Attachment-strategy baselines already implemented: Sequential, Lexical, PRE, Numbering.

## N. Metrics
PAA (immediate parent), **APA** (full ancestor context — the retrieval-relevant metric), ACP (context precision), confidence–error point-biserial r. Structural guarantees (0 cycles / 0 orphans / depth ≤ 8) tracked throughout.

## O. Experiments (done — real numbers)
Mean over 11 docs × 5 seeds:
- **Reading-order noise ρ (0→0.5):** PAA 97.7 → 97.3, APA 96.8 → 96.4 — *near-flat*.
- **Heading-number dropout δ (0→0.5):** PAA 97.7 → 92.7, **APA 96.8 → 89.2**, ACP 97.3 → 91.2 — *clear monotonic degradation*.
- **Confidence–error (δ=0.3):** correct-node conf 0.611 vs wrong-node 0.548 (gap +0.062); r = +0.16 (n=980).
- **Gated repair (δ=0.3, τ=0.60):** APA 92.8 → 93.3, ACP 94.1 → 94.4.

## P. Ablation Study
- ρ-only vs δ-only vs both → isolates which error source matters (result: δ dominates; numbering confers ρ-robustness).
- Repair on/off, and τ sweep → shows the confidence gate's contribution and its operating point.

## Q. Expected vs. Verified Results
**Verified (measured):** H1 holds for heading-number loss; H1 is *weak* for reading-order (an interesting positive: numbering-primary design is inherently reading-order-robust). H2 holds but is *modest* (r≈0.16). H3 holds but the gain is *small* with the current heuristic KCE. **No values are fabricated.**

## R. Limitations (state honestly)
1. Metrics are structural proxies for retrieval context, not end-task QA (no LLM in the loop yet).
2. Current KCE confidence is a hand-crafted heuristic → only a weak error detector (r≈0.16), so gated repair gains are small. *(This is the strongest motivation for the Major.)*
3. Gold structure is numbering-derived; unnumbered-heading correctness is not directly graded.
4. Corpus is 11 academic PDFs; non-academic layouts untested.

## S. Major Extension (this Minor → Objective 1)
This Minor establishes (i) that structure quality measurably affects retrieval context and (ii) that confidence can guard it — but weakly. The Major then: **O1 =** this robustness framework (foundation); **O2 =** a *learned* structural-confidence estimator (fixes the weak r≈0.16); **O3 =** faithful, bounding-box-grounded attribution built on validated structure; **O4 =** robustness under *heterogeneous / federated* document sources; **O5 =** end-to-end QA evaluation with an LLM in the loop.

---

## How to defend this TOMORROW (honesty script)
- **Frame:** "My Minor is a robustness study of structure-aware RAG — a question the whole sub-field leaves open — plus an inference-time mitigation."
- **Show real numbers:** the degradation curves and the repair ablation above (all measured, reproducible via `robustness_eval.py`).
- **Own the limits:** "The mitigation is currently modest because my confidence signal is heuristic; that is precisely what my Major's learned-confidence objective addresses." This turns a weakness into the research trajectory — exactly what a guide wants to hear.
