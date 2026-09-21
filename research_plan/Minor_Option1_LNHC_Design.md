# Minor Project — Full Research Design (Option 1)

**Working title:** *Layout-Native Hierarchical Chunking (LNHC): A Training-Free Alternative to Fine-Tuned Document Structuring for RAG*

**Base paper being extended:** Lu, Chen, Qiao, Sun. *HiChunk: Evaluating and Enhancing Retrieval-Augmented Generation with Hierarchical Chunking*, arXiv:2509.11552 (ACL 2026). Code: github.com/TencentCloudADP/hichunk · Benchmark: HuggingFace Youtu-RAG/HiCBench.

---

## 1. What the base paper (HiChunk) does — accurately
- **Problem it raises:** standard RAG chunking is evaluated poorly because most QA benchmarks are *evidence-sparse* (answer sits in 1–2 sentences), so chunking quality barely affects the score. HiChunk argues we need *evidence-dense* evaluation and better multi-level chunking.
- **Method:** a **fine-tuned LLM** (instruction-tuned on Gov-report, Qasper, Wiki-727) that converts two subtasks — *identify chunk points* and *assign hierarchy level* — into text generation, producing multi-level chunk points; then fixed-size balancing; then an **Auto-Merge** retrieval step that merges sibling child chunks up to their parent under three conditions (≥2 siblings retrieved, an adaptive length threshold θ\*, and a token-budget check).
- **Benchmark (HiCBench):** built from OHRBench documents, with **manually annotated multi-level chunk points** and three QA types — T0 evidence-sparse, T1 single-chunk-dense, T2 multi-chunk-dense (evidence forced to ≥10% of context).
- **Metrics:** chunking-point F1 at each level (F1\_L1, F1\_L2, F1\_all); downstream RAG QA quality and a fact-coverage measure; time cost of chunking.
- **Baselines:** fixed-size, semantic, LumberChunker, PIC, Late Chunking.

## 2. The exact gap we attack
1. **Fine-tuning dependence.** HiChunk's structuring needs a trained LLM + training corpora + GPU, and is domain/model-specific.
2. **Layout-blindness.** It operates on raw sentence tokens and **never uses the document's physical layout** — font size, bold, section numbering ("3.1"), indentation, reading order — even though these are the very cues humans use to see structure, and they are free to extract.
> Neither HiChunk nor its baselines test whether **native layout alone** can recover the same hierarchical chunking that a fine-tuned model learns.

## 3. Problem Statement
To determine whether a **training-free, layout-native** method can recover multi-level document chunking as well as a fine-tuned LLM, and to quantify its effect on retrieval and answer quality under the HiCBench evidence-dense evaluation.

## 4. Research Questions
- **RQ1:** How close is training-free layout-native chunking to fine-tuned HiChunk on chunk-point F1 (per level and overall)?
- **RQ2:** Does layout-native chunking, fed into the same Auto-Merge retrieval, match HiChunk on downstream QA — especially on the evidence-dense T1/T2 tasks?
- **RQ3:** Which layout signals matter most, and where does layout fail (motivating a hybrid)?

## 5. Hypotheses
- **H1:** Native layout recovers most L1/L2 chunk points, so LNHC's chunk-F1 is within a small margin of fine-tuned HiChunk and clearly beats fixed-size/semantic chunking.
- **H2:** On evidence-dense QA (T1/T2), LNHC + Auto-Merge is competitive with HiChunk at **near-zero structuring cost and no training**.
- **H3:** A **confidence-gated hybrid** (use the LLM only where layout is ambiguous) closes most of any remaining gap while calling the LLM on a small fraction of boundaries.

## 6. Novelty (precise, defensible)
> HiChunk assumes hierarchical structure must be **learned** by a fine-tuned LLM from token text. We propose that the structure is **already encoded in the document's layout** and can be recovered **without any training**. LNHC is a training-free chunk-point + level detector from layout signals, evaluated head-to-head with HiChunk on its own benchmark and retrieval algorithm. Contribution = (a) the layout-native chunker, (b) a controlled comparison showing how much of HiChunk's benefit is attributable to *learning* vs to *hierarchy itself*, and (c) a confidence-gated hybrid that trades a few LLM calls for accuracy. This is a scientific claim with an ablation, not a mere combination.

## 7. Proposed Method (LNHC)
1. **Layout extraction** (reuses your engine): per line, get text, font size, bold, bounding box, reading-order rank (your column-gutter algorithm).
2. **Chunk-point detection (training-free):** a line is a chunk point if it is a heading — matches a section-number regex `^\d+(\.\d+)*`, or font ≥ 1.15× body, or bold + short + title-case, or preceded by large vertical whitespace. Each rule contributes to a **boundary score**.
3. **Hierarchy-level assignment:** numbered → level = count of number components (3→L1, 3.1→L2, 3.1.1→L3); unnumbered → level from font-size rank. Produces the same multi-level chunk-point format HiChunk outputs.
4. **Granularity balancing:** apply HiChunk's fixed-size balancing so chunk sizes are comparable (fair comparison).
5. **Retrieval:** feed LNHC chunks into **HiChunk's own Auto-Merge** unchanged, so any QA difference is attributable to *chunking*, not retrieval.
6. **Confidence-gated hybrid (contribution 3):** where the boundary score is in an uncertain band, call the LLM only for those lines; everywhere else use layout. Report the LLM-call fraction.

## 8. Architecture / Data Flow
`PDF → layout extraction (your engine) → LNHC chunk-point + level detector → fixed-size balancing → hierarchical chunks → HiChunk Auto-Merge retrieval → LLM answer → HiCBench metrics (chunk-F1, QA, cost)`. Baselines swap only the chunker block.

## 9. Implementation (what you build vs reuse)
- **Reuse (yours):** layout extraction, reading-order, numbering/font heading detection.
- **New code:** `lnhc_chunker.py` (boundary scoring + level assignment → HiChunk chunk-point format); `hybrid_gate.py` (uncertain-band LLM fallback); an adapter to HiChunk's Auto-Merge; an evaluation runner over HiCBench.
- **External:** clone HiChunk repo (for Auto-Merge + their eval harness + the fine-tuned baseline), download HiCBench.

## 10. Dataset
- **HiCBench** (Youtu-RAG/HiCBench) — the base paper's own benchmark; use its manual multi-level chunk-point annotations and its T0/T1/T2 QA. Evaluating on the *same* benchmark is what makes the comparison credible.

## 11. Baselines
Fixed-size, semantic, LumberChunker, **HiChunk (fine-tuned — the paper's method)**, and **LNHC (ours)** + **LNHC-Hybrid (ours)**. All share the same Auto-Merge retriever and the same answer LLM.

## 12. Metrics
- **Chunking:** F1\_L1, F1\_L2, F1\_all vs HiCBench gold chunk points.
- **QA:** the benchmark's answer-quality metric + fact-coverage, reported separately for T0/T1/T2.
- **Cost:** structuring time and **LLM calls per document** (LNHC ≈ 0; hybrid = small; HiChunk = full).

## 13. Experiments
E1 chunk-F1: LNHC vs HiChunk vs fixed/semantic. E2 QA on T1/T2 (evidence-dense) with identical retriever + LLM. E3 cost/latency + LLM-call fraction. E4 hybrid sweep (uncertainty band width vs accuracy vs LLM calls).

## 14. Ablation
Remove each layout signal (numbering / font / bold / whitespace) → which drives chunk-F1. Layout-only vs hybrid. τ (uncertainty band) sweep.

## 15. Expected Results (hypothesised — NOT fabricated)
We expect LNHC to beat fixed-size/semantic clearly, to come within a modest margin of fine-tuned HiChunk on chunk-F1 and T1/T2 QA, at ~zero structuring cost, and the hybrid to close most of the remaining gap using few LLM calls. **Actual numbers to be measured; none are invented here.**

## 16. Limitations (state honestly)
- Layout signals are weaker on documents without clear numbering/formatting (scanned, single-column narrative) → where the hybrid matters.
- HiCBench is English/academic-leaning.
- Requires running an LLM for the QA metric and for the HiChunk baseline (needs your GPU).

## 17. How this becomes the Major (Objective map)
- **O1 (this Minor):** training-free layout-native chunking vs fine-tuned, on HiCBench.
- **O2:** the confidence-gated hybrid, learned uncertainty for *when* to call the LLM.
- **O3:** structure-grounded **faithful** attribution (the SIGIR-2025 gap) on top of the chunks.
- **O4:** robustness of the whole pipeline to parsing noise (the earlier study becomes a stress-test objective).
- **O5:** cross-lingual / heterogeneous / federated documents.

---

## 18. Honest feasibility & timeline (given your 1-day deadline)
This is a **multi-day** build (clone HiChunk, download HiCBench, run the fine-tuned baseline + LLM QA on your GPU, then LNHC + hybrid). It is **not** finishable by tomorrow with real HiCBench numbers.

Realistic staging:
- **Tomorrow:** present the **design** above (problem from HiChunk → the layout-blindness gap → LNHC novelty → experiment plan). I can also prototype `lnhc_chunker.py` and demo it producing multi-level chunk points on your existing PDFs — a concrete artifact, though not the HiCBench score.
- **Days 2–4:** you run (on your GPU) the HiCBench eval: fixed-size + LNHC first (fast), then the HiChunk fine-tuned baseline; get chunk-F1 + T1/T2 QA.
- **Week 2:** hybrid + ablation + write-up.

I will write the actual `lnhc_chunker.py`, the HiCBench evaluation runner, and setup instructions so you can execute on your GPU; I can run the CPU parts (the chunker prototype, chunk-F1 once HiCBench text is available) here.

**Sources:** HiChunk (arXiv:2509.11552, ACL 2026); HiCBench (HuggingFace Youtu-RAG/HiCBench); HiChunk code (github.com/TencentCloudADP/hichunk).
