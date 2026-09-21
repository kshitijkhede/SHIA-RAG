# Minor Project — Presentation Content (Robustness of Structure-Aware RAG)

Use your existing NITK template. ~11 slides. Figure `robustness_figure.png` goes on the Results slide.

---

**Slide 1 — Title**
Layout title: *How Robust is Structure-Aware Retrieval? Quantifying and Mitigating Document-Structure Errors in RAG* · Minor Project (IT897) · your name/roll · guide · NITK.

**Slide 2 — Table of Contents**
Introduction · Literature Survey · Problem & Objectives · Methodology · Experimental Setup · Results & Analysis · Discussion · Conclusion & Future Work · References.

**Slide 3 — Introduction**
- RAG answers from retrieved document text; how the document is represented matters.
- Modern systems are *structure-aware*: they build a title→section→subsection hierarchy and attach ancestor context.
- **Every such system assumes the recovered structure is correct.**
- Real PDFs produce imperfect structure (multi-column mis-reads, missed heading numbers).
- *Say:* "My project asks what happens when that assumption breaks — and how to guard against it."

**Slide 4 — Literature Survey (overview)**
- Flat RAG (RAG, REALM, RETRO, Atlas) — no structure.
- Structure-aware RAG (RAPTOR, TreeRAG, HiChunk, BookRAG, LAD-RAG) — build hierarchy, evaluate on clean structure.
- **Gap:** none study degradation under *imperfect* structure, none localise errors, none repair at inference time.

**Slide 5 — Problem & Objectives**
- **Problem (one line):** the sensitivity of structure-aware retrieval context to parser errors, and its detection/repair, is unquantified.
- **RQ1** degradation? **RQ2** predictable from confidence? **RQ3** repairable?
- Objectives: metrics (PAA/APA/ACP) · perturbation harness (ρ, δ) · degradation study · confidence–error analysis · confidence-gated repair + ablation.

**Slide 6 — Methodology (pipeline + new blocks)**
- Reused instrument: ingest → reading-order → concept extraction → numbering/PRE hierarchy → validation → KCE confidence.
- **New:** perturbation injector (ρ, δ) and confidence-gated repair (τ).
- Metrics: PAA (parent), APA (ancestor context = what a retriever attaches), ACP (context precision).
- *Figure:* the architecture with the two new blocks highlighted (I can generate this).

**Slide 7 — Experimental Setup**
- Corpus: 11 real research-paper PDFs; gold structure from section numbering.
- 5 random seeds per level; perturbation levels 0–50%.
- Python; reuses the implemented `shia_minor` pipeline; new `robustness_eval.py`. No LLM (isolates the structural effect).

**Slide 8 — Results & Analysis**  *(put `robustness_figure.png` here)*
- Reading-order noise ρ: APA ~96.8 → 96.4 — **near-flat** (numbering-primary confers robustness).
- Heading-number loss δ: PAA 97.7 → 92.7; **APA 96.8 → 89.2**; ACP 97.3 → 91.2 — **monotonic drop**.
- Confidence–error: correct 0.61 vs wrong 0.55 (r = +0.16) — a real but weak detector.
- Gated repair (δ=0.3): APA 92.8 → 93.3.

**Slide 9 — Discussion (7 points)**
1. Numbering-primary attachment is inherently robust to reading-order errors (design insight).
2. The true vulnerability is heading-number loss — the semantic-fallback regime.
3. Errors degrade the *ancestor context* the LLM would see — silently.
4. KCE confidence predicts errors weakly (r≈0.16) but usefully.
5. Confidence-gated repair recovers part of the loss with no re-parse.
6. Limitation: APA is a structural proxy, not end-task QA.
7. Limitation: heuristic confidence caps the repair gain → motivates learned confidence.

**Slide 10 — Conclusion & Future Work**
- First robustness characterisation of structure-aware retrieval + a confidence-gated repair, validated on 11 papers.
- Future: learned structural confidence; LLM-in-the-loop QA; block-level perturbation; larger corpora; heterogeneous/federated documents.
- *Say:* "This Minor becomes Objective 1 of my Major, which turns the weak heuristic confidence into a learned signal and adds end-task evaluation."

**Slide 11 — References + Thank You**
RAPTOR (ICLR 2024), TreeRAG (ACL 2025), HiChunk (2025/26), BookRAG (2025), LAD-RAG (2025), "Correctness is not Faithfulness" (SIGIR 2025); TF–IDF (Salton), Kahn (1962).

---

**Talking-point discipline:** lead with the research question; show the figure; state the two honest limitations and tie each to a Major objective; never quote QA numbers you don't have.
