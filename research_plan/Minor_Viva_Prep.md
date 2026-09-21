# Minor Project — Viva Preparation (Robustness of Structure-Aware RAG)

**How to open (30-second pitch):**
"Structure-aware RAG methods — RAPTOR, TreeRAG, HiChunk, BookRAG — all build a document hierarchy and assume it is correct. Real parsers make mistakes. My project is the first to *quantify* how much retrieval context degrades when the recovered structure is wrong, to test whether a per-node confidence signal can *detect* those errors, and to add a confidence-gated *repair* step at inference time. I built a controlled perturbation framework on 11 real papers and measured all three."

**One-line contribution:** A robustness characterisation of structure-aware retrieval + a confidence-gated inference-time repair, with measured degradation curves, a confidence–error correlation, and a repair ablation.

---

## A. Basic questions

**Q. What is your project in one sentence?**
A study of how structural-parsing errors affect the retrieval context in structure-aware RAG, and a confidence-gated method to detect and repair them.

**Q. What is RAG?**
Retrieval-Augmented Generation: an LLM answers using text retrieved from an external store at query time, instead of parametric memory alone.

**Q. What is "structure-aware" RAG?**
Instead of flat fixed-size chunks, it represents the document as a hierarchy (title → section → subsection) and retrieves along that structure, attaching ancestor context.

**Q. What is the retrieval context / ancestor path?**
When a hierarchical retriever returns a node, it also attaches the node's ancestors (its parent section, grandparent, …) so the LLM has the surrounding context. If the parent is wrong, that attached context is wrong.

**Q. What are PAA, APA, ACP?**
PAA = is the immediate parent correct. APA = is the *full* root→node ancestor chain correct (the retrieval context). ACP = of the ancestors we attached, what fraction are correct.

---

## B. Method questions

**Q. Where do the structural errors come from in reality?**
Two main sources: (1) multi-column reading-order mistakes, and (2) heading mis-detection — a heading's section number is missed or garbled. I model these as ρ (reading-order swaps) and δ (heading-number dropout).

**Q. How do you inject the perturbations?**
ρ: randomly swap adjacent concepts' reading-order rank at rate ρ. δ: each numbered heading loses its number with probability δ, which forces the semantic fallback (PRE) instead of exact numbering.

**Q. What is confidence-gated repair?**
Each node has a KCE confidence score. If a node's confidence is below τ (=0.60), I distrust its predicted parent and reattach it to the nearest earlier shallower heading — a safe structural prior — without re-parsing the PDF.

**Q. Why no LLM in your metrics?**
Deliberate: I isolate the *structural* effect. APA measures exactly the context a hierarchical retriever would attach, so I can attribute degradation to structure, not to LLM variance. Adding an LLM-in-the-loop QA evaluation is my first Major objective.

---

## C. Results questions (know these numbers)

**Q. What did you find?** (11 papers × 5 seeds)
- Reading-order noise ρ (0→50%): APA ~96.8 → 96.4 — **almost flat**.
- Heading-number loss δ (0→50%): PAA 97.7 → 92.7; **APA 96.8 → 89.2**; ACP 97.3 → 91.2 — **clear monotonic drop**.
- Confidence–error: correct nodes conf 0.61 vs wrong 0.55 (gap +0.06); point-biserial r = +0.16.
- Gated repair (δ=0.3, τ=0.60): APA 92.8 → 93.3, ACP 94.1 → 94.4.

**Q. Why is reading-order noise almost harmless but number-loss isn't?**
Because attachment is *numbering-primary*: the section number itself encodes the parent (3.1→3), independent of reading order. So the hierarchy inherits robustness to reading-order errors — a positive finding. The vulnerability is exactly where numbering is unavailable and the system must guess semantically.

**Q. Your repair gain is small (0.5 APA). Isn't that weak?**
Yes, and I state it honestly. The gain is bounded by the *confidence signal quality*: my KCE is a hand-crafted heuristic and only weakly predicts errors (r≈0.16). The finding is "confidence is a real but weak error detector; a simple repair recovers part of the loss." Making the confidence signal *learned* is the first objective of my Major — this experiment is precisely what motivates it.

---

## D. Research-gap / novelty questions

**Q. What exactly is novel — isn't this just testing existing code?**
The novelty is the *research contribution*, not the parser: (1) the first robustness characterisation of structure-aware retrieval under controlled structural error; (2) the retrieval-context metric APA; (3) a confidence-gated inference-time repair. No prior structure-aware RAG paper reports degradation under structural error or a mitigation for it.

**Q. How is this different from LAD-RAG / BookRAG / TreeRAG?**
They *build* structure and evaluate on clean inputs. I take structure as *given but noisy* and study its fragility and repair. Orthogonal and complementary — my framework could stress-test any of them.

**Q. Isn't robustness testing just engineering?**
It's a research question with a hypothesis (H1–H3) and measured evidence, and it produces a design principle (numbering-primary attachment is robustness-conferring) plus a mechanism (confidence-gated repair). That is empirical systems research.

---

## E. Critical / hard questions (be ready)

**Q. Your gold is numbering-derived — isn't PAA circular?**
For the *clean* case, partly — which is why the study is about the *perturbed* case: once numbering is dropped (δ), the system must recover structure without the gold signal, so the metric is no longer trivial. The interesting numbers are all at δ>0.

**Q. Only 11 documents — is that enough?**
It's enough to show consistent, monotonic trends across 5 seeds (55 runs per level), but I state corpus size as a limitation and scaling to a large public set (e.g., MMLongBench-Doc) is planned in the Major.

**Q. APA is a proxy, not real QA accuracy.**
Correct, and stated as limitation #1. APA measures the exact object a hierarchical retriever attaches, so it is a faithful *structural* proxy; end-task QA with an LLM is Major Objective 5.

**Q. Why should anyone care if the drop is only ~8 points?**
Because it is silent — the LLM receives wrong context with no error flag, and in a multi-hop answer one wrong ancestor can flip the conclusion. Quantifying and flagging it is the point.

**Q. What would you do with more time?**
Learned confidence (to lift r well above 0.16), LLM-in-the-loop QA, block-level (pre-heading) perturbation, and larger corpora — all mapped into the Major.

---

## F. Defense posture
- Lead with the **research question**, not the parser.
- Show the **figure** (degradation curves) and the **repair ablation**.
- **Own the two honest weaknesses** (proxy metric; weak confidence signal) and immediately tie each to a Major objective. Examiners reward a candidate who knows their own limits and has a plan.
- Never claim QA/end-task numbers you don't have.
