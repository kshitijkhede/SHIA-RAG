# SHIA-RAG — Literature Notes (14 papers, read in full)

**Purpose:** Working notes for the Major Project-I literature survey. Every paper below was read in full from its PDF; numbers are quoted from the papers themselves. Each entry ends with a plain-spoken *relevance to SHIA-RAG* so the survey can position the project honestly.
**SHIA-RAG in one line:** a structure-aware, dual-tier RAG system — a document tree (title→section→subsection) derived *training-free from PDF layout geometry* and validated, a semantic concept forest on top, precedence-constrained retrieval (never a fact without its parent context), adaptive-depth query routing, and online self-evolution.
**Last updated:** 18 September 2026

---

## How the 14 papers group

| # | Paper | Year / venue | Group | Role for SHIA-RAG |
| :-- | :-- | :-- | :-- | :-- |
| 1 | RAG (Lewis et al.) | 2020, NeurIPS | Foundational | Names the paradigm; flat-chunk baseline |
| 2 | REALM | 2020, ICML | Foundational | Learned-retriever ancestor; background |
| 3 | RETRO | 2022, ICML | Foundational | Frozen/training-free retriever; background |
| 4 | Atlas | 2023, JMLR | Foundational | SOTA retrieval-augmented LM; strong baseline |
| 5 | RAPTOR | 2024, ICLR | Tree / hierarchy | Canonical tree-RAG; **must-beat baseline** |
| 6 | TreeRAG | 2025, ACL Findings | Tree / hierarchy | Closest methodological cousin |
| 7 | Ψ-RAG (Hierarchical Abstract Tree) | 2026, ICML (preprint) | Tree / hierarchy | Current tree-RAG SOTA; **hard baseline** |
| 8 | T-RAG | 2024, arXiv (case study) | Tree / hierarchy | Background / motivation |
| 9 | GraphRAG | 2024, MS (preprint) | Graph / chunk | Related; global-sensemaking baseline |
| 10 | HiChunk | 2025, arXiv (Tencent) | Graph / chunk | **Primary chunking baseline** + benchmark |
| 11 | Self-RAG | 2023, ICLR | Graph / chunk | Background for routing + self-evolution |
| 12 | DTE-RAG | 2026, IEEE ISBDAS | Recent competitor | Closest "dual-tree" competitor |
| 13 | T-Retriever | 2026, AAAI (arXiv) | Recent competitor | Closest "tree-based retrieval" competitor |
| 14 | Agentic-RAG | 2026, IEEE ATiGB | Recent competitor | Closest "structure-aware + agentic" competitor |

**Bottom line up front:** all 14 are legitimately relevant to a RAG project, but they are not equal. The must-cite, must-benchmark competitors are **RAPTOR, Ψ-RAG, HiChunk, DTE-RAG, T-Retriever** (plus **Atlas** on knowledge-intensive QA). **TreeRAG** is the closest methodological cousin. **RAG, REALM, RETRO, GraphRAG, Self-RAG, T-RAG** are foundational background or motivation — cite them, don't necessarily run them head-to-head.

---

## Group A — Foundational retrieval-augmented LMs

These four define the RAG lineage. Common gap: **all retrieve flat, fixed-granularity chunks** (100 words in RAG, 288 word-pieces in REALM, 64 tokens in RETRO, <200-word sections in Atlas). None builds a document tree, enforces parent context, routes by depth, or self-evolves — which is exactly SHIA-RAG's white space.

### 1. RAG — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
Lewis et al., FAIR/UCL/NYU, **NeurIPS 2020**.
Couples a BART generator (parametric memory) with a DPR-retrieved dense index of 21M Wikipedia chunks (non-parametric memory); the retrieved document is a latent variable that is marginalized. Two variants: RAG-Sequence (one doc per output) and RAG-Token (a different doc per token). Trained end-to-end, tuning the query encoder + generator while the document index stays frozen. SOTA open-domain QA at the time (NQ 44.5 EM, TriviaQA 56.8/68.0); more factual, specific and diverse generation than BART; knowledge updatable by hot-swapping the index.
*Relevance:* Core reference — SHIA-RAG's name descends from here, and RAG is the canonical **flat-chunk baseline** SHIA-RAG improves structurally. Only loosely related to the tree focus (no hierarchy, no precedence). Its "swap the index to update knowledge" argument is the seed of SHIA-RAG's self-evolution framing.

### 2. REALM — Retrieval-Augmented Language Model Pre-Training
Guu et al., Google Research, **ICML 2020**.
First to *pre-train* a neural retriever using the masked-LM signal, backpropagating through retrieval over millions of documents (top-K MIPS approximation, asynchronous index refresh every ~500 steps). Key inductive biases: salient-span masking, a null document, ICT warm-start. Beats prior open-QA systems by 4–16% absolute (NQ-Open 40.4 EM) while ~30× smaller than T5-11B; extractive only.
*Relevance:* Background / conceptual ancestor of the "learned retriever" and the interpretability/updatability arguments. SHIA-RAG's training-free tree is a deliberate contrast to REALM's expensive learned retriever. Not a structure baseline.

### 3. RETRO — Improving Language Models by Retrieving from Trillions of Tokens
Borgeaud et al., DeepMind, **ICML 2022**.
Retrieval-enhanced Transformer that retrieves, per 64-token chunk, k nearest neighbours from a database up to **2 trillion tokens**, using a **frozen BERT retriever** (so neighbours are precomputed offline — no retriever training or index refresh). Integrates neighbours via chunked cross-attention (cost linear in retrieved data). A 7.5B RETRO matches GPT-3/Jurassic-1 (178B) on the Pile; introduces a leakage-aware evaluation.
*Relevance:* Background, but an important ally: the strongest demonstration that a **training-free / frozen retriever** works at scale — supports SHIA-RAG's training-free philosophy. Its leakage-aware evaluation is a tool SHIA-RAG can adopt to prove gains are real. Retrieval is flat 64-token chunks — the opposite of hierarchical.

### 4. Atlas — Few-shot Learning with Retrieval Augmented Language Models
Izacard, Lewis et al., Meta AI, **JMLR 2023**.
Contriever retriever + T5 with Fusion-in-Decoder (each doc encoded independently, decoder cross-attends over the concatenation). Central finding: **jointly pre-training retriever + LM is what unlocks few-shot** knowledge tasks. Studies four LM-supervised retriever losses (PDist chosen) and cheap index-refresh (query-side fine-tuning). Splits Wikipedia by **section** into <200-word passages — a small amount of structure. NQ 45.1 with 64 examples (beats PaLM-540B with 50× fewer params); MMLU 47.9% 5-shot beats GPT-3-175B; updatable via index swap.
*Relevance:* The most relevant of the four and a **strong direct baseline** on knowledge-intensive QA. Its updatability is exactly what SHIA-RAG's self-evolution targets. Crucially, Atlas *uses section structure only as a chunking heuristic* — no retrieval-time hierarchy or precedence — so it is the ideal contrast for "structure left on the table."

---

## Group B — Tree / hierarchy RAG (the core lineage)

This is SHIA-RAG's home territory. Natural progression: RAPTOR → T-RAG → TreeRAG → Ψ-RAG. **None of them derives the tree from real document layout geometry, and none is fully training-free at construction** — that is SHIA-RAG's distinctive claim.

### 5. RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval
Sarthi et al., Stanford, **ICLR 2024**.
Builds a tree **bottom-up**: ~100-token leaf chunks → soft clustering (GMM + UMAP + BIC) → LLM summary per cluster → re-embed → repeat. Query time uses "collapsed tree" (flatten all layers, score every node, fill a token budget), which beats layer-by-layer traversal because it picks the right granularity per question. SOTA on QASPER (55.7 F1) and QuALITY (82.6%, +20% absolute on the hard subset).
*Relevance:* **Primary must-beat baseline** and the canonical tree-RAG reference. Direct foil to SHIA-RAG: RAPTOR's tree is induced from embedding clusters and needs an LLM summary per node; SHIA-RAG's is derived top-down from layout geometry, training-free, and faithful to the document's real structure. RAPTOR's "collapsed tree picks the right granularity" is the motivation for adaptive-depth routing. RAPTOR has no parent-context guarantee.

### 6. TreeRAG — Unleashing the Power of Hierarchical Storage for Enhanced Knowledge Retrieval in Long Documents
Tao et al., SCUT / Pazhou Lab, **ACL 2025 Findings**.
Two parts: (1) **Tree-Chunking** — an LLM (GPT-4o) categorizes the document top-down into a numbered title→section tree, and each chunk is embedded with its **ancestor titles prepended** (fixes pronoun ambiguity, raises similarity). (2) **Bidirectional Traversal Retrieval** — an LLM "step-back" intent check; for summarization/enumeration intents it walks root→leaves and leaf→roots to recover sibling/child chunks distant in vector space. Beats Late-Chunking, Meta-Chunking, RAPTOR on the Dragonball benchmark (Recall/Precision/EIR).
*Relevance:* **Closest methodological cousin.** It independently validates SHIA-RAG's two pillars — a title→section→subsection tree and retrieval that pulls ancestor context (their prefix + traversal ≈ precedence-constrained retrieval), and its intent "step-back" gate prefigures adaptive-depth routing. Differentiator: TreeRAG builds the tree with an expensive LLM and no geometric validation; SHIA-RAG does it training-free from layout, deterministically. Use TreeRAG's **EIR** metric.

### 7. Ψ-RAG — Hierarchical Abstract Tree for Cross-Document RAG
Zhao & Yang, HKUST (Guangzhou), **ICML 2026 (forward-dated preprint — flag as cutting-edge, not independently reproduced).**
Replaces RAPTOR's clustering with agglomerative-style **merging / leaf-collapse / abstract-collapse** (n−1 steps), proven (Dasgupta cost) to avoid k-means' "uniform effect" on skewed corpora; nodes get summary or keyword abstracts. Retrieval is a multi-granular **agentic** loop (retrieve-vs-answer decision, query reorganization, dense-tree + BM25 hybrid). Reports +25.9% avg F1 and +23.7% Recall@5 over RAPTOR; first tree-RAG to beat Graph-RAG (and HippoRAG 2 by 7.4%). Shows RAPTOR+Q3R underperforming plain DPR on 5/6 QA sets — evidence of classic tree-RAG's corpus-scale collapse.
*Relevance:* **Current tree-RAG SOTA and a hard baseline.** Its critique of cluster-built trees is the sharpest published argument *for* SHIA-RAG's geometry-derived tree. Its agentic query reorganization is a strong instance of adaptive-depth routing to compare against. SHIA-RAG must show it can match Ψ-RAG's multi-hop numbers; its differentiators remain geometry-grounding, precedence, and self-evolution.

### 8. T-RAG — Lessons From the LLM Trenches
Fatehkia, Lucas, Chawla, QCRI/HBKU, **arXiv 2024 (applied case study).**
On-premise QA over a private governance manual: RAG (section-header chunks) + a **QLoRA-fine-tuned Llama-2-7B** + a hand-built **entities tree** (org chart). spaCy NER detects entities in the query and injects their tree location as text. Doubles correctness on entity/hierarchy questions; Needle-in-a-Haystack 64.1% vs RAG 33.3%. Introduces the Correct-Verbose metric.
*Relevance:* **Background / motivation**, not a competing algorithm. Its tree is a manually curated *entity* ontology used for context injection, and it relies on fine-tuning — mechanistically far from SHIA-RAG. Useful for: the finding that hierarchical context injected with chunks reduces hallucination, and the **Needle-in-a-Haystack** + **Correct-Verbose** evaluation ideas.

---

## Group C — Graph & chunk-structure RAG

### 9. GraphRAG — From Local to Global: A GraphRAG Approach to Query-Focused Summarization
Edge, Trinh et al., Microsoft Research, **2024 (preprint, under review).**
LLM extracts an entity knowledge graph from chunks → Leiden hierarchical community detection → bottom-up community summaries → map-reduce over community summaries at query time (each summary gives a partial answer + helpfulness score; top ones combined). Beats vector RAG on global "sensemaking" questions (comprehensiveness/diversity win rates 72–83%); root-level summaries need 97% fewer query tokens. Indexing is expensive (LLM entity extraction; 281 min for a 1M-token set).
*Relevance:* Related work and a **baseline for global/thematic queries.** Its community hierarchy ≈ SHIA-RAG's concept forest, and its C0–C3 level selection prefigures adaptive-depth routing. Contrast: GraphRAG's structure is an LLM-induced entity graph (expensive, co-occurrence-driven), not the document's own layout; it targets sensemaking, not precedence-constrained fact retrieval.

### 10. HiChunk — Evaluating and Enhancing RAG with Hierarchical Chunking
Lu, Chen et al., Tencent Youtu Lab, **arXiv 2025.**
Contributes both a benchmark (**HiCBench**, with multi-level chunk-point annotations and *evidence-dense* T1/T2 QA — fixing the "evidence sparsity" that makes existing benchmarks unable to tell good chunkers apart) and a framework: a fine-tuned Qwen3-4B turns a document into a multi-level chunk tree (chunk points + hierarchy levels as text generation), and an **Auto-Merge** retriever merges query-ranked leaf chunks upward into their parent once enough siblings are hit. Big chunk-point F1 gains over Semantic Chunker / LumberChunker; evidence-recall and Fact-Coverage gains on evidence-dense QA.
*Relevance:* **The single most directly competing paper — primary baseline.** Same problem (multi-level document structuring + adaptive-granularity retrieval); **Auto-Merge is a close analogue of precedence-constrained retrieval**, and its max-level study supports adaptive-depth routing. Reuse **HiCBench + chunk-point F1** as an evaluation harness, and the evidence-density insight. Clean differentiator: HiChunk's tree comes from a *fine-tuned LLM on text*; SHIA-RAG's is *training-free from layout geometry*.

### 11. Self-RAG — Learning to Retrieve, Generate, and Critique through Self-Reflection
Asai et al., UW / AI2 / IBM, **ICLR 2024.**
Trains one LM to emit "reflection tokens" — Retrieve (yes/no), ISREL (relevant?), ISSUP (supported?), ISUSE (utility) — via offline critic distillation from GPT-4, so at inference the model does **on-demand adaptive retrieval** and segment-level self-critique with test-time-tunable weighting. Beats non-proprietary baselines on 6 tasks and surpasses ChatGPT on several (incl. citation precision).
*Relevance:* **Background for pillars 4–5** (adaptive-depth routing and self-evolution), not a structure baseline. Self-RAG's learned "retrieve or not, how much" and self-critique are templates for a routing/quality gate. Contrast: it buys adaptivity by *fine-tuning the generator* (GPT-4-dependent) — the opposite of SHIA-RAG's training-free structural approach, which should realize adaptive depth *structurally / at inference*.

---

## Group D — Recent competitors (2026) — read these most carefully

These three are the closest current work and the sharpest positioning targets.

### 12. DTE-RAG — Dual-Tree Structure Contextual Enhanced Knowledge Retrieval (Multi-Hop)
Fan & Zhang, USST, **IEEE ISBDAS 2026.**
Three parts: context-enhanced chunking (each chunk stores text + LLM cross-chunk context + generated related questions; retrieval score fuses query-to-text and query-to-context similarity), **Dual-Tree indexing** (a semantic-similarity tree, RAPTOR-style, + a lexical/entity-relatedness tree), and dual-channel retrieval fused by MMR. HotpotQA + 2WikiMultiHopQA: ~54–55% EM / ~64% F1, topping Naïve RAG, GraphRAG, HippoRAG, RAPTOR, SiReRAG — but highest latency (~157 TPQ). *(Honesty flag: abstract says 54.10/63.90, table says 54.98/64.73 — internal inconsistency.)*
*Relevance:* **Closest "dual-tree" competitor — positioning must be sharp.** Its two trees are *retrieval-signal* trees (similarity vs. relatedness) over enhanced chunks, **not a validated document structure**. SHIA-RAG's Tier-1 is a geometry-derived, validated title→section→subsection tree it doesn't have; SHIA-RAG adds precedence guarantees, adaptive-depth routing, and self-evolution (DTE-RAG is fully static/offline). Report HotpotQA + 2Wiki to be directly comparable. Also cite **SiReRAG** (its strongest baseline, another "similar + related" method).

### 13. T-Retriever — Tree-based Hierarchical RAG for Textual Graphs
Wei, Qin et al., Renmin University, **AAAI 2026 (arXiv Jan 2026).**
Reformulates attributed-graph retrieval as tree retrieval via a **learning-free** encoding tree built by minimizing **Semantic-Structural Entropy (S²-Entropy)** (structural entropy + a KDE-based semantic term) with adaptive compression (Partition / Prune / Regulate) instead of fixed layer quotas. Beats G-Retriever, GRAG, RAPTOR, ArchRAG on SceneGraphs/WebQSP/BookGraphs (gains grow with graph size, +6.63% on 77k-node graphs); cuts LLM context tokens up to 84% ; no parameter updates.
*Relevance:* **Closest "tree-based retrieval" competitor**, and its strengths overlap SHIA-RAG's best claims (training-free hierarchy; structure + semantics beats single-signal). But it needs a **pre-existing attributed graph** and hours of offline entropy optimization; SHIA-RAG builds its tree directly from PDF layout with no graph-construction step. T-Retriever retrieves **uniformly across levels** — SHIA-RAG's adaptive-depth routing is a direct improvement — and has no precedence guarantee or self-evolution.

### 14. Agentic-RAG — Agentic RAG for Structure-Aware Long-Document Reasoning
Lam, Tran, Do, PTIT / PTHub (Vietnam), **IEEE ATiGB 2026.**
Adds an LLM agentic control layer (Planner → Executor → LLM reranker → groundedness gate → relevance gate → CoT query rewriter, ≤3 iterations) on top of an *unmodified* MinerU-parsed, VLM-captioned LightRAG graph index. On HotpotQA context precision 0.168→0.246; on ASQA precision 0.50→0.65, recall 0.28→0.48 (RAGAS metrics). Authors concede **sparse multi-hop bridge-evidence chaining remains the bottleneck**.
*Relevance:* **Closest "structure-aware + agentic" competitor.** Its MinerU layout parsing is adjacent to SHIA-RAG's layout tree, and its planner "local-then-read" resembles adaptive-depth routing. But structure is only *consulted* by a planner, not the retrieval substrate; it has **no parent-context guarantee** — and its own admitted failure (missing bridge/parent evidence on HotpotQA) is exactly what SHIA-RAG's precedence-constrained retrieval is built to prevent. Strong motivating citation. Its RAGAS context-precision/recall are metrics SHIA-RAG can adopt.

---

## Synthesis — where SHIA-RAG sits

**The gap every one of the 14 leaves open.** Not a single paper derives its primary hierarchy from *validated PDF layout geometry*; not one enforces *precedence / parent-context constraints at retrieval time*; and not one combines *adaptive-depth routing* with *online self-evolution*. The tree-RAG systems either cluster embeddings (RAPTOR, Ψ-RAG, DTE-RAG's semantic tree), parse with an expensive LLM (TreeRAG, HiChunk), optimize over a pre-existing graph (T-Retriever, GraphRAG), or curate by hand (T-RAG). That intersection — geometry-grounded document tree + concept forest + precedence-constrained, depth-adaptive, self-evolving retrieval — is SHIA-RAG's contribution.

**Recommended baselines to actually run** (in priority order): RAPTOR and Ψ-RAG (tree-RAG SOTA), HiChunk (chunking + its HiCBench benchmark), DTE-RAG and T-Retriever (2026 competitors), Atlas (knowledge-intensive QA), and Flat/vanilla RAG as the floor. TreeRAG is the closest cousin to compare *methodology* against.

**Recommended datasets / metrics** (chosen so results are directly comparable to the competitors): HotpotQA and 2WikiMultiHopQA (multi-hop; shared with DTE-RAG, Ψ-RAG, Agentic-RAG); HiCBench with chunk-point F1 (chunking quality, shared with HiChunk); QuALITY / QASPER (long-doc, shared with RAPTOR); RAGAS context-precision/recall (shared with Agentic-RAG); plus SHIA-RAG's own hierarchy metrics (Parent-Assignment Accuracy, orphan rate, tree-edit distance) and TreeRAG's EIR.

**Honesty flags carried from the readings:** Ψ-RAG and the three 2026 competitors are forward-dated preprints / early IEEE papers, not independently reproduced; DTE-RAG's headline numbers are internally inconsistent between its abstract and table. Treat their reported numbers as targets to verify, not settled facts.
