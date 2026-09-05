## 9. Evaluation Framework (SHEF 2.0) & Experimental Methodology

### 9.1 Benchmark Corpus Suite

To validate performance under both evidence-sparse and multi-hop conditions, SHEF 2.0 tests across four diverse benchmark corpora:

| Dataset | Type / Domain | Number of QA Pairs | Primary Evaluation Focus |
| :--- | :--- | :--- | :--- |
| **HiCBench** (Tencent 2025) | Evidence-Dense Technical Docs | 1,200 | Multi-level chunking quality under varying evidence density |
| **HotpotQA** (Yang et al.) | Multi-Hop Wikipedia QA | 7,405 (Dev distractor) | Multi-hop reasoning chains across disparate documents |
| **QuALITY** (Pang et al.) | Long-Document Story / Book QA | 2,524 | Narrative understanding over 5,000+ token context |
| **CS-Textbook Gold Corpus** | OS (Silberschatz) & Networks (Kurose) | 500 (Annotated) | Parent Assignment Accuracy (PAA) vs. Human Gold Standard Taxonomy |

---

### 9.2 Complete Metric Definitions Across 4 Performance Pillars

#### Pillar A: Retrieval & Reasoning Quality
1. **Hop Precision & Recall (HPR):**
   $$\text{HopRecall} = \frac{|\text{Retrieved Grounding Path} \cap \text{Gold Reasoning Path}|}{|\text{Gold Reasoning Path}|}$$
2. **Ancestor Chain Recall (ACR):**
   $$\text{ACR} = \frac{|\text{Retrieved Ancestors} \cap \text{Gold Ancestors}|}{|\text{Gold Ancestors}|}$$
3. **Context Density Score (CDS):** Measures the ratio of useful evidence tokens to total tokens fed to the LLM:
   $$\text{CDS} = \frac{\sum_{s \in \text{EvidenceSpans}} \text{Tokens}(s)}{\text{Total Context Tokens Selected}} \in [0, 1]$$

#### Pillar B: Hierarchy Induction Quality
4. **Parent Assignment Accuracy (PAA):**
   $$\text{PAA} = \frac{\sum_{i=1}^N \mathbb{1}[\text{Parent}_{\text{pred}}(v_i) = \text{Parent}_{\text{gold}}(v_i)]}{N}$$
5. **Normalized Tree Edit Distance (NTED):**
   $$\text{NTED}(\mathcal{T}_{\text{pred}}, \mathcal{T}_{\text{gold}}) = 1.0 - \frac{\text{TED}(\mathcal{T}_{\text{pred}}, \mathcal{T}_{\text{gold}})}{\max(|\mathcal{T}_{\text{pred}}|, |\mathcal{T}_{\text{gold}}|)}$$
6. **Orphan Rate (OR):** Fraction of non-root nodes that fail integration:
   $$\text{OR} = \frac{|\{v \in \mathcal{V} : \text{deg}_{\text{in}}(v) = 0 \land v \notin \text{DomainRoots}\}|}{|\mathcal{V}|}$$
7. **Forest Density (FD):** Reconciled ratio of connections to concepts:
   $$FD = \frac{|\mathcal{E}|}{|\mathcal{V}|}$$

#### Pillar C: Generation & Verification Quality
8. **Citation Faithfulness Score (CFS):** Fraction of generated claims directly supported by referenced $E_{\text{proj}}$ spans.
9. **ROUGE-L / BERTScore:** Standard lexical and semantic generation metrics against gold reference answers.

#### Pillar D: Graph Dynamics & Online Stability
10. **Cumulative Regret ($R_T$):** Quantifies bandit convergence efficiency:
    $$R_T = \sum_{t=1}^T \left( U^*_t - U_{\text{selected}, t} \right)$$

---

### 9.3 Statistical Significance Protocols & Hypothesis Testing

* **Paired Student’s $t$-test & Wilcoxon Signed-Rank Test:** Executed on per-query retrieval metrics (Hop Recall, PAA, BERTScore) across all baselines.
* **Holm-Bonferroni Correction:** Applied across all multiple pairwise comparisons against the 5 primary baselines (Flat RAG, RAPTOR, TreeRAG, GraphRAG, HiChunk) to maintain Family-Wise Error Rate $\alpha \le 0.01$.

---

### 9.4 Comprehensive Ablation Study Protocol

1. **Ablation 1 (No Precedence Constraints):** Replace DC-Knapsack with standard greedy top-$k$ knapsack. *Hypothesis: Significant drop in Context Coherence and spike in ungrounded hallucinations.*
2. **Ablation 2 (No Dual-Tier HKF):** Discard Tier 1 syntactic document trees and retain only Tier 2 concepts. *Hypothesis: Substantial degradation on sequential narrative and reading-order QA.*
3. **Ablation 3 (No Online Thompson Evolution):** Freeze graph weights post-construction. *Hypothesis: Stagnant retrieval performance failing to adapt to iterative query patterns.*
4. **Ablation 4 (No Adaptive Depth Router):** Fix traversal depth universally at $\tau = 2$. *Hypothesis: Severe token bloat on simple queries and recall failure on multi-hop comparisons.*

---

### 9.5 Empirical Test Suite Execution (77 Tests) & Live TreeRAG PDF Benchmark

#### 9.5.1 Comprehensive Unit & Integration Test Matrix

To guarantee mathematical and behavioral correctness, the codebase includes an automated test harness consisting of **77 test cases distributed across 13 specialized test modules**. All 77 tests pass with a 100% success rate in under 1 second:

| Test Module | Target Architecture Component | Key Invariants & Behaviors Verified | Status |
| :--- | :--- | :--- | :--- |
| `test_schemas_and_invariants.py` | Layer 0: Schemas & Invariants | Regex validation, DAG acyclicity, PRE weight normalization ($\sum w_i = 1$) | **7/7 PASSED** |
| `test_cycle_detection.py` | Layer 5: `HierarchyValidator` | Upward DFS cycle detection, self-loops, strict tree depth bound ($d \le 8$) | **7/7 PASSED** |
| `test_kce_scorer.py` | Layer 4: `KnowledgeConfidenceEngine` | Topological propagation, baseline prior $\text{Conf}_0$, leaf damping | **2/2 PASSED** |
| `test_structural_parser.py` | Layer 2: `StructuralDocumentParser` | 2D geometric sorting, reading order monotonicity, empty block filtering | **3/3 PASSED** |
| `test_pdf_ingestion.py` | Layer 1 & 3: `PDFLoader` & Concept Extractor | Real-world PDF layout parsing, bounding box extraction, live hierarchy derivation | **5/5 PASSED** |
| `test_dc_knapsack.py` | Layer 6: `DAGKnapsackOptimizer` | Branch-and-bound exactness, zero-orphan precedence, token budget bounds | **8/8 PASSED** |
| `test_srdr_router.py` | Layer 6: `SelfReflectiveDepthRouter` | Word boundary entity detection, query complexity $\Psi$, traversal depth $\tau$ | **10/10 PASSED** |
| `test_citation_verifier.py` | Layer 7: `ClaimAttributionVerifier` | Claim splitting, token overlap entailment, citation rewards $R \in [0, 1]$ | **4/4 PASSED** |
| `test_thompson_evolution.py` | Layer 8: `ThompsonEvolutionEngine` | Beta posterior updating, reward clipping $[-1, +1]$, graph Laplacian smoothing | **5/5 PASSED** |
| `test_shef_evaluator.py` | Evaluation: `SHEFEvaluator` | Context Density (CDS), Parent Assignment Accuracy (PAA), Forest Density ($FD$) | **5/5 PASSED** |
| `test_end_to_end_pipeline.py` | System Orchestrator (`pipeline.py`) | Multi-hop comparative routing, online belief updates, precedence context | **6/6 PASSED** |
| `test_baselines.py` | Comparative Baselines | Flat RAG chunking and similarity retrieval benchmarks | **2/2 PASSED** |
| `test_audited_fixes.py` | Historical Flaw Regressions | Regression tests for all 24 historical and audited fixes | **13/13 PASSED** |
| **TOTAL** | **Full System Verification** | **Zero failures, zero regressions, 100% invariant compliance** | **77/77 PASSED (0.91s)** |

#### 9.5.2 Live Document Ingestion Benchmark on TreeRAG (ACL 2025)

The live document ingestion pipeline was experimentally benchmarked using the actual 14-page research publication:  
*`TreeRAG Unleashing the Power of Hierarchical Storage for Enhanced.pdf` (1.56 MB)*.

```
============================================================================
  LIVE PDF INGESTION BENCHMARK RESULTS (TreeRAG ACL 2025)
============================================================================
  Document Node ID:            DOC-TreeRAG Unleash
  Total Pages Extracted:       14
  Layout Blocks Identified:    121 blocks (reading-order monotonic)
  Document Sections Mapped:    14 major sections (Abstract, Intro, Tree Induction, 
                               Precedence Traversal, Experiments, Ablations, etc.)
  Concepts Derived:            14 hierarchical concept units
  Forest Topology Induced:     6 independent domain trees (Root max depth = 4)
  Invariant Verification:      PASSED (0 cycles, 0 self-loops, 0 orphaned concepts)
  
  Query Benchmark:
  "What is the TreeRAG architecture and how does it organize hierarchical documents?"
  --> Detected Mode:           MODE_1_THEMATIC (Depth limit tau = 1)
  --> Knapsack Token Usage:    165 / 2048 tokens
  --> Context Density Score:   0.882
  --> Claim Citation Reward:   R = 1.000 (100% verified against extracted blocks)
============================================================================
```
