## 4. The 5 Core Upgraded Research Novelties

### 4.1 Novelty 1: Dual-Tier Heterogeneous Knowledge Forest (HKF)

To resolve the structural-semantic schism, SHIA-RAG 2.0 introduces the **Dual-Tier Heterogeneous Knowledge Forest**:

```mermaid
graph TD
    subgraph "Tier 1: Syntactic Document Tree (Preserves Narrative Flow)"
        Doc["Document Node"] --> Sec1["Section: Network Protocols"]
        Doc --> Sec2["Section: Routing Algorithms"]
        Sec1 --> Blk1["Paragraph: Link-State Fundamentals"]
        Sec2 --> Blk2["Paragraph: OSPF Convergence Dynamics"]
    end
    
    subgraph "Tier 2: Semantic Concept Forest (Preserves Domain Taxonomy)"
        RootNode["Concept: Routing Protocols"] -->|"HIERARCHICAL: IS_A"| Concept1["Concept: Link-State Protocol"]
        Concept1 -->|"HIERARCHICAL: IS_A"| Concept2["Concept: OSPF"]
        Concept2 -->|"SEMANTIC: CAUSES"| Concept3["Concept: Fast Convergence"]
        Concept2 -.->|"SEMANTIC: COMPARED_TO"| Concept4["Concept: RIP"]
    end
    
    Blk1 -.->|"E_proj: Grounding Anchor"| Concept1
    Blk2 -.->|"E_proj: Grounding Anchor"| Concept2
```

* **Tier 1 (Syntactic Document Tree):** Preserves explicit document structure ($Doc \to Section \to Block$) and sequential reading order. Chunks in Tier 1 retain exact character offsets and surrounding paragraphs.
* **Tier 2 (Semantic Concept Forest):** Organizes deduplicated, canonical `KnowledgeNode` concepts into an acyclic taxonomy (`HIERARCHICAL` edges: $IS\_A, PART\_OF$) overlaid with cross-cutting `SEMANTIC` relations ($USES, CAUSES, COMPARED\_TO$).
* **Bidirectional Projection Anchors ($E_{\text{proj}}$):** Every concept node in Tier 2 maintains strict cryptographic provenance pointers ($E_{\text{proj}}$) to the exact text blocks in Tier 1 from which its claims were extracted, ensuring 100% verifiable citation attribution.

---

### 4.2 Novelty 2: Precedence-Constrained DAG Knapsack Optimizer (DC-Knapsack)

We mathematically formulate context assembly under an LLM token budget $B$ as a **Precedence-Constrained 0-1 Knapsack Problem on Directed Acyclic Graphs**:

$$\max_{\mathbf{x}} \sum_{i \in \mathcal{V}} U_i \cdot x_i$$
$$\text{subject to} \quad \sum_{i \in \mathcal{V}} C_i \cdot x_i \le B, \quad x_i \in \{0, 1\} \quad \forall i \in \mathcal{V}$$
$$\text{Precedence Invariant:} \quad x_i \le x_p \quad \forall p \in \text{Parents}(i) \quad \text{where } (p, i) \in E_{\text{hierarchical}}$$

Where:
* $U_i = \text{Rel}(v_i, q) \cdot \text{Conf}(v_i)$ represents the query-relevance utility of concept $v_i$.
* $C_i = \text{Tokens}(v_i)$ is the exact BPE token footprint of the concept's definition and evidence.
* $B$ is the maximum token capacity allocated for retrieval context.
* **The Precedence Invariant guarantees that no child concept can be included in the context unless its prerequisite parent concept is also selected**, mathematically eliminating orphan hallucinations.

---

### 4.3 Novelty 3: Self-Reflective Query & Depth Traversal Router (SRDR)

Rather than traversing the graph uniformly, the **Self-Reflective Router (SRDR)** inspects query intent, linguistic complexity, and entity distribution to dynamically configure traversal parameters across **4 operational modes**:

```
                                  USER QUERY
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │  Self-Reflective Query Router │
                      └───────────────┬───────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
    Mode 1: Thematic             Mode 2: Factual             Mode 3: Multi-Hop
     Sensemaking                  Needle Search               Comparative
     Depth: Tau = 1               Depth: Tau = 1              Depth: Tau = 3+
    Focus: Roots + Summaries     Focus: Leaf + Direct Parent Focus: Dual Tree + Cross-Links
```

| Mode | Trigger Conditions | Traversal Strategy | Target Nodes |
| :--- | :--- | :--- | :--- |
| **Mode 1: Thematic Sensemaking** | High generality keywords (*"overview"*, *"summarize all"*, *"trends"*); low entity specificity. | **Top-Down Breadth-First:** Retrieves forest roots and high-level parent abstractions. | Roots + Tier 1 Section Headers |
| **Mode 2: Factual Needle** | Single entity lookup; exact property inquiry (*"What is the default timeout of X?"*). | **Leaf-to-Root Local:** Vector search finds leaf node; traverses directly to its single parent for definition. | Leaf Concept + Immediate Parent |
| **Mode 3: Multi-Hop Comparative** | Multiple distinct entities; comparative tokens (*"compare X vs Y"*, *"how does A affect B?"*). | **Bidirectional Dual-Subtree:** Traverses up from both entities and traces connecting `SEMANTIC` cross-links. | Sibling Trees + Intersection Edges |
| **Mode 4: Parametric / Direct** | Syntactic logic, general reasoning, or conversational chit-chat requiring no external grounding. | **Zero Retrieval:** Directly forwards query to LLM parametric memory, bypassing index. | None (Zero API / DB Latency) |

---

### 4.4 Novelty 4: Bayesian Thompson-Sampling Graph Evolution with Laplacian Smoothing

To eliminate the runaway positive feedback drift of naive multipliers, SHIA-RAG 2.0 treats retrieval path selection as an **Online Multi-Armed Bandit**:

1. **Edge State Representation:** Every hierarchical and semantic edge $e = (u, v)$ maintains a conjugate Beta prior:
   $$w_e \sim \text{Beta}(\alpha_e, \beta_e)$$
   Where $\alpha_e \ge 1$ represents accumulated retrieval success tokens, and $\beta_e \ge 1$ represents retrieval failure tokens.
2. **Thompson Sampling Traversal:** During graph traversal, the effective weight $\tilde{w}_e$ is stochastically sampled from its posterior distribution:
   $$\tilde{w}_e \sim \text{Beta}(\alpha_e, \beta_e)$$
   This inherently balances **exploitation** of proven reasoning paths with **exploration** of under-utilized conceptual links.
3. **Attribution Feedback Update:** Given downstream verification reward $R \in \{0, 1\}$:
   $$\alpha_e \leftarrow \alpha_e + R, \qquad \beta_e \leftarrow \beta_e + (1 - R) \quad \forall e \in \text{TraversalPath}$$
4. **Graph Laplacian Regularization:** To prevent popular nodes from starving neighboring subtrees, we apply periodic Laplacian diffusion:
   $$\mathbf{W}^{(t+1)} = (1 - \gamma) \mathbf{A} + \gamma (\mathbf{I} - \mathcal{L}_{\text{sym}}) \mathbf{A}$$
   Where $\mathcal{L}_{\text{sym}} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$ is the normalized graph Laplacian, propagating learned utility smoothly to adjacent conceptual nodes.

---

### 4.5 Novelty 5: Evidence-Calibrated Multi-Metric Evaluation Framework (SHEF 2.0)

SHEF 2.0 merges **HiChunk’s HiCBench** (testing chunk granularity under evidence sparsity) with multi-hop reasoning datasets (**HotpotQA, QuALITY**) to evaluate retrieval quality across four orthogonal axes:
* **Structural Hierarchy Fidelity:** Parent Assignment Accuracy (PAA) and Tree Edit Distance (TED) against human gold-standard taxonomies.
* **Context Token Efficiency (Context Density):** Ratio of verified factual evidence tokens to total tokens consumed in the LLM prompt window.
* **Hop Reasoning Precision:** Exact path-matching accuracy along multi-hop relation chains.
* **Index Stability & Regret:** Cumulative regret bounds proving graph convergence without semantic degeneration.
