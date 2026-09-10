## 5. End-to-End System Architecture

### 5.1 The 9-Layer Unified Architecture Pipeline

The architecture is structured into 9 cohesive layers spanning three operational phases:

```mermaid
graph TD
    subgraph "Phase 1: Offline Ingestion & Forest Construction"
        L1["Layer 1: Input Ingestion & Multi-Modal Preprocessing<br/>PDF/DOCX extraction, OCR, Hash Deduplication"]
        L2["Layer 2: Structural Document Parsing<br/>LayoutLMv3, Reading Order, Tier 1 Tree Builder"]
        L3["Layer 3: Knowledge Extraction & Canonicalization<br/>spaCy NER, MinHash LSH Alias Resolution"]
        L4["Layer 4: Relation Management & KCE<br/>Triple Extraction, Ontology Typing, KCE Scorer"]
        L5["Layer 5: SHIA Core Hierarchy Induction<br/>CPG++ → PRE → HV → FI → CLD"]
        
        L1 --> L2 --> L3 --> L4 --> L5
        L5 --> KF[("Knowledge Forest Store<br/>PostgreSQL + Neo4j + Milvus")]
    end

    subgraph "Phase 2: Online Retrieval & Verified Generation"
        Query["User Query"] --> L6["Layer 6: Retrieval Engine & DC-Knapsack<br/>SRDR Router → Traversal → Precedence Knapsack"]
        KF <--> L6
        L6 --> L7["Layer 7: Context Assembly & Generation<br/>Hierarchical Prompting → LLM → Claim Attribution"]
        L7 --> Answer["Verified Response + Provenance Citations"]
    end

    subgraph "Phase 3: Continuous Operations & Feedback"
        Answer --> L8["Layer 8: Operations & Self-Evolution<br/>Thompson Sampling Edge Update + Laplacian Diffusion"]
        L8 -->|"Weight Adaptation"| KF
    end
```

---

### 5.2 Inter-Layer Data Contracts & Lifecycle Transitions

| Stage | Producer $\to$ Consumer | Input Schema | Output Schema | Validation Contract |
| :--- | :--- | :--- | :--- | :--- |
| **L1 $\to$ L2** | Layer 1 $\to$ Layer 2 | `RawDocumentPayload` | `NormalizedDocument` | UTF-8 encoded; MIME validated; MD5 hash verified. |
| **L2 $\to$ L3** | Layer 2 $\to$ Layer 3 | `NormalizedDocument` | `DocumentTree` (`Block[]`) | Strictly positive bounding boxes; monotonically increasing reading order. |
| **L3 $\to$ L4** | Layer 3 $\to$ Layer 4 | `DocumentTree` | `KnowledgeNode[]` | MinHash LSH deduplicated; canonical names unique per domain. |
| **L4 $\to$ L5** | Layer 4 $\to$ Layer 5 | `KnowledgeNode[]` | `KnowledgeEdge[]` | Relation typed (`HIERARCHICAL` vs `SEMANTIC`); confidence $\in [0, 1]$. |
| **L5 $\to$ Storage** | Layer 5 $\to$ DB Store | `KnowledgeNode[] + Edge[]` | `KnowledgeForest` | Hierarchical subgraphs must satisfy `is_acyclic_addition()`. |
| **L6 $\to$ L7** | Layer 6 $\to$ Layer 7 | `UserQuery` | `PackedContextPlan` | Cumulative tokens $\le B$; all parent prerequisites satisfied. |
| **L7 $\to$ L8** | Layer 7 $\to$ Layer 8 | `PackedContextPlan` | `AttributedAnswer` | Every factual claim mapped to verified `E_proj` source span. |
| **L8 $\to$ L5** | Layer 8 $\to$ Forest | `AttributedAnswer + Feedback`| `EdgeWeightDelta[]` | Beta posterior parameters $\alpha_e, \beta_e$ updated atomically. |

---

### 5.3 System-Wide Architectural Data Flow

```
RAW DOCUMENT (PDF / DOCX / TEXT / HTML)
    │
    ▼ Layer 1: Validate MIME, compute SHA-256, OCR scanned text (Tesseract)
NORMALIZED TEXT STREAM + METADATA
    │
    ▼ Layer 2: LayoutLMv3 visual layout segmentation, 2D reading order sort
[PAGE → SECTION → PARAGRAPH BLOCK] (Tier 1 Syntactic Document Tree)
    │
    ▼ Layer 3: spaCy dependency parsing + 8B LLM concept extraction
CANDIDATE KNOWLEDGE UNITS (Facts, Definitions, Properties)
    │
    ▼ Layer 3b: MinHash LSH (128 perms) + Cosine ANN (threshold >= 0.85)
CANONICAL KNOWLEDGE NODES (Deduplicated Concepts & Aliases)
    │
    ▼ Layer 4: Predicate classification, ontology matching, KCE confidence scoring
TYPED KNOWLEDGE EDGES (HIERARCHICAL vs SEMANTIC)
    │
    ▼ Layer 5: CPG++ (Top-5 ANN) → PRE (5-factor score) → HV (Cycle check) → FI → CLD
KNOWLEDGE FOREST (Acyclic Hierarchy + Semantic Cross-Links + E_proj Anchors)
    │
    ▼ Persisted in: PostgreSQL (Blocks) + Neo4j (Graph) + Milvus (Vectors)
INDEX READY FOR RETRIEVAL

USER QUERY
    │
    ▼ Layer 6: Self-Reflective Router (SRDR) classifies intent into Mode 1-4 (Depth tau)
    ▼ Layer 6: Asymmetric Embedding Search (bge-base-en-v1.5) retrieves seed nodes
    ▼ Layer 6: Adaptive Forest Traversal (Upward ancestors, downward children, cross-links)
    ▼ Layer 6: Precedence-Constrained DAG Knapsack (DC-Knapsack) optimizes tokens <= B
PACKED OPTIMAL CONTEXT PLAN (Topologically sorted concept definitions + evidence)
    │
    ▼ Layer 7: LLM generates answer with explicit [KN-XXXXXX] citations
    ▼ Layer 7: Claim Attribution Verifier validates claims against E_proj text spans
ATTRIBUTED ANSWER + PROVENANCE CITATIONS + ATTRIBUTION REWARD R in {0, 1}
    │
    ▼ Layer 8: Thompson Sampling updates Beta(alpha, beta) priors on traversed path
    ▼ Layer 8: Graph Laplacian smoothing propagates rewards to conceptual neighbors
UPDATED & SELF-EVOLVED KNOWLEDGE FOREST
```

---

### 5.4 Full-Stack Web Application & Real-Time Cytoscape Visualization Architecture

To transition SHIA-RAG from an offline research script to a production-grade enterprise platform, the system includes a high-performance **Full-Stack Web Application**:

```mermaid
graph TD
    subgraph "Frontend Client (Modern Responsive Single-Page App)"
        UI["Chat & Query Interface"]
        Graph["Interactive Cytoscape DAG Visualizer (Dagre Layout)"]
        Inspector["Live Reasoning & Verification Inspector"]
        DocManager["Document Library & Upload Manager"]
    end

    subgraph "FastAPI REST Server (src/api.py)"
        EP1["POST /api/query (Multi-Turn Chat History)"]
        EP2["POST /api/upload (Multi-Part File Ingestion)"]
        EP3["GET /api/forest (Cytoscape JSON Serialization)"]
        EP4["DELETE /api/document/{doc_id} (Subtree Pruning)"]
        EP5["POST /api/reset_chat (Conversational Reset)"]
    end

    subgraph "SHIA-RAG Core Orchestration Pipeline"
        Pipe["Pipeline Orchestrator (pipeline.py)"]
        Synthesizer["Synthesizer (synthesizer.py)"]
        Knapsack["DAG Knapsack (dc_knapsack.py)"]
    end

    UI --> EP1
    DocManager --> EP2
    Graph --> EP3
    DocManager --> EP4
    UI --> EP5

    EP1 --> Pipe
    EP2 --> Pipe
    EP3 --> Pipe
    EP4 --> Pipe
    Pipe --> Knapsack
    Pipe --> Synthesizer
```

1. **Interactive Cytoscape Knowledge Forest:**
   Renders the dual-tier graph dynamically in real time using Cytoscape.js with a hierarchical Dagre layout. Concept nodes are color-coded by abstraction depth (Forest Roots $\to$ Intermediate Concepts $\to$ Evidence Leaves). Directed arrows clearly delineate `HIERARCHICAL` parent-child containment from `SEMANTIC` cross-cutting edges.
2. **Real-Time Reasoning Inspector:**
   Displays live execution telemetry for every query:
   * **Router Mode:** Displays detected operational mode (e.g., `MODE_1_THEMATIC`, `MODE_2_FACTUAL`, `MODE_3_MULTIHOP`).
   * **Knapsack Token Packing:** Shows exact tokens utilized vs. hard budget ($C \le B$).
   * **Verification Score:** Reports empirical verification reward ($R \in [0, 1]$) based on token-overlap entailment against source PDF spans.
   * **Provable Citations:** Displays verified source bounding blocks, page numbers, and character offsets.
3. **Multi-Document Subtree Isolation & Lifecycle Management:**
   Users can upload multiple heterogeneous PDFs simultaneously. The UI provides a document selector to scope retrieval strictly to an individual document or across the entire library. A dedicated deletion endpoint (`DELETE /api/document/{doc_id}`) guarantees that removing a document cleanly purges all its Tier 1 syntactic blocks and Tier 2 semantic nodes, preventing cross-document contamination.
4. **Pluggable Model Acceleration:**
   The synthesis engine (`synthesizer.py`) features pluggable support for state-of-the-art hosted models:
   * **Google Gemini Flash (`gemini-2.5-flash` / `gemini-1.5-flash` via `google-genai` SDK):** High-speed streaming synthesis with sub-second latency.
   * **OpenAI API (`gpt-4o-mini` / `gpt-3.5-turbo`):** Standard external LLM synthesis.
   * **Deterministic Local Synthesizer:** Fully offline fallback ensuring that the system functions with 100% test passing even in air-gapped environments without external API keys.
