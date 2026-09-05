# SHIA-RAG 2.0
## Semantic Hierarchy Induction Architecture for Retrieval-Augmented Generation

> A Dual-Tier Heterogeneous Knowledge Forest for structured, grounded, and evolving retrieval-augmented generation.

---

## Overview

SHIA-RAG replaces arbitrary text chunking with an evolving **Knowledge Forest** that:

1. **Tier 1 (Syntactic Document Tree):** Preserves verbatim document structure (Document → Section → Paragraph).
2. **Tier 2 (Semantic Concept Forest):** Induces acyclic concept taxonomies with typed cross-links.
3. **DC-Knapsack:** Guarantees no orphan concepts reach the LLM via precedence-constrained context selection.
4. **Thompson Sampling:** Continuously evolves the graph from retrieval feedback.

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker & Docker Compose

### 2. Start Databases

```bash
cd docker/
docker compose up -d
```

This brings up PostgreSQL (pgvector), Neo4j 5, Milvus 2.4, and Redis 7.2.

### 3. Install Dependencies

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Download spaCy model
python -m spacy download en_core_web_lg
```

### 4. Initialize Neo4j Schema

```bash
cypher-shell -u neo4j -p <password> < docker/init_db.cypher
```

### 5. Run Interactive Console or PDF Ingestion
```bash
# Automated 4-mode benchmark test
python run_demo.py

# Interactive chat console
python run_demo.py --interactive

# Ingest and query any real uploaded PDF
python run_demo.py --pdf "research_papers/TreeRAG Unleashing the Power of Hierarchical Storage for Enhanced.pdf" --interactive
```

### 6. Run Tests (78/78 Passing)
```bash
pytest tests/ -v
```

## Project Structure

```
shia-rag-core/
├── config/
│   ├── system_config.yaml         # Tunable parameters (weights, thresholds, ports)
│   └── relation_ontology.yaml     # Permitted edge predicates
├── src/
│   ├── layer0_data_model/         # Pydantic schemas & invariant validators
│   ├── layer1_ingestion/          # Multi-format document loading & OCR
│   ├── layer2_parsing/            # LayoutLMv3 segmentation & reading order
│   ├── layer3_extraction/         # Concept extraction & LSH deduplication
│   ├── layer4_relations/          # Relation mining & KCE confidence engine
│   ├── layer5_shia_core/          # CPG++, PRE, HV, FI, CLD pipeline
│   ├── layer6_retrieval/          # SRDR router, traversal, DC-Knapsack
│   ├── layer7_generation/         # Prompt assembly, LLM, citation verifier
│   └── layer8_operations/         # Thompson evolution & telemetry
├── evaluation/                    # SHEF 2.0 benchmarks
├── docker/                        # Containerization & DB init
├── tests/                         # Unit & integration tests
├── pyproject.toml                 # Dependencies & build config
└── README.md                      # This file
```

## Key Modules

| Module | Purpose |
|:---|:---|
| `hv_validator.py` | Cycle-free DAG invariant enforcement via upward DFS |
| `dc_knapsack.py` | Precedence-constrained context selection under token budgets |
| `srdr_router.py` | 4-mode query routing with integrated entity detection |
| `thompson_evolution.py` | Beta-Bernoulli bandit with Laplacian smoothing |
| `fi_integrator.py` | Forest node placement & root promotion |
| `cld_crosslinker.py` | Semantic cross-link discovery across subtrees |
| `kce_scorer.py` | Topological confidence propagation (fixes exponential decay) |

## Configuration

All tunable parameters are in `config/system_config.yaml`. Key settings:

- **PRE weights:** `w1=0.30, w2=0.25, w3=0.15, w4=0.20, w5=0.10` (must sum to 1.0)
- **Token budget:** `default_token_budget: 3072`
- **Max tree depth:** `max_tree_depth: 8`
- **Laplacian γ:** `laplacian_gamma: 0.05`

## Evaluation (SHEF 2.0)

Benchmarks across 4 datasets:

| Dataset | Focus |
|:---|:---|
| HiCBench (Tencent 2025) | Evidence density & chunk quality |
| HotpotQA | Multi-hop reasoning chains |
| QuALITY | Long-document narrative understanding |
| CS-Textbook Gold | Parent Assignment Accuracy vs human taxonomy |

## License

MIT

## References

See `SHIA_RAG_Complete_Report.md` for the full 12-chapter specification, literature review, and mathematical formulations.
