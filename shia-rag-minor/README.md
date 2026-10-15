# SHIA-RAG (Minor Project) v2 — Layout-Aware Hierarchical RAG

Reference implementation for the IT897 minor-project report *"Layout-Aware
Hierarchical Knowledge Representation for Retrieval-Augmented Generation."*
It turns a multi-column PDF into a validated hierarchy of concept nodes while
preserving physical reading order — **and now supports semantic retrieval
and question answering** over the induced hierarchy.

## What's New in v2

| Feature | v1 | v2 |
|---------|----|----|
| Semantic similarity | TF-IDF only | **Sentence-transformers** (with TF-IDF fallback) |
| Body text | Headings only | **Paragraphs chunked under headings** |
| Retrieval | ❌ None | **Hierarchy-aware semantic retrieval** |
| Persistence | ❌ Ephemeral | **JSON document store** |
| Query UI | ❌ None | **Interactive question box in web UI** |
| Version | 1.0.0 | **2.0.0** |

## Ten modules (`shia_minor/`)

| # | Module | Class | Responsibility |
|---|--------|-------|----------------|
| 1 | `pdf_ingestion.py`   | `PDFIngestionService` | Extract text blocks with bounding boxes and font size |
| 2 | `structural_parser.py` | `StructuralParser`   | Column-gutter reading-order recovery + heading classification |
| 3 | `concept_extractor.py` | `ConceptExtractor`   | Heading blocks → concept nodes with bbox evidence |
| 4 | `hierarchy_builder.py` | `HierarchyBuilder`   | CPG++ candidate generation + PRE ranking + cross-links |
| 5 | `graph_validator.py`   | `GraphValidator`     | Upward-DFS cycle / depth / single-parent / doc-isolation checks |
| 6 | `kce_scorer.py`        | `KCEScorer`          | Kahn topological confidence propagation (δ = 0.85) |
| 7 | `embedder.py`          | `Embedder`           | **NEW** Dense semantic embeddings (sentence-transformers / TF-IDF) |
| 8 | `chunker.py`           | `Chunker`            | **NEW** Assign body paragraphs to heading concepts |
| 9 | `retriever.py`         | `HierarchyRetriever` | **NEW** Hierarchy-aware semantic retrieval with ancestor context |
| 10 | `store.py`            | `DocumentStore`      | **NEW** JSON persistence for processed documents |

`schemas.py` holds the Pydantic Tier 1 / Tier 2 data model; `pipeline.py` wires
all ten modules together.

## Install
```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` is recommended for best retrieval quality.
> If not installed, the system automatically falls back to TF-IDF embeddings.

## Run
```bash
# Web UI (recommended) — upload PDF, see hierarchy, ask questions
python app.py
# then open http://127.0.0.1:5000

# CLI: ingest one PDF and print the induced hierarchy
python run_demo.py "path/to/paper.pdf"

# Reproduce the report's corpus results
python evaluate.py "path/to/folder_of_pdfs"

# Unit tests
pytest tests/ -v
```

## Architecture (v2)

```
PDF → PDFIngestionService → StructuralParser (reading order + headings)
    → ConceptExtractor → HierarchyBuilder (CPG++/PRE + cross-links)
    → GraphValidator (inside builder) → Chunker (body text)
    → KCEScorer → Embedder + HierarchyRetriever → DocumentStore
                                    ↑
                              User Query → Top-k Results with Ancestor Context
```

## Method (matches the report)
- **Reading order (Algorithm 1):** split blocks at the gutter (W/2), pull out
  full-width spanning blocks to top/bottom, sort each column top-to-bottom with a
  vertical tolerance τ_y = 20 pt.
- **Hierarchy:** section numbering is the primary attachment signal; the Parent
  Ranking Engine `Score = 0.40·S_struct + 0.40·S_sem + 0.20·S_level` (S_sem is
  TF-IDF cosine) is the fallback for unnumbered headings.
- **Invariants:** acyclic, single-parent, depth ≤ 8, per-document; unattached
  concepts are promoted to roots (no orphans).
- **Confidence:** local typographic score propagated root→leaf with damping 0.85.
- **Retrieval (NEW):** dense embeddings over concept name + body text; query
  results enriched with full ancestor-chain context for hierarchical RAG.

## Reproduced results (11-paper corpus)
- Reading-order monotonicity: **81.6% → 98.4%**.
- PAA over 197 numbered sections: Sequential 22.8%, Lexical 43.2%, **PRE 52.8%,
  Numbering 100%**.
- 0 cycles, 0 orphans, depth ≤ 8 on every document.
