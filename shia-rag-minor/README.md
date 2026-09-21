# SHIA-RAG (Minor Project) — Layout-Aware Hierarchical Knowledge Representation

Reference implementation for the IT897 minor-project report *"Layout-Aware
Hierarchical Knowledge Representation for Retrieval-Augmented Generation."*
It turns a multi-column PDF into a validated hierarchy of concept nodes while
preserving physical reading order.

## Six modules (`shia_minor/`)

| # | Module | Class | Responsibility |
|---|--------|-------|----------------|
| 1 | `pdf_ingestion.py`   | `PDFIngestionService` | Extract text blocks with bounding boxes and font size |
| 2 | `structural_parser.py` | `StructuralParser`   | Column-gutter reading-order recovery + heading classification |
| 3 | `concept_extractor.py` | `ConceptExtractor`   | Heading blocks → concept nodes with bbox evidence |
| 4 | `hierarchy_builder.py` | `HierarchyBuilder`   | CPG++ candidate generation + PRE ranking + cross-links |
| 5 | `graph_validator.py`   | `GraphValidator`     | Upward-DFS cycle / depth / single-parent / doc-isolation checks |
| 6 | `kce_scorer.py`        | `KCEScorer`          | Kahn topological confidence propagation (δ = 0.85) |

`schemas.py` holds the Pydantic Tier 1 / Tier 2 data model; `pipeline.py` wires
the six modules together.

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
# ingest one PDF and print the induced hierarchy
python run_demo.py "path/to/paper.pdf"

# reproduce the report's corpus results
python evaluate.py "path/to/folder_of_pdfs"

# unit tests
pytest tests/ -v
```

## Method (matches the report)
- **Reading order (Algorithm 1):** split blocks at the gutter (W/2), pull out
  full-width spanning blocks to top/bottom, sort each column top-to-bottom with a
  vertical tolerance τ_y = 20 pt.
- **Hierarchy:** section numbering is the primary attachment signal; the Parent
  Ranking Engine `Score = 0.40·S_struct + 0.40·S_sem + 0.20·S_level` (S_sem is
  TF-IDF cosine) is the fallback for unnumbered headings. A dense sentence encoder
  can replace the TF-IDF term without other changes.
- **Invariants:** acyclic, single-parent, depth ≤ 8, per-document; unattached
  concepts are promoted to roots (no orphans).
- **Confidence:** local typographic score propagated root→leaf with damping 0.85.

## Reproduced results (11-paper corpus)
- Reading-order monotonicity: **81.6% → 98.4%**.
- PAA over 197 numbered sections: Sequential 22.8%, Lexical 43.2%, **PRE 52.8%,
  Numbering 100%**.
- 0 cycles, 0 orphans, depth ≤ 8 on every document.
