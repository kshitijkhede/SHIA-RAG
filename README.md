# SHIA-RAG 2.0: Project Workspace

> **Semantic Hierarchy Induction Architecture for Retrieval-Augmented Generation**  
> A Dual-Tier Heterogeneous Knowledge Forest (HKF) for verifiable, structured, and evolving LLM retrieval.

---

## 🌟 Where to Start Reading

If you want to understand this project, you only need **ONE** file:

* 📄 **[SHIA_RAG_Complete_Report.pdf](SHIA_RAG_Complete_Report.pdf)**: Complete Master Document (All 12 Chapters + Full Viva Defense Guide).
* 🌐 **[SHIA_RAG_Report.html](SHIA_RAG_Report.html)**: Interactive browser version with KaTeX mathematical formulas, tables, and Mermaid flowcharts.
* 📝 **[SHIA_RAG_Complete_Report.md](SHIA_RAG_Complete_Report.md)**: Master markdown source.

---

## 📁 Workspace Structure

```text
Rag/
├── SHIA_RAG_Complete_Report.pdf   # ⭐ THE MASTER REPORT (Everything in 1 file)
├── SHIA_RAG_Report.html           # ⭐ Interactive Web Report (Math + Diagrams)
├── SHIA_RAG_Complete_Report.md    # Master Markdown Source
├── README.md                      # This directory guide
│
├── shia-rag-core/                 # 💻 Core Python Package & Implementation
│   ├── src/                       # 9-Layer Architecture (Layer 0 to Layer 8)
│   ├── tests/                     # Automated Test Suite (78/78 Passing)
│   ├── config/                    # System & Ontology YAML Configurations
│   ├── docker/                    # Docker Compose (Postgres, Neo4j, Milvus, Redis)
│   ├── run_demo.py                # Interactive Console & Live PDF Runner
│   └── pyproject.toml             # Python Dependencies & Metadata
│
├── docs/                          # 📚 Documentation Sources & Build Scripts
│   ├── report_chapters/           # Modular Source Chapters (01 to 12)
│   ├── viva_guide/                # Standalone Copies of Viva/Interview Defense
│   └── scripts/                   # Compilation & HTML Render Scripts
│
├── research_papers/               # 📑 Referenced Academic Research Papers
│   ├── GraphRAG (Microsoft)
│   ├── RAPTOR
│   ├── TreeRAG, HiChunk, T-RAG, etc.
│
└── archive/                       # 🗄️ Early Scratch Notes & Raw Extracted Texts
    ├── Files Ka Detailed Analysis.pdf
    ├── Layer Explanation Plan.pdf
    ├── Layer Zero समझाना.pdf
    └── Raw Text Extraction Logs
```

---

## 🚀 Quick Commands

### Run Interactive Console or PDF Ingestion
```bash
cd shia-rag-core
# Interactive chat over the knowledge forest
python run_demo.py --interactive

# Ingest and query any real uploaded PDF
python run_demo.py --pdf "research_papers/TreeRAG Unleashing the Power of Hierarchical Storage for Enhanced.pdf" --interactive
```

### Run All Unit Tests
```bash
cd shia-rag-core
pytest tests/ -v
```

### Start Local Infrastructure (Neo4j, Milvus, Postgres, Redis)
```bash
cd shia-rag-core/docker
docker compose up -d
```

### Recompile Master Report
```bash
python3 docs/scripts/compile_master_report.py
python3 docs/scripts/convert_report_to_html.py
```
