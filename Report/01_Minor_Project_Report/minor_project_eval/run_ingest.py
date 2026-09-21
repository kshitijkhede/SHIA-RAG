import sys, time
from pathlib import Path

# Auto-detect project virtualenv if running in environment without required packages
try:
    import pydantic
except ImportError:
    for candidate in [
        Path(__file__).resolve().parents[3] / "shia-rag-core" / ".venv" / "bin" / "python",
        Path(__file__).resolve().parents[3] / "shia-rag-minor" / ".venv" / "bin" / "python",
    ]:
        if candidate.exists() and candidate != Path(sys.executable):
            import subprocess
            res = subprocess.run([str(candidate)] + sys.argv)
            sys.exit(res.returncode)
    raise

ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / "shia-rag-core"
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(CORE))

from src.layer1_ingestion.pdf_loader import PDFLoader
from src.layer3_extraction.concept_extractor import ConceptExtractor
from src.pipeline import SHIARAGPipeline

PDF = ROOT / "research_papers" / "TreeRAG Unleashing the Power of Hierarchical Storage for Enhanced.pdf"

t0 = time.time()
loader = PDFLoader(min_block_chars=5)
doc_node, blocks = loader.load_pdf(PDF)
t1 = time.time()
print(f"PAGES={doc_node.total_pages}")
print(f"BLOCKS={len(blocks)}")

# distinct sections detected
secs = {}
for b in blocks:
    secs[b.section_path] = secs.get(b.section_path, 0) + 1
print(f"DISTINCT_SECTIONS={len(secs)}")

ext = ConceptExtractor()
extracted = ext.extract_from_document(doc_node, blocks)
print(f"CONCEPTS_EXTRACTED={len(extracted)}")

pipe = SHIARAGPipeline()
for node, parents in extracted:
    pipe.add_concept_node(node, candidate_parents=parents)
pipe.run_confidence_propagation()
try:
    pipe.validate_invariants()
    inv = "PASS (0 cycles)"
except Exception as e:
    inv = f"FAIL: {e}"
stats = pipe.get_forest_statistics()
t2 = time.time()

print(f"NODES={stats['total_nodes']}")
print(f"HIER_EDGES={stats['hierarchical_edges']}")
print(f"SEM_EDGES={stats['semantic_edges']}")
print(f"ROOTS={stats['root_count']}")
print(f"MAX_DEPTH={stats['max_depth']}")
print(f"AVG_CONF={stats['avg_confidence']:.3f}")
print(f"INVARIANTS={inv}")
# orphan rate: non-root nodes with no parent
nonroot = [n for n in pipe.nodes.values() if n.parent_id is not None]
orph = [n for n in nonroot if n.parent_id not in pipe.nodes]
print(f"NONROOT={len(nonroot)} ORPHANS={len(orph)}")
print(f"INGEST_SEC={t1-t0:.2f}  TOTAL_SEC={t2-t0:.2f}")

# dump the induced tree (name -> parent name, depth) for building ground truth
print("=== INDUCED TREE ===")
for nid, n in sorted(pipe.nodes.items(), key=lambda kv: kv[1].depth):
    pname = pipe.nodes[n.parent_id].canonical_name if n.parent_id in pipe.nodes else "(ROOT)"
    print(f"  d{n.depth} | {n.canonical_name[:60]!r:62} <- {pname[:40]!r}")
