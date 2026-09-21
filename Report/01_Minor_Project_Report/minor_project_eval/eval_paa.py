import sys, re
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

loader = PDFLoader(min_block_chars=5)
doc_node, blocks = loader.load_pdf(PDF)
extracted = ConceptExtractor().extract_from_document(doc_node, blocks)
pipe = SHIARAGPipeline()
for n, p in extracted:
    pipe.add_concept_node(n, candidate_parents=p)
pipe.run_confidence_propagation()

# Gold: TreeRAG paper's true section hierarchy, keyed by leading section number.
# parent given as section-number key, or "ROOT".
gold = {
    "abstract":"ROOT","1":"ROOT","2":"ROOT","3":"ROOT","4":"ROOT","5":"ROOT",
    "3.1":"3","3.2":"3","3.1.1":"3.1","3.1.2":"3.1",
    "4.1":"4","4.2":"4","4.2.1":"4.2","4.2.2":"4.2","4.2.3":"4.2",
}
def secnum(name):
    m = re.match(r"^(\d+(?:\.\d+)*)", name.strip())
    if m: return m.group(1)
    if name.strip().lower().startswith("abstract") and "detail" not in name.lower():
        return "abstract"
    return None

# predicted parent (by section number) for each real-section node
pred = {}
for nid, node in pipe.nodes.items():
    nm = node.canonical_name
    if "Details" in nm or "Novelties" in nm or "Contributions" in nm:
        continue
    k = secnum(nm)
    if k is None or k not in gold:
        continue
    if node.parent_id and node.parent_id in pipe.nodes:
        pk = secnum(pipe.nodes[node.parent_id].canonical_name)
        pred[k] = pk if pk else "ROOT"
    else:
        pred[k] = "ROOT"

matched = [k for k in gold if k in pred]
correct = [k for k in matched if pred[k] == gold[k]]
print(f"GOLD_SECTIONS={len(gold)}")
print(f"MATCHED(extracted)={len(matched)}  COVERAGE={len(matched)/len(gold)*100:.1f}%")
print(f"CORRECT_PARENT={len(correct)}")
print(f"PAA_matched={len(correct)/max(1,len(matched))*100:.1f}%")
print(f"PAA_overall_gold={len(correct)/len(gold)*100:.1f}%")
missing = [k for k in gold if k not in pred]
wrong = [(k, pred[k], gold[k]) for k in matched if pred[k] != gold[k]]
print("MISSING(not extracted):", missing)
print("WRONG_PARENT:", wrong)
