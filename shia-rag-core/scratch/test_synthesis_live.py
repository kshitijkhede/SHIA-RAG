import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import SHIARAGPipeline
from src.api import _format_natural_grounded_answer

pdf_path = Path("/home/itadmin/Desktop/Rag/shia-rag-core/uploads/Design_and_Evaluation_of_Agentic_RAG_Systems_Native_Self-RAG_and_Adaptive_RAG.pdf")
if not pdf_path.exists():
    print(f"File not found: {pdf_path}")
    sys.exit(1)

pipeline = SHIARAGPipeline()
print("Ingesting PDF...")
ingest_res = pipeline.ingest_pdf(str(pdf_path), clear_existing=True)
print(f"Ingested {ingest_res['concepts_extracted']} concepts.")

query = "what is the research paper about"
print(f"\nQuerying: '{query}'")
raw_res = pipeline.run_query(query)
formatted = _format_natural_grounded_answer(query, raw_res, pipeline)

print("=" * 80)
print(formatted)
print("=" * 80)
print(f"Attribution Reward: {raw_res.get('attribution_reward', 'N/A')}")
print(f"Routing Mode: {raw_res.get('routing_mode', 'N/A')}")
print(f"Verified: {raw_res.get('verified', 'N/A')}")
