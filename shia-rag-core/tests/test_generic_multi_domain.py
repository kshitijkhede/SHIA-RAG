"""
Verification tests for Generic PDF Ingestion & Cross-Domain Querying
=====================================================================
Ensures that the pipeline and concept extractor work 100% generically across
different research domains (GraphRAG, RAPTOR, TreeRAG, and Post-Quantum Swarms)
without any domain-specific hardcoding.
"""

from pathlib import Path
import pytest

from src.pipeline import SHIARAGPipeline
from src.layer1_ingestion.pdf_loader import PDFLoader
from src.layer3_extraction.concept_extractor import ConceptExtractor

RESEARCH_PAPERS_DIR = Path(__file__).resolve().parent.parent.parent / "research_papers"


def test_generic_pipeline_has_no_hardcoded_cryptographic_or_drone_words():
    """Verifies that no paper-specific keywords exist in the generic scoring logic."""
    import inspect
    from src.pipeline import SHIARAGPipeline
    from src import api

    pipeline_src = inspect.getsource(SHIARAGPipeline._score_node_relevance)
    api_src = inspect.getsource(api._format_natural_grounded_answer)

    forbidden_hardcoded_terms = ["kyber", "dilithium", "sphincs", "uavi", "drone"]
    for term in forbidden_hardcoded_terms:
        assert term not in pipeline_src.lower(), f"Forbidden hardcoded term '{term}' found in pipeline._score_node_relevance"
        assert term not in api_src.lower(), f"Forbidden hardcoded term '{term}' found in api._format_natural_grounded_answer"


def test_generic_graphrag_ingestion_and_novelties():
    """Verifies generic ingestion and novelty query handling on GraphRAG paper."""
    graphrag_pdf = RESEARCH_PAPERS_DIR / "A Graph RAG Approach to Query-Focused Summarization (GraphRAG).pdf"
    if not graphrag_pdf.exists():
        pytest.skip("GraphRAG PDF not found")

    p = SHIARAGPipeline()
    p.ingest_pdf(str(graphrag_pdf))

    root_node = next(n for n in p.nodes.values() if n.depth == 0)
    assert "GraphRAG" in root_node.canonical_name or "Summarization" in root_node.canonical_name
    assert len(root_node.definition_text) > 50

    # Query novelties
    res = p.run_query("What are the main novelties and contributions of this paper?")
    selected_names = [n["node_id"] for n in res["selected_nodes"]]
    assert len(selected_names) > 0
    # Must retrieve grounded nodes and positive attribution support
    assert res["attribution_reward"] > 0.3


def test_generic_raptor_ingestion_and_novelties():
    """Verifies generic ingestion and title cleaning on RAPTOR paper."""
    raptor_pdf = RESEARCH_PAPERS_DIR / "Recursive Abstractive Processing for Tree-Organized Retrieval.pdf"
    if not raptor_pdf.exists():
        pytest.skip("RAPTOR PDF not found")

    loader = PDFLoader()
    doc, blocks = loader.load_pdf(str(raptor_pdf))

    extractor = ConceptExtractor()
    extracted = extractor.extract_from_document(doc, blocks)
    root_node = extracted[0][0]

    # Title must not be conference banner 'Published as a conference paper at ICLR 2024'
    assert "ICLR" not in root_node.canonical_name
    assert "RAPTOR" in root_node.canonical_name or "RETRIEVAL" in root_node.canonical_name
