import pytest
from src.pipeline import SHIARAGPipeline
from src.api import _format_natural_grounded_answer


from pathlib import Path

def _find_paper(filename: str) -> str:
    candidates = [
        Path(filename),
        Path("research_papers") / filename,
        Path("../research_papers") / filename,
        Path(__file__).parent.parent.parent / "research_papers" / filename,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return filename


def test_heading_extraction_and_query():
    pipeline = SHIARAGPipeline()
    res = pipeline.ingest_pdf(
        _find_paper("Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented.pdf"),
        clear_existing=True
    )
    assert res["concepts_extracted"] > 0

    # Query for headings
    query = "give me heading from the paper"
    query_res = pipeline.run_query(query)
    formatted = _format_natural_grounded_answer(query, query_res, pipeline)

    # Must contain main sections
    assert "1 Introduction" in formatted
    assert "2 Preliminary" in formatted
    assert "3 Ψ-RAG" in formatted
    assert "3.1 Abstract Tree Indexing" in formatted
    assert "3.2 Multi-granular Agentic Retrieval" in formatted
    assert "4 Distribution Adaptability of Tree-RAG" in formatted
    assert "5 Experiments" in formatted
    assert "6 Conclusion" in formatted
    assert "Appendix A: Algorithms of Ψ-RAG" in formatted
    assert "Appendix B: Theoretical Proofs" in formatted
    assert "B.1 Proof of Theorem 4.1" in formatted

    # Must NOT contain prompt noise or TOC dot leaders
    assert "LLM Prompt for R" not in formatted
    assert ". . . ." not in formatted
    assert "Y ArchRAG" not in formatted

    # Must contain attribution verification badge and be clean of raw [KN- noise
    assert "SHIA-RAG 2.0 Attribution Verification" in formatted
    assert "[KN-" not in formatted
    assert any(n["node_id"].startswith("KN-") for n in query_res.get("selected_nodes", []))


def test_treerag_headings_query():
    pipeline = SHIARAGPipeline()
    res = pipeline.ingest_pdf(
        _find_paper("TreeRAG Unleashing the Power of Hierarchical Storage for Enhanced.pdf"),
        clear_existing=True
    )
    assert res["concepts_extracted"] > 0

    query = "what are the sections in this paper"
    query_res = pipeline.run_query(query)
    formatted = _format_natural_grounded_answer(query, query_res, pipeline)

    assert "Introduction" in formatted
    assert "TreeRAG" in formatted
    assert "Conclusion" in formatted
    assert "[KN-" not in formatted
    assert any(n["node_id"].startswith("KN-") for n in query_res.get("selected_nodes", []))
