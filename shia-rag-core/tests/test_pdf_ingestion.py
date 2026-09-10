"""
Unit & Integration Tests for SHIA-RAG PDF Document Ingestion
============================================================
Tests end-to-end PDF ingestion, syntactic block decomposition,
conceptual hierarchy induction, forest integration, and query verification
on real research paper PDF documents.
"""

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.layer0_data_model.schemas import DocumentMimeType
from src.layer1_ingestion.pdf_loader import PDFLoader
from src.layer3_extraction.concept_extractor import ConceptExtractor
from src.pipeline import SHIARAGPipeline


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
TREERAG_PDF = WORKSPACE_ROOT / "research_papers" / "TreeRAG Unleashing the Power of Hierarchical Storage for Enhanced.pdf"


class TestPDFLoader:
    def test_pdf_not_found(self):
        loader = PDFLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_pdf("non_existent_file.pdf")

    @pytest.mark.skipif(not TREERAG_PDF.exists(), reason="TreeRAG sample PDF not found")
    def test_pdf_loading_structure(self):
        loader = PDFLoader(min_block_chars=5)
        doc_node, blocks = loader.load_pdf(TREERAG_PDF)

        assert doc_node.filename == TREERAG_PDF.name
        assert doc_node.mime_type == DocumentMimeType.PDF
        assert len(doc_node.sha256_hash) == 64
        assert doc_node.total_pages == 16
        assert len(blocks) > 200

        # Verify block metadata
        b0 = blocks[0]
        assert b0.doc_id == doc_node.doc_id
        assert b0.reading_order == 1
        assert b0.page_number == 1
        assert b0.bbox_coords is not None
        assert len(b0.bbox_coords) == 4


class TestConceptExtractor:
    @pytest.mark.skipif(not TREERAG_PDF.exists(), reason="TreeRAG sample PDF not found")
    def test_hierarchy_extraction(self):
        loader = PDFLoader(min_block_chars=5)
        extractor = ConceptExtractor()
        doc_node, blocks = loader.load_pdf(TREERAG_PDF)

        extracted = extractor.extract_from_document(doc_node, blocks)
        assert len(extracted) >= 10

        # Root node check
        root_node, root_parents = extracted[0]
        assert root_node.abstraction_level == 1.0
        assert root_parents == []
        assert any("TreeRAG" in a for a in root_node.aliases)

        # Child section check: Tree-Chunking should be present
        section_names = [n.canonical_name for n, _ in extracted]
        assert any("Tree-Chunking" in s or "TreeRAG" in s for s in section_names)


class TestPipelinePDFIngestion:
    @pytest.mark.skipif(not TREERAG_PDF.exists(), reason="TreeRAG sample PDF not found")
    def test_end_to_end_pdf_ingestion_and_invariants(self):
        pipeline = SHIARAGPipeline()
        stats = pipeline.ingest_pdf(TREERAG_PDF)

        assert stats["document"] == TREERAG_PDF.name
        assert stats["total_pages"] == 16
        assert stats["blocks_ingested"] > 200
        assert stats["concepts_extracted"] >= 10
        assert stats["forest_stats"]["root_count"] >= 1
        assert stats["forest_stats"]["max_depth"] <= 8
        assert stats["forest_stats"]["hierarchical_edges"] >= stats["concepts_extracted"] - stats["forest_stats"]["root_count"]

        # Structural invariants must hold
        pipeline.validate_invariants()

    @pytest.mark.skipif(not TREERAG_PDF.exists(), reason="TreeRAG sample PDF not found")
    def test_pdf_query_retrieval_and_citation(self):
        pipeline = SHIARAGPipeline()
        pipeline.ingest_pdf(TREERAG_PDF)

        res = pipeline.run_query("What is TreeRAG and what are its components?")

        assert res["routing_mode"] in ("THEMATIC", "FACTUAL", "MULTIHOP", "PARAMETRIC")
        assert len(res["selected_nodes"]) > 0
        assert res["total_tokens"] <= 2048
        assert len(res["answer"].strip()) > 50
        assert res["attribution_reward"] >= 0.0
        assert res["evolution_update"]["query_count"] >= 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
