"""
Tests for Multi-Depth Knowledge Forest (Depth 0..5+) and Universal PDF Ingestion & Verification.
"""

import sys
from pathlib import Path
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.pipeline import SHIARAGPipeline
from src.layer0_data_model.schemas import DocumentMimeType, NodeType, KnowledgeNode


def test_multi_depth_tree_structure():
    """Verify that parent linking and multi-tier concepts reach depths 4, 5, and beyond."""
    pipeline = SHIARAGPipeline()
    doc = pipeline.register_document("Deep_Taxonomy_Test.pdf", DocumentMimeType.PDF, pages=5)

    # 1. Document Root (depth 0)
    root = KnowledgeNode(
        node_id="KN-ROOT",
        canonical_name="Quantum Machine Learning",
        node_type=NodeType.CONCEPT,
        definition_text="Integration of quantum computation with machine learning algorithms.",
        evidence_texts=["Evidence root"],
        source_block_ids=[],
        abstraction_level=1.0,
        token_cost=50,
        doc_id=doc.doc_id,
    )
    pipeline.add_concept_node(root)
    assert pipeline.nodes["KN-ROOT"].depth == 0

    # 2. Section (depth 1)
    sec3 = KnowledgeNode(
        node_id="KN-SEC3",
        canonical_name="3 Quantum Variational Circuits",
        node_type=NodeType.CONCEPT,
        definition_text="Parameterized quantum circuits optimized via classical gradient descent.",
        evidence_texts=["Evidence sec 3"],
        source_block_ids=[],
        abstraction_level=0.8,
        token_cost=60,
        doc_id=doc.doc_id,
    )
    pipeline.add_concept_node(sec3, candidate_parents=[("KN-ROOT", 0.95)])
    assert pipeline.nodes["KN-SEC3"].depth == 1

    # 3. Subsection 3.1 (depth 2)
    sec31 = KnowledgeNode(
        node_id="KN-SEC31",
        canonical_name="3.1 Ansatz Architectures",
        node_type=NodeType.CONCEPT,
        definition_text="Specific parameterization layers for variational quantum eigensolvers.",
        evidence_texts=["Evidence sec 3.1"],
        source_block_ids=[],
        abstraction_level=0.6,
        token_cost=60,
        doc_id=doc.doc_id,
    )
    pipeline.add_concept_node(sec31, candidate_parents=[("KN-SEC3", 0.95)])
    assert pipeline.nodes["KN-SEC31"].depth == 2

    # 4. Sub-subsection 3.1.2 (depth 3)
    sec312 = KnowledgeNode(
        node_id="KN-SEC312",
        canonical_name="3.1.2 Hardware-Efficient Ansatz",
        node_type=NodeType.CONCEPT,
        definition_text="Ansatz tailored to physical qubit connectivity and native gate sets.",
        evidence_texts=["Evidence sec 3.1.2"],
        source_block_ids=[],
        abstraction_level=0.4,
        token_cost=60,
        doc_id=doc.doc_id,
    )
    pipeline.add_concept_node(sec312, candidate_parents=[("KN-SEC31", 0.95)])
    assert pipeline.nodes["KN-SEC312"].depth == 3

    # 5. Core Concept under 3.1.2 (depth 4)
    concept = KnowledgeNode(
        node_id="KN-CONC",
        canonical_name="Entangling Layer Design",
        node_type=NodeType.CONCEPT,
        definition_text="Layer of CNOT gates arranged in circular topology across adjacent qubits.",
        evidence_texts=["Evidence entangling layer"],
        source_block_ids=[],
        abstraction_level=0.25,
        token_cost=70,
        doc_id=doc.doc_id,
    )
    pipeline.add_concept_node(concept, candidate_parents=[("KN-SEC312", 0.92)])
    assert pipeline.nodes["KN-CONC"].depth == 4

    # 6. Mechanism/Rule under Core Concept (depth 5)
    mechanism = KnowledgeNode(
        node_id="KN-MECH",
        canonical_name="Circular Entanglement Rule",
        node_type=NodeType.PROCEDURE,
        definition_text="CNOT gates are applied cyclically connecting qubit i to qubit (i+1) mod N.",
        evidence_texts=["Evidence circular rule"],
        source_block_ids=[],
        abstraction_level=0.15,
        token_cost=70,
        doc_id=doc.doc_id,
    )
    pipeline.add_concept_node(mechanism, candidate_parents=[("KN-CONC", 0.93)])
    assert pipeline.nodes["KN-MECH"].depth == 5

    # 7. Parameter / Evidence Leaf (depth 6)
    param = KnowledgeNode(
        node_id="KN-PARAM",
        canonical_name="Gate Count Complexity O(N)",
        node_type=NodeType.EVIDENCE,
        definition_text="Total two-qubit gate count scales linearly with N qubits with depth 2.",
        evidence_texts=["Gate count O(N)"],
        source_block_ids=[],
        abstraction_level=0.1,
        token_cost=50,
        doc_id=doc.doc_id,
    )
    pipeline.add_concept_node(param, candidate_parents=[("KN-MECH", 0.90)])
    assert pipeline.nodes["KN-PARAM"].depth == 6

    # Verify invariants hold across this 7-level tree (depth 0..6)
    pipeline.validate_invariants()
    stats = pipeline.get_forest_statistics()
    assert stats["max_depth"] == 6
    assert stats["total_nodes"] == 7


def test_citation_verifier_unbracketed_and_intro_support():
    """Verify that ClaimAttributionVerifier correctly supports claims from retrieved nodes even without explicit tags."""
    from src.layer7_generation.citation_verifier import ClaimAttributionVerifier

    verifier = ClaimAttributionVerifier()
    selected_nodes = {
        "KN-MATH1": {
            "evidence": [
                "The cube root of 125 is 5 because 5 multiplied by itself three times equals 125.",
                "Cube root is the inverse operation of cubing a number.",
            ],
            "canonical_name": "Cube Roots",
        }
    }

    # Generation without explicit [KN-MATH1] tag
    answer = (
        "Based on grounded analysis of the textbook: "
        "The cube root of 125 is 5. "
        "This is because 5 multiplied by itself three times equals 125."
    )

    reward, records = verifier.verify_generation(answer, selected_nodes)
    assert reward >= 0.70
    assert any(r["supported"] for r in records)


def test_real_pdf_ingestion_and_depth():
    """Ingest a real PDF and ensure max_depth exceeds 3."""
    pipeline = SHIARAGPipeline()
    pdf_path = Path("uploads/hemh106.pdf")
    if not pdf_path.exists():
        pytest.skip("hemh106.pdf not found in uploads")

    result = pipeline.ingest_pdf(pdf_path, clear_existing=True)
    assert result["forest_stats"]["max_depth"] >= 3
    assert result["concepts_extracted"] > 10

    # Test exact query answering
    retrieval = pipeline.retrieve("what is cube root of 125")
    assert len(retrieval.selected_node_ids) > 0

    generation = pipeline.generate_and_verify(retrieval)
    assert generation.attribution_reward >= 0.70
    assert len(generation.answer) > 0
