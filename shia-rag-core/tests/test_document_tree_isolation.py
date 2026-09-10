"""
Unit and Integration tests for Document Tree Isolation in SHIA-RAG 2.0.
Verifies:
  1. Pipeline initializes completely empty (clean slate).
  2. Ingesting multiple documents produces separate, isolated trees.
  3. No cross-document tree entanglement when allow_cross_document=False.
  4. Query retrieval can be strictly scoped to a single document's tree.
  5. Document deletion cleanly purges only the target document's tree.
"""

import sys
from pathlib import Path
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.layer0_data_model.schemas import DocumentMimeType, KnowledgeNode, NodeType
from src.pipeline import SHIARAGPipeline
from src.api import delete_document, get_forest_graph, list_documents, pipeline as api_pipeline


class TestDocumentTreeIsolation:
    def test_clean_pipeline_initialization(self):
        pipe = SHIARAGPipeline()
        assert len(pipe.documents) == 0
        assert len(pipe.nodes) == 0
        assert len(pipe.edges) == 0
        stats = pipe.get_forest_statistics()
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0

    def test_separate_trees_multi_document(self):
        pipe = SHIARAGPipeline()

        # Doc 1: Tree A
        doc_a = pipe.register_document("Paper_A_Attention.pdf", DocumentMimeType.PDF, pages=5)
        b_a1 = pipe.add_text_block(doc_a.doc_id, "Attention Is All You Need introduces Transformer architectures.", 1)
        b_a2 = pipe.add_text_block(doc_a.doc_id, "Self-attention computes dynamic weights across input tokens.", 2)

        root_a = KnowledgeNode(
            node_id="KN-DOCA01",
            canonical_name="Attention Mechanisms",
            node_type=NodeType.CONCEPT,
            definition_text="Transformer self-attention architecture.",
            source_block_ids=[b_a1.block_id],
            doc_id=doc_a.doc_id,
        )
        pipe.add_concept_node(root_a)

        child_a = KnowledgeNode(
            node_id="KN-DOCA02",
            canonical_name="Multi-Head Attention",
            node_type=NodeType.CONCEPT,
            definition_text="Parallel attention heads computing subspace projections.",
            source_block_ids=[b_a2.block_id],
            doc_id=doc_a.doc_id,
        )
        pipe.add_concept_node(child_a, candidate_parents=[("KN-DOCA01", 0.95)])

        # Doc 2: Tree B
        doc_b = pipe.register_document("Paper_B_GraphRAG.pdf", DocumentMimeType.PDF, pages=8)
        b_b1 = pipe.add_text_block(doc_b.doc_id, "GraphRAG leverages knowledge graphs to augment retrieval.", 1)
        b_b2 = pipe.add_text_block(doc_b.doc_id, "Community detection groups nodes into hierarchical clusters.", 2)

        root_b = KnowledgeNode(
            node_id="KN-DOCB01",
            canonical_name="Graph Retrieval",
            node_type=NodeType.CONCEPT,
            definition_text="Knowledge graph augmented generation.",
            source_block_ids=[b_b1.block_id],
            doc_id=doc_b.doc_id,
        )
        pipe.add_concept_node(root_b)

        child_b = KnowledgeNode(
            node_id="KN-DOCB02",
            canonical_name="Community Detection",
            node_type=NodeType.CONCEPT,
            definition_text="Hierarchical graph clustering algorithm.",
            source_block_ids=[b_b2.block_id],
            doc_id=doc_b.doc_id,
        )
        pipe.add_concept_node(child_b, candidate_parents=[("KN-DOCB01", 0.93)])

        # Verify separate roots and document IDs
        assert root_a.node_id in pipe.nodes
        assert root_b.node_id in pipe.nodes
        assert pipe.nodes[root_a.node_id].depth == 0
        assert pipe.nodes[root_b.node_id].depth == 0
        assert pipe.nodes[root_a.node_id].parent_id is None
        assert pipe.nodes[root_b.node_id].parent_id is None

        # Verify children are strictly attached to their own document root
        assert pipe.nodes[child_a.node_id].parent_id == root_a.node_id
        assert pipe.nodes[child_b.node_id].parent_id == root_b.node_id

        # Verify crosslink discovery with allow_cross_document=False
        new_cl = pipe.run_crosslink_discovery(allow_cross_document=False)
        for e in new_cl:
            src_doc = pipe.nodes[e.source_id].doc_id
            tgt_doc = pipe.nodes[e.target_id].doc_id
            assert src_doc == tgt_doc, f"Crosslink {e.edge_id} improperly connected across docs {src_doc} and {tgt_doc}"

        # Test scoped retrieval on Doc A only
        res_a = pipe.run_query("Tell me about attention", doc_id=doc_a.doc_id)
        for n in res_a["selected_nodes"]:
            assert pipe.nodes[n["node_id"]].doc_id == doc_a.doc_id

        # Test scoped retrieval on Doc B only
        res_b = pipe.run_query("Explain community clusters", doc_id=doc_b.doc_id)
        for n in res_b["selected_nodes"]:
            assert pipe.nodes[n["node_id"]].doc_id == doc_b.doc_id

        # Test deleting Document A
        del_res = pipe.delete_document(doc_a.doc_id)
        assert del_res is True
        assert doc_a.doc_id not in pipe.documents
        assert root_a.node_id not in pipe.nodes
        assert child_a.node_id not in pipe.nodes
        # Ensure Document B's tree remains intact
        assert doc_b.doc_id in pipe.documents
        assert root_b.node_id in pipe.nodes
        assert child_b.node_id in pipe.nodes
