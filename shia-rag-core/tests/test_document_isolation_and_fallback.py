"""
Integration and regression tests for SHIA-RAG Document Isolation,
Auto-Scoping, and Robust Single-Document Retrieval Fallback.

Validates:
1. Single arbitrary PDF never yields empty answer on general queries
   ("what is this paper about", "summarize", "main novelties").
2. Uploading Document B after Document A does NOT contaminate answers
   when querying without doc_id (auto-scopes to latest document).
3. Explicit scoping with doc_id strictly enforces 100% isolation.
4. Intent-aware grounded synthesis produces clean, informative text.
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
from src.api import QueryRequest, execute_query, pipeline as api_pipeline, _format_natural_grounded_answer


class TestDocumentIsolationAndFallback:
    def test_single_document_broad_queries_never_empty(self):
        """Verify any uploaded document gets grounded responses for broad questions."""
        pipe = SHIARAGPipeline()
        doc = pipe.register_document("QuantumDroneSwarm.pdf", DocumentMimeType.PDF, pages=10)
        
        b0 = pipe.add_text_block(doc.doc_id, "A Post-Quantum Authentication Scheme for Drone Swarms. Authors: Alice et al.", 1)
        b1 = pipe.add_text_block(doc.doc_id, "Abstract: Traditional public-key cryptography fails against quantum threats. We propose a lattice-based authentication scheme for UAV drone swarms.", 1)
        b2 = pipe.add_text_block(doc.doc_id, "Novelty and Contributions: 1) First lattice-based group key exchange for UAVs. 2) 40% reduction in communication overhead.", 2)
        b3 = pipe.add_text_block(doc.doc_id, "Evaluation and Results: Simulations demonstrate robust packet delivery and sub-millisecond handshake latency under quantum attacks.", 8)

        # Root abstract node
        root = KnowledgeNode(
            node_id="KN-ROOT01",
            canonical_name="Quantum Drone Swarm Authentication",
            node_type=NodeType.CONCEPT,
            definition_text="Lattice-based post-quantum cryptographic protocol for UAV swarms.",
            source_block_ids=[b0.block_id, b1.block_id],
            doc_id=doc.doc_id,
        )
        pipe.add_concept_node(root)

        # Contribution node
        contrib = KnowledgeNode(
            node_id="KN-CONTRIB01",
            canonical_name="Lattice Group Key Agreement",
            node_type=NodeType.CONCEPT,
            definition_text="Novel key exchange reducing communication overhead by 40% in UAV swarms.",
            source_block_ids=[b2.block_id],
            doc_id=doc.doc_id,
        )
        pipe.add_concept_node(contrib, candidate_parents=[("KN-ROOT01", 0.95)])

        # Evaluation node
        eval_node = KnowledgeNode(
            node_id="KN-EVAL01",
            canonical_name="Protocol Latency Evaluation",
            node_type=NodeType.CONCEPT,
            definition_text="Sub-millisecond handshake latency tested in simulated quantum adversarial setting.",
            source_block_ids=[b3.block_id],
            doc_id=doc.doc_id,
        )
        pipe.add_concept_node(eval_node, candidate_parents=[("KN-ROOT01", 0.90)])

        queries = [
            "what is this paper about",
            "summarize this paper",
            "list all the novelties in this paper",
            "give me detail about project",
            "what are the main contributions"
        ]

        for q in queries:
            retrieval_res = pipe.retrieve(q)
            assert len(retrieval_res.selected_node_ids) > 0, f"Query '{q}' returned empty nodes on valid document!"
            assert retrieval_res.total_utility > 0.0, f"Query '{q}' had zero utility!"
            
            res = pipe.run_query(q)
            assert len(res["selected_nodes"]) > 0
            
            answer = _format_natural_grounded_answer(q, res["selected_nodes"], pipe, scoped_doc_id=doc.doc_id)
            assert "Quantum Drone Swarm Authentication" in answer or "lattice" in answer.lower()
            assert len(answer.strip()) > 20, "Answer must contain verified grounded content"

    def test_multi_doc_unspecified_doc_id_auto_scopes_to_latest(self):
        """When 2 documents exist and user doesn't specify doc_id, retrieve strictly from latest document."""
        pipe = SHIARAGPipeline()

        # Doc 1: Drone Swarm
        doc_a = pipe.register_document("DroneSwarm.pdf", DocumentMimeType.PDF, pages=6)
        b_a1 = pipe.add_text_block(doc_a.doc_id, "Drone swarm authentication using post-quantum lattices.", 1)
        node_a = KnowledgeNode(
            node_id="KN-DRONE01",
            canonical_name="Drone Swarm Security",
            node_type=NodeType.CONCEPT,
            definition_text="Post-quantum cryptographic UAV protection.",
            source_block_ids=[b_a1.block_id],
            doc_id=doc_a.doc_id,
        )
        pipe.add_concept_node(node_a)

        # Doc 2: TreeRAG
        doc_b = pipe.register_document("TreeRAG_Paper.pdf", DocumentMimeType.PDF, pages=12)
        b_b1 = pipe.add_text_block(doc_b.doc_id, "TreeRAG: Hierarchical Structure-Aware Retrieval Augmented Generation.", 1)
        b_b2 = pipe.add_text_block(doc_b.doc_id, "TreeRAG builds hierarchical tree abstractions across document chapters.", 2)
        node_b = KnowledgeNode(
            node_id="KN-TREE01",
            canonical_name="TreeRAG Architecture",
            node_type=NodeType.CONCEPT,
            definition_text="Hierarchical tree-structured retrieval system.",
            source_block_ids=[b_b1.block_id, b_b2.block_id],
            doc_id=doc_b.doc_id,
        )
        pipe.add_concept_node(node_b)

        # Retrieval without doc_id -> MUST auto-scope to Doc B (latest)
        res = pipe.retrieve("what is this paper about")
        assert len(res.selected_node_ids) > 0
        for nid in res.selected_node_ids:
            node = pipe.nodes[nid]
            assert node.doc_id == doc_b.doc_id, f"Node {node.node_id} has doc_id {node.doc_id}, expected {doc_b.doc_id}"
            assert "DRONE" not in node.node_id

        # Answer formatting must only reference Doc B
        selected_nodes_dict = [pipe.nodes[nid].model_dump() for nid in res.selected_node_ids]
        answer = _format_natural_grounded_answer("what is this paper about", selected_nodes_dict, pipe)
        assert "TreeRAG" in answer
        assert "Drone" not in answer, "Cross-document leakage detected in natural answer!"

    def test_multi_doc_explicit_scoping_strictness(self):
        """Explicitly specifying doc_id must retrieve 100% target doc nodes and 0% other doc nodes."""
        pipe = SHIARAGPipeline()

        doc_a = pipe.register_document("AlphaDoc.pdf", DocumentMimeType.PDF)
        b_a = pipe.add_text_block(doc_a.doc_id, "AlphaDoc details quantum computing and entanglement.", 1)
        node_a = KnowledgeNode(
            node_id="KN-ALPHA01",
            canonical_name="Quantum Entanglement",
            node_type=NodeType.CONCEPT,
            definition_text="Physical phenomenon of entangled states.",
            source_block_ids=[b_a.block_id],
            doc_id=doc_a.doc_id,
        )
        pipe.add_concept_node(node_a)

        doc_b = pipe.register_document("BetaDoc.pdf", DocumentMimeType.PDF)
        b_b = pipe.add_text_block(doc_b.doc_id, "BetaDoc details convolutional neural networks for vision.", 1)
        node_b = KnowledgeNode(
            node_id="KN-BETA01",
            canonical_name="Convolutional Networks",
            node_type=NodeType.CONCEPT,
            definition_text="Deep learning layers with sliding spatial kernels.",
            source_block_ids=[b_b.block_id],
            doc_id=doc_b.doc_id,
        )
        pipe.add_concept_node(node_b)

        # Query scoped to Alpha
        res_a = pipe.retrieve("tell me about the findings", doc_id=doc_a.doc_id)
        assert len(res_a.selected_node_ids) > 0
        for nid in res_a.selected_node_ids:
            assert pipe.nodes[nid].doc_id == doc_a.doc_id

        # Query scoped to Beta
        res_b = pipe.retrieve("tell me about the findings", doc_id=doc_b.doc_id)
        assert len(res_b.selected_node_ids) > 0
        for nid in res_b.selected_node_ids:
            assert pipe.nodes[nid].doc_id == doc_b.doc_id

    def test_api_endpoint_scoping_and_meta(self):
        """Test the FastAPI execute_query handler auto-scoping and response metadata."""
        # Reset and prepare api_pipeline
        api_pipeline.clear()

        doc = api_pipeline.register_document("SamplePaper.pdf", DocumentMimeType.PDF, pages=4)
        b = api_pipeline.add_text_block(doc.doc_id, "Sample paper introducing adaptive retrieval trees.", 1)
        node = KnowledgeNode(
            node_id="KN-SAMPLE01",
            canonical_name="Adaptive Retrieval",
            node_type=NodeType.CONCEPT,
            definition_text="Dynamic tree structure adapting to query complexity.",
            source_block_ids=[b.block_id],
            doc_id=doc.doc_id,
        )
        api_pipeline.add_concept_node(node)

        # Query without doc_id -> must auto-scope to doc.doc_id
        req = QueryRequest(query="summarize this paper")
        resp = execute_query(req)
        assert resp["doc_id"] == doc.doc_id
        assert len(resp["selected_nodes"]) > 0
        assert "Adaptive Retrieval" in resp["answer"]
