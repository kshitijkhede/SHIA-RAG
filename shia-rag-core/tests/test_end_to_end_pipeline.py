"""
End-to-End Integration Tests for SHIA-RAG 2.0 Pipeline
======================================================
Verifies that all 9 layers (L0 through L8) function together harmoniously:
  - Forest construction, acyclicity, and depth limit invariants (L0, L5)
  - Topological confidence propagation (L4)
  - Cross-link discovery (L5)
  - SRDR query routing and complexity Ψ(q) scoring (L6)
  - DC-Knapsack precedence-constrained context optimization (L6)
  - Claim attribution verification with continuous reward R ∈ [0, 1] (L7)
  - Thompson Sampling Bayesian posterior updates (L8)
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


@pytest.fixture
def pipeline():
    """Initializes and returns a pipeline with the standard CS/AI knowledge forest."""
    p = SHIARAGPipeline(token_budget=2048, max_depth=8)
    p.load_sample_knowledge_base()
    return p


class TestEndToEndPipeline:
    def test_forest_construction_and_invariants(self, pipeline):
        """Verifies that the forest initializes cleanly with zero invariant violations."""
        stats = pipeline.get_forest_statistics()
        assert stats["total_nodes"] >= 12
        assert stats["total_edges"] >= 15
        assert stats["root_count"] == 1
        assert stats["roots"] == ["Computer Science"]
        assert stats["max_depth"] <= 8
        assert stats["avg_confidence"] > 0.60

        # Global invariant verification
        pipeline.validate_invariants()

    def test_thematic_query_retrieval(self, pipeline):
        """Verifies that thematic queries route to MODE_1_THEMATIC and select high-level concepts."""
        query = "Summarize the architecture and overview of Artificial Intelligence and Machine Learning"
        result = pipeline.run_query(query)

        assert result["routing_mode"] == "THEMATIC"
        selected_ids = {n["node_id"] for n in result["selected_nodes"]}
        assert "KN-CS001" in selected_ids
        assert "KN-AI001" in selected_ids
        assert "KN-ML001" in selected_ids

        # Verify no orphan concept in selection
        for node in result["selected_nodes"]:
            nid = node["node_id"]
            parents = pipeline.forest_parents_map.get(nid, [])
            for pid in parents:
                assert pid in selected_ids, f"Orphan concept {nid} selected without parent {pid}"

        # Continuous attribution reward should be high
        assert 0.80 <= result["attribution_reward"] <= 1.0

    def test_factual_needle_and_precedence(self, pipeline):
        """Verifies factual query routing and strict vertical precedence constraints."""
        query = "What is the exact mechanism of Transmission Control Protocol (TCP) flow control?"
        result = pipeline.run_query(query)

        assert result["routing_mode"] == "FACTUAL"
        selected_ids = {n["node_id"] for n in result["selected_nodes"]}
        assert "KN-TCP01" in selected_ids

        # Precedence: TCP must include Transport Layer Protocols, Networking, and Computer Science
        assert "KN-TRN01" in selected_ids
        assert "KN-NET01" in selected_ids
        assert "KN-CS001" in selected_ids

        # Subtree isolation: TCP query must NOT include AI/ML leaf nodes like SGD or Transformer
        assert "KN-SGD01" not in selected_ids
        assert "KN-TRF01" not in selected_ids

        # Attribution reward should be high
        assert result["attribution_reward"] >= 0.80

    def test_multihop_comparative_retrieval(self, pipeline):
        """Verifies multi-hop routing across semantic cross-links."""
        query = "Compare TCP versus UDP in transport protocols, and explain how Transformer architectures utilize optimization algorithms like Stochastic Gradient Descent and Backpropagation."
        result = pipeline.run_query(query)

        assert result["routing_mode"] == "MULTIHOP"
        selected_ids = {n["node_id"] for n in result["selected_nodes"]}

        # Both subtrees should be pulled in
        assert "KN-TCP01" in selected_ids
        assert "KN-UDP01" in selected_ids
        assert "KN-BP001" in selected_ids
        assert "KN-SGD01" in selected_ids
        assert "KN-TRF01" in selected_ids

        # Attribution reward
        assert result["attribution_reward"] >= 0.85
        assert result["evolution_update"]["traversed_edge_count"] > 0

    def test_thompson_sampling_feedback_update(self, pipeline):
        """Verifies that verification rewards update edge Beta posteriors and shift weights."""
        # Record initial edge priors
        edge_id = pipeline.edges[0].edge_id
        init_alpha, _ = pipeline.edge_priors[edge_id]

        # Run query traversing this edge
        query = "Summarize the architecture and overview of Artificial Intelligence and Machine Learning"
        res = pipeline.run_query(query)

        # Confirm update
        assert res["evolution_update"]["query_count"] >= 1
        assert len(pipeline.edge_priors) == len(pipeline.edges)
        assert pipeline.edge_priors[edge_id][0] >= init_alpha

    def test_tight_budget_precedence_no_orphans(self, pipeline):
        """Verifies DC-Knapsack under a very tight token budget: child never selected without parent."""
        # Set budget small enough that not all nodes fit
        tight_budget = 160
        res = pipeline.retrieve("Transmission Control Protocol (TCP)", token_budget=tight_budget)

        assert res.total_tokens <= tight_budget
        selected_set = set(res.selected_node_ids)

        # Invariant: Every selected child must have all hierarchical parents in selected_set
        for nid in res.selected_node_ids:
            for pid in pipeline.forest_parents_map.get(nid, []):
                assert pid in selected_set, f"Violated precedence: child {nid} in context without parent {pid}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
