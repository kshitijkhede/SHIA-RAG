"""
Unit tests covering the 22 bug fixes and mitigations from the PDF Structure Summary audit.
"""

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.layer0_data_model.schemas import KnowledgeNode, NodeType
from src.layer4_relations.kce_scorer import KnowledgeConfidenceEngine
from src.layer5_shia_core.cld_crosslinker import CrossLinkDiscovery
from src.layer5_shia_core.fi_integrator import ForestIntegrator
from src.layer6_retrieval.dc_knapsack import DAGKnapsackOptimizer, KnapsackItem
from src.layer6_retrieval.srdr_router import SelfReflectiveDepthRouter
from src.layer7_generation import ClaimAttributionVerifier
from src.layer8_operations.thompson_evolution import ThompsonEvolutionEngine


# ============================================================================
# Layer 7 Fixes Tests: Citation Verifier
# ============================================================================

class TestCitationVerifierAuditFixes:
    def test_multi_citation_checking(self):
        """L7.1: Verify all citations are checked, not just first match."""
        verifier = ClaimAttributionVerifier(overlap_threshold=0.30)
        # First citation is unrelated, second citation supports the claim
        answer = "BGP uses TCP port 179 [KN-000001] [KN-000002]."
        selected_nodes = {
            "KN-000001": {"evidence": ["Unrelated evidence about something else entirely."]},
            "KN-000002": {"evidence": ["BGP uses TCP port 179 for establishing peering sessions."]}
        }
        reward, records = verifier.verify_generation(answer, selected_nodes)
        assert reward == 1.0
        assert len(records) == 1
        assert records[0]["supported"] is True
        assert records[0]["grounded_node"] == "KN-000002"
        assert "KN-000001" in records[0]["all_checked_nodes"]
        assert "KN-000002" in records[0]["all_checked_nodes"]

    def test_contradiction_detection(self):
        """L7.2: Contradictions with high lexical overlap must be flagged and rejected."""
        verifier = ClaimAttributionVerifier(overlap_threshold=0.30)
        # Premise says NOT possible; claim says IS possible
        answer = "Preemption is possible in FIFO scheduling [KN-000003]."
        selected_nodes = {
            "KN-000003": {"evidence": ["Preemption is impossible in FIFO scheduling."]}
        }
        reward, records = verifier.verify_generation(answer, selected_nodes)
        assert reward == 0.0
        assert len(records) == 1
        assert records[0]["supported"] is False
        assert records[0]["contradicted"] is True

    def test_compound_sentence_splitting(self):
        """L7.3: Compound sentences must be split into distinct claims."""
        verifier = ClaimAttributionVerifier(overlap_threshold=0.30)
        answer = (
            "TCP provides reliable byte stream delivery [KN-000004]; "
            "whereas UDP provides unreliable datagram transport [KN-000005]."
        )
        selected_nodes = {
            "KN-000004": {"evidence": ["TCP provides reliable byte stream delivery."]},
            "KN-000005": {"evidence": ["UDP provides unreliable datagram transport."]}
        }
        reward, records = verifier.verify_generation(answer, selected_nodes)
        assert len(records) == 2
        assert reward == 1.0
        assert all(r["supported"] for r in records)


# ============================================================================
# Layer 6 Fixes Tests: SRDR Router & DC Knapsack
# ============================================================================

class TestSRDRRouterAuditFixes:
    def test_word_boundary_regex(self):
        """L6.1: Word-boundary regex prevents false substring matches."""
        router = SelfReflectiveDepthRouter()
        # Query contains "indifferent" which contains substring "differ", but is NOT a comparative query
        decision = router.route("Explain indifferent attitude in psychology.", entity_count=1)
        assert decision["mode"] == "MODE_2_FACTUAL_NEEDLE"

        # Explicit comparison with word boundary
        decision2 = router.route("Compare TCP and UDP latency.", entity_count=2)
        assert decision2["mode"] == "MODE_3_MULTIHOP_COMPARATIVE"

    def test_psi_and_esthops_computation(self):
        """Q1, Q2: Formal query complexity Ψ(q) and EstHops calculation."""
        router = SelfReflectiveDepthRouter()
        query = "Compare TCP congestion control versus UDP transmission protocols."
        decision = router.route(query, entity_count=2)
        assert "psi_score" in decision
        assert 0.0 <= decision["psi_score"] <= 1.0
        assert decision["max_depth"] >= 3


class TestDCKnapsackAuditFixes:
    def test_branch_and_bound_exactness(self):
        """L6.6: Branch-and-Bound solver finds optimal combination on small DAG."""
        optimizer = DAGKnapsackOptimizer(token_budget=100, max_subgraph_nodes=150)
        items = {
            "root": KnapsackItem(node_id="root", relevance_score=2.0, token_cost=30, parent_ids=[]),
            "child_a": KnapsackItem(node_id="child_a", relevance_score=5.0, token_cost=40, parent_ids=["root"]),
            "child_b": KnapsackItem(node_id="child_b", relevance_score=3.0, token_cost=50, parent_ids=["root"]),
        }
        selected, total_util, total_tokens = optimizer.solve(items)
        assert "root" in selected
        assert "child_a" in selected
        assert "child_b" not in selected
        assert total_util == pytest.approx(7.0)
        assert total_tokens == 70

    def test_marginal_density_recalculation_greedy(self):
        """Bundle density recalculation for greedy fallback."""
        optimizer = DAGKnapsackOptimizer(token_budget=100, max_subgraph_nodes=2)
        items = {
            "p": KnapsackItem(node_id="p", relevance_score=1.0, token_cost=20, parent_ids=[]),
            "c1": KnapsackItem(node_id="c1", relevance_score=4.0, token_cost=30, parent_ids=["p"]),
            "c2": KnapsackItem(node_id="c2", relevance_score=4.0, token_cost=30, parent_ids=["p"]),
        }
        selected, total_util, total_tokens = optimizer.solve(items)
        assert "p" in selected
        assert total_tokens <= 100


# ============================================================================
# Layer 8 Fixes Tests: Thompson Evolution
# ============================================================================

class TestThompsonEvolutionAuditFixes:
    def test_reward_clipping_and_credit_assignment(self):
        """L8.1, L8.2: Reward clipping and edge credit assignment."""
        engine = ThompsonEvolutionEngine(min_reward_clip=0.05, max_reward_clip=0.95)
        priors = {"edge_1": (1.0, 1.0), "edge_2": (1.0, 1.0)}
        
        engine.update_edge_feedback(
            edge_priors=priors,
            traversed_edges=["edge_1", "edge_2"],
            reward=1.0,  # Clamped to 0.95
            edge_contributions={"edge_1": 1.0, "edge_2": 0.5}
        )
        assert priors["edge_1"][0] == pytest.approx(1.95, abs=0.05)
        assert priors["edge_2"][0] == pytest.approx(1.475, abs=0.05)

    def test_temporal_decay(self):
        """L8.7: Temporal decay prevents concept drift."""
        engine = ThompsonEvolutionEngine(temporal_decay_rate=0.90)
        priors = {"edge_1": (11.0, 1.0)}  # 10 observed successes above prior 1.0
        engine.update_edge_feedback(
            edge_priors=priors,
            traversed_edges=["edge_1"],
            reward=0.5
        )
        assert priors["edge_1"][0] < 11.5


# ============================================================================
# Layer 4 & 5 Fixes Tests: KCE, CLD, FI
# ============================================================================

class TestLayer4And5AuditFixes:
    def test_kce_baseline_conf(self):
        """Error 10: Configurable baseline confidence prior."""
        kce = KnowledgeConfidenceEngine(baseline_conf=0.60)
        assert kce.baseline_conf == 0.60
        node = KnowledgeNode(
            node_id="KN-000001",
            canonical_name="Root Concept",
            node_type=NodeType.CONCEPT,
            depth=0,
            confidence_score=0.80
        )
        scores = kce.propagate_confidence({"KN-000001": node}, [])
        assert "KN-000001" in scores
        assert scores["KN-000001"] >= 0.10

    def test_cld_candidate_cap(self):
        """L6.3: CLD candidate cap prevents cross-link explosion."""
        cld = CrossLinkDiscovery(max_expansion_candidates=50)
        assert cld.max_expansion_candidates == 50

    def test_fi_tie_breaker_and_null_safety(self):
        """Error 23: Null safety and deterministic tie-breaker for equal scores."""
        fi = ForestIntegrator()
        node = KnowledgeNode(
            node_id="KN-000001",
            canonical_name="Test Node",
            node_type=NodeType.CONCEPT,
            depth=0,
            confidence_score=0.85
        )
        # Null parent candidate -> promotes to root
        up_node, edge = fi.integrate_node(
            node=node,
            ranked_valid_parents=[(None, 0.9)],
            forest_nodes={},
            forest_edges=[],
            node_depth_map={}
        )
        assert edge is None  # Promoted to root
        assert up_node.depth == 0

        # Equal score tie-breaker: KN-000002 vs KN-000003 -> KN-000002 selected deterministically
        parent_a = KnowledgeNode(node_id="KN-000002", canonical_name="Parent A", node_type=NodeType.CONCEPT, depth=1)
        parent_b = KnowledgeNode(node_id="KN-000003", canonical_name="Parent B", node_type=NodeType.CONCEPT, depth=1)
        forest_nodes = {"KN-000002": parent_a, "KN-000003": parent_b}
        node_depth_map = {"KN-000002": 1, "KN-000003": 1}

        up_node2, edge2 = fi.integrate_node(
            node=node,
            ranked_valid_parents=[("KN-000003", 0.8), ("KN-000002", 0.8)],
            forest_nodes=forest_nodes,
            forest_edges=[],
            node_depth_map=node_depth_map
        )
        assert edge2 is not None
        assert edge2.target_id == "KN-000002"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
