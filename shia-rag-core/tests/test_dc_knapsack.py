"""
Unit tests for DAGKnapsackOptimizer (precedence-constrained context selection).
"""

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.layer6_retrieval.dc_knapsack import DAGKnapsackOptimizer, KnapsackItem


@pytest.fixture
def optimizer():
    return DAGKnapsackOptimizer(token_budget=100)


class TestBasicSelection:
    def test_empty_items(self, optimizer):
        selected, utility, tokens = optimizer.solve({})
        assert selected == []
        assert utility == 0.0
        assert tokens == 0

    def test_single_item_fits(self, optimizer):
        items = {"A": KnapsackItem("A", relevance_score=1.0, token_cost=50, parent_ids=[])}
        selected, utility, tokens = optimizer.solve(items)
        assert "A" in selected
        assert utility == 1.0
        assert tokens == 50

    def test_single_item_too_large(self, optimizer):
        items = {"A": KnapsackItem("A", relevance_score=1.0, token_cost=200, parent_ids=[])}
        selected, utility, tokens = optimizer.solve(items)
        assert selected == []
        assert utility == 0.0
        assert tokens == 0


class TestPrecedenceConstraints:
    def test_child_pulls_parent(self, optimizer):
        """Selecting child B must also pull its prerequisite parent A."""
        items = {
            "A": KnapsackItem("A", relevance_score=0.1, token_cost=30, parent_ids=[]),
            "B": KnapsackItem("B", relevance_score=5.0, token_cost=30, parent_ids=["A"]),
        }
        selected, utility, tokens = optimizer.solve(items)
        # B is high utility so both A and B should be selected
        assert "B" in selected
        assert "A" in selected
        assert tokens == 60
        assert utility == pytest.approx(5.1)

    def test_child_rejected_if_closure_exceeds_budget(self, optimizer):
        """B requires A, but A+B token cost exceeds budget."""
        items = {
            "A": KnapsackItem("A", relevance_score=0.1, token_cost=60, parent_ids=[]),
            "B": KnapsackItem("B", relevance_score=5.0, token_cost=60, parent_ids=["A"]),
        }
        selected, utility, tokens = optimizer.solve(items)
        # A+B = 120 > budget 100, neither should be selected
        assert "B" not in selected

    def test_no_orphan_nodes_ever_selected(self, optimizer):
        """Property test: For any random graph, no selected node lacks its selected parent."""
        items = {
            "ROOT": KnapsackItem("ROOT", relevance_score=0.2, token_cost=20, parent_ids=[]),
            "MID": KnapsackItem("MID", relevance_score=0.5, token_cost=25, parent_ids=["ROOT"]),
            "LEAF": KnapsackItem("LEAF", relevance_score=3.0, token_cost=30, parent_ids=["MID"]),
        }
        selected, _, _ = optimizer.solve(items)
        for node_id in selected:
            item = items[node_id]
            for pid in item.parent_ids:
                if pid in items:
                    assert pid in selected, f"Orphan: {node_id} selected without parent {pid}"


class TestBudgetRespect:
    def test_never_exceeds_budget(self, optimizer):
        items = {
            f"N{i}": KnapsackItem(f"N{i}", relevance_score=0.5, token_cost=15, parent_ids=[])
            for i in range(20)
        }
        _, _, tokens = optimizer.solve(items)
        assert tokens <= 100

    def test_maximizes_utility(self, optimizer):
        """Should prefer high-density items."""
        items = {
            "HIGH": KnapsackItem("HIGH", relevance_score=10.0, token_cost=50, parent_ids=[]),
            "LOW": KnapsackItem("LOW", relevance_score=0.1, token_cost=50, parent_ids=[]),
        }
        selected, _, _ = optimizer.solve(items)
        assert "HIGH" in selected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
