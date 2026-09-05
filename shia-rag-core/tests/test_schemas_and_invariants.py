"""
Unit tests for Layer 0 Schemas and Invariants.
"""

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.layer0_data_model.invariants import (
    ForestInvariantChecker,
    validate_pre_weights,
)
from src.layer0_data_model.schemas import (
    EdgeCategory,
    HierarchicalRelation,
    KnowledgeEdge,
    KnowledgeNode,
    NodeType,
)


class TestKnowledgeNodeSchema:
    def test_valid_node_creation(self):
        node = KnowledgeNode(
            node_id="KN-000001",
            canonical_name="Artificial Intelligence",
            node_type=NodeType.CONCEPT,
            depth=0,
        )
        assert node.node_id == "KN-000001"
        assert node.depth == 0
        assert node.node_type == NodeType.CONCEPT

    def test_invalid_node_id_pattern(self):
        with pytest.raises(ValueError):
            KnowledgeNode(
                node_id="INVALID_ID",
                canonical_name="Invalid",
                node_type=NodeType.CONCEPT,
                depth=0,
            )


class TestInvariants:
    def test_pre_weight_validation(self):
        assert validate_pre_weights(0.3, 0.25, 0.2, 0.15, 0.1) is True
        with pytest.raises(ValueError, match="PRE weights must sum to 1.0"):
            validate_pre_weights(0.5, 0.5, 0.5, 0.0, 0.0)

    def test_forest_invariant_checker_clean(self):
        checker = ForestInvariantChecker(max_depth=4)
        n1 = KnowledgeNode(node_id="KN-000001", canonical_name="Root", node_type=NodeType.CONCEPT, depth=0)
        n2 = KnowledgeNode(
            node_id="KN-000002",
            canonical_name="Child",
            node_type=NodeType.CONCEPT,
            depth=1,
            parent_id="KN-000001",
        )
        nodes = {"KN-000001": n1, "KN-000002": n2}
        edges = [
            KnowledgeEdge(
                edge_id="E-00000001",
                source_id="KN-000002",
                target_id="KN-000001",
                category=EdgeCategory.HIERARCHICAL,
                predicate=HierarchicalRelation.IS_A.value,
            )
        ]
        violations = checker.check_all(nodes, edges)
        assert len(violations) == 0

    def test_forest_invariant_checker_self_loop(self):
        # Model level catches self-loops on edge creation
        with pytest.raises(ValueError, match="Self-loop detected"):
            KnowledgeEdge(
                edge_id="E-00000001",
                source_id="KN-000001",
                target_id="KN-000001",
                category=EdgeCategory.HIERARCHICAL,
                predicate=HierarchicalRelation.IS_A.value,
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
