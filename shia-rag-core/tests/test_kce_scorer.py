"""
Unit tests for KnowledgeConfidenceEngine (Layer 4).
"""

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.layer0_data_model import EdgeCategory, HierarchicalRelation, KnowledgeEdge, KnowledgeNode, NodeType
from src.layer4_relations import KnowledgeConfidenceEngine


@pytest.fixture
def kce():
    return KnowledgeConfidenceEngine()


class TestKCEScorer:
    def test_single_node_propagation(self, kce):
        node = KnowledgeNode(
            node_id="KN-000001",
            canonical_name="Root Concept",
            node_type=NodeType.CONCEPT,
            depth=0,
            confidence=0.9,
            evidence_texts=["Some evidence 1", "Some evidence 2"],
        )
        scores = kce.propagate_confidence({"KN-000001": node}, [])
        assert "KN-000001" in scores
        assert scores["KN-000001"] > 0.5

    def test_parent_child_topological_propagation(self, kce):
        root = KnowledgeNode(
            node_id="KN-000001",
            canonical_name="Root Concept",
            node_type=NodeType.CONCEPT,
            depth=0,
            confidence=1.0,
            evidence_texts=["Solid ground truth evidence"],
        )
        child = KnowledgeNode(
            node_id="KN-000002",
            canonical_name="Child Concept",
            node_type=NodeType.CONCEPT,
            depth=1,
            confidence=0.8,
            parent_id="KN-000001",
            evidence_texts=["Child evidence"],
        )
        edge = KnowledgeEdge(
            edge_id="E-00000001",
            source_id="KN-000002",
            target_id="KN-000001",
            category=EdgeCategory.HIERARCHICAL,
            predicate=HierarchicalRelation.IS_A.value,
        )
        nodes = {"KN-000001": root, "KN-000002": child}
        edges = [edge]
        scores = kce.propagate_confidence(nodes, edges)
        assert len(scores) == 2
        assert scores["KN-000001"] >= kce.min_conf
        assert scores["KN-000002"] >= kce.min_conf


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
