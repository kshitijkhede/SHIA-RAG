"""
Unit Tests for SHEF 2.0 Evaluation Framework
"""

import json
import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src"), str(_PROJECT_ROOT / "evaluation")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.shef_evaluator import SHEFEvaluator
from src.layer0_data_model.schemas import EdgeCategory, HierarchicalRelation, KnowledgeEdge, KnowledgeNode, NodeType
from src.pipeline import SHIARAGPipeline


class TestSHEFEvaluator:
    def test_context_density_score(self):
        nodes = [
            {"name": "TCP", "definition": "Transmission Control Protocol"},
            {"name": "UDP", "definition": "User Datagram Protocol"},
        ]
        score = SHEFEvaluator.compute_context_density_score(nodes, total_context_tokens=100)
        assert 0.0 < score <= 1.0

        # Empty nodes
        assert SHEFEvaluator.compute_context_density_score([], 100) == 0.0

    def test_parent_assignment_accuracy(self):
        gold = {"A": "ROOT", "B": "A", "C": "B"}
        pred_perfect = {"A": "ROOT", "B": "A", "C": "B"}
        assert SHEFEvaluator.compute_parent_assignment_accuracy(pred_perfect, gold) == 1.0

        pred_imperfect = {"A": "ROOT", "B": "ROOT", "C": "B"}
        assert SHEFEvaluator.compute_parent_assignment_accuracy(pred_imperfect, gold) == pytest.approx(2 / 3)

    def test_ancestor_chain_recall(self):
        gold_chain = {"ROOT", "NETWORKING", "TRANSPORT"}
        retrieved_full = {"ROOT", "NETWORKING", "TRANSPORT", "TCP", "UDP"}
        assert SHEFEvaluator.compute_ancestor_chain_recall(retrieved_full, gold_chain) == 1.0

        retrieved_partial = {"ROOT", "TCP"}
        assert SHEFEvaluator.compute_ancestor_chain_recall(retrieved_partial, gold_chain) == pytest.approx(1 / 3)

        assert SHEFEvaluator.compute_ancestor_chain_recall(set(), set()) == 1.0

    def test_orphan_rate_and_forest_density(self):
        root = KnowledgeNode(
            node_id="KN-1",
            canonical_name="Root",
            node_type=NodeType.CONCEPT,
            definition_text="Root node",
        )
        child = KnowledgeNode(
            node_id="KN-2",
            canonical_name="Child",
            node_type=NodeType.CONCEPT,
            definition_text="Child node",
            parent_id="KN-1",
        )
        nodes = {"KN-1": root, "KN-2": child}
        edges = [
            KnowledgeEdge(
                edge_id="E-1",
                source_id="KN-2",
                target_id="KN-1",
                category=EdgeCategory.HIERARCHICAL,
                predicate=HierarchicalRelation.IS_A.value,
            )
        ]

        rate = SHEFEvaluator.compute_orphan_rate(nodes, edges)
        assert rate == 0.0

        density = SHEFEvaluator.compute_forest_density(nodes, edges)
        assert density == 0.5

        summary = SHEFEvaluator.evaluate_forest(nodes, edges)
        assert summary["orphan_rate"] == 0.0
        assert summary["forest_density"] == 0.5

    def test_evaluate_pipeline_against_sample_benchmark(self):
        dataset_path = _PROJECT_ROOT / "evaluation" / "datasets" / "sample_benchmark.json"
        assert dataset_path.exists()

        with open(dataset_path) as f:
            benchmark = json.load(f)

        pipeline = SHIARAGPipeline()
        pipeline.load_sample_knowledge_base()

        for item in benchmark:
            res = pipeline.run_query(item["query"])
            selected_ids = {n["node_id"] for n in res["selected_nodes"]}
            gold_ancestors = set(item["gold_ancestor_chain"])
            acr = SHEFEvaluator.compute_ancestor_chain_recall(selected_ids, gold_ancestors)
            assert acr >= 0.5  # Crucial ancestors recalled without orphans


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
