"""
Unit tests for HierarchyValidator (cycle detection, depth limits, self-loops).
"""

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.layer5_shia_core.hv_validator import HierarchyValidator


@pytest.fixture
def validator():
    return HierarchyValidator(max_depth=4)


class TestSelfLoopDetection:
    def test_rejects_self_loop(self, validator):
        assert validator.validate_placement("A", "A", {}, {"A": 0}) is False

    def test_accepts_non_self_loop(self, validator):
        assert validator.validate_placement("A", "B", {"A": []}, {"A": 0}) is True


class TestDepthLimit:
    def test_rejects_at_max_depth(self, validator):
        """Parent at depth 3, child would be 4 which equals max_depth=4."""
        assert validator.validate_placement("A", "B", {"A": []}, {"A": 3}) is False

    def test_accepts_within_depth_limit(self, validator):
        assert validator.validate_placement("A", "B", {"A": []}, {"A": 2}) is True

    def test_accepts_at_root(self, validator):
        assert validator.validate_placement("A", "B", {"A": []}, {"A": 0}) is True


class TestCycleDetection:
    def test_detects_simple_cycle(self, validator):
        """B is parent of A, so placing B under A creates A→B→A cycle."""
        parents_map = {"A": ["B"], "B": []}
        depth_map = {"A": 1, "B": 0}
        # child_id=B is ancestor of parent_id=A, so this should be rejected
        assert validator.validate_placement("A", "B", parents_map, depth_map) is False

    def test_detects_transitive_cycle(self, validator):
        """C→B→A chain. Placing C under A would create A→C→B→A."""
        parents_map = {"A": ["B"], "B": ["C"], "C": []}
        depth_map = {"A": 2, "B": 1, "C": 0}
        assert validator.validate_placement("A", "C", parents_map, depth_map) is False

    def test_accepts_valid_placement(self, validator):
        """D is unrelated to A→B→C chain."""
        parents_map = {"A": [], "B": ["A"], "C": ["B"], "D": []}
        depth_map = {"A": 0, "B": 1, "C": 2, "D": 0}
        assert validator.validate_placement("A", "D", parents_map, depth_map) is True

    def test_accepts_sibling_placement(self, validator):
        """E placed under A alongside B (both children of A)."""
        parents_map = {"A": [], "B": ["A"], "E": []}
        depth_map = {"A": 0, "B": 1, "E": 0}
        assert validator.validate_placement("A", "E", parents_map, depth_map) is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
