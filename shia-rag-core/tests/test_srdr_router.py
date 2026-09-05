"""
Unit tests for SelfReflectiveDepthRouter (SRDR) and EntityCounter.
"""

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.layer6_retrieval.srdr_router import EntityCounter, SelfReflectiveDepthRouter


@pytest.fixture
def router():
    return SelfReflectiveDepthRouter()


@pytest.fixture
def counter():
    return EntityCounter()


class TestEntityCounter:
    def test_no_entities(self, counter):
        assert counter.count_entities("what is this?") == 0

    def test_quoted_entity(self, counter):
        assert counter.count_entities('What is "OSPF"?') >= 1

    def test_multiple_quoted(self, counter):
        assert counter.count_entities('Compare "OSPF" and "RIP"') >= 2

    def test_capitalized_words(self, counter):
        result = counter.count_entities("How does OSPF work with BGP?")
        assert result >= 1


class TestSRDRRouting:
    def test_thematic_mode(self, router):
        result = router.route("Give me an overview of networking protocols")
        assert result["mode"] == "MODE_1_THEMATIC"
        assert result["max_depth"] == 1
        assert result["strategy"] == "ROOT_BREADTH_FIRST"

    def test_thematic_summarize(self, router):
        result = router.route("Summarize all routing algorithms")
        assert result["mode"] == "MODE_1_THEMATIC"

    def test_factual_needle(self, router):
        result = router.route("What is the default timeout?", entity_count=1)
        assert result["mode"] == "MODE_2_FACTUAL_NEEDLE"
        assert result["max_depth"] == 1
        assert result["strategy"] == "LEAF_TO_ROOT"

    def test_multihop_comparative_by_keyword(self, router):
        result = router.route("Compare OSPF vs RIP", entity_count=2)
        assert result["mode"] == "MODE_3_MULTIHOP_COMPARATIVE"
        assert result["max_depth"] == 3
        assert result["include_crosslinks"] is True

    def test_multihop_by_entity_count(self, router):
        result = router.route("How does X relate to Y?", entity_count=2)
        assert result["mode"] == "MODE_3_MULTIHOP_COMPARATIVE"

    def test_parametric_mode(self, router):
        result = router.route("Hello, how are you?", entity_count=0)
        assert result["mode"] == "MODE_4_PARAMETRIC"
        assert result["max_depth"] == 0
        assert result["strategy"] == "SKIP_RETRIEVAL"

    def test_entity_count_auto_detection(self, router):
        """When entity_count is not provided, auto-detect via EntityCounter."""
        result = router.route("Summarize all topics")
        # "summarize" triggers thematic regardless of entity count
        assert result["mode"] == "MODE_1_THEMATIC"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
