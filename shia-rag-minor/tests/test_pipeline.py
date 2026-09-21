"""Unit tests for the minor-project modules (invariants, reading order, PRE)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pytest
except ModuleNotFoundError:  # allow running without pytest installed
    import contextlib

    class _Pytest:
        @staticmethod
        @contextlib.contextmanager
        def raises(exc):
            try:
                yield
            except exc:
                return
            raise AssertionError(f"expected {exc.__name__}")

        @staticmethod
        def main(argv):
            import traceback
            g = dict(globals())
            passed = failed = 0
            for cname, cls in [(k, v) for k, v in g.items() if k.startswith("Test")]:
                obj = cls()
                for m in dir(obj):
                    if m.startswith("test_"):
                        try:
                            getattr(obj, m)(); passed += 1
                        except Exception:
                            failed += 1; print(f"FAIL {cname}.{m}"); traceback.print_exc()
            print(f"{passed} passed, {failed} failed")
            return 1 if failed else 0

    pytest = _Pytest()

from shia_minor.schemas import BoundingBox, TextBlock, ConceptNode
from shia_minor.structural_parser import StructuralParser
from shia_minor.graph_validator import GraphValidator
from shia_minor.hierarchy_builder import HierarchyBuilder
from shia_minor.kce_scorer import KCEScorer


def _blk(bid, x0, y0, x1, y1, text="body text that is quite long for a block", size=10.0, page=1):
    return TextBlock(block_id=bid, page=page, text=text,
                     bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1), font_size=size)


class TestSchemas:
    def test_bbox_order_rejected(self):
        with pytest.raises(ValueError):
            BoundingBox(x0=10, y0=0, x1=5, y1=10)

    def test_depth_bound(self):
        with pytest.raises(ValueError):
            ConceptNode(concept_id="a", name="A", doc_id="d", level=1, order=0, depth=9)


class TestReadingOrder:
    def test_two_column_order(self):
        # left column (x<300) rows, right column (x>300) rows, page width 600
        parser = StructuralParser()
        blocks = [
            _blk(0, 50, 50, 250, 70, "left top"),
            _blk(1, 350, 50, 550, 70, "right top"),
            _blk(2, 50, 150, 250, 170, "left bottom"),
            _blk(3, 350, 150, 550, 170, "right bottom"),
        ]
        ordered = parser.sort_reading_order(blocks, page_w=600, page_h=800)
        texts = [b.text for b in ordered]
        assert texts == ["left top", "left bottom", "right top", "right bottom"]


class TestValidator:
    def test_self_loop_and_cycle(self):
        v = GraphValidator()
        nodes = {c.concept_id: c for c in [
            ConceptNode(concept_id="A", name="A", doc_id="d", level=1, order=0, depth=0),
            ConceptNode(concept_id="B", name="B", doc_id="d", level=2, order=1, depth=1),
        ]}
        parent_of = {"B": "A"}
        assert v.validate_edge("A", "A", nodes, parent_of) is False       # self loop
        assert v.validate_edge("A", "B", nodes, parent_of) is True        # ok
        assert v.validate_edge("B", "A", nodes, parent_of) is False       # cycle

    def test_depth_bound(self):
        v = GraphValidator(max_depth=8)
        deep = ConceptNode(concept_id="P", name="P", doc_id="d", level=1, order=0, depth=8)
        child = ConceptNode(concept_id="C", name="C", doc_id="d", level=2, order=1)
        nodes = {"P": deep, "C": child}
        assert v.validate_edge("P", "C", nodes, {}) is False

    def test_document_isolation(self):
        v = GraphValidator()
        a = ConceptNode(concept_id="A", name="A", doc_id="d1", level=1, order=0)
        b = ConceptNode(concept_id="B", name="B", doc_id="d2", level=2, order=1)
        assert v.validate_edge("A", "B", {"A": a, "B": b}, {}) is False


class TestHierarchyAndKCE:
    def _concepts(self):
        return [
            ConceptNode(concept_id="c0", name="3 System", doc_id="d", level=1, order=0, section_number="3"),
            ConceptNode(concept_id="c1", name="3.1 Method", doc_id="d", level=2, order=1, section_number="3.1"),
            ConceptNode(concept_id="c2", name="3.1.1 Step", doc_id="d", level=3, order=2, section_number="3.1.1"),
        ]

    def test_numbering_builds_correct_tree(self):
        edges, concepts = HierarchyBuilder().build(self._concepts())
        pmap = {c.concept_id: c.parent_id for c in concepts}
        assert pmap["c0"] is None
        assert pmap["c1"] == "c0"
        assert pmap["c2"] == "c1"
        assert all(c.depth <= 8 for c in concepts)

    def test_no_orphans(self):
        _, concepts = HierarchyBuilder().build(self._concepts())
        ids = {c.concept_id for c in concepts}
        for c in concepts:
            assert c.parent_id is None or c.parent_id in ids

    def test_kce_monotone_damping(self):
        _, concepts = HierarchyBuilder().build(self._concepts())
        KCEScorer().score(concepts)
        conf = {c.concept_id: c.confidence for c in concepts}
        # child confidence cannot exceed its own local confidence
        for c in concepts:
            assert c.confidence <= c.local_confidence + 1e-9


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
