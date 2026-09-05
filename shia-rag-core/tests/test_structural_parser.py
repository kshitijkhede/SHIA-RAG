"""
Tests for Layer 2: Structural Document Parsing & Reading Order Recovery
"""

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.layer0_data_model.schemas import TextBlock
from src.layer2_parsing.structural_parser import StructuralDocumentParser


class TestStructuralDocumentParser:
    def test_2d_sorting(self):
        parser = StructuralDocumentParser(y_tolerance=20.0)
        raw_blocks = [
            (200.0, 50.0, 300.0, 70.0, "Right column top", 2, 0),
            (50.0, 50.0, 150.0, 70.0, "Left column top", 1, 0),
            (50.0, 150.0, 150.0, 170.0, "Left column bottom", 3, 0),
        ]
        sorted_blocks = parser.sort_reading_order_2d(raw_blocks)
        # First should be Left column top (y=50, x=50), then Right column top (y=50, x=200), then bottom
        assert sorted_blocks[0][4] == "Left column top"
        assert sorted_blocks[1][4] == "Right column top"
        assert sorted_blocks[2][4] == "Left column bottom"

    def test_reading_order_monotonicity(self):
        parser = StructuralDocumentParser()
        b1 = TextBlock(doc_id="DOC-1", reading_order=1, text_content="Paragraph 1")
        b2 = TextBlock(doc_id="DOC-1", reading_order=2, text_content="Paragraph 2")
        b3 = TextBlock(doc_id="DOC-1", reading_order=3, text_content="Paragraph 3")

        assert parser.verify_reading_order_monotonicity([b1, b2, b3]) is True

        # Non-monotonic sequence
        b_invalid = TextBlock(doc_id="DOC-1", reading_order=2, text_content="Out of order")
        assert parser.verify_reading_order_monotonicity([b1, b3, b_invalid]) is False

    def test_filter_empty_blocks(self):
        parser = StructuralDocumentParser()
        b1 = TextBlock(doc_id="DOC-1", reading_order=1, text_content="Valid text")
        b2 = TextBlock(doc_id="DOC-1", reading_order=2, text_content="   ")
        b3 = TextBlock(doc_id="DOC-1", reading_order=3, text_content="Hi")

        filtered = parser.filter_empty_blocks([b1, b2, b3], min_chars=5)
        assert len(filtered) == 1
        assert filtered[0].text_content == "Valid text"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
