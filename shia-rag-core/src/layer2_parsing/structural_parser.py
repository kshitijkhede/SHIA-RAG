"""
SHIA-RAG Layer 2: Structural Document Parsing & Reading Order Recovery
======================================================================
Provides layout-aware structural block segmentation, 2D reading order sort,
and monotonic reading order validation for Tier 1 document trees.

Reference: Chapter 6.3 (Layer 2), SHIA_RAG_Complete_Report.md
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Tuple

_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.layer0_data_model.schemas import TextBlock

logger = logging.getLogger(__name__)


class StructuralDocumentParser:
    """
    Parses and orders raw document layout blocks, enforcing reading order
    consistency and geometric layout constraints.
    """

    def __init__(self, y_tolerance: float = 20.0):
        """
        Args:
            y_tolerance: Vertical pixel tolerance for row clustering in 2D reading order sort.
        """
        self.y_tolerance = y_tolerance

    def sort_reading_order_2d(
        self,
        raw_blocks: List[Tuple[float, float, float, float, str, int, int]],
    ) -> List[Tuple[float, float, float, float, str, int, int]]:
        """
        Sorts raw bounding box blocks primarily top-to-bottom and secondarily left-to-right.

        Args:
            raw_blocks: List of tuples (x0, y0, x1, y1, text, block_no, block_type)

        Returns:
            Sorted list of blocks in monotonic reading order.
        """
        return sorted(
            raw_blocks,
            key=lambda b: (round(b[1] / self.y_tolerance) * self.y_tolerance, b[0]),
        )

    def verify_reading_order_monotonicity(self, blocks: List[TextBlock]) -> bool:
        """
        Validates that a sequence of TextBlocks has strictly monotonically
        increasing reading_order indices starting from 1.

        Args:
            blocks: List of TextBlock objects.

        Returns:
            True if reading order is strictly monotonic, False otherwise.
        """
        if not blocks:
            return True

        for i in range(len(blocks) - 1):
            if blocks[i + 1].reading_order <= blocks[i].reading_order:
                logger.warning(
                    f"Monotonicity violation at index {i}: "
                    f"{blocks[i].reading_order} >= {blocks[i + 1].reading_order}"
                )
                return False
        return True

    def filter_empty_blocks(
        self,
        blocks: List[TextBlock],
        min_chars: int = 5,
    ) -> List[TextBlock]:
        """Filters out whitespace-only or excessively short text blocks."""
        return [b for b in blocks if len(b.text_content.strip()) >= min_chars]
