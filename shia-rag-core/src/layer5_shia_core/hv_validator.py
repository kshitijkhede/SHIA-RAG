"""
SHIA-RAG Layer 5: Hierarchy Validator (HV)
===========================================
Validates that placing a child node under a parent maintains
all forest invariants: no self-loops, no cycles, depth limits.

Uses upward DFS ancestor closure check (corrected algorithm).

Reference: Chapter 8.1, SHIA_RAG_Complete_Report.md
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Set

_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)


class HierarchyValidator:
    """Validates placement invariants before inserting a hierarchical edge."""

    def __init__(self, max_depth: int = 8, min_sibling_coherence: float = 0.40):
        self.max_depth = max_depth
        self.min_sibling_coherence = min_sibling_coherence

    def validate_placement(
        self,
        parent_id: str,
        child_id: str,
        forest_parents_map: Dict[str, List[str]],
        node_depth_map: Dict[str, int],
    ) -> bool:
        """
        Validates that placing child_id under parent_id maintains all invariants:
        1. No self-loops.
        2. No cycles (child cannot be an ancestor of parent).
        3. Maximum depth invariant (parent.depth + 1 < max_depth).
        """
        # Invariant 1: Self-loop check
        if parent_id == child_id:
            logger.debug(f"Rejected: self-loop {parent_id} → {child_id}")
            return False

        # Invariant 2: Depth limit check
        parent_depth = node_depth_map.get(parent_id, 0)
        if parent_depth + 1 >= self.max_depth:
            logger.debug(
                f"Rejected: depth limit. Parent depth={parent_depth}, "
                f"child would be {parent_depth + 1} >= {self.max_depth}"
            )
            return False

        # Invariant 3: Cycle detection via upward DFS from parent
        visited: Set[str] = set()
        stack: List[str] = [parent_id]

        while stack:
            curr = stack.pop()
            if curr == child_id:
                logger.debug(
                    f"Rejected: cycle detected. {child_id} is ancestor of {parent_id}"
                )
                return False

            if curr not in visited:
                visited.add(curr)
                for p in forest_parents_map.get(curr, []):
                    stack.append(p)

        return True
