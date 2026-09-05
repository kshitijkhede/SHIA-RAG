"""
SHIA-RAG Layer 0: Graph Invariant Validators
==============================================
Runtime validators that enforce the structural invariants
of the Dual-Tier Heterogeneous Knowledge Forest:

1. Acyclicity: No cycles in hierarchical edges (DAG invariant).
2. Depth limit: No node exceeds D_max depth.
3. Self-loop prohibition: No edge connects a node to itself.
4. Orphan detection: All non-root nodes must have a parent.
5. PRE weight normalization: w1+w2+w3+w4+w5 = 1.0
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.layer0_data_model.schemas import EdgeCategory, KnowledgeEdge, KnowledgeNode

logger = logging.getLogger(__name__)


class ForestInvariantChecker:
    """
    Validates global invariants across the entire Knowledge Forest.
    Should be invoked after batch insertions or periodic maintenance.
    """

    def __init__(self, max_depth: int = 8):
        self.max_depth = max_depth

    def check_all(
        self,
        nodes: Dict[str, KnowledgeNode],
        edges: List[KnowledgeEdge],
    ) -> List[str]:
        """
        Runs all invariant checks and returns a list of violation messages.
        An empty list means all invariants hold.
        """
        violations: List[str] = []

        # Build parent map from hierarchical edges
        parents_map: Dict[str, List[str]] = {nid: [] for nid in nodes}
        for edge in edges:
            if edge.category == EdgeCategory.HIERARCHICAL:
                parents_map.setdefault(edge.source_id, []).append(edge.target_id)

        # Check 1: Self-loops
        for edge in edges:
            if edge.source_id == edge.target_id:
                violations.append(f"SELF_LOOP: Edge {edge.edge_id} connects {edge.source_id} to itself.")

        # Check 2: Acyclicity (DFS-based)
        cycle_nodes = self._detect_cycles(nodes, parents_map)
        for nid in cycle_nodes:
            violations.append(f"CYCLE: Node {nid} is part of a hierarchical cycle.")

        # Check 3: Depth violations
        depth_map = self._compute_depths(nodes, parents_map)
        for nid, depth in depth_map.items():
            if depth >= self.max_depth:
                violations.append(
                    f"DEPTH_EXCEEDED: Node {nid} at depth {depth} exceeds D_max={self.max_depth}."
                )

        # Check 4: Orphan detection (non-root nodes without a hierarchical parent)
        root_ids = {nid for nid, node in nodes.items() if node.parent_id is None and node.depth == 0}
        for nid, node in nodes.items():
            if nid not in root_ids and not parents_map.get(nid):
                violations.append(f"ORPHAN: Non-root node {nid} ('{node.canonical_name}') has no hierarchical parent.")

        if violations:
            logger.warning(f"Forest invariant check found {len(violations)} violation(s).")
        else:
            logger.info("All forest invariants hold.")

        return violations

    def _detect_cycles(
        self,
        nodes: Dict[str, KnowledgeNode],
        parents_map: Dict[str, List[str]],
    ) -> Set[str]:
        """
        Detects cycles in the hierarchical DAG using DFS coloring.
        Returns set of node IDs involved in cycles.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {nid: WHITE for nid in nodes}
        cycle_nodes: Set[str] = set()

        def dfs(nid: str) -> bool:
            color[nid] = GRAY
            for parent_id in parents_map.get(nid, []):
                if parent_id not in color:
                    continue
                if color[parent_id] == GRAY:
                    cycle_nodes.add(nid)
                    cycle_nodes.add(parent_id)
                    return True
                if color[parent_id] == WHITE:
                    if dfs(parent_id):
                        cycle_nodes.add(nid)
                        return True
            color[nid] = BLACK
            return False

        for nid in nodes:
            if color[nid] == WHITE:
                dfs(nid)

        return cycle_nodes

    def _compute_depths(
        self,
        nodes: Dict[str, KnowledgeNode],
        parents_map: Dict[str, List[str]],
    ) -> Dict[str, int]:
        """
        Computes depth for each node by traversing up to roots.
        Handles disconnected subtrees gracefully.
        """
        depth_cache: Dict[str, int] = {}

        def get_depth(nid: str, visited: Optional[Set[str]] = None) -> int:
            if nid in depth_cache:
                return depth_cache[nid]
            if visited is None:
                visited = set()
            if nid in visited:
                return 0  # Cycle guard
            visited.add(nid)

            parents = parents_map.get(nid, [])
            if not parents:
                depth_cache[nid] = 0
                return 0

            max_parent_depth = max(get_depth(pid, visited) for pid in parents if pid in nodes)
            depth_cache[nid] = max_parent_depth + 1
            return depth_cache[nid]

        for nid in nodes:
            get_depth(nid)

        return depth_cache


def validate_pre_weights(w1: float, w2: float, w3: float, w4: float, w5: float) -> bool:
    """
    Validates that PRE weights sum to 1.0 (Structural Invariant 4).
    Returns True if valid, or raises ValueError if violated.
    """
    weight_sum = w1 + w2 + w3 + w4 + w5
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"PRE weights must sum to 1.0 (Structural Invariant 4). "
            f"Got: w1={w1} + w2={w2} + w3={w3} + w4={w4} + w5={w5} = {weight_sum}"
        )
    return True
