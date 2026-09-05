"""
SHIA-RAG Layer 4: Knowledge Confidence Engine (KCE)
=====================================================
Computes and propagates confidence scores through the Knowledge Forest.

Confidence is computed from three signals:
1. Extraction confidence: How reliably the concept was extracted from text.
2. Evidence density: Number and quality of supporting Tier 1 blocks.
3. Topological propagation: Parent confidence flows down to children.

Historical Fix (Fatal Flaw 3 from Audit):
  Previous versions used naive multiplicative propagation which caused
  exponential confidence decay in deep trees. This version uses
  TOPOLOGICAL ORDER propagation with a damping factor, ensuring stable
  confidence values at all depths.

Fixes from PDF Structure Summary Audit:
  - Error #10: baseline_conf now defined as configurable prior Conf₀ = 0.50.
  - P4: Evidence frequency can be inflated by duplicate documents —
    evidence_count uses source_block_ids which are unique per block.

Known Limitations:
  - Confidence ≠ truth — a high-confidence parent can still be wrong.
  - Evidence frequency from duplicate sources can inflate scores.
  - Parent confidence propagation can reinforce wrong hierarchies.

Reference: Chapter 2.3 (Flaw 3 Resolution), SHIA_RAG_Complete_Report.md
"""

from __future__ import annotations

import logging
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Set

_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.layer0_data_model.schemas import EdgeCategory, KnowledgeEdge, KnowledgeNode

logger = logging.getLogger(__name__)


class KnowledgeConfidenceEngine:
    """
    Computes and propagates confidence scores across the Knowledge Forest.
    
    Uses topological order traversal (roots → leaves) with damped propagation
    to avoid the exponential decay bug identified in the project audit.
    """

    def __init__(
        self,
        damping_factor: float = 0.85,
        min_confidence: float = 0.10,
        baseline_conf: float = 0.50,
        extraction_weight: float = 0.50,
        evidence_weight: float = 0.30,
        parent_weight: float = 0.20,
    ):
        """
        Args:
            damping_factor: How much parent confidence flows to children.
                           0.85 means child gets 85% of parent signal.
            min_confidence: Floor confidence to prevent degenerate zero values.
            baseline_conf: Configurable prior confidence Conf₀ for nodes with
                          no extraction signal (Error #10 fix). Default 0.50.
            extraction_weight: Weight for the extraction confidence signal.
            evidence_weight: Weight for the evidence density signal.
            parent_weight: Weight for the parent propagation signal.
        """
        self.damping = damping_factor
        self.min_conf = min_confidence
        self.baseline_conf = baseline_conf
        self.w_extract = extraction_weight
        self.w_evidence = evidence_weight
        self.w_parent = parent_weight

        # Validate weights
        total = self.w_extract + self.w_evidence + self.w_parent
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"KCE weights must sum to 1.0, got {total}")

    def propagate_confidence(
        self,
        nodes: Dict[str, KnowledgeNode],
        edges: List[KnowledgeEdge],
    ) -> Dict[str, float]:
        """
        Propagates confidence scores from roots to leaves in topological order.

        This replaces the buggy naive multiplicative approach from prior specs.
        Uses Kahn's algorithm for topological sorting to ensure that every
        parent's confidence is finalized before computing its children's scores.

        Args:
            nodes: All KnowledgeNodes keyed by node_id.
            edges: All KnowledgeEdges.

        Returns:
            Map of node_id → final confidence score in [min_confidence, 1.0].
        """
        # Build child→parent and parent→children maps (hierarchical only)
        children_map: Dict[str, List[str]] = {nid: [] for nid in nodes}
        parent_map: Dict[str, str] = {}

        for edge in edges:
            if edge.category == EdgeCategory.HIERARCHICAL:
                child_id = edge.source_id
                parent_id = edge.target_id
                if child_id in nodes and parent_id in nodes:
                    children_map.setdefault(parent_id, []).append(child_id)
                    parent_map[child_id] = parent_id

        # Identify roots (nodes with no hierarchical parent)
        roots = [nid for nid in nodes if nid not in parent_map]

        # Topological order via BFS (Kahn's algorithm)
        topo_order = self._topological_sort(roots, children_map, nodes)

        # Propagate confidence in topological order
        confidence_map: Dict[str, float] = {}

        for nid in topo_order:
            node = nodes[nid]

            # Signal 1: Extraction confidence (intrinsic)
            # Use baseline_conf if node has no extraction confidence (Error #10 fix)
            extract_conf = node.confidence if node.confidence > 0.0 else self.baseline_conf

            # Signal 2: Evidence density (number of supporting blocks)
            evidence_count = len(node.source_block_ids)
            evidence_conf = min(1.0, evidence_count / 5.0)  # Saturates at 5 blocks

            # Signal 3: Parent propagation (damped)
            parent_id = parent_map.get(nid)
            if parent_id and parent_id in confidence_map:
                parent_conf = confidence_map[parent_id] * self.damping
            else:
                parent_conf = 1.0  # Roots get full parent signal

            # Weighted combination
            final_conf = (
                self.w_extract * extract_conf
                + self.w_evidence * evidence_conf
                + self.w_parent * parent_conf
            )

            # Apply floor
            final_conf = max(self.min_conf, min(1.0, final_conf))
            confidence_map[nid] = final_conf

        logger.info(
            f"KCE propagated confidence for {len(confidence_map)} nodes. "
            f"Range: [{min(confidence_map.values()):.3f}, {max(confidence_map.values()):.3f}]"
        )

        return confidence_map

    def _topological_sort(
        self,
        roots: List[str],
        children_map: Dict[str, List[str]],
        nodes: Dict[str, KnowledgeNode],
    ) -> List[str]:
        """
        BFS-based topological sort starting from root nodes.
        Ensures parents are processed before children.
        """
        order: List[str] = []
        visited: Set[str] = set()
        queue: deque[str] = deque(roots)

        while queue:
            nid = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            order.append(nid)

            for child_id in children_map.get(nid, []):
                if child_id not in visited and child_id in nodes:
                    queue.append(child_id)

        # Add any remaining unvisited nodes (disconnected components)
        for nid in nodes:
            if nid not in visited:
                order.append(nid)

        return order

    def compute_single_node_confidence(
        self,
        node: KnowledgeNode,
        parent_confidence: float = 1.0,
    ) -> float:
        """
        Computes confidence for a single node (used during incremental insertion).

        Args:
            node: The node to score.
            parent_confidence: The finalized confidence of this node's parent.

        Returns:
            Confidence score in [min_confidence, 1.0].
        """
        extract_conf = node.confidence
        evidence_conf = min(1.0, len(node.source_block_ids) / 5.0)
        parent_signal = parent_confidence * self.damping

        final = (
            self.w_extract * extract_conf
            + self.w_evidence * evidence_conf
            + self.w_parent * parent_signal
        )

        return max(self.min_conf, min(1.0, final))
