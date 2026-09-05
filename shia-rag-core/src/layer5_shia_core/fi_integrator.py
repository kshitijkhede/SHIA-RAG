"""
SHIA-RAG Layer 5: Forest Integrator (FI)
=========================================
Responsible for the final placement of a KnowledgeNode into the
Semantic Concept Forest after it has been validated by HV.

Key responsibilities:
1. Place the node under its highest-ranked valid parent.
2. If NO valid parent exists (all candidates rejected by HV),
   promote the node to a new domain root.
3. Update depth maps, sibling lists, and parent pointers.
4. Trigger Tier 1 ↔ Tier 2 projection anchor (E_proj) creation.

Reference: Chapter 6.6 (Layer 5), SHIA_RAG_Complete_Report.md
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.layer0_data_model.schemas import (
    EdgeCategory,
    HierarchicalRelation,
    KnowledgeEdge,
    KnowledgeNode,
)

logger = logging.getLogger(__name__)


class ForestIntegrator:
    """
    Integrates validated KnowledgeNodes into the Semantic Concept Forest.
    
    Called after the CPG++ → PRE → HV pipeline has produced a ranked,
    validated list of candidate parents for each new node.
    """

    def __init__(
        self,
        max_depth: int = 8,
        min_confidence_for_root: float = 0.60,
    ):
        self.max_depth = max_depth
        self.min_confidence_for_root = min_confidence_for_root

    def integrate_node(
        self,
        node: KnowledgeNode,
        ranked_valid_parents: Sequence[Tuple[Optional[str], float]],
        forest_nodes: Dict[str, KnowledgeNode],
        forest_edges: List[KnowledgeEdge],
        node_depth_map: Dict[str, int],
    ) -> Tuple[KnowledgeNode, Optional[KnowledgeEdge]]:
        """
        Places a node into the forest under the best valid parent,
        or promotes it to a root if no valid parent exists.

        Args:
            node: The KnowledgeNode to integrate.
            ranked_valid_parents: List of (parent_id, score) from PRE+HV,
                                  sorted descending by score. Only includes
                                  parents that passed HV validation.
            forest_nodes: Current map of node_id → KnowledgeNode.
            forest_edges: Current list of all edges in the forest.
            node_depth_map: Current map of node_id → depth.

        Returns:
            Tuple of (updated_node, new_edge_or_None).
            If promoted to root, new_edge is None.
        """
        if ranked_valid_parents:
            # Case 1: Place under the highest-ranked valid parent
            # Error #23 FIX: Null safety guard + deterministic UUID tie-breaker
            # If multiple parents have the same score, use node_id as tie-breaker
            # to ensure deterministic placement
            sorted_parents = sorted(
                ranked_valid_parents,
                key=lambda p: (-p[1], p[0] if p[0] is not None else ""),  # Descending score, ascending ID for ties
            )
            best_parent_id, best_score = sorted_parents[0]

            if best_parent_id is None:
                logger.warning(
                    f"Best parent ID is None for node '{node.node_id}'. "
                    f"Promoting to root."
                )
                return self._promote_to_root(node, forest_nodes, node_depth_map)

            parent_node = forest_nodes.get(best_parent_id)

            if parent_node is None:
                logger.warning(
                    f"Best parent '{best_parent_id}' not found in forest. "
                    f"Promoting node '{node.node_id}' to root."
                )
                return self._promote_to_root(node, forest_nodes, node_depth_map)

            # Update node metadata
            parent_depth = node_depth_map.get(best_parent_id, 0)
            new_depth = parent_depth + 1

            updated_node = node.model_copy(update={
                "parent_id": best_parent_id,
                "depth": new_depth,
            })

            # Create hierarchical edge: child → parent (IS_A)
            new_edge = KnowledgeEdge(
                source_id=node.node_id,
                target_id=best_parent_id,
                category=EdgeCategory.HIERARCHICAL,
                predicate=HierarchicalRelation.IS_A.value,
                extraction_confidence=best_score,
                extraction_method="pre_ranking",
            )

            # Register in maps
            forest_nodes[node.node_id] = updated_node
            forest_edges.append(new_edge)
            node_depth_map[node.node_id] = new_depth

            logger.info(
                f"Integrated '{updated_node.canonical_name}' (depth={new_depth}) "
                f"under parent '{parent_node.canonical_name}' (score={best_score:.3f})"
            )

            return updated_node, new_edge

        else:
            # Case 2: No valid parent — promote to domain root
            logger.info(
                f"No valid parents for '{node.canonical_name}'. "
                f"Promoting to domain root."
            )
            return self._promote_to_root(node, forest_nodes, node_depth_map)

    def _promote_to_root(
        self,
        node: KnowledgeNode,
        forest_nodes: Dict[str, KnowledgeNode],
        node_depth_map: Dict[str, int],
    ) -> Tuple[KnowledgeNode, None]:
        """
        Promotes a node to a new domain root (depth=0, no parent).
        
        This is invoked when:
        - All candidate parents were rejected by HV (cycles, depth violations).
        - The node represents a genuinely new domain concept.
        - The node's confidence is below min_confidence_for_root (logged as warning).
        """
        if node.confidence < self.min_confidence_for_root:
            logger.warning(
                f"Node '{node.canonical_name}' promoted to root with low confidence "
                f"({node.confidence:.2f} < {self.min_confidence_for_root}). "
                f"Consider manual review."
            )

        root_node = node.model_copy(update={
            "parent_id": None,
            "depth": 0,
            "abstraction_level": max(node.abstraction_level, 0.8),  # Roots are abstract
        })

        forest_nodes[node.node_id] = root_node
        node_depth_map[node.node_id] = 0

        logger.info(f"Created new domain root: '{root_node.canonical_name}' [{root_node.node_id}]")

        return root_node, None

    def compute_sibling_coherence(
        self,
        parent_id: str,
        forest_nodes: Dict[str, KnowledgeNode],
        forest_edges: List[KnowledgeEdge],
        embedding_store: Dict[str, list],
    ) -> float:
        """
        Computes the mean pairwise cosine similarity among all children
        of a given parent node. Used for quality monitoring.

        Returns:
            Mean cosine similarity in [0, 1], or 1.0 if <= 1 child.
        """
        import numpy as np

        child_ids = [
            e.source_id for e in forest_edges
            if e.target_id == parent_id and e.category == EdgeCategory.HIERARCHICAL
        ]

        if len(child_ids) <= 1:
            return 1.0

        embeddings = []
        for cid in child_ids:
            emb = embedding_store.get(cid)
            if emb is not None:
                embeddings.append(np.array(emb))

        if len(embeddings) <= 1:
            return 1.0

        # Compute mean pairwise cosine similarity
        total_sim = 0.0
        pair_count = 0
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                cos_sim = float(
                    np.dot(embeddings[i], embeddings[j])
                    / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-9)
                )
                total_sim += cos_sim
                pair_count += 1

        return total_sim / max(1, pair_count)
