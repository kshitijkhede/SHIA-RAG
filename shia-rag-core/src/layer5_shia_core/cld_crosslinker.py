"""
SHIA-RAG Layer 5: Cross-Link Discovery (CLD)
==============================================
Discovers and creates semantic cross-links between KnowledgeNodes
in the Concept Forest that are NOT related hierarchically.

Semantic edge types discovered:
- USES: concept A utilizes/depends on concept B
- CAUSES: concept A triggers/enables concept B  
- COMPARED_TO: concepts A and B are frequently contrasted
- PREREQUISITE_OF: concept A must be understood before B
- CONTRADICTS: claims from different sources conflict

Discovery methods:
1. Embedding proximity (high cosine sim but different subtrees)
2. Co-occurrence in the same document blocks
3. LLM-based relation extraction for candidate pairs
4. Pattern-based detection (comparative language, causal connectors)

Reference: Chapter 6.6 (Layer 5), SHIA_RAG_Complete_Report.md
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.layer0_data_model.schemas import (
    EdgeCategory,
    KnowledgeEdge,
    KnowledgeNode,
    SemanticRelation,
)

logger = logging.getLogger(__name__)


# ── Linguistic patterns for relation detection ──

CAUSAL_PATTERNS = [
    r"\b(causes?|triggers?|leads?\s+to|results?\s+in|enables?|induces?)\b",
    r"\b(because|therefore|consequently|as\s+a\s+result)\b",
]

COMPARATIVE_PATTERNS = [
    r"\b(compare[ds]?\s+(?:to|with)|versus|vs\.?|in\s+contrast|unlike|differs?\s+from)\b",
    r"\b(similar\s+to|analogous|equivalent)\b",
]

PREREQUISITE_PATTERNS = [
    r"\b(requires?|prerequisite|depends?\s+on|assumes?\s+knowledge\s+of|builds?\s+upon)\b",
    r"\b(before\s+understanding|must\s+first)\b",
]

USAGE_PATTERNS = [
    r"\b(uses?|utilizes?|employs?|leverages?|relies?\s+on|invokes?)\b",
    r"\b(implemented\s+(?:using|with|via)|based\s+on)\b",
]


class CrossLinkDiscovery:
    """
    Discovers semantic cross-links between KnowledgeNodes
    that exist in different subtrees of the concept forest.
    """

    def __init__(
        self,
        cosine_threshold: float = 0.70,
        cooccurrence_threshold: int = 2,
        max_crosslinks_per_node: int = 10,
        max_expansion_candidates: int = 500,
    ):
        self.cosine_threshold = cosine_threshold
        self.cooccurrence_threshold = cooccurrence_threshold
        self.max_crosslinks_per_node = max_crosslinks_per_node
        # L6.3 FIX: Cap total candidates to prevent cross-link explosion
        self.max_expansion_candidates = max_expansion_candidates

    def discover_crosslinks(
        self,
        nodes: Dict[str, KnowledgeNode],
        existing_edges: List[KnowledgeEdge],
        embeddings: Dict[str, np.ndarray],
        block_to_nodes: Optional[Dict[str, List[str]]] = None,
    ) -> List[KnowledgeEdge]:
        """
        Main entry point: discovers semantic cross-links across the forest.

        Args:
            nodes: All KnowledgeNodes in the forest.
            existing_edges: All existing edges (hierarchical + semantic).
            embeddings: Map of node_id → embedding vector.
            block_to_nodes: Map of block_id → list of node_ids extracted from it.

        Returns:
            List of new KnowledgeEdge objects (category=SEMANTIC).
        """
        # Build a set of existing edge pairs to avoid duplicates
        existing_pairs: Set[Tuple[str, str]] = set()
        for edge in existing_edges:
            existing_pairs.add((edge.source_id, edge.target_id))
            existing_pairs.add((edge.target_id, edge.source_id))

        # Build subtree membership to avoid linking nodes in the same subtree
        subtree_roots = self._compute_subtree_roots(nodes, existing_edges)

        new_edges: List[KnowledgeEdge] = []
        crosslink_counts: Dict[str, int] = {nid: 0 for nid in nodes}

        # Method 1: Embedding proximity across different subtrees
        proximity_candidates = self._find_proximity_candidates(
            nodes, embeddings, subtree_roots, existing_pairs
        )

        # Method 2: Co-occurrence in same document blocks
        cooccurrence_candidates = self._find_cooccurrence_candidates(
            nodes, block_to_nodes, subtree_roots, existing_pairs
        ) if block_to_nodes else []

        # Merge and deduplicate candidates
        all_candidates: Dict[Tuple[str, str], float] = {}
        for src, tgt, score in proximity_candidates:
            key = (min(src, tgt), max(src, tgt))
            all_candidates[key] = max(all_candidates.get(key, 0), score)

        for src, tgt, score in cooccurrence_candidates:
            key = (min(src, tgt), max(src, tgt))
            all_candidates[key] = max(all_candidates.get(key, 0), score)

        # Classify and create edges (with expansion cap enforcement — L6.3 fix)
        candidates_processed = 0
        for (src, tgt), score in sorted(all_candidates.items(), key=lambda x: -x[1]):
            if crosslink_counts[src] >= self.max_crosslinks_per_node:
                continue
            if crosslink_counts[tgt] >= self.max_crosslinks_per_node:
                continue
            # L6.3 FIX: Enforce total candidate expansion cap
            if candidates_processed >= self.max_expansion_candidates:
                logger.warning(
                    f"Reached max expansion limit ({self.max_expansion_candidates}). "
                    f"Stopping cross-link discovery to prevent explosion."
                )
                break

            relation = self._classify_relation(nodes[src], nodes[tgt])

            edge = KnowledgeEdge(
                source_id=src,
                target_id=tgt,
                category=EdgeCategory.SEMANTIC,
                predicate=relation.value,
                extraction_confidence=min(score, 1.0),
                extraction_method="crosslink_discovery",
            )
            new_edges.append(edge)
            crosslink_counts[src] += 1
            crosslink_counts[tgt] += 1
            candidates_processed += 1

        logger.info(f"CLD discovered {len(new_edges)} new semantic cross-links.")
        return new_edges

    def _compute_subtree_roots(
        self,
        nodes: Dict[str, KnowledgeNode],
        edges: List[KnowledgeEdge],
    ) -> Dict[str, str]:
        """
        Computes the root of each node's subtree to determine
        whether two nodes are in different subtrees.
        """
        parent_map: Dict[str, str] = {}
        for edge in edges:
            if edge.category == EdgeCategory.HIERARCHICAL:
                parent_map[edge.source_id] = edge.target_id

        subtree_root: Dict[str, str] = {}

        def find_root(nid: str, visited: Optional[Set[str]] = None) -> str:
            if nid in subtree_root:
                return subtree_root[nid]
            if visited is None:
                visited = set()
            if nid in visited:
                return nid  # Cycle guard
            visited.add(nid)

            parent = parent_map.get(nid)
            if parent is None or parent not in nodes:
                subtree_root[nid] = nid
                return nid
            root = find_root(parent, visited)
            subtree_root[nid] = root
            return root

        for nid in nodes:
            find_root(nid)

        return subtree_root

    def _find_proximity_candidates(
        self,
        nodes: Dict[str, KnowledgeNode],
        embeddings: Dict[str, np.ndarray],
        subtree_roots: Dict[str, str],
        existing_pairs: Set[Tuple[str, str]],
    ) -> List[Tuple[str, str, float]]:
        """
        Finds candidate cross-links based on high embedding cosine similarity
        between nodes in DIFFERENT subtrees.
        """
        candidates: List[Tuple[str, str, float]] = []
        node_ids = [nid for nid in nodes if nid in embeddings]

        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                nid_a, nid_b = node_ids[i], node_ids[j]

                # Skip if same subtree
                if subtree_roots.get(nid_a) == subtree_roots.get(nid_b):
                    continue

                # Skip if edge already exists
                if (nid_a, nid_b) in existing_pairs:
                    continue

                # Compute cosine similarity
                emb_a = embeddings[nid_a]
                emb_b = embeddings[nid_b]
                cos_sim = float(
                    np.dot(emb_a, emb_b)
                    / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-9)
                )

                if cos_sim >= self.cosine_threshold:
                    candidates.append((nid_a, nid_b, cos_sim))

        return candidates

    def _find_cooccurrence_candidates(
        self,
        nodes: Dict[str, KnowledgeNode],
        block_to_nodes: Dict[str, List[str]],
        subtree_roots: Dict[str, str],
        existing_pairs: Set[Tuple[str, str]],
    ) -> List[Tuple[str, str, float]]:
        """
        Finds candidate cross-links based on co-occurrence
        in the same document blocks (Tier 1 proximity).
        """
        cooccurrence_counts: Dict[Tuple[str, str], int] = {}

        for block_id, node_ids in block_to_nodes.items():
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    a, b = node_ids[i], node_ids[j]
                    if a not in nodes or b not in nodes:
                        continue
                    if subtree_roots.get(a) == subtree_roots.get(b):
                        continue
                    if (a, b) in existing_pairs:
                        continue

                    key = (min(a, b), max(a, b))
                    cooccurrence_counts[key] = cooccurrence_counts.get(key, 0) + 1

        candidates = []
        for (a, b), count in cooccurrence_counts.items():
            if count >= self.cooccurrence_threshold:
                # Normalize to [0, 1] score
                score = min(1.0, count / 5.0)
                candidates.append((a, b, score))

        return candidates

    def _classify_relation(
        self,
        source_node: KnowledgeNode,
        target_node: KnowledgeNode,
    ) -> SemanticRelation:
        """
        Classifies the semantic relation type between two nodes
        using pattern matching on their definition/evidence texts.

        Falls back to COMPARED_TO if no specific pattern matches.
        """
        combined_text = " ".join([
            source_node.definition_text,
            target_node.definition_text,
            " ".join(source_node.evidence_texts[:3]),
            " ".join(target_node.evidence_texts[:3]),
        ]).lower()

        # Check patterns in priority order
        for pattern in CAUSAL_PATTERNS:
            if re.search(pattern, combined_text):
                return SemanticRelation.CAUSES

        for pattern in PREREQUISITE_PATTERNS:
            if re.search(pattern, combined_text):
                return SemanticRelation.PREREQUISITE_OF

        for pattern in USAGE_PATTERNS:
            if re.search(pattern, combined_text):
                return SemanticRelation.USES

        for pattern in COMPARATIVE_PATTERNS:
            if re.search(pattern, combined_text):
                return SemanticRelation.COMPARED_TO

        # Default: COMPARED_TO (two related concepts in different subtrees)
        return SemanticRelation.COMPARED_TO
