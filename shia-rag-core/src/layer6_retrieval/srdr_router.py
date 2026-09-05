"""
SHIA-RAG Layer 6: Self-Reflective Depth Router (SRDR)
======================================================
Routes user queries into one of four operational modes based on
query intent, linguistic complexity, and entity distribution.

Fixes from PDF Structure Summary Audit:
  - L6.1: Add word-boundary regex to prevent false positive substring matches.
  - Q1: Add formal Ψ(q) computation matching the mathematical formulation.
  - Q2: Add EstHops(q) estimator.
  - SRDR docstring: "Self-reflective" refers to automated rule-based query
    analysis — NOT an LLM-based reflection mechanism.

Known Limitations (documented per PDF audit):
  - Rule-based keyword detection is brittle (⚠️ Part 20).
  - Entity-count quality directly affects routing accuracy.
  - English-centric keyword vocabulary.
  - Formal Ψ(q) formulation is an approximation of reasoning complexity.
  - Incorrect routing can permanently remove relevant evidence (⚠️ Part 20).
  - Higher τ ≠ better answer — too much traversal introduces noise.

Reference: Chapter 8.4, SHIA_RAG_Complete_Report.md
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)


class EntityCounter:
    """
    Lightweight entity counter using pattern matching and optional spaCy NER.
    Provides the entity_count input required by SelfReflectiveDepthRouter.
    """

    def __init__(self, spacy_model: Optional[Any] = None):
        self._nlp = spacy_model

    def count_entities(self, query: str) -> int:
        """
        Counts distinct named entities in a query.
        Falls back to quoted-term and capitalized-word heuristic if spaCy unavailable.
        """
        if self._nlp is not None:
            try:
                doc = self._nlp(query)
                entities = {ent.text.lower() for ent in doc.ents}
                return len(entities)
            except Exception as e:
                logger.warning(f"spaCy NER failed, using fallback: {e}")

        return self._heuristic_count(query)

    @staticmethod
    def _heuristic_count(query: str) -> int:
        """Heuristic entity count: quoted terms + capitalized multi-word phrases."""
        entities: Set[str] = set()

        # Quoted terms
        for match in re.findall(r'"([^"]+)"', query):
            entities.add(match.lower().strip())
        for match in re.findall(r"'([^']+)'", query):
            entities.add(match.lower().strip())

        stopwords = {
            "what", "how", "why", "when", "where", "which",
            "the", "a", "an", "is", "are", "does", "do", "can",
            "and", "or", "in", "on", "at", "to", "for", "with", "of"
        }

        # Find capitalized sequences / phrases (excluding sentence starts)
        words = query.split()
        current_entity: List[str] = []

        for i, word in enumerate(words):
            clean = re.sub(r"[^\w]", "", word)
            if not clean:
                continue

            is_first_word = (i == 0)
            is_capitalized = clean[0].isupper() and clean.lower() not in stopwords
            is_acronym = clean.isupper() and len(clean) >= 2

            if (is_capitalized and not is_first_word) or is_acronym:
                current_entity.append(clean.lower())
            else:
                if current_entity:
                    entities.add(" ".join(current_entity))
                    current_entity = []

        if current_entity:
            entities.add(" ".join(current_entity))

        return max(0, len(entities))


class SelfReflectiveDepthRouter:
    """
    Routes user queries into one of four operational modes based on
    query intent, linguistic complexity, and entity distribution.

    "Self-reflective" refers to the router's ability to automatically
    analyze query characteristics and adapt retrieval strategy — this
    is implemented via rule-based query analysis, NOT via an explicit
    LLM self-reflection mechanism. This design choice prioritizes
    deterministic, low-latency routing over LLM inference overhead.

    Includes formal Ψ(q) query complexity scoring as defined in §7.5
    of the specification.
    """

    # Use compiled word-boundary regex to prevent false positives (L6.1 fix)
    # e.g., "differences" should not match "differ" without word boundary
    _COMPARATIVE_PATTERN = re.compile(
        r'\b(compare[ds]?|vs\.?|versus|difference[s]?|differ[s]?|contrast[s]?|'
        r'better|worse|advantage[s]?|disadvantage[s]?)\b',
        re.IGNORECASE
    )
    _THEMATIC_PATTERN = re.compile(
        r'\b(overview|summarize|summarise|landscape|survey|theme[s]?|'
        r'all\s+(?:about|types|kinds)|explain\s+(?:all|everything))\b',
        re.IGNORECASE
    )
    _MULTIHOP_PATTERN = re.compile(
        r'\b(how\s+does\s+\w+\s+affect|relationship\s+between|'
        r'connected\s+to|leads?\s+to|causes?|impact[s]?\s+on)\b',
        re.IGNORECASE
    )

    def __init__(
        self,
        spacy_model: Optional[Any] = None,
        # Ψ(q) formula weights from §7.5 (Q4)
        w_entity: float = 0.35,
        w_comparative: float = 0.35,
        w_hops: float = 0.30,
    ):
        self.entity_counter = EntityCounter(spacy_model)
        self.w_entity = w_entity
        self.w_comparative = w_comparative
        self.w_hops = w_hops

    def route(self, query: str, entity_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Routes user query into one of four operational modes.

        Args:
            query: The user's natural language query.
            entity_count: Optional override. If None, auto-detected via EntityCounter.

        Returns:
            Dict with mode, max_depth (τ), strategy, traversal flags,
            and the formal Ψ(q) complexity score.
        """
        if entity_count is None:
            entity_count = self.entity_counter.count_entities(query)

        # L6.1 FIX: Use compiled word-boundary regex instead of raw substring match
        has_comparison = bool(self._COMPARATIVE_PATTERN.search(query))
        has_thematic = bool(self._THEMATIC_PATTERN.search(query))
        has_multihop = bool(self._MULTIHOP_PATTERN.search(query))

        # Q1, Q2 FIX: Compute formal Ψ(q) matching §7.5 mathematical formulation
        psi_score = self._compute_psi(query, entity_count, has_comparison, has_multihop)

        # Map Ψ(q) to τ (traversal depth) — per §7.5 mapping
        tau = self._map_psi_to_tau(psi_score)

        if has_thematic:
            mode = {
                "mode": "MODE_1_THEMATIC",
                "max_depth": 1,
                "strategy": "ROOT_BREADTH_FIRST",
                "include_parents": False,
                "include_children": True,
                "include_crosslinks": False,
                "psi_score": psi_score,
                "tau_override": 1,
            }
        elif has_comparison or entity_count >= 2:
            mode = {
                "mode": "MODE_3_MULTIHOP_COMPARATIVE",
                "max_depth": max(3, tau),  # Use τ but minimum 3 for comparisons
                "strategy": "DUAL_SUBTREE_BIDIRECTIONAL",
                "include_parents": True,
                "include_children": True,
                "include_crosslinks": True,
                "psi_score": psi_score,
                "tau_override": tau,
            }
        elif entity_count == 1:
            mode = {
                "mode": "MODE_2_FACTUAL_NEEDLE",
                "max_depth": max(1, tau),
                "strategy": "LEAF_TO_ROOT",
                "include_parents": True,
                "include_children": False,
                "include_crosslinks": False,
                "psi_score": psi_score,
                "tau_override": tau,
            }
        else:
            mode = {
                "mode": "MODE_4_PARAMETRIC",
                "max_depth": 0,
                "strategy": "SKIP_RETRIEVAL",
                "include_parents": False,
                "include_children": False,
                "include_crosslinks": False,
                "psi_score": psi_score,
                "tau_override": 0,
            }

        logger.info(
            f"SRDR routed query to {mode['mode']} "
            f"(entities={entity_count}, Ψ={psi_score:.3f}, τ={mode['max_depth']})"
        )
        return mode

    # Alias for convenience and backward compatibility
    route_query = route

    def _compute_psi(
        self,
        query: str,
        entity_count: int,
        has_comparison: bool,
        has_multihop: bool,
    ) -> float:
        """
        Computes the formal query complexity score Ψ(q) per §7.5:

        Ψ(q) = w_entity * S_entity(q) + w_comparative * S_comp(q) + w_hops * EstHops(q)

        where:
        - S_entity(q) = min(|E_q| / 3, 1.0) — normalized entity count
        - S_comp(q) ∈ {0, 1} — binary comparative indicator
        - EstHops(q) ∈ [0, 1] — estimated reasoning hops (Q2 fix)
        """
        # S_entity: normalized entity count (saturates at 3)
        s_entity = min(entity_count / 3.0, 1.0)

        # S_comp: binary comparative flag
        s_comp = 1.0 if has_comparison else 0.0

        # EstHops: estimated reasoning hops (Q2 fix — was under-specified)
        est_hops = self._estimate_hops(query, entity_count, has_multihop)

        psi = (
            self.w_entity * s_entity
            + self.w_comparative * s_comp
            + self.w_hops * est_hops
        )

        return min(1.0, max(0.0, psi))

    def _estimate_hops(
        self,
        query: str,
        entity_count: int,
        has_multihop: bool,
    ) -> float:
        """
        Estimates the number of reasoning hops required for a query (Q2 fix).

        Heuristic based on:
        - Multi-hop language patterns (causal chains, relationships)
        - Query word count (longer queries tend to be more complex)
        - Entity count (more entities → more potential hops)

        Returns a normalized score in [0, 1].
        """
        hop_score = 0.0

        # Multi-hop pattern presence
        if has_multihop:
            hop_score += 0.5

        # Query length proxy (more words → potentially more complex)
        word_count = len(query.split())
        if word_count > 15:
            hop_score += 0.25
        elif word_count > 8:
            hop_score += 0.10

        # Multiple entities suggest multi-hop reasoning
        if entity_count >= 3:
            hop_score += 0.25
        elif entity_count >= 2:
            hop_score += 0.15

        return min(1.0, hop_score)

    @staticmethod
    def _map_psi_to_tau(psi_score: float) -> int:
        """
        Maps Ψ(q) → τ (traversal depth) per §7.5:

        Ψ ∈ [0.0, 0.25) → τ = 1 (shallow)
        Ψ ∈ [0.25, 0.50) → τ = 2 (moderate)
        Ψ ∈ [0.50, 0.75) → τ = 3 (deep)
        Ψ ∈ [0.75, 1.00] → τ = 4 (maximum)
        """
        if psi_score >= 0.75:
            return 4
        elif psi_score >= 0.50:
            return 3
        elif psi_score >= 0.25:
            return 2
        else:
            return 1
