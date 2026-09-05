"""
SHIA-RAG Layer 7: Claim Attribution Verifier
==============================================
Extracts claim-level statements from generated answers and verifies
factual consistency against E_proj text anchors.

Returns an attribution reward R in [0.0, 1.0] for Thompson Sampling feedback.

Fixes from PDF Structure Summary Audit:
  - L7.1: Check ALL citations for each claim (not just first match).
  - L7.2: Add negation/contradiction detection before lexical overlap.
  - L7.3: Sub-sentence claim splitting for compound sentences.
  - L7.4: Citation presence ≠ citation correctness — documented.
  - L7.7: Reward is continuous R∈[0,1] (matching implementation, not binary).

Known Limitations (documented per PDF audit):
  - Lexical overlap is a weak substitute for semantic entailment (L7.1).
  - Contradictions can have high lexical overlap (L7.2).
  - Verifier errors can poison Layer 8 learning (L7.5).
  - Source correctness itself is not established by this verifier (L7.6).
  - NLI model details need separate evaluation (L7.8).

Reference: Chapter 8.6, SHIA_RAG_Complete_Report.md
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)

# Negation cues for contradiction detection (L7.2 fix)
NEGATION_PATTERNS = [
    r"\b(not|no|never|neither|nor|cannot|can't|won't|doesn't|didn't|isn't|aren't|wasn't|weren't)\b",
    r"\b(hardly|barely|scarcely|seldom|rarely|impossible|incorrect|invalid|false|wrong|untrue)\b",
    r"\b(deny|denies|denied|refuse|refused|reject|rejected|contradict|contradicts)\b",
]

# Clause splitting delimiters for compound sentence handling (L7.3 fix)
CLAUSE_DELIMITERS = r'(?:;\s+|\s+and\s+|\s+but\s+|\s+however\s+|\s+whereas\s+|\s+while\s+|\s+although\s+)'


class ClaimAttributionVerifier:
    """
    Verifies that generated claims are attributable to the retrieved
    KnowledgeNode evidence (E_proj anchors).

    NOTE: This verifier checks attribution — whether a claim can be traced
    back to source evidence. It does NOT guarantee that the source evidence
    itself is correct, nor does it fully "remove" hallucination. Attribution
    verification is a necessary but not sufficient condition for factual
    correctness. (L7.4, L7.6 acknowledgement)
    """

    def __init__(
        self,
        nli_model: Optional[Any] = None,
        overlap_threshold: float = 0.50,
        contradiction_sensitivity: float = 0.30,
    ):
        """
        Args:
            nli_model: Optional NLI model (e.g., DeBERTa-v3) for entailment checking.
                       Falls back to lexical overlap if None.
            overlap_threshold: Minimum word overlap ratio for lexical fallback.
            contradiction_sensitivity: Threshold for negation pattern density
                                        that flags a potential contradiction.
        """
        self.nli_model = nli_model
        self.overlap_threshold = overlap_threshold
        self.contradiction_sensitivity = contradiction_sensitivity

    def verify_generation(
        self,
        generated_answer: str,
        selected_nodes: Dict[str, Any],
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Extracts claim-level statements and verifies factual consistency.

        Args:
            generated_answer: The LLM's generated response text.
            selected_nodes: Map of node_id → node data dict with 'evidence' field.

        Returns:
            Tuple of (attribution_reward, attribution_records).
            attribution_reward is continuous R ∈ [0.0, 1.0] — NOT binary.
            This continuous reward reflects the proportion of verified claims.
        """
        # Step 1: Extract sentence-level claims
        sentences = [
            s.strip()
            for s in re.split(r'(?<=[.!?])\s+', generated_answer)
            if len(s.strip()) > 5
        ]

        if not sentences:
            logger.warning("No sentences found in generated answer.")
            return 0.0, []

        # Step 2: Split compound sentences into sub-claims (L7.3 fix)
        claims = []
        for sentence in sentences:
            sub_claims = self._split_compound_claims(sentence)
            for claim in sub_claims:
                claims.append({"text": claim, "original_sentence": sentence})

        if not claims:
            logger.warning("No claims extracted after splitting.")
            return 0.0, []

        verified_count = 0
        attribution_records: List[Dict[str, Any]] = []

        for claim_info in claims:
            claim_text = claim_info["text"]

            # Extract citation tags e.g. [KN-000456], [KN-CS001]
            # Also check the original sentence for citations
            citations = re.findall(r'\[(KN-[A-Za-z0-9_]+)\]', claim_text)
            if not citations:
                citations = re.findall(
                    r'\[(KN-[A-Za-z0-9_]+)\]', claim_info["original_sentence"]
                )

            is_supported = False
            is_contradicted = False
            matched_node = None
            all_checked_nodes: List[str] = []

            if citations:
                # L7.1 FIX: Check ALL citations, not just the first match
                for cite_id in citations:
                    if cite_id in selected_nodes:
                        all_checked_nodes.append(cite_id)
                        node_data = selected_nodes[cite_id]
                        evidence_list = node_data.get("evidence", [])
                        node_evidence = " ".join(
                            [a.get("text", "") if isinstance(a, dict) else str(a)
                             for a in evidence_list]
                        )

                        # L7.2 FIX: Check for contradiction BEFORE entailment
                        if self._check_contradiction(
                            premise=node_evidence, hypothesis=claim_text
                        ):
                            is_contradicted = True
                            matched_node = cite_id
                            # Don't break — continue checking other citations

                        elif self._check_entailment(
                            premise=node_evidence, hypothesis=claim_text
                        ):
                            is_supported = True
                            matched_node = cite_id
                            # Don't break — continue to check all citations
                            # (a later citation might contradict)

            # Contradictions override support
            if is_contradicted:
                is_supported = False

            if is_supported:
                verified_count += 1

            attribution_records.append({
                "claim": claim_text,
                "original_sentence": claim_info["original_sentence"],
                "citations": citations,
                "supported": is_supported,
                "contradicted": is_contradicted,
                "grounded_node": matched_node,
                "all_checked_nodes": all_checked_nodes,
            })

        # Continuous reward R ∈ [0.0, 1.0] (L7.7 — matches implementation)
        overall_reward = float(verified_count) / max(1, len(claims))

        logger.info(
            f"Attribution verification: {verified_count}/{len(claims)} claims supported "
            f"(reward={overall_reward:.2f})"
        )

        return overall_reward, attribution_records

    def _split_compound_claims(self, sentence: str) -> List[str]:
        """
        Splits compound sentences into individual sub-claims (L7.3 fix).

        For example:
          "A uses B and C relies on D" → ["A uses B", "C relies on D"]

        Returns the original sentence if no clause delimiter is found,
        or if the resulting sub-claims are too short to be meaningful.
        """
        parts = re.split(CLAUSE_DELIMITERS, sentence)
        sub_claims = [p.strip() for p in parts if len(p.strip()) > 10]

        if len(sub_claims) <= 1:
            return [sentence]  # No meaningful split

        return sub_claims

    def _check_contradiction(self, premise: str, hypothesis: str) -> bool:
        """
        Checks if the premise contradicts the hypothesis (L7.2 fix).

        Uses NLI model if available, otherwise falls back to
        negation-aware lexical heuristic.
        """
        if self.nli_model is not None:
            try:
                result = self.nli_model(premise=premise, hypothesis=hypothesis)
                return result.get("label", "").lower() == "contradiction"
            except Exception as e:
                logger.warning(f"NLI contradiction check failed, using fallback: {e}")

        # Heuristic: high lexical overlap + negation patterns = likely contradiction
        if not premise or not hypothesis:
            return False

        hypo_words = set(re.findall(r'\w+', hypothesis.lower()))
        premise_words = set(re.findall(r'\w+', premise.lower()))
        overlap = len(hypo_words & premise_words) / max(1, len(hypo_words))

        # Check negation patterns in hypothesis that aren't in premise (or vice versa)
        hypo_negations = sum(
            1 for pattern in NEGATION_PATTERNS
            if re.search(pattern, hypothesis.lower())
        )
        premise_negations = sum(
            1 for pattern in NEGATION_PATTERNS
            if re.search(pattern, premise.lower())
        )

        # Contradiction heuristic: high overlap + different negation polarity
        negation_mismatch = abs(hypo_negations - premise_negations) > 0
        return overlap >= self.overlap_threshold and negation_mismatch

    def _check_entailment(self, premise: str, hypothesis: str) -> bool:
        """
        Checks if the premise entails the hypothesis.
        Uses NLI model if available, otherwise falls back to lexical overlap.

        Known limitation: Lexical overlap is a weak proxy for semantic
        entailment. Two sentences can share words but express different
        or even contradictory meanings. (L7.1 acknowledgement)
        """
        if self.nli_model is not None:
            try:
                result = self.nli_model(premise=premise, hypothesis=hypothesis)
                return result.get("label", "").lower() == "entailment"
            except Exception as e:
                logger.warning(f"NLI model failed, using lexical fallback: {e}")

        # Lexical overlap fallback
        if not premise:
            return False
        hypo_words = set(re.findall(r'\w+', hypothesis.lower()))
        premise_words = set(re.findall(r'\w+', premise.lower()))
        overlap = len(hypo_words & premise_words) / max(1, len(hypo_words))
        return overlap >= self.overlap_threshold
