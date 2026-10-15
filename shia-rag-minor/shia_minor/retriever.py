"""
Module 9 --- Hierarchy-aware retriever.

Given a natural-language query, finds the top-k most relevant concepts using
dense (or TF-IDF) embeddings, then enriches each result with its ancestor
chain context — the full root→node lineage that a hierarchical retriever would
attach to the prompt.

This is the 'R' in RAG — the component that makes the induced hierarchy
useful for question answering.

Reference: report Chapter 3 (Retrieval-Augmented Generation Layer).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .schemas import ConceptNode
from .embedder import Embedder


@dataclass
class RetrievalResult:
    """A single retrieved concept together with its hierarchy context."""
    concept_id: str
    name: str
    score: float
    confidence: float
    section_number: str
    depth: int
    ancestor_path: str              # "Root → Ch 3 → §3.1 → ..."
    body_snippet: str               # first ~400 chars of body text
    context: str                    # full formatted context for LLM/display


class HierarchyRetriever:
    """Semantic retrieval over a concept hierarchy.

    Usage::

        ret = HierarchyRetriever(embedder)
        ret.index(concepts)
        results = ret.query("What is the gutter algorithm?")
    """

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self._concepts: List[ConceptNode] = []
        self._doc_embs: Optional[np.ndarray] = None
        self._nodes: Dict[str, ConceptNode] = {}

    @property
    def is_indexed(self) -> bool:
        return self._doc_embs is not None and len(self._concepts) > 0

    # ------------------------------------------------------------------
    def index(self, concepts: List[ConceptNode]) -> None:
        """Build the retrieval index from a list of concept nodes."""
        self._concepts = list(concepts)
        self._nodes = {c.concept_id: c for c in concepts}
        # Embed using name + definition + body text for rich representations
        texts: List[str] = []
        for c in concepts:
            parts = [c.name]
            if c.definition:
                parts.append(c.definition)
            if c.body_text:
                parts.append(c.body_text[:500])
            texts.append(". ".join(parts))
        self._doc_embs = self.embedder.embed(texts)

    # ------------------------------------------------------------------
    def query(self, question: str, top_k: int = 5,
              confidence_weight: float = 0.1) -> List[RetrievalResult]:
        """Retrieve the top-*k* concepts most relevant to *question*.

        ``score = semantic_similarity + confidence_weight × KCE_confidence``
        so high-confidence nodes get a small boost.
        """
        if not self.is_indexed:
            return []
        q_emb = self.embedder.embed_query(question)
        raw = Embedder.cosine_scores(q_emb, self._doc_embs)  # type: ignore[arg-type]

        # Blend semantic score with KCE confidence
        confs = np.array([c.confidence for c in self._concepts])
        scores = raw + confidence_weight * confs

        top_idx = scores.argsort()[::-1][:top_k]
        results: List[RetrievalResult] = []
        for i in top_idx:
            c = self._concepts[int(i)]
            ancestors = self._ancestor_chain(c)
            context = self._build_context(c, ancestors)
            path = self._format_path(c, ancestors)
            results.append(RetrievalResult(
                concept_id=c.concept_id,
                name=c.name,
                score=round(float(scores[int(i)]), 4),
                confidence=round(c.confidence, 3),
                section_number=c.section_number or "",
                depth=c.depth,
                ancestor_path=path,
                body_snippet=(c.body_text or "")[:400],
                context=context,
            ))
        return results

    # ------------------------------------------------------------------
    def _ancestor_chain(self, concept: ConceptNode) -> List[ConceptNode]:
        """Walk up to root, returning ``[parent, grandparent, …, root]``."""
        chain: List[ConceptNode] = []
        seen = {concept.concept_id}
        cur = concept.parent_id
        while cur and cur in self._nodes and cur not in seen:
            parent = self._nodes[cur]
            chain.append(parent)
            seen.add(cur)
            cur = parent.parent_id
        return chain

    @staticmethod
    def _format_path(concept: ConceptNode,
                     ancestors: List[ConceptNode]) -> str:
        if not ancestors:
            return concept.name
        names = [a.name for a in reversed(ancestors)]
        names.append(concept.name)
        return " → ".join(names)

    @staticmethod
    def _build_context(concept: ConceptNode,
                       ancestors: List[ConceptNode]) -> str:
        """Format retrieval context: ancestor path + concept content."""
        parts: List[str] = []
        # Ancestor path (root → … → parent)
        if ancestors:
            path = " → ".join(a.name for a in reversed(ancestors))
            parts.append(f"[Section path: {path} → {concept.name}]")
        else:
            parts.append(f"[Section: {concept.name}]")
        if concept.section_number:
            parts.append(f"[Section number: {concept.section_number}]")
        if concept.definition:
            parts.append(concept.definition)
        if concept.body_text:
            parts.append(concept.body_text[:1000])
        return "\n".join(parts)
