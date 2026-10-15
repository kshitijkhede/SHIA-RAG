"""
Module 8 --- Body-text chunker.

Assigns the body-text paragraphs between consecutive headings to their owning
concept node.  This gives each concept rich retrievable content beyond just
the heading name — which is critical for the 'R' in RAG.

Reference: report Chapter 3 (Concept Enrichment).
"""
from __future__ import annotations

from typing import List

from .schemas import ConceptNode, TextBlock
from .structural_parser import StructuralParser


class Chunker:
    """Attach body paragraphs to the heading concept they belong to."""

    def __init__(self, parser: StructuralParser | None = None):
        self.parser = parser or StructuralParser()

    def assign(self, ordered_blocks: List[TextBlock],
               concepts: List[ConceptNode]) -> List[ConceptNode]:
        """Populate each concept's ``body_text`` with the paragraphs that
        appear between its heading and the next heading.

        Parameters
        ----------
        ordered_blocks : list[TextBlock]
            All blocks in reading order (as produced by ``Pipeline.sort_blocks``).
        concepts : list[ConceptNode]
            Concept nodes extracted from those blocks (already have ``order``
            indices that correspond to positions in *ordered_blocks*).

        Returns
        -------
        list[ConceptNode]
            The same list, with ``body_text`` populated in-place.
        """
        if not concepts or not ordered_blocks:
            return concepts

        body_fs = self.parser.body_font_size(ordered_blocks)

        # Build a set of heading-block indices for quick lookup
        sorted_concepts = sorted(concepts, key=lambda c: c.order)
        head_orders = {c.order for c in sorted_concepts}

        # Reset body text
        for c in concepts:
            c.body_text = ""

        for ci, concept in enumerate(sorted_concepts):
            start = concept.order + 1
            # Stop at the next concept's heading block (or end of doc)
            end = (sorted_concepts[ci + 1].order
                   if ci + 1 < len(sorted_concepts)
                   else len(ordered_blocks))

            chunks: List[str] = []
            for idx in range(start, min(end, len(ordered_blocks))):
                if idx in head_orders:
                    break  # safety: don't cross into another heading
                blk = ordered_blocks[idx]
                # Quick check: skip if it looks like another heading we missed
                is_head, _, _ = self.parser.classify_heading(blk, body_fs)
                if is_head:
                    break
                text = blk.text.strip()
                if len(text) > 20:          # skip tiny fragments / page numbers
                    chunks.append(text)

            concept.body_text = "\n".join(chunks)

        return concepts

