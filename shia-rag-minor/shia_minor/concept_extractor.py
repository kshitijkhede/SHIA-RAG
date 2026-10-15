"""
Module 3 --- Concept extraction.

Turns heading blocks into Tier 2 ConceptNodes, each grounded in a Tier 1 block by
an EvidenceSpan. If the first body line after a heading begins with a definitional
cue, it is stored as the concept's definition.

Reference: report Chapter 3 (Heading Classification and Concept Extraction).
"""
from __future__ import annotations

import re
from typing import List

from .schemas import ConceptNode, EvidenceSpan, TextBlock
from .structural_parser import StructuralParser

_DEF_RE = re.compile(r"^\s*[A-Z0-9].*\b(is|are|refers to|is defined as|denotes)\b", re.I)


class ConceptExtractor:
    def __init__(self, parser: StructuralParser | None = None):
        self.parser = parser or StructuralParser()

    @staticmethod
    def _clean_name(text: str) -> str:
        """Heading name = first line, trimmed to the title (drop trailing body text)."""
        line = re.sub(r"\s+", " ", text.split("\n", 1)[0].strip())
        words = line.split()
        # keep the section number (if any) plus up to 7 title words
        return " ".join(words[:8])[:80]

    def extract(self, ordered_blocks: List[TextBlock], doc_id: str) -> List[ConceptNode]:
        body = self.parser.body_font_size(ordered_blocks)

        # Identify running headers / footers / repeated table labels: the same
        # short unnumbered text appearing on 3+ different pages (e.g. a "Method"
        # column header). These are not section headings and must be dropped.
        from collections import defaultdict
        seen_pages: defaultdict = defaultdict(set)
        for b in ordered_blocks:
            is_head, _, num = self.parser.classify_heading(b, body)
            if is_head and not num:
                seen_pages[self._clean_name(b.text)].add(b.page)
        repeated = {name for name, pages in seen_pages.items() if len(pages) >= 3}

        concepts: List[ConceptNode] = []
        for i, b in enumerate(ordered_blocks):
            is_head, level, num = self.parser.classify_heading(b, body)
            if not is_head:
                continue
            name = self._clean_name(b.text)
            if not num and name in repeated:        # drop running header/label
                continue
            definition = ""
            for nb in ordered_blocks[i + 1:i + 3]:
                nis_head, _, _ = self.parser.classify_heading(nb, body)
                if not nis_head and len(nb.text) > 40:
                    if _DEF_RE.match(nb.text[:140]):
                        definition = nb.text[:300]
                    break
            concepts.append(ConceptNode(
                concept_id=f"{doc_id}_c{len(concepts)}",
                name=name,
                doc_id=doc_id, level=level,
                section_number=num or None, order=i,
                definition=definition, is_bold=b.is_bold,
                font_ratio=round(b.font_size / body, 3),
                evidence=[EvidenceSpan(page=b.page, block_id=b.block_id,
                                       bbox=b.bbox, text=b.text[:200])],
            ))
        return concepts
