"""
End-to-end pipeline that wires the six modules together.

    PDF -> PDFIngestionService -> StructuralParser (reading order + headings)
        -> ConceptExtractor -> HierarchyBuilder (CPG++/PRE, cross-links)
        -> GraphValidator (inside the builder) -> KCEScorer

Reference: report Chapter 3 (Overview) and Chapter 4 (Implementation).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .schemas import ConceptNode, HierarchyEdge, TextBlock
from .pdf_ingestion import PDFIngestionService
from .structural_parser import StructuralParser
from .concept_extractor import ConceptExtractor
from .hierarchy_builder import HierarchyBuilder
from .graph_validator import GraphValidator
from .kce_scorer import KCEScorer


class Pipeline:
    def __init__(self):
        self.ingest = PDFIngestionService()
        self.parser = StructuralParser()
        self.extractor = ConceptExtractor(self.parser)
        self.builder = HierarchyBuilder(GraphValidator())
        self.kce = KCEScorer()

    def sort_blocks(self, blocks: List[TextBlock]) -> List[TextBlock]:
        dims = self.parser.page_dimensions(blocks)
        ordered: List[TextBlock] = []
        for page in sorted(dims):
            pblocks = [b for b in blocks if b.page == page]
            w, h = dims[page]
            ordered.extend(self.parser.sort_reading_order(pblocks, w, h))
        for i, b in enumerate(ordered):
            b.reading_order = i
        return ordered

    def run(self, pdf_path: str) -> Dict:
        doc_id = Path(pdf_path).stem[:24]
        blocks = self.ingest.extract(pdf_path)
        ordered = self.sort_blocks(blocks)
        concepts = self.extractor.extract(ordered, doc_id)
        edges, concepts = self.builder.build(concepts)
        self.kce.score(concepts)
        roots = [c for c in concepts if c.parent_id is None]
        return {
            "doc_id": doc_id,
            "pages": max((b.page for b in blocks), default=0),
            "blocks": len(blocks),
            "concepts": concepts,
            "edges": edges,
            "roots": len(roots),
            "max_depth": max((c.depth for c in concepts), default=0),
            "orphans": sum(1 for c in concepts
                           if c.parent_id is not None
                           and c.parent_id not in {x.concept_id for x in concepts}),
        }
