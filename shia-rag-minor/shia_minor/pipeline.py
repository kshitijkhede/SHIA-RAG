"""
End-to-end pipeline that wires all modules together.

    PDF -> PDFIngestionService -> StructuralParser (reading order + headings)
        -> ConceptExtractor -> HierarchyBuilder (CPG++/PRE, cross-links)
        -> GraphValidator (inside the builder) -> Chunker (body text)
        -> KCEScorer -> Embedder + HierarchyRetriever -> DocumentStore

New in v2: body-text chunking, semantic embeddings, hierarchy-aware
retrieval, persistent storage.  The ``query()`` method is the 'R' in RAG.

Reference: report Chapter 3 (Overview) and Chapter 4 (Implementation).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .schemas import ConceptNode, HierarchyEdge, TextBlock
from .pdf_ingestion import PDFIngestionService
from .structural_parser import StructuralParser
from .concept_extractor import ConceptExtractor
from .hierarchy_builder import HierarchyBuilder
from .graph_validator import GraphValidator
from .chunker import Chunker
from .kce_scorer import KCEScorer
from .embedder import Embedder
from .retriever import HierarchyRetriever, RetrievalResult
from .store import DocumentStore


class Pipeline:
    def __init__(self, store_dir: str = "./shia_store",
                 use_embeddings: bool = True):
        self.ingest = PDFIngestionService()
        self.parser = StructuralParser()
        self.extractor = ConceptExtractor(self.parser)
        self.builder = HierarchyBuilder(GraphValidator())
        self.chunker = Chunker(self.parser)
        self.kce = KCEScorer()
        self.embedder = Embedder(force_tfidf=not use_embeddings)
        self.retriever = HierarchyRetriever(self.embedder)
        self.store = DocumentStore(store_dir)

    # ------------------------------------------------------------------
    def sort_blocks(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Sort all blocks across pages into reading order."""
        dims = self.parser.page_dimensions(blocks)
        ordered: List[TextBlock] = []
        for page in sorted(dims):
            pblocks = [b for b in blocks if b.page == page]
            w, h = dims[page]
            ordered.extend(self.parser.sort_reading_order(pblocks, w, h))
        for i, b in enumerate(ordered):
            b.reading_order = i
        return ordered

    # ------------------------------------------------------------------
    def run(self, pdf_path: str) -> Dict:
        """Run the full pipeline on a PDF.

        Returns a dict with the same keys as v1, plus ``"body_text_chars"``
        and ``"retriever_backend"``.  The retriever is automatically indexed
        for subsequent ``query()`` calls.
        """
        doc_id = Path(pdf_path).stem[:24]

        # --- ingestion + reading order ---
        blocks = self.ingest.extract(pdf_path)
        ordered = self.sort_blocks(blocks)

        # --- concept extraction + hierarchy ---
        concepts = self.extractor.extract(ordered, doc_id)
        edges, concepts = self.builder.build(concepts)

        # --- body-text chunking (NEW) ---
        self.chunker.assign(ordered, concepts)

        # --- confidence scoring ---
        self.kce.score(concepts)

        # --- retrieval indexing (NEW) ---
        self.retriever.index(concepts)

        # --- persist (NEW) ---
        roots = [c for c in concepts if c.parent_id is None]
        meta = {
            "pages": max((b.page for b in blocks), default=0),
            "blocks": len(blocks),
            "roots": len(roots),
            "max_depth": max((c.depth for c in concepts), default=0),
        }
        self.store.save(doc_id, concepts, edges, meta)

        return {
            "doc_id": doc_id,
            "pages": meta["pages"],
            "blocks": meta["blocks"],
            "concepts": concepts,
            "edges": edges,
            "roots": len(roots),
            "max_depth": meta["max_depth"],
            "orphans": sum(1 for c in concepts
                           if c.parent_id is not None
                           and c.parent_id not in {x.concept_id for x in concepts}),
            "body_text_chars": sum(len(c.body_text) for c in concepts),
            "retriever_backend": self.embedder.backend,
        }

    # ------------------------------------------------------------------
    def query(self, question: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve the top-*k* concepts relevant to *question*.

        The retriever must have been indexed first (via ``run()`` or
        ``load_document()``).
        """
        return self.retriever.query(question, top_k=top_k)

    # ------------------------------------------------------------------
    def load_document(self, doc_id: str) -> Optional[Dict]:
        """Load a previously processed document from the store and re-index
        the retriever so ``query()`` works immediately."""
        data = self.store.load(doc_id)
        if data is None:
            return None
        self.retriever.index(data["concepts"])
        return data

    def list_documents(self) -> List[Dict]:
        """List all stored documents."""
        return self.store.list_documents()
