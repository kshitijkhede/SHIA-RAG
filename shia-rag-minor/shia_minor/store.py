"""
Module 10 --- Document store (JSON persistence).

Saves processed document hierarchies to disk so previously analysed PDFs can
be loaded and queried instantly without re-running the extraction pipeline.

Reference: report Chapter 4 (Operational Layer).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .schemas import ConceptNode, HierarchyEdge


class DocumentStore:
    """JSON-file-based persistence for processed documents."""

    def __init__(self, store_dir: str = "./shia_store"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, doc_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in doc_id)
        return self.store_dir / f"{safe}.json"

    # ------------------------------------------------------------------
    def save(self, doc_id: str, concepts: List[ConceptNode],
             edges: List[HierarchyEdge],
             meta: Optional[Dict] = None) -> Path:
        """Persist a processed document; returns the file path."""
        data = {
            "doc_id": doc_id,
            "meta": meta or {},
            "concepts": [c.model_dump() for c in concepts],
            "edges": [e.model_dump() for e in edges],
        }
        p = self._path(doc_id)
        p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return p

    def load(self, doc_id: str) -> Optional[Dict]:
        """Load a previously saved document.  Returns *None* if not found."""
        p = self._path(doc_id)
        if not p.exists():
            return None
        raw = json.loads(p.read_text(encoding="utf-8"))
        concepts = [ConceptNode(**c) for c in raw["concepts"]]
        edges = [HierarchyEdge(**e) for e in raw["edges"]]
        return {
            "doc_id": raw["doc_id"],
            "meta": raw.get("meta", {}),
            "concepts": concepts,
            "edges": edges,
        }

    def has(self, doc_id: str) -> bool:
        return self._path(doc_id).exists()

    def list_documents(self) -> List[Dict]:
        """Return a list of ``{doc_id, pages, blocks, concepts}`` summaries."""
        docs: List[Dict] = []
        for p in sorted(self.store_dir.glob("*.json")):
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                meta = raw.get("meta", {})
                docs.append({
                    "doc_id": raw["doc_id"],
                    "pages": meta.get("pages", "?"),
                    "blocks": meta.get("blocks", "?"),
                    "concepts": len(raw.get("concepts", [])),
                })
            except Exception:
                continue
        return docs

    def delete(self, doc_id: str) -> bool:
        p = self._path(doc_id)
        if p.exists():
            p.unlink()
            return True
        return False

