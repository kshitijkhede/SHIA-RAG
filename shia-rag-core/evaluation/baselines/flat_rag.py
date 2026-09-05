"""
Flat RAG Baseline Implementation
================================
Implements naive fixed-size chunking and flat top-k vector retrieval
for comparative benchmarking against SHIA-RAG 2.0 (per SHEF 2.0 Chapter 9).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List

_BASELINES_DIR = Path(__file__).resolve().parent
_EVAL_DIR = _BASELINES_DIR.parent
_PROJECT_ROOT = _EVAL_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src"), str(_EVAL_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


class FlatRAGBaseline:
    """
    Standard naive chunking baseline:
    - Fixed chunk size (e.g., 250 words)
    - Sliding window overlap (e.g., 50 words)
    - Flat lexical/vector retrieval (no hierarchical DAG, no precedence constraints)
    """

    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[Dict[str, Any]] = []

    def chunk_document(self, doc_id: str, text: str) -> List[Dict[str, Any]]:
        """Splits a document text into fixed overlapping chunks."""
        words = text.split()
        if not words:
            return []

        chunks: List[Dict[str, Any]] = []
        start = 0
        chunk_idx = 1
        while start < len(words):
            end = min(len(words), start + self.chunk_size)
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "chunk_id": f"CHUNK-{doc_id}-{chunk_idx:04d}",
                "doc_id": doc_id,
                "text": chunk_text,
                "token_cost": len(chunk_words),
            })
            chunk_idx += 1
            if end >= len(words):
                break
            start += max(1, self.chunk_size - self.chunk_overlap)

        self.chunks.extend(chunks)
        return chunks

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves top-k flat chunks based on query overlap."""
        if not self.chunks:
            return []

        query_words = set(re.findall(r"\w+", query.lower()))
        scored = []
        for c in self.chunks:
            chunk_words = set(re.findall(r"\w+", c["text"].lower()))
            overlap = len(query_words & chunk_words) / max(1, len(query_words))
            scored.append((overlap, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]
