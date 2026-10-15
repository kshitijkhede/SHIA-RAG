"""
Module 7 --- Semantic Embedder.

Provides dense vector embeddings for concept text using sentence-transformers.
Falls back gracefully to TF-IDF vectors when sentence-transformers is not
installed, so the system always works out of the box.

Reference: report Chapter 3 (Semantic Similarity Enhancement).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder:
    """Encode texts into normalised dense vectors for semantic similarity.

    Uses ``sentence-transformers`` when available (recommended); otherwise
    falls back to a TF-IDF bag-of-words representation.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 force_tfidf: bool = False):
        self._use_st = _HAS_ST and not force_tfidf
        self._model: Optional[SentenceTransformer] = None
        self._model_name = model_name
        if self._use_st:
            self._model = SentenceTransformer(model_name)
        # Lazy TF-IDF state (fitted on the index corpus)
        self._tfidf: Optional[TfidfVectorizer] = None

    @property
    def backend(self) -> str:
        """Return which backend is active."""
        return "sentence-transformers" if self._use_st else "tfidf"

    # ------------------------------------------------------------------
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return *(N, D)* L2-normalised embeddings for *texts*."""
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        if self._use_st:
            return self._model.encode(  # type: ignore[union-attr]
                texts, normalize_embeddings=True, show_progress_bar=False)
        # TF-IDF fallback -------------------------------------------------
        self._tfidf = TfidfVectorizer(stop_words="english",
                                       ngram_range=(1, 2), max_features=4096)
        try:
            m = self._tfidf.fit_transform(texts).toarray().astype(np.float32)
        except ValueError:
            return np.zeros((len(texts), 1), dtype=np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return m / norms

    def embed_query(self, query: str) -> np.ndarray:
        """Return *(1, D)* embedding for a single query string."""
        if self._use_st:
            return self._model.encode(  # type: ignore[union-attr]
                [query], normalize_embeddings=True, show_progress_bar=False)
        if self._tfidf is None:
            return np.zeros((1, 1), dtype=np.float32)
        try:
            m = self._tfidf.transform([query]).toarray().astype(np.float32)
            n = np.linalg.norm(m)
            return m / n if n > 0 else m
        except Exception:
            dim = len(self._tfidf.vocabulary_) if self._tfidf.vocabulary_ else 1
            return np.zeros((1, dim), dtype=np.float32)

    # ------------------------------------------------------------------
    @staticmethod
    def cosine_scores(query_emb: np.ndarray,
                      doc_embs: np.ndarray) -> np.ndarray:
        """Return *(N,)* cosine-similarity scores."""
        if query_emb.size == 0 or doc_embs.size == 0:
            return np.zeros(doc_embs.shape[0], dtype=np.float32)
        return (doc_embs @ query_emb.T).flatten()

