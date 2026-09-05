"""
Unit Tests for Baselines (Flat RAG)
"""

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src"), str(_PROJECT_ROOT / "evaluation")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.baselines.flat_rag import FlatRAGBaseline


class TestFlatRAGBaseline:
    def test_chunking(self):
        baseline = FlatRAGBaseline(chunk_size=10, chunk_overlap=2)
        text = "This is word one two three four five six seven eight nine ten eleven twelve thirteen."
        chunks = baseline.chunk_document("DOC-1", text)
        assert len(chunks) >= 2
        assert all(c["doc_id"] == "DOC-1" for c in chunks)
        assert all(c["token_cost"] > 0 for c in chunks)

    def test_retrieval(self):
        baseline = FlatRAGBaseline(chunk_size=5, chunk_overlap=1)
        baseline.chunk_document("DOC-A", "Computer networks connect hosts using IP protocols.")
        baseline.chunk_document("DOC-B", "Machine learning trains neural networks on empirical data.")

        results = baseline.retrieve("What is neural networks learning?", top_k=1)
        assert len(results) == 1
        assert "Machine learning" in results[0]["text"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
