"""
SHIA-RAG 2.0 Core Package
=========================
Semantic Hierarchy Induction Architecture for Retrieval-Augmented Generation.
"""

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from .pipeline import SHIARAGPipeline

__all__ = ["SHIARAGPipeline"]
