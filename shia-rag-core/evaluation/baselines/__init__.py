"""
Baseline Models for SHEF 2.0 Evaluation
"""

import sys
from pathlib import Path

_BASELINES_DIR = Path(__file__).resolve().parent
_EVAL_DIR = _BASELINES_DIR.parent
_PROJECT_ROOT = _EVAL_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src"), str(_EVAL_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from .flat_rag import FlatRAGBaseline

__all__ = ["FlatRAGBaseline"]
