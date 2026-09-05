"""
SHIA-RAG Evaluation Framework: SHEF 2.0
"""

import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVAL_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src"), str(_EVAL_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from .shef_evaluator import SHEFEvaluator

__all__ = ["SHEFEvaluator"]
