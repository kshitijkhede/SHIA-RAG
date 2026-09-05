"""Test configuration: adds shia-rag-core, src/, evaluation/, and workspace root to sys.path."""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
workspace_root = project_root.parent

for p in [str(project_root), str(project_root / "src"), str(project_root / "evaluation"), str(workspace_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)
