#!/usr/bin/env python3
"""
SHIA-RAG 2.0 Web Application Launcher
======================================
Starts the ChatGPT & Claude AI web interface and REST API server.

Usage:
  # Start on default port 8000:
  python run_web.py

  # Custom port:
  python run_web.py --port 8080 --host 0.0.0.0
"""

import argparse
import sys
from pathlib import Path

# Add project root and src/ to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
for _p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(WORKSPACE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import uvicorn
from src.api import app, pipeline


def main():
    parser = argparse.ArgumentParser(description="SHIA-RAG 2.0 ChatGPT & Claude AI Web Interface")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    parser.add_argument("--preload-sample", action="store_true", default=True, help="Preload CS/AI sample ontology")
    args = parser.parse_args()

    print("\n" + "=" * 76)
    print("  🚀 SHIA-RAG 2.0: CHATGPT & CLAUDE AI INTERFACE")
    print("=" * 76)
    print(f"  --> Local Web URL:    http://{args.host}:{args.port}")
    print(f"  --> API Documentation: http://{args.host}:{args.port}/docs")
    print(f"  --> System Health:    http://{args.host}:{args.port}/api/health")

    stats = pipeline.get_forest_statistics()
    print(f"  --> Active Forest:    {stats['total_nodes']} nodes, {stats['total_edges']} edges, max depth {stats['max_depth']}")
    print("=" * 76 + "\n")

    uvicorn.run("src.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
