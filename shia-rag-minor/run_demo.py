#!/usr/bin/env python3
"""Ingest one PDF and print the induced concept hierarchy.

Usage:  python run_demo.py path/to/paper.pdf
"""
import sys
from shia_minor.pipeline import Pipeline


def main(path):
    res = Pipeline().run(path)
    print(f"Document : {res['doc_id']}")
    print(f"Pages    : {res['pages']}")
    print(f"Blocks   : {res['blocks']}")
    print(f"Concepts : {len(res['concepts'])}   Edges: {len(res['edges'])}   "
          f"Roots: {res['roots']}   Max depth: {res['max_depth']}   Orphans: {res['orphans']}")
    print("\nInduced hierarchy:")
    nodes = {c.concept_id: c for c in res["concepts"]}
    for c in sorted(res["concepts"], key=lambda c: c.order):
        indent = "    " * c.depth
        print(f"  {indent}- {c.name[:60]}  (depth {c.depth}, conf {c.confidence:.2f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python run_demo.py <pdf>"); sys.exit(1)
    main(sys.argv[1])
