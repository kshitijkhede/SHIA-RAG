"""
Module 5 --- Graph validator.

Enforces the structural invariants before a hierarchy edge is inserted:
  1. no self-loops
  2. bounded depth (parent.depth + 1 <= D_max)
  3. acyclicity (upward DFS from the parent must not reach the child)
  4. document isolation (parent and child from the same document)

Reference: report Chapter 3 (Structural Validation), Algorithm 2.
"""
from __future__ import annotations

from typing import Dict

from .schemas import ConceptNode, MAX_DEPTH


class GraphValidator:
    def __init__(self, max_depth: int = MAX_DEPTH):
        self.max_depth = max_depth

    def validate_edge(self, parent_id: str, child_id: str,
                      nodes: Dict[str, ConceptNode],
                      parent_of: Dict[str, str]) -> bool:
        if parent_id == child_id:
            return False
        p = nodes.get(parent_id)
        c = nodes.get(child_id)
        if p is None or c is None:
            return False
        if p.doc_id != c.doc_id:                       # document isolation
            return False
        if p.depth + 1 > self.max_depth:               # depth bound
            return False
        # acyclicity: walk up from parent; if we meet the child, it is a cycle
        seen = {child_id}
        cur = parent_id
        hops = 0
        while cur is not None and hops <= self.max_depth + 1:
            if cur == child_id:
                return False
            if cur in seen:
                break
            seen.add(cur)
            cur = parent_of.get(cur)
            hops += 1
        return True
