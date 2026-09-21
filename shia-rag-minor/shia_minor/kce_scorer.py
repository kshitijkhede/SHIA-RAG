"""
Module 6 --- Knowledge Confidence Estimation (KCE).

Each concept gets a local confidence from typographic cues, then confidence is
propagated from roots to leaves in topological order (Kahn's algorithm) with
damping delta:

    C_local(v) = min(1, 0.5 + 0.15*[font>1.25x body] + 0.10*[bold]
                            + 0.15*[numbered] + 0.10*[has definition])
    C(v)       = C_local(v)                          if v is a root
                 C_local(v) * (delta + (1-delta)*C(parent))   otherwise

Reference: report Chapter 3 (Confidence Estimation).
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List

from .schemas import ConceptNode

DELTA = 0.85


class KCEScorer:
    def __init__(self, delta: float = DELTA):
        self.delta = delta

    def local(self, c: ConceptNode) -> float:
        s = 0.5
        if c.font_ratio >= 1.25:
            s += 0.15
        if c.is_bold:
            s += 0.10
        if c.section_number:
            s += 0.15
        if c.definition:
            s += 0.10
        return min(1.0, s)

    def score(self, concepts: List[ConceptNode]) -> Dict[str, float]:
        nodes = {c.concept_id: c for c in concepts}
        children: Dict[str, List[str]] = {c.concept_id: [] for c in concepts}
        indeg: Dict[str, int] = {c.concept_id: 0 for c in concepts}
        for c in concepts:
            if c.parent_id and c.parent_id in nodes:
                children[c.parent_id].append(c.concept_id)
                indeg[c.concept_id] = 1
        conf: Dict[str, float] = {}
        q = deque([cid for cid, d in indeg.items() if d == 0])
        for cid in q:
            conf[cid] = self.local(nodes[cid])
        while q:
            u = q.popleft()
            for v in children[u]:
                conf[v] = self.local(nodes[v]) * (self.delta + (1 - self.delta) * conf[u])
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        # write back
        for c in concepts:
            c.local_confidence = round(self.local(c), 3)
            c.confidence = round(conf.get(c.concept_id, c.local_confidence), 3)
        return conf
