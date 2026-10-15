"""
Module 4 --- Hierarchy builder: CPG++ candidate generation + PRE ranking.

Attachment policy (report Chapter 4): section numbering is the primary signal, and
the Parent Ranking Engine (PRE) is a fallback for unnumbered headings. PRE scores a
candidate parent u for a child c as

    Score(u,c) = 0.40*S_struct + 0.40*S_sem + 0.20*S_level

with
    S_struct = 1 / (1 + log(1 + |order(c) - order(u)|))
    S_sem    = TF-IDF cosine similarity of the two concepts' text
    S_level  = max(0, 1 - 0.2*(level(c) - level(u) - 1))

Every chosen edge is checked by GraphValidator; if none is valid the concept
becomes a root (no orphans). The class also exposes the individual baselines used
in the report's comparison (sequential, lexical, numbering, pre).

Reference: report Chapter 3 (Hierarchy Induction) and Chapter 4 (results).
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schemas import ConceptNode, CrossLinkEdge, HierarchyEdge
from .graph_validator import GraphValidator

W_STRUCT, W_SEM, W_LEVEL = 0.40, 0.40, 0.20
THETA_CROSS = 0.75


class HierarchyBuilder:
    def __init__(self, validator: GraphValidator | None = None):
        self.validator = validator or GraphValidator()

    # ---------- similarity ----------
    def _sim_matrix(self, concepts: List[ConceptNode]) -> np.ndarray:
        texts = [f"{c.name} {c.definition}".strip() for c in concepts]
        try:
            m = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(texts)
            return cosine_similarity(m)
        except ValueError:
            return np.zeros((len(concepts), len(concepts)))

    # ---------- CPG++ ----------
    @staticmethod
    def candidate_parents(c: ConceptNode, concepts: List[ConceptNode]) -> List[ConceptNode]:
        # Only strictly shallower earlier headings are valid parents. When none
        # exist, return [] so the caller attaches the concept to the document
        # title instead of chaining it under an earlier same-level sibling
        # (which turned flat sibling sections, e.g. a resume's Education /
        # Projects / Skills, into a spurious deep chain).
        return [u for u in concepts
                if u.order < c.order and u.level < c.level and u.doc_id == c.doc_id]

    # ---------- PRE ----------
    def _pre_best(self, c: ConceptNode, cands: List[ConceptNode],
                  sim: np.ndarray, idx: Dict[str, int]) -> Optional[ConceptNode]:
        best, best_s = None, -1.0
        for u in cands:
            dpos = abs(c.order - u.order)
            s_struct = 1.0 / (1.0 + math.log(1 + dpos))
            s_sem = max(0.0, float(sim[idx[c.concept_id]][idx[u.concept_id]]))
            dl = c.level - u.level
            s_level = max(0.0, 1.0 - 0.2 * (dl - 1)) if dl >= 1 else 0.0
            score = W_STRUCT * s_struct + W_SEM * s_sem + W_LEVEL * s_level
            if score > best_s:
                best_s, best = score, u
        return best

    # ---------- build (numbering primary, PRE fallback) ----------
    def build(self, concepts: List[ConceptNode]) -> Tuple[List[HierarchyEdge], List[ConceptNode]]:
        concepts = sorted(concepts, key=lambda c: c.order)

        # document title/root: earliest concept with the largest font ratio
        title = max(concepts, key=lambda c: (c.font_ratio, -c.order)) if concepts else None
        if title is not None:
            title.level = 0
            title.parent_id = None
            title.depth = 0
            # Fold title-continuation lines back into the title: a title that
            # wraps onto a second line uses the SAME (unique, largest) title
            # font and sits on the title's page, so it must not become its own
            # section node. Match on font + page, not reading-order distance
            # (other blocks can sort between the two title lines).
            if title.font_ratio >= 1.3:
                tpage = title.evidence[0].page if title.evidence else None
                cont = [c for c in concepts
                        if c is not title and c.section_number is None
                        and c.font_ratio >= 0.98 * title.font_ratio
                        and (tpage is None or (c.evidence and c.evidence[0].page == tpage))]
                for c in sorted(cont, key=lambda c: c.order):
                    title.name = f"{title.name} {c.name}".strip()[:120]
                if cont:
                    drop = {c.concept_id for c in cont}
                    concepts = [c for c in concepts if c.concept_id not in drop]

        nodes = {c.concept_id: c for c in concepts}
        by_num = {c.section_number: c for c in concepts if c.section_number}
        sim = self._sim_matrix(concepts)
        idx = {c.concept_id: i for i, c in enumerate(concepts)}
        parent_of: Dict[str, str] = {}
        edges: List[HierarchyEdge] = []

        for c in concepts:
            if c is title:
                continue
            chosen: Optional[ConceptNode] = None
            # primary signal: dotted section numbering (3.1 -> 3)
            if c.section_number and "." in c.section_number:
                pnum = c.section_number.rsplit(".", 1)[0]
                chosen = by_num.get(pnum)
            # fallback: CPG++ + PRE (for unnumbered or top-level numbered sections)
            if chosen is None:
                cands = self.candidate_parents(c, concepts)
                chosen = self._pre_best(c, cands, sim, idx) if cands else None
            # last resort: attach to the document title so the forest is well-rooted
            if (chosen is None or not self.validator.validate_edge(
                    chosen.concept_id, c.concept_id, nodes, parent_of)) and title is not None:
                if self.validator.validate_edge(title.concept_id, c.concept_id, nodes, parent_of):
                    chosen = title
                else:
                    chosen = None
            if chosen is not None:
                parent_of[c.concept_id] = chosen.concept_id
                c.parent_id = chosen.concept_id
                c.depth = nodes[chosen.concept_id].depth + 1
                edges.append(HierarchyEdge(child_id=c.concept_id, parent_id=chosen.concept_id))
            else:
                c.parent_id = None
                c.depth = 0
        return edges, concepts

    # ---------- cross-links ----------
    def cross_links(self, concepts: List[ConceptNode],
                    theta: float = THETA_CROSS) -> List[CrossLinkEdge]:
        sim = self._sim_matrix(concepts)
        root = self._subtree_root(concepts)
        links = []
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                a, b = concepts[i], concepts[j]
                if root.get(a.concept_id) == root.get(b.concept_id):
                    continue
                s = float(sim[i][j])
                if s >= theta:
                    links.append(CrossLinkEdge(source_id=a.concept_id,
                                               target_id=b.concept_id, similarity=round(s, 3)))
        return links

    @staticmethod
    def _subtree_root(concepts: List[ConceptNode]) -> Dict[str, str]:
        nodes = {c.concept_id: c for c in concepts}
        root: Dict[str, str] = {}
        def find(cid: str) -> str:
            seen = set()
            cur = cid
            while nodes[cur].parent_id and cur not in seen:
                seen.add(cur); cur = nodes[cur].parent_id
            return cur
        for c in concepts:
            root[c.concept_id] = find(c.concept_id)
        return root

    # ---------- baselines (for the report's PAA comparison) ----------
    def assign_baseline(self, concepts: List[ConceptNode], method: str) -> Dict[str, str]:
        concepts = sorted(concepts, key=lambda c: c.order)
        present = {c.section_number for c in concepts if c.section_number}
        parent: Dict[str, str] = {}
        if method == "sequential":
            for i, c in enumerate(concepts):
                parent[c.concept_id] = concepts[i - 1].concept_id if i > 0 else "ROOT"
        elif method == "lexical":
            for i, c in enumerate(concepts):
                toks = set(re.findall(r"[a-z]+", c.name.lower()))
                best, ov = "ROOT", 0
                for u in concepts[:i]:
                    o = len(toks & set(re.findall(r"[a-z]+", u.name.lower())))
                    if o > ov:
                        ov, best = o, u.concept_id
                parent[c.concept_id] = best
        elif method == "numbering":
            by_num = {c.section_number: c for c in concepts if c.section_number}
            for c in concepts:
                if c.section_number and "." in c.section_number:
                    pn = c.section_number.rsplit(".", 1)[0]
                    parent[c.concept_id] = by_num[pn].concept_id if pn in by_num else "ROOT"
                else:
                    parent[c.concept_id] = "ROOT"
        elif method == "pre":
            sim = self._sim_matrix(concepts)
            idx = {c.concept_id: i for i, c in enumerate(concepts)}
            for c in concepts:
                cands = self.candidate_parents(c, concepts)
                best = self._pre_best(c, cands, sim, idx) if cands else None
                parent[c.concept_id] = best.concept_id if best else "ROOT"
        else:
            raise ValueError(f"unknown baseline: {method}")
        return parent
