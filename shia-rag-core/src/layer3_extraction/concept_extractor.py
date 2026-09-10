"""
SHIA-RAG Layer 3: Concept & Relation Extractor
================================================
Extracts canonical KnowledgeNodes, definitions, and candidate hierarchical
relationships from Tier 1 TextBlocks.

Uses definitional pattern matching, section hierarchy alignment,
and evidence anchoring to build candidate nodes for Tier 2.

Reference: Chapter 6.4 (Layer 3), SHIA_RAG_Complete_Report.md
"""

from __future__ import annotations

import logging
import re
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Set, Tuple

_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.layer0_data_model.schemas import (
    DocumentNode,
    KnowledgeNode,
    NodeType,
    TextBlock,
)

logger = logging.getLogger(__name__)

# Definitional & Propositional Extraction Patterns
DEFINITION_PATTERNS = [
    re.compile(r"\b([A-Z][A-Za-z0-9\s\-]{2,40})\s+(?:is defined as|is a type of|is a form of|refers to|represents|denotes)\s+([^.?!]+[.?!])"),
    re.compile(r"\b(?:We propose|We introduce|We design)\s+([A-Z][A-Za-z0-9\s\-]{2,35}),?\s+(?:which|a method that|an approach that)\s+([^.?!]+[.?!])"),
    re.compile(r"^\s*([A-Z][A-Za-z0-9\s\-]{2,40}(?:phase|algorithm|scheme|protocol|step|mechanism|verification|generation|computation|agreement))\s*:\s*([^.?!]{15,300}[.?!])", re.M),
    re.compile(r"\b([A-Z][A-Za-z0-9\-]{2,30}(?:\s+[A-Z][A-Za-z0-9\-]{2,30})?)\s+(?:is used to|serves to|operates by|relies on|utilizes|employs|guarantees|computes|provides)\s+([^.?!]{15,250}[.?!])"),
    re.compile(r"\b([A-Z][A-Za-z0-9\s\-]{2,35})\s*\(([A-Z0-9\+]{2,10})\)\s*(?:is|are|provides|enables)\s+([^.?!]{15,250}[.?!])"),
]

# Filtering out non-conceptual sentence starters
STOP_WORDS = {
    "however", "therefore", "moreover", "furthermore", "specifically",
    "in addition", "for example", "as a result", "consequently",
    "in this paper", "in section", "table", "figure", "equation",
    "copyright", "licensee", "academic", "received", "accepted", "published"
}


def _clean_sentence_text(text: str, max_chars: int = 450) -> str:
    """Cleans whitespace and truncates at complete sentence or word boundaries."""
    cleaned = " ".join(text.split()).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[:max_chars]
    last_p = max(cut.rfind("."), cut.rfind(";"))
    if last_p > 100:
        return cut[:last_p + 1].strip()
    last_s = cut.rfind(" ")
    if last_s > 100:
        return cut[:last_s].strip() + "..."
    return cut + "..."


class ConceptExtractor:
    """
    Deconstructs Tier 1 TextBlocks into structured KnowledgeNodes
    with candidate parent linkages for Tier 2 forest integration.
    """

    def __init__(self, max_concepts_per_section: int = 5):
        self.max_concepts_per_section = max_concepts_per_section

    def extract_from_document(
        self,
        doc_node: DocumentNode,
        blocks: List[TextBlock],
    ) -> List[Tuple[KnowledgeNode, List[Tuple[str, float]]]]:
        """
        Extracts a list of (KnowledgeNode, candidate_parents) from document blocks,
        establishing a multi-level taxonomy rooted at the document and branching
        through sections down to fine-grained conceptual propositions.

        Returns:
            List of tuples (node, [(parent_id, score), ...]).
        """
        if not blocks:
            return []

        # Step 1: Detect Document Title and Abstract
        doc_title = doc_node.filename.replace(".pdf", "").replace("_", " ").strip()
        abstract_text = ""

        # Search for abstract block
        for i, b in enumerate(blocks[:20]):
            b_txt = b.text_content.strip()
            if b_txt.lower() == "abstract" or b_txt.lower().startswith("abstract:"):
                # Check next blocks, skipping banners like arXiv or preprint notices
                for next_b in blocks[i + 1: i + 5]:
                    n_txt = next_b.text_content.strip()
                    if len(n_txt) > 60 and not re.search(r"arxiv:\d+|doi:|preprint|copyright|http", n_txt, re.I):
                        abstract_text = n_txt
                        break
                break
            elif "abstract" in b_txt.lower() and len(b_txt) > 80:
                abstract_text = b_txt
                break

        # Clean abstract: remove leading "Abstract" word and de-hyphenate broken words across linebreaks
        if abstract_text:
            abstract_text = re.sub(r"^(?:abstract[:\s\-]*|abstract\s+)", "", abstract_text, flags=re.I).strip()
            abstract_text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", abstract_text)

        # Candidate title from early non-metadata blocks
        DISALLOWED_TITLE_PATTERNS = re.compile(
            r"^(?:published as|conference paper|workshop|symposium|proceedings|findings|transactions|"
            r"journal of|under review|preprint|arxiv:\d+|doi:|copyright|citation|academic editor|"
            r"received|accepted|published|licensee|open access|creative commons|attribution|conditions of|"
            r"issn|pages\s+\d+|volume\s+\d+|vol\.\s*\d+|no\.\s*\d+|http|https|\d+[-–]\d+|©|"
            r"\([ivxldcm0-9a-z]+\)|we provide|we evaluate|we present|we propose|we study|in this paper|"
            r"figure|fig\.|table)\b",
            re.I
        )
        AFFILIATION_PATTERNS = re.compile(
            r"@|department|university|school of|institute|laboratory|faculty|center for|college|corporation|inc\.|ltd\.|author",
            re.I
        )

        clean_fn = re.sub(r"^\d+[\).\s\-]+", "", doc_node.filename.replace(".pdf", "")).strip()
        fn_words = {w.lower() for w in re.findall(r"[A-Za-z0-9]+", clean_fn.replace("_", " ")) if len(w) > 2}

        candidates = []
        for idx, b in enumerate(blocks[:15]):
            t = b.text_content.strip()
            t = re.sub(r"^(Article|Review|Communication|Paper)\s+", "", t, flags=re.I).strip()
            if DISALLOWED_TITLE_PATTERNS.search(t) or AFFILIATION_PATTERNS.search(t) or ";" in t:
                continue
            if 10 < len(t) < 200 and not re.search(r"arxiv|proceedings|findings|copyright|http|doi:|citation:|editor:", t, re.I):
                if idx + 1 < len(blocks):
                    next_t = blocks[idx + 1].text_content.strip()
                    if 3 < len(next_t) < 120 and not DISALLOWED_TITLE_PATTERNS.search(next_t) and not AFFILIATION_PATTERNS.search(next_t) and ";" not in next_t:
                        if re.search(r"\b(for|of|in|on|and|to|with|by|via|a|an|the)\s*$", t, re.I) or t.endswith(":"):
                            t = f"{t} {next_t}"
                c_clean = re.sub(r"\s+", " ", t).strip()
                c_words = {w.lower() for w in re.findall(r"[A-Za-z0-9]+", c_clean) if len(w) > 2}
                overlap = len(c_words & fn_words)
                candidates.append((overlap, len(c_clean), c_clean))

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        if candidates:
            doc_title = candidates[0][2]
        else:
            doc_title = clean_fn

        # Derive root aliases (e.g. acronyms like TreeRAG, GraphRAG, or first words)
        root_aliases: List[str] = [doc_node.filename.replace(".pdf", "")]
        acronym_match = re.search(r"\b([A-Z][A-Za-z0-9\-]{2,15})\b", doc_title)
        if acronym_match:
            root_aliases.append(acronym_match.group(1))
        # Add title prefix
        title_prefix = doc_title.split(":")[0].strip()
        if title_prefix and title_prefix != doc_title:
            root_aliases.append(title_prefix)

        root_id = f"KN-{uuid.uuid4().hex[:6].upper()}"
        root_def = _clean_sentence_text(abstract_text, 600) if abstract_text else f"Master document taxonomy root for '{doc_title}'."
        root_node = KnowledgeNode(
            node_id=root_id,
            canonical_name=doc_title,
            node_type=NodeType.CONCEPT,
            definition_text=root_def,
            evidence_texts=[b.text_content[:250] for b in blocks[:3]],
            source_block_ids=[b.block_id for b in blocks[:3]],
            abstraction_level=1.0,
            token_cost=60,
            aliases=list(set(root_aliases)),
            doc_id=doc_node.doc_id,
        )

        results: List[Tuple[KnowledgeNode, List[Tuple[str, float]]]] = [
            (root_node, [])
        ]

        # Step 2: Group blocks by section
        sections: Dict[str, List[TextBlock]] = {}
        for b in blocks:
            sec_key = b.section_path if b.section_path and not b.section_path.startswith("0.0") else "Document Overview"
            sections.setdefault(sec_key, []).append(b)

        # Track section nodes by section number prefix for hierarchical parent resolution
        # e.g., '3.1.2' -> parent is '3.1', '3.1' -> parent is '3', '3' -> parent is root
        sec_num_to_id: Dict[str, str] = {}
        seen_concepts: Set[str] = {doc_title.lower(), root_id.lower()}
        for a in root_aliases:
            seen_concepts.add(a.lower())

        # Step 3: Extract Section Concepts with Hierarchical Parent Resolution
        for sec_name, sec_blocks in sections.items():
            if sec_name in ("Document Overview", "SEC-REFE References"):
                continue

            combined_sec_text = " ".join(b.text_content for b in sec_blocks)
            sec_block_ids = [b.block_id for b in sec_blocks[:4]]

            # Synthesize section summary definition (skip heading echoes, pick substantive sentences)
            sentences = re.split(r"(?<=[.?!])\s+", combined_sec_text)
            substantive_sentences = [
                s.strip() for s in sentences
                if len(s.strip()) > 35 and not re.match(r"^\d+(\.\d+)*\.?\s*[A-Z]", s.strip())
            ]
            is_contrib_sec = any(w in sec_name.lower() for w in ("objective", "contribution", "novel", "conclusion", "abstract", "overview"))
            if substantive_sentences:
                max_s = 4 if is_contrib_sec else 2
                sec_def = " ".join(substantive_sentences[:max_s]).strip()
            elif sentences:
                sec_def = sentences[0].strip()
            else:
                sec_def = combined_sec_text[:180].strip()

            # Parse section number and clean title
            chap_match = re.match(r"^(?:Chapter|CHAPTER)\s+([0-9IVXLCDM]+)(?::|\.|\s*)\s*(.*)", sec_name, re.I)
            app_match = re.match(r"^Appendix\s+([A-H])(?::|\.|\s)\s*(.*)", sec_name, re.I)
            num_match = re.match(r"^([1-9]\d*(?:\.\d+)*|[A-H]\.\d+(?:\.\d+)*)\.?\s*(.*)", sec_name)
            sec_num = ""
            sec_display_name = sec_name
            sec_aliases: List[str] = [sec_name]

            if chap_match:
                chap_num = chap_match.group(1)
                clean_title = chap_match.group(2).strip()
                sec_num = chap_num
                sec_display_name = f"Chapter {chap_num}: {clean_title}" if clean_title else f"Chapter {chap_num}"
                sec_aliases.extend([f"Chapter {chap_num}", sec_display_name, clean_title])
            elif app_match:
                app_letter = app_match.group(1).upper()
                clean_title = app_match.group(2).strip()
                sec_num = f"Appendix {app_letter}"
                sec_display_name = f"Appendix {app_letter}: {clean_title}" if clean_title else sec_num
                sec_aliases.extend([sec_num, f"Appendix {app_letter}", clean_title])
            elif num_match:
                sec_num = num_match.group(1)
                clean_title = num_match.group(2).strip()
                if clean_title:
                    sec_display_name = f"{sec_num} {clean_title}"
                    sec_aliases.extend([sec_num, clean_title, f"Section {sec_num}"])
                else:
                    sec_display_name = f"Section {sec_num}"
                    sec_aliases.append(sec_num)
            elif sec_name.startswith("SEC-"):
                clean_title = re.sub(r"^SEC-[A-Z0-9]+\s*", "", sec_name).strip()
                sec_display_name = clean_title.title()
                sec_aliases.append(clean_title)

            # Determine parent in taxonomy DAG
            parent_id = root_id
            if sec_num and "." in sec_num:
                parent_prefix = sec_num.rsplit(".", 1)[0]
                if parent_prefix in sec_num_to_id:
                    parent_id = sec_num_to_id[parent_prefix]
                elif f"Appendix {parent_prefix}" in sec_num_to_id:
                    parent_id = sec_num_to_id[f"Appendix {parent_prefix}"]
                elif f"Chapter {parent_prefix}" in sec_num_to_id:
                    parent_id = sec_num_to_id[f"Chapter {parent_prefix}"]
                else:
                    top_prefix = sec_num.split(".")[0]
                    if top_prefix in sec_num_to_id:
                        parent_id = sec_num_to_id[top_prefix]
            elif sec_num and sec_num in sec_num_to_id:
                parent_id = sec_num_to_id[sec_num]

            sec_id = f"KN-{uuid.uuid4().hex[:6].upper()}"
            if sec_num:
                sec_num_to_id[sec_num] = sec_id
                if app_match:
                    sec_num_to_id[app_letter] = sec_id
                if chap_match:
                    sec_num_to_id[f"Chapter {chap_num}"] = sec_id

            # Abstraction level: Top sections = 0.8, Level 2 = 0.6, Level 3 = 0.4
            depth_level = 1 if app_match or chap_match else (sec_num.count(".") if sec_num else 0)
            abstraction = max(0.3, 0.8 - depth_level * 0.15)

            if is_contrib_sec:
                sec_aliases.extend(["Novelties", "Novelty", "Key Novelties", "Contributions", "Contribution", "Research Objectives"])

            sec_node = KnowledgeNode(
                node_id=sec_id,
                canonical_name=sec_display_name,
                node_type=NodeType.CONCEPT,
                definition_text=_clean_sentence_text(sec_def, 1200 if is_contrib_sec else 450),
                evidence_texts=[b.text_content for b in sec_blocks[:2]],
                source_block_ids=sec_block_ids,
                abstraction_level=abstraction,
                token_cost=min(180, len(sec_def.split()) + 20),
                aliases=list(set(sec_aliases)),
                doc_id=doc_node.doc_id,
            )
            results.append((sec_node, [(parent_id, 0.95)]))

            # Step 4: Extract Fine-Grained Concepts within Section
            extracted_in_section = 0

            # Generic scientific/technical contribution & novelty discourse markers
            GENERIC_CONTRIBUTION_PATTERNS = [
                re.compile(r"\b(?:our|the|this\s+paper'?s?)\s+(?:main|primary|key|core|central)?\s*(?:contributions?|novelties|novelty|innovations?)\b", re.I),
                re.compile(r"\b(?:in\s+this\s+(?:paper|work|study|article|report)|herein)\s*,\s*(?:we|this\s+work)\s+(?:propose|introduce|present|develop|design|formulate|demonstrate|construct|implement)\b", re.I),
                re.compile(r"\bwe\s+(?:propose|introduce|present|develop|design|formulate|construct)\s+(?:a|an|the|this)\s+(?:novel|new|hierarchical|lightweight|efficient|generalized|robust)?\s*([A-Za-z0-9\s\-]{3,45})\b", re.I),
                re.compile(r"\b(?:specific\s+objectives\s+include|the\s+primary\s+objective\s+is|this\s+study\s+aims\s+to|our\s+goal\s+is\s+to)\b", re.I),
                re.compile(r"\bto\s+(?:address|tackle|overcome|solve)\s+this\s+(?:problem|challenge|issue|limitation|shortcoming)\s*,\s*we\b", re.I),
                re.compile(r"\bwe\s+summarize\s+(?:our|the)\s+contributions\s+as\s+follows\b", re.I),
            ]

            for b in sec_blocks:
                txt = b.text_content.strip()
                if any(p.search(txt) for p in GENERIC_CONTRIBUTION_PATTERNS):
                    contrib_name = "Key Contributions and Novelties"
                    if contrib_name.lower() not in seen_concepts:
                        seen_concepts.add(contrib_name.lower())
                        c_id = f"KN-{uuid.uuid4().hex[:6].upper()}"
                        c_def = _clean_sentence_text(txt, 1400)
                        c_node = KnowledgeNode(
                            node_id=c_id,
                            canonical_name=contrib_name,
                            node_type=NodeType.CONCEPT,
                            definition_text=c_def,
                            evidence_texts=[txt],
                            source_block_ids=[b.block_id],
                            abstraction_level=0.5,
                            token_cost=min(220, len(c_def.split()) + 25),
                            aliases=[
                                "Novelties", "Novelty", "Key Novelties", "Contributions", "Contribution",
                                "Main Contributions", "Innovations", "Research Objectives", "Proposed Scheme",
                                "Proposed Method", "Proposed Architecture"
                            ],
                            doc_id=doc_node.doc_id,
                        )
                        results.append((c_node, [(sec_id, 0.95)]))
                        extracted_in_section += 1
                        break

            for b in sec_blocks:
                if extracted_in_section >= self.max_concepts_per_section:
                    break

                for pattern in DEFINITION_PATTERNS:
                    if extracted_in_section >= self.max_concepts_per_section:
                        break

                    for match in pattern.finditer(b.text_content):
                        term = match.group(1).strip()
                        definition = match.group(2).strip()

                        term_lower = term.lower()
                        if term_lower in seen_concepts or term_lower in STOP_WORDS or len(term) < 3:
                            continue

                        seen_concepts.add(term_lower)
                        child_id = f"KN-{uuid.uuid4().hex[:6].upper()}"

                        # Derive aliases for child node
                        child_aliases = [term]
                        if "(" in term and ")" in term:
                            acro = re.search(r"\((.*?)\)", term)
                            if acro:
                                child_aliases.append(acro.group(1).strip())

                        child_node = KnowledgeNode(
                            node_id=child_id,
                            canonical_name=term,
                            node_type=NodeType.CONCEPT,
                            definition_text=_clean_sentence_text(definition, 400),
                            evidence_texts=[b.text_content],
                            source_block_ids=[b.block_id],
                            abstraction_level=0.25,
                            token_cost=min(100, len(definition.split()) + 25),
                            aliases=list(set(child_aliases)),
                            doc_id=doc_node.doc_id,
                        )
                        # Candidate parent: The enclosing Section Node
                        results.append((child_node, [(sec_id, 0.92)]))
                        extracted_in_section += 1

                        # Multi-Tier Decomposition: Extract sub-mechanisms, rules, or formulas under core concept
                        sub_patterns = [
                            re.compile(r"\b([A-Z][A-Za-z0-9\s\-]{2,30}\s+(?:phase|step|mechanism|algorithm|rule|formula|property|operation))\s*:\s*([^.?!]{15,250}[.?!])", re.I),
                            re.compile(r"\b(?:Step\s+[0-9IVX]+|Rule\s+[0-9IVX]+|Property\s+[0-9IVX]+)\s*:\s*([^.?!]{15,250}[.?!])", re.I),
                            re.compile(r"\b(The\s+(?:cube|square)\s+root\s+of\s+[0-9\.,]+)\s+(?:is|=|equals)\s+([0-9\.,]+(?:[^.?!]{0,100}[.?!]))", re.I),
                            re.compile(r"\b([A-Z][A-Za-z0-9\s\-]{2,25}\s+(?:footprint|overhead|complexity|runtime|latency|accuracy|bound))\s+(?:is|totals|amounts to)\s+([^.?!]{10,180}[.?!])", re.I),
                        ]
                        for sub_pat in sub_patterns:
                            for sub_m in sub_pat.finditer(b.text_content):
                                sub_title = sub_m.group(1).strip()
                                sub_desc = sub_m.group(2).strip()
                                sub_lower = sub_title.lower()
                                if sub_lower not in seen_concepts and len(sub_title) > 3:
                                    seen_concepts.add(sub_lower)
                                    sub_id = f"KN-{uuid.uuid4().hex[:6].upper()}"
                                    sub_node = KnowledgeNode(
                                        node_id=sub_id,
                                        canonical_name=sub_title,
                                        node_type=NodeType.PROCEDURE,
                                        definition_text=_clean_sentence_text(f"{sub_title}: {sub_desc}", 350),
                                        evidence_texts=[b.text_content],
                                        source_block_ids=[b.block_id],
                                        abstraction_level=0.15,
                                        token_cost=min(90, len(sub_desc.split()) + 20),
                                        aliases=[sub_title],
                                        doc_id=doc_node.doc_id,
                                    )
                                    # Candidate parent: The child concept node (Hierarchy Depth + 1)
                                    results.append((sub_node, [(child_id, 0.93)]))
                                    break

            # Fallback: If section has blocks but no pattern matched, extract key substantive proposition
            if extracted_in_section == 0 and len(sec_blocks) >= 1:
                for b in sec_blocks:
                    txt = b.text_content.strip()
                    if len(txt) < 80 or txt.startswith("Citation:") or txt.startswith("Copyright:"):
                        continue
                    s_list = [s.strip() for s in re.split(r"(?<=[.?!])\s+", txt) if len(s.strip()) > 30]
                    if s_list:
                        prop_def = s_list[0] if len(s_list) == 1 else f"{s_list[0]} {s_list[1]}"
                        prop_name = f"{sec_display_name} Details"
                        prop_id = f"KN-{uuid.uuid4().hex[:6].upper()}"
                        prop_node = KnowledgeNode(
                            node_id=prop_id,
                            canonical_name=prop_name,
                            node_type=NodeType.PROCEDURE,
                            definition_text=_clean_sentence_text(prop_def, 450),
                            evidence_texts=[txt],
                            source_block_ids=[b.block_id],
                            abstraction_level=0.3,
                            token_cost=min(120, len(prop_def.split()) + 25),
                            aliases=[sec_display_name, prop_name],
                            doc_id=doc_node.doc_id,
                        )
                        results.append((prop_node, [(sec_id, 0.90)]))
                        break

        logger.info(f"Extracted {len(results)} candidate KnowledgeNodes from '{doc_node.filename}'.")
        return results
