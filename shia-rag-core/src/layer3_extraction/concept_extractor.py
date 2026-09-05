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
    re.compile(r"\b([A-Z][A-Za-z0-9\s\-]{2,40})\s+(?:is defined as|is a type of|is a form of)\s+([^.?!]+[.?!])"),
    re.compile(r"\b([A-Z][A-Za-z0-9\s\-]{2,35})\s+(?:refers to|represents|denotes)\s+([^.?!]+[.?!])"),
    re.compile(r"\b(?:We propose|We introduce|We design)\s+([A-Z][A-Za-z0-9\s\-]{2,35}),?\s+(?:which|a method that|an approach that)\s+([^.?!]+[.?!])"),
]

# Filtering out non-conceptual sentence starters
STOP_WORDS = {
    "however", "therefore", "moreover", "furthermore", "specifically",
    "in addition", "for example", "as a result", "consequently",
    "in this paper", "in section", "table", "figure", "equation"
}


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
        for i, b in enumerate(blocks[:15]):
            b_txt = b.text_content.strip()
            if b_txt.lower() == "abstract" or b_txt.lower().startswith("abstract:"):
                if i + 1 < len(blocks):
                    abstract_text = blocks[i + 1].text_content.strip()
                break
            elif "abstract" in b_txt.lower() and len(b_txt) > 60:
                abstract_text = b_txt
                break

        # Candidate title from early non-metadata blocks
        for b in blocks[:5]:
            t = b.text_content.strip()
            if 15 < len(t) < 180 and not re.search(r"arxiv|proceedings|findings|copyright|http|pages\s+\d+|doi:", t, re.I):
                doc_title = t
                break

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
        root_def = abstract_text[:300] if abstract_text else f"Master document taxonomy root for '{doc_title}'."
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

            # Synthesize section summary definition (first 2 sentences)
            sentences = re.split(r"(?<=[.?!])\s+", combined_sec_text)
            sec_def = " ".join(sentences[:2]).strip() if sentences else combined_sec_text[:180]

            # Parse section number and clean title
            num_match = re.match(r"^(\d+(\.\d+)*)\s*(.*)", sec_name)
            sec_num = ""
            sec_display_name = sec_name
            sec_aliases: List[str] = [sec_name]

            if num_match:
                sec_num = num_match.group(1)
                clean_title = num_match.group(3).strip()
                if clean_title:
                    sec_display_name = clean_title
                    sec_aliases.extend([sec_num, f"{sec_num} {clean_title}", clean_title])
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
                else:
                    top_prefix = sec_num.split(".")[0]
                    if top_prefix in sec_num_to_id:
                        parent_id = sec_num_to_id[top_prefix]

            sec_id = f"KN-{uuid.uuid4().hex[:6].upper()}"
            if sec_num:
                sec_num_to_id[sec_num] = sec_id

            # Abstraction level: Top sections = 0.8, Level 2 = 0.6, Level 3 = 0.4
            depth_level = sec_num.count(".") if sec_num else 0
            abstraction = max(0.4, 0.8 - depth_level * 0.2)

            sec_node = KnowledgeNode(
                node_id=sec_id,
                canonical_name=sec_display_name,
                node_type=NodeType.CONCEPT,
                definition_text=sec_def[:250],
                evidence_texts=[b.text_content for b in sec_blocks[:2]],
                source_block_ids=sec_block_ids,
                abstraction_level=abstraction,
                token_cost=min(120, len(sec_def.split()) + 20),
                aliases=list(set(sec_aliases)),
            )
            results.append((sec_node, [(parent_id, 0.95)]))

            # Step 4: Extract Fine-Grained Concepts within Section
            extracted_in_section = 0
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

                        child_node = KnowledgeNode(
                            node_id=child_id,
                            canonical_name=term,
                            node_type=NodeType.CONCEPT,
                            definition_text=definition[:250],
                            evidence_texts=[b.text_content],
                            source_block_ids=[b.block_id],
                            abstraction_level=0.25,
                            token_cost=min(100, len(definition.split()) + 25),
                            aliases=[term],
                        )
                        # Candidate parent: The enclosing Section Node
                        results.append((child_node, [(sec_id, 0.92)]))
                        extracted_in_section += 1

        logger.info(f"Extracted {len(results)} candidate KnowledgeNodes from '{doc_node.filename}'.")
        return results
