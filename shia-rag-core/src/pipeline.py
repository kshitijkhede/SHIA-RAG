"""
SHIA-RAG 2.0: Master End-to-End Pipeline Orchestrator
======================================================
Integrates and coordinates all 9 layers (L0 to L8) of the
Semantic Hierarchy Induction Architecture for RAG:

- Layer 0: Schemas & Invariant Checking (KnowledgeNode, KnowledgeEdge, ForestInvariantChecker)
- Layer 1: Syntactic Document & Text Block Management
- Layer 4: Knowledge Confidence Engine (KCE - Topological Damped Propagation)
- Layer 5: Hierarchy Validator (HV), Forest Integrator (FI), Cross-Link Discovery (CLD)
- Layer 6: Self-Reflective Depth Router (SRDR), DC-Knapsack Precedence Optimizer
- Layer 7: Grounded Context Assembly & Claim Attribution Verifier
- Layer 8: Thompson Sampling Graph Evolution with Laplacian Smoothing

Reference: SHIA_RAG_Complete_Report.md
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root and src/ are on sys.path for direct script invocation
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for _p in [str(PROJECT_ROOT), str(CURRENT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Layer imports
from src.layer0_data_model.schemas import (
    DocumentMimeType,
    DocumentNode,
    EdgeCategory,
    GenerationResult,
    KnowledgeEdge,
    KnowledgeNode,
    NodeType,
    RetrievalResult,
    SemanticRelation,
    TextBlock,
)
from src.layer0_data_model.invariants import ForestInvariantChecker
from src.layer1_ingestion.pdf_loader import PDFLoader
from src.layer3_extraction.concept_extractor import ConceptExtractor
from src.layer4_relations.kce_scorer import KnowledgeConfidenceEngine
from src.layer5_shia_core.hv_validator import HierarchyValidator
from src.layer5_shia_core.fi_integrator import ForestIntegrator
from src.layer5_shia_core.cld_crosslinker import CrossLinkDiscovery
from src.layer6_retrieval.srdr_router import SelfReflectiveDepthRouter
from src.layer6_retrieval.dc_knapsack import DAGKnapsackOptimizer, KnapsackItem
from src.layer7_generation.citation_verifier import ClaimAttributionVerifier
from src.layer8_operations.thompson_evolution import ThompsonEvolutionEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SHIARAGPipeline")


class SHIARAGPipeline:
    """
    Unified Pipeline Orchestrator for SHIA-RAG 2.0.
    
    Provides an end-to-end interface for:
      1. Building and maintaining the Dual-Tier Heterogeneous Knowledge Forest.
      2. Validating acyclicity and hierarchy depth invariants.
      3. Discovering semantic cross-links across disparate subtrees.
      4. Propagating confidence scores topologically.
      5. Routing queries dynamically via SRDR.
      6. Selecting optimal, precedence-constrained context under token budgets via DC-Knapsack.
      7. Grounding and verifying claims with continuous attribution rewards.
      8. Evolving graph retrieval beliefs online via Thompson Sampling and Laplacian smoothing.
    """

    def __init__(
        self,
        token_budget: int = 2048,
        max_depth: int = 8,
        laplacian_gamma: float = 0.05,
        smoothing_interval: int = 5,
        min_confidence_for_root: float = 0.60,
        baseline_conf: float = 0.50,
        kce_damping: float = 0.85,
    ):
        self.token_budget = token_budget
        self.max_depth = max_depth
        self.laplacian_gamma = laplacian_gamma
        self.smoothing_interval = smoothing_interval

        # Core Engines
        self.validator = HierarchyValidator(max_depth=max_depth)
        self.integrator = ForestIntegrator(
            max_depth=max_depth,
            min_confidence_for_root=min_confidence_for_root,
        )
        self.kce = KnowledgeConfidenceEngine(
            damping_factor=kce_damping,
            baseline_conf=baseline_conf,
        )
        self.cld = CrossLinkDiscovery()
        self.router = SelfReflectiveDepthRouter()
        self.knapsack = DAGKnapsackOptimizer(token_budget=token_budget)
        self.verifier = ClaimAttributionVerifier()
        self.evolution = ThompsonEvolutionEngine(
            smoothing_gamma=laplacian_gamma,
            smoothing_interval=smoothing_interval,
        )
        self.invariant_checker = ForestInvariantChecker(max_depth=max_depth)
        self.pdf_loader = PDFLoader()
        self.concept_extractor = ConceptExtractor()

        # Knowledge Forest State
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        self.node_depth_map: Dict[str, int] = {}
        self.forest_parents_map: Dict[str, List[str]] = {}
        self.forest_children_map: Dict[str, List[str]] = {}
        self.edge_priors: Dict[str, Tuple[float, float]] = {}  # edge_id -> (alpha, beta)
        self.edge_id_to_nodes: Dict[str, Tuple[str, str]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.block_to_nodes: Dict[str, List[str]] = {}

        # Tier 1 Document Store
        self.documents: Dict[str, DocumentNode] = {}
        self.blocks: Dict[str, TextBlock] = {}

    # ─────────────────────────────────────────────────────────────
    # Tier 1 & Tier 2 Forest Construction
    # ─────────────────────────────────────────────────────────────

    def register_document(self, filename: str, mime_type: DocumentMimeType = DocumentMimeType.PLAIN, pages: int = 1) -> DocumentNode:
        """Registers a source document in Tier 1."""
        sha256 = f"{abs(hash(filename)):064x}"
        doc = DocumentNode(
            filename=filename,
            mime_type=mime_type,
            sha256_hash=sha256,
            total_pages=pages,
        )
        self.documents[doc.doc_id] = doc
        return doc

    def add_text_block(self, doc_id: str, text: str, reading_order: int, section_path: str = "1.0") -> TextBlock:
        """Adds a structural text block to Tier 1."""
        block = TextBlock(
            doc_id=doc_id,
            reading_order=reading_order,
            text_content=text,
            section_path=section_path,
            char_start=0,
            char_end=len(text),
        )
        self.blocks[block.block_id] = block
        return block

    def add_concept_node(
        self,
        node: KnowledgeNode,
        candidate_parents: Optional[List[Tuple[str, float]]] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> KnowledgeNode:
        """
        Integrates a concept node into Tier 2 of the Knowledge Forest.
        
        Enforces:
          - Upward DFS cycle prevention via HierarchyValidator
          - Max depth bounds (parent.depth + 1 < max_depth)
          - Deterministic placement / root promotion via ForestIntegrator
          - Thompson Sampling edge prior registration
        """
        # Validate candidate parents with HV
        validated_candidates: List[Tuple[str, float]] = []
        if candidate_parents:
            for pid, score in candidate_parents:
                if pid in self.nodes:
                    is_valid = self.validator.validate_placement(
                        parent_id=pid,
                        child_id=node.node_id,
                        forest_parents_map=self.forest_parents_map,
                        node_depth_map=self.node_depth_map,
                    )
                    if is_valid:
                        validated_candidates.append((pid, score))

        # Integrate via ForestIntegrator (places or promotes to root)
        updated_node, new_edge = self.integrator.integrate_node(
            node=node,
            ranked_valid_parents=validated_candidates,
            forest_nodes=self.nodes,
            forest_edges=self.edges,
            node_depth_map=self.node_depth_map,
        )

        self.nodes[updated_node.node_id] = updated_node
        self.forest_parents_map.setdefault(updated_node.node_id, [])
        self.forest_children_map.setdefault(updated_node.node_id, [])

        if new_edge:
            self.forest_parents_map[updated_node.node_id].append(new_edge.target_id)
            self.forest_children_map.setdefault(new_edge.target_id, []).append(updated_node.node_id)
            self.edge_priors[new_edge.edge_id] = (new_edge.alpha, new_edge.beta)
            self.edge_id_to_nodes[new_edge.edge_id] = (new_edge.source_id, new_edge.target_id)
            self.evolution.register_new_edge(new_edge.edge_id)

        # Store embedding
        if embedding is not None:
            self.embeddings[updated_node.node_id] = embedding
        else:
            # Deterministic pseudo-embedding from canonical name
            rng = np.random.RandomState(abs(hash(updated_node.canonical_name)) % (2**32))
            emb = rng.randn(64)
            self.embeddings[updated_node.node_id] = emb / np.linalg.norm(emb)

        # Register block provenance anchors
        for bid in updated_node.source_block_ids:
            self.block_to_nodes.setdefault(bid, []).append(updated_node.node_id)

        return updated_node

    def run_confidence_propagation(self) -> Dict[str, float]:
        """Runs Layer 4 topological confidence propagation across the forest."""
        confidence_map = self.kce.propagate_confidence(self.nodes, self.edges)
        for nid, conf in confidence_map.items():
            if nid in self.nodes:
                self.nodes[nid] = self.nodes[nid].model_copy(update={"confidence": conf})
        logger.info(f"KCE propagated confidence across {len(confidence_map)} nodes.")
        return confidence_map

    def run_crosslink_discovery(self) -> List[KnowledgeEdge]:
        """Runs Layer 5 Cross-Link Discovery to find semantic inter-tree relationships."""
        new_edges = self.cld.discover_crosslinks(
            nodes=self.nodes,
            existing_edges=self.edges,
            embeddings=self.embeddings,
            block_to_nodes=self.block_to_nodes,
        )
        for edge in new_edges:
            self.edges.append(edge)
            self.edge_priors[edge.edge_id] = (edge.alpha, edge.beta)
            self.edge_id_to_nodes[edge.edge_id] = (edge.source_id, edge.target_id)
            self.evolution.register_new_edge(edge.edge_id)
        return new_edges

    def validate_invariants(self) -> None:
        """Verifies forest invariants (acyclicity, depth limits, valid predicates)."""
        violations = self.invariant_checker.check_all(self.nodes, self.edges)
        if violations:
            raise ValueError(f"Forest invariant violation(s): {violations}")
        logger.info("Forest invariant check PASSED: Acyclic taxonomy DAG maintained.")

    def ingest_pdf(self, pdf_path: str | Path) -> Dict[str, Any]:
        """
        Ingests a real PDF document into Tier 1 TextBlocks, extracts Tier 2 concepts
        and hierarchical relations, integrates them into the Knowledge Forest,
        discovers semantic cross-links, propagates confidence, and validates structural invariants.

        Args:
            pdf_path: Filepath to the PDF document.

        Returns:
            Dictionary summarizing ingestion metrics and forest statistics.
        """
        path = Path(pdf_path).resolve()
        logger.info(f"Ingesting PDF document: {path}")
        doc_node, blocks = self.pdf_loader.load_pdf(path)
        self.documents[doc_node.doc_id] = doc_node
        for b in blocks:
            self.blocks[b.block_id] = b

        extracted_nodes = self.concept_extractor.extract_from_document(doc_node, blocks)

        integrated_nodes = []
        for node, candidate_parents in extracted_nodes:
            integrated = self.add_concept_node(node, candidate_parents=candidate_parents)
            integrated_nodes.append(integrated)

        # Layer 4: Topological confidence propagation
        self.run_confidence_propagation()

        # Layer 5: Semantic cross-link discovery
        new_crosslinks = self.run_crosslink_discovery()

        # Forest invariant checking
        self.validate_invariants()

        stats = self.get_forest_statistics()
        logger.info(
            f"PDF ingestion successful: '{doc_node.filename}' -> "
            f"{len(integrated_nodes)} concepts, {len(new_crosslinks)} crosslinks, "
            f"depth {stats['max_depth']}."
        )
        return {
            "document": doc_node.filename,
            "doc_id": doc_node.doc_id,
            "sha256": doc_node.sha256_hash,
            "total_pages": doc_node.total_pages,
            "blocks_ingested": len(blocks),
            "concepts_extracted": len(integrated_nodes),
            "crosslinks_discovered": len(new_crosslinks),
            "forest_stats": stats,
        }

    # ─────────────────────────────────────────────────────────────
    # Preloaded Knowledge Forest (Realistic CS & AI Domain)
    # ─────────────────────────────────────────────────────────────

    def load_sample_knowledge_base(self) -> None:
        """
        Preloads a comprehensive Computer Science & Artificial Intelligence
        Knowledge Forest exercising multi-depth trees and cross-links.
        """
        doc = self.register_document("CS_AI_Fundamentals.pdf", DocumentMimeType.PDF, pages=12)

        # Tier 1 Blocks
        b1 = self.add_text_block(doc.doc_id, "Computer Science encompasses computation, information architecture, and systems.", 1, "1.0")
        b2 = self.add_text_block(doc.doc_id, "Computer Networking organizes communication protocols into layered architectural stacks.", 2, "2.0")
        b3 = self.add_text_block(doc.doc_id, "The Transport Layer provides end-to-end communication services for applications.", 3, "2.1")
        b4 = self.add_text_block(doc.doc_id, "Transmission Control Protocol (TCP) guarantees reliable, in-order byte stream delivery with congestion control.", 4, "2.1.1")
        b5 = self.add_text_block(doc.doc_id, "User Datagram Protocol (UDP) offers lightweight, connectionless datagram transmission without reliability guarantees.", 5, "2.1.2")
        b6 = self.add_text_block(doc.doc_id, "Artificial Intelligence studies computational agents capable of perception, reasoning, and decision-making.", 6, "3.0")
        b7 = self.add_text_block(doc.doc_id, "Machine Learning constructs statistical models that learn representations and patterns from empirical data.", 7, "3.1")
        b8 = self.add_text_block(doc.doc_id, "Optimization Algorithms minimize empirical loss functions over model parameter manifolds.", 8, "3.1.1")
        b9 = self.add_text_block(doc.doc_id, "Stochastic Gradient Descent (SGD) updates parameters iteratively using mini-batch gradient approximations.", 9, "3.1.1.1")
        b10 = self.add_text_block(doc.doc_id, "Neural Networks represent compositions of parameterized non-linear mathematical transformations.", 10, "3.1.2")
        b11 = self.add_text_block(doc.doc_id, "Backpropagation calculates loss gradients with respect to weights using the multivariate chain rule.", 11, "3.1.2.1")
        b12 = self.add_text_block(doc.doc_id, "Transformer Architecture computes self-attention across sequence tokens to model long-range dependencies.", 12, "3.1.2.2")
        b13 = self.add_text_block(doc.doc_id, "Operating Systems manage computer hardware resources and provide common services for programs.", 13, "4.0")
        b14 = self.add_text_block(doc.doc_id, "CPU Scheduling decides which process in the ready queue is allocated the CPU for execution.", 14, "4.1")
        b15 = self.add_text_block(doc.doc_id, "Round Robin is a preemptive CPU scheduling algorithm designed for time-sharing systems where each process gets a fixed time quantum.", 15, "4.1.1")

        # Tier 2 Concept Nodes
        # 1. Root: Computer Science
        cs_root = KnowledgeNode(
            node_id="KN-CS001",
            canonical_name="Computer Science",
            node_type=NodeType.CONCEPT,
            definition_text="The study of computation, automation, and information processing.",
            evidence_texts=[b1.text_content],
            source_block_ids=[b1.block_id],
            abstraction_level=1.0,
            token_cost=60,
        )
        self.add_concept_node(cs_root)

        # 2. Networking Subtree
        net_node = KnowledgeNode(
            node_id="KN-NET01",
            canonical_name="Computer Networking",
            node_type=NodeType.CONCEPT,
            definition_text="Interconnected computing systems that exchange data using standardized protocols.",
            evidence_texts=[b2.text_content],
            source_block_ids=[b2.block_id],
            abstraction_level=0.8,
            token_cost=70,
        )
        self.add_concept_node(net_node, candidate_parents=[("KN-CS001", 0.95)])

        trn_node = KnowledgeNode(
            node_id="KN-TRN01",
            canonical_name="Transport Layer Protocols",
            node_type=NodeType.CONCEPT,
            definition_text="Protocols operating at Layer 4 of the OSI model delivering host-to-host communication.",
            evidence_texts=[b3.text_content],
            source_block_ids=[b3.block_id],
            abstraction_level=0.6,
            token_cost=75,
        )
        self.add_concept_node(trn_node, candidate_parents=[("KN-NET01", 0.92)])

        tcp_node = KnowledgeNode(
            node_id="KN-TCP01",
            canonical_name="Transmission Control Protocol (TCP)",
            node_type=NodeType.CONCEPT,
            definition_text="Connection-oriented protocol providing reliable, ordered, and error-checked byte stream delivery.",
            evidence_texts=[b4.text_content],
            source_block_ids=[b4.block_id],
            abstraction_level=0.3,
            token_cost=85,
        )
        self.add_concept_node(tcp_node, candidate_parents=[("KN-TRN01", 0.96)])

        udp_node = KnowledgeNode(
            node_id="KN-UDP01",
            canonical_name="User Datagram Protocol (UDP)",
            node_type=NodeType.CONCEPT,
            definition_text="Connectionless transport protocol offering minimal latency and no delivery guarantees.",
            evidence_texts=[b5.text_content],
            source_block_ids=[b5.block_id],
            abstraction_level=0.3,
            token_cost=80,
        )
        self.add_concept_node(udp_node, candidate_parents=[("KN-TRN01", 0.94)])

        # 3. AI / ML Subtree
        ai_node = KnowledgeNode(
            node_id="KN-AI001",
            canonical_name="Artificial Intelligence",
            node_type=NodeType.CONCEPT,
            definition_text="Engineering and science of building intelligent agents that emulate cognitive functions.",
            evidence_texts=[b6.text_content],
            source_block_ids=[b6.block_id],
            abstraction_level=0.85,
            token_cost=70,
        )
        self.add_concept_node(ai_node, candidate_parents=[("KN-CS001", 0.98)])

        ml_node = KnowledgeNode(
            node_id="KN-ML001",
            canonical_name="Machine Learning",
            node_type=NodeType.CONCEPT,
            definition_text="Subfield of AI where algorithms learn representations from data without explicit rules.",
            evidence_texts=[b7.text_content],
            source_block_ids=[b7.block_id],
            abstraction_level=0.7,
            token_cost=75,
        )
        self.add_concept_node(ml_node, candidate_parents=[("KN-AI001", 0.96)])

        opt_node = KnowledgeNode(
            node_id="KN-OPT01",
            canonical_name="Optimization Algorithms",
            node_type=NodeType.CONCEPT,
            definition_text="Techniques used to minimize objective loss functions during model training.",
            evidence_texts=[b8.text_content],
            source_block_ids=[b8.block_id],
            abstraction_level=0.5,
            token_cost=70,
        )
        self.add_concept_node(opt_node, candidate_parents=[("KN-ML001", 0.88)])

        sgd_node = KnowledgeNode(
            node_id="KN-SGD01",
            canonical_name="Stochastic Gradient Descent (SGD)",
            node_type=NodeType.PROCEDURE,
            definition_text="Iterative parameter optimization method using random subset gradients.",
            evidence_texts=[b9.text_content],
            source_block_ids=[b9.block_id],
            abstraction_level=0.2,
            token_cost=80,
        )
        self.add_concept_node(sgd_node, candidate_parents=[("KN-OPT01", 0.95)])

        nn_node = KnowledgeNode(
            node_id="KN-NN001",
            canonical_name="Neural Networks",
            node_type=NodeType.CONCEPT,
            definition_text="Layered computational models inspired by biological neural connections.",
            evidence_texts=[b10.text_content],
            source_block_ids=[b10.block_id],
            abstraction_level=0.5,
            token_cost=75,
        )
        self.add_concept_node(nn_node, candidate_parents=[("KN-ML001", 0.92)])

        bp_node = KnowledgeNode(
            node_id="KN-BP001",
            canonical_name="Backpropagation",
            node_type=NodeType.PROCEDURE,
            definition_text="Efficient algorithm for computing partial derivatives of loss with respect to network parameters.",
            evidence_texts=[b11.text_content],
            source_block_ids=[b11.block_id],
            abstraction_level=0.2,
            token_cost=85,
        )
        self.add_concept_node(bp_node, candidate_parents=[("KN-NN001", 0.97)])

        trf_node = KnowledgeNode(
            node_id="KN-TRF01",
            canonical_name="Transformer Architecture",
            node_type=NodeType.CONCEPT,
            definition_text="Deep neural network architecture utilizing stacked multi-head self-attention mechanisms.",
            evidence_texts=[b12.text_content],
            source_block_ids=[b12.block_id],
            abstraction_level=0.2,
            token_cost=90,
        )
        self.add_concept_node(trf_node, candidate_parents=[("KN-NN001", 0.93)])

        # 4. Operating Systems Subtree
        os_node = KnowledgeNode(
            node_id="KN-OS001",
            canonical_name="Operating Systems",
            node_type=NodeType.CONCEPT,
            definition_text="System software managing computer hardware resources and providing common services for applications.",
            evidence_texts=[b13.text_content],
            source_block_ids=[b13.block_id],
            abstraction_level=0.8,
            token_cost=70,
        )
        self.add_concept_node(os_node, candidate_parents=[("KN-CS001", 0.95)])

        sched_node = KnowledgeNode(
            node_id="KN-SCH01",
            canonical_name="CPU Scheduling",
            node_type=NodeType.CONCEPT,
            definition_text="Mechanism determining which ready process receives CPU core execution time.",
            evidence_texts=[b14.text_content],
            source_block_ids=[b14.block_id],
            abstraction_level=0.5,
            token_cost=70,
        )
        self.add_concept_node(sched_node, candidate_parents=[("KN-OS001", 0.93)])

        rr_node = KnowledgeNode(
            node_id="KN-RR001",
            canonical_name="Round Robin Scheduling",
            node_type=NodeType.PROCEDURE,
            definition_text="Preemptive scheduling algorithm where each process receives a cyclic, equal time quantum.",
            evidence_texts=[b15.text_content],
            source_block_ids=[b15.block_id],
            abstraction_level=0.2,
            token_cost=80,
            aliases=["Round Robin", "RR scheduling", "round robin"],
        )
        self.add_concept_node(rr_node, candidate_parents=[("KN-SCH01", 0.96)])

        # Explicit Semantic Cross-Links
        cross_links = [
            ("KN-BP001", "KN-SGD01", SemanticRelation.USES.value, 0.95),
            ("KN-TRF01", "KN-BP001", SemanticRelation.USES.value, 0.92),
            ("KN-TCP01", "KN-UDP01", SemanticRelation.COMPARED_TO.value, 0.88),
            ("KN-NN001", "KN-TRF01", SemanticRelation.PREREQUISITE_OF.value, 0.90),
            ("KN-RR001", "KN-SCH01", SemanticRelation.USES.value, 0.92),
            ("KN-NET01", "KN-OS001", SemanticRelation.USES.value, 0.85),
        ]
        for src, tgt, pred, conf in cross_links:
            edge = KnowledgeEdge(
                source_id=src,
                target_id=tgt,
                category=EdgeCategory.SEMANTIC,
                predicate=pred,
                extraction_confidence=conf,
                extraction_method="domain_ontology",
            )
            self.edges.append(edge)
            self.edge_priors[edge.edge_id] = (edge.alpha, edge.beta)
            self.edge_id_to_nodes[edge.edge_id] = (src, tgt)
            self.evolution.register_new_edge(edge.edge_id)

        # Propagate confidence & discover further cross-links
        self.run_confidence_propagation()
        self.run_crosslink_discovery()
        self.validate_invariants()
        logger.info(f"Loaded sample Knowledge Forest: {len(self.nodes)} nodes, {len(self.edges)} edges.")

    # ─────────────────────────────────────────────────────────────
    # Layer 6: Retrieval Engine (SRDR + DC-Knapsack)
    # ─────────────────────────────────────────────────────────────

    def _score_node_relevance(self, query: str, node: KnowledgeNode, mode: str) -> float:
        """Computes query-node relevance based on textual similarity and routing mode."""
        query_words = set(re.findall(r"\w+", query.lower()))
        stopwords = {
            "what", "is", "the", "exact", "of", "and", "or", "in", "for", "to",
            "how", "does", "like", "which", "one", "define", "summarize", "overview",
            "architecture", "give", "me", "an", "all", "topics", "mechanism"
        }
        meaningful_query_words = {w for w in query_words if w not in stopwords and len(w) > 2}
        if not meaningful_query_words:
            meaningful_query_words = query_words

        # Term frequency match in canonical name, aliases, and definition
        name_words = {w for w in re.findall(r"\w+", node.canonical_name.lower()) if len(w) > 2}
        for alias in node.aliases:
            name_words.update(w for w in re.findall(r"\w+", alias.lower()) if len(w) > 2)
        def_words = {w for w in re.findall(r"\w+", node.definition_text.lower()) if len(w) > 2}
        ev_words = {w for w in re.findall(r"\w+", " ".join(node.evidence_texts).lower()) if len(w) > 2}

        overlap = len(meaningful_query_words & name_words)
        query_coverage = overlap / max(1, len(meaningful_query_words))
        name_coverage = overlap / max(1, len(name_words))
        name_match = max(query_coverage, name_coverage)

        # Check direct substring matching on canonical name and aliases
        query_lower = query.lower()
        if node.canonical_name.lower() in query_lower or any(w in node.canonical_name.lower() for w in meaningful_query_words if len(w) > 3):
            name_match = max(name_match, 0.70)
        for alias in node.aliases:
            if alias.lower() in query_lower or any(w in alias.lower() for w in meaningful_query_words if len(w) > 3):
                name_match = max(name_match, 0.70)

        def_match = len(meaningful_query_words & def_words) / max(1, len(meaningful_query_words))
        ev_match = len(meaningful_query_words & ev_words) / max(1, len(meaningful_query_words))

        base_relevance = 0.6 * name_match + 0.25 * def_match + 0.15 * ev_match

        if base_relevance <= 0.02:
            return 0.0

        # Mode-dependent weighting
        if mode == "THEMATIC":
            # Favor abstract / high-level nodes (near root)
            relevance = base_relevance * (0.6 + 0.4 * node.abstraction_level)
            if node.depth <= 2:
                relevance += 0.25
        elif mode == "FACTUAL":
            # Favor deep, concrete evidence nodes
            relevance = base_relevance * (0.7 + 0.3 * (1.0 - node.abstraction_level))
            if ev_match > 0.10:
                relevance += 0.25
        elif mode == "MULTIHOP":
            # Balanced, boosted by confidence
            relevance = base_relevance * 0.8 + node.confidence * 0.2
        else:  # PARAMETRIC
            relevance = base_relevance * 0.9

        return min(1.0, max(0.0, relevance))

    def retrieve(self, query: str, token_budget: Optional[int] = None) -> RetrievalResult:
        """
        Executes Layer 6 Retrieval:
          1. Routes query via SRDR (THEMATIC, FACTUAL, MULTIHOP, PARAMETRIC).
          2. Samples edge traversal weights via Thompson Sampling.
          3. Evaluates candidate node utilities under mode bias.
          4. Solves DC-Knapsack with strict ancestor-closure precedence.
          5. Formats structured context with provenance anchors.
        """
        effective_budget = token_budget or self.token_budget

        # Step 1: SRDR Routing
        routing_info = self.router.route(query)
        raw_mode = routing_info.get("mode", "MODE_4_PARAMETRIC")
        if "THEMATIC" in raw_mode:
            mode = "THEMATIC"
        elif "FACTUAL" in raw_mode:
            mode = "FACTUAL"
        elif "MULTIHOP" in raw_mode:
            mode = "MULTIHOP"
        else:
            mode = "PARAMETRIC"

        # Step 2: Sample Thompson edge weights for exploration/exploitation
        self.evolution.sample_edge_weights(self.edge_priors)

        # Step 3: Compute candidate relevance and items for DC-Knapsack
        knapsack_items: Dict[str, KnapsackItem] = {}
        for nid, node in self.nodes.items():
            rel_score = self._score_node_relevance(query, node, mode)
            # Parent prerequisites
            parents = self.forest_parents_map.get(nid, [])
            # Token cost (definition + evidence)
            cost = max(20, node.token_cost if node.token_cost > 0 else (len(node.definition_text.split()) + 25))

            knapsack_items[nid] = KnapsackItem(
                node_id=nid,
                relevance_score=rel_score,
                token_cost=cost,
                parent_ids=list(parents),
            )

        # Step 4: Solve DC-Knapsack Optimizer
        self.knapsack.budget = effective_budget
        selected_ids, total_utility, total_tokens = self.knapsack.solve(knapsack_items)

        # Step 5: Identify traversed edges among selected nodes
        selected_set = set(selected_ids)
        traversed_edge_ids: List[str] = []
        for edge in self.edges:
            if edge.source_id in selected_set and edge.target_id in selected_set:
                traversed_edge_ids.append(edge.edge_id)

        # Step 6: Assemble formatted context text
        context_lines: List[str] = []
        # Sort by depth so ancestors appear first
        sorted_selected = sorted(selected_ids, key=lambda nid: self.nodes[nid].depth)
        for nid in sorted_selected:
            n = self.nodes[nid]
            ev_str = " | ".join(n.evidence_texts) if n.evidence_texts else "N/A"
            context_lines.append(
                f"[{n.node_id}] {n.canonical_name} (Depth {n.depth}, Conf {n.confidence:.2f}):\n"
                f"  Definition: {n.definition_text}\n"
                f"  Evidence: {ev_str}"
            )

        context_text = "\n\n".join(context_lines)

        return RetrievalResult(
            query=query,
            mode=mode,
            selected_node_ids=sorted_selected,
            total_tokens=total_tokens,
            total_utility=total_utility,
            traversed_edge_ids=traversed_edge_ids,
            context_text=context_text,
        )

    # ─────────────────────────────────────────────────────────────
    # Layer 7: Generation & Attribution Verification
    # ─────────────────────────────────────────────────────────────

    def generate_and_verify(self, retrieval_result: RetrievalResult) -> GenerationResult:
        """
        Executes Layer 7:
          1. Assembles grounded synthesis from retrieved context.
          2. Explicitly tags citations [KN-xxxx] for all factual claims.
          3. Evaluates claim factual support via ClaimAttributionVerifier.
          4. Returns continuous attribution reward R in [0.0, 1.0].
        """
        selected_nodes = {nid: self.nodes[nid] for nid in retrieval_result.selected_node_ids}

        # Synthesize answer using selected nodes with explicit citations
        if not selected_nodes:
            answer = "I could not find sufficient grounded evidence in the Knowledge Forest to answer this query."
        else:
            answer_parts: List[str] = []
            for nid, node in selected_nodes.items():
                clean_def = node.definition_text.rstrip(". ")
                answer_parts.append(
                    f"{node.canonical_name} is defined as {clean_def} [{node.node_id}]."
                )
            answer = " ".join(answer_parts)

        # Prepare evidence map for verifier: node_id -> {"evidence": list of strings}
        verifier_node_data = {
            nid: {
                "evidence": node.evidence_texts + [node.definition_text],
                "canonical_name": node.canonical_name,
            }
            for nid, node in selected_nodes.items()
        }

        reward, records = self.verifier.verify_generation(
            generated_answer=answer,
            selected_nodes=verifier_node_data,
        )

        return GenerationResult(
            query=retrieval_result.query,
            answer=answer,
            attribution_reward=reward,
            attribution_records=records,
            selected_node_ids=retrieval_result.selected_node_ids,
            traversed_edge_ids=retrieval_result.traversed_edge_ids,
            total_context_tokens=retrieval_result.total_tokens,
            generation_tokens=len(answer.split()),
        )

    # ─────────────────────────────────────────────────────────────
    # Layer 8: Online Evolution (Thompson Sampling Update)
    # ─────────────────────────────────────────────────────────────

    def feedback_and_evolve(self, gen_result: GenerationResult) -> Dict[str, Any]:
        """
        Executes Layer 8 Online Evolution:
          1. Feeds attribution reward R into Thompson conjugate Beta posterior update.
          2. Applies temporal decay and edge contribution weights.
          3. Triggers periodic Laplacian smoothing across the adjacency matrix.
        """
        traversed = gen_result.traversed_edge_ids
        reward = gen_result.attribution_reward

        # Update edge posteriors
        self.evolution.update_edge_feedback(
            edge_priors=self.edge_priors,
            traversed_edges=traversed,
            reward=reward,
        )

        # Check and apply Laplacian smoothing if interval reached
        smoothed = self.evolution.maybe_apply_smoothing(
            edge_priors=self.edge_priors,
            edge_id_to_nodes=self.edge_id_to_nodes,
        )

        return {
            "reward": reward,
            "traversed_edge_count": len(traversed),
            "smoothed_this_step": smoothed,
            "query_count": self.evolution._query_count,
        }

    # ─────────────────────────────────────────────────────────────
    # Unified End-to-End Execution
    # ─────────────────────────────────────────────────────────────

    def run_query(self, query: str, token_budget: Optional[int] = None) -> Dict[str, Any]:
        """
        Runs the complete SHIA-RAG 2.0 loop for a user query:
        Retrieve -> Generate & Verify -> Evolve Online.
        """
        retrieval_res = self.retrieve(query, token_budget=token_budget)
        gen_res = self.generate_and_verify(retrieval_res)
        feedback_res = self.feedback_and_evolve(gen_res)

        return {
            "query": query,
            "routing_mode": retrieval_res.mode,
            "selected_nodes": [
                {
                    "node_id": nid,
                    "name": self.nodes[nid].canonical_name,
                    "depth": self.nodes[nid].depth,
                    "confidence": self.nodes[nid].confidence,
                }
                for nid in retrieval_res.selected_node_ids
            ],
            "total_tokens": retrieval_res.total_tokens,
            "answer": gen_res.answer,
            "attribution_reward": gen_res.attribution_reward,
            "attribution_records": gen_res.attribution_records,
            "evolution_update": feedback_res,
        }

    def get_forest_statistics(self) -> Dict[str, Any]:
        """Computes comprehensive forest structural metrics."""
        roots = [nid for nid, node in self.nodes.items() if node.parent_id is None]
        hierarchical_edges = [e for e in self.edges if e.category == EdgeCategory.HIERARCHICAL]
        semantic_edges = [e for e in self.edges if e.category == EdgeCategory.SEMANTIC]
        depths = [node.depth for node in self.nodes.values()]

        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "hierarchical_edges": len(hierarchical_edges),
            "semantic_edges": len(semantic_edges),
            "root_count": len(roots),
            "roots": [self.nodes[r].canonical_name for r in roots],
            "max_depth": max(depths) if depths else 0,
            "avg_depth": float(np.mean(depths)) if depths else 0.0,
            "avg_confidence": float(np.mean([n.confidence for n in self.nodes.values()])) if self.nodes else 0.0,
            "edge_priors_tracked": len(self.edge_priors),
        }


def main():
    """Standalone CLI entry point demonstration."""
    print("=" * 70)
    print("  SHIA-RAG 2.0: Master End-to-End Pipeline Demonstration")
    print("=" * 70)

    pipeline = SHIARAGPipeline()
    print("\n[1/4] Preloading Knowledge Forest (Computer Science & AI)...")
    pipeline.load_sample_knowledge_base()

    stats = pipeline.get_forest_statistics()
    print("\n[Forest Stats]")
    print(f"  Total Nodes: {stats['total_nodes']}")
    print(f"  Total Edges: {stats['total_edges']} ({stats['hierarchical_edges']} Hierarchical, {stats['semantic_edges']} Semantic)")
    print(f"  Forest Roots: {stats['roots']}")
    print(f"  Max Depth: {stats['max_depth']}")
    print(f"  Avg Confidence: {stats['avg_confidence']:.3f}")

    test_queries = [
        "Summarize the architecture and overview of Artificial Intelligence and Machine Learning",
        "What is the exact mechanism of Transmission Control Protocol (TCP) flow control?",
        "Compare TCP versus UDP in transport protocols, and explain which one Transformer architectures rely on for distributed training.",
        "Define Stochastic Gradient Descent.",
    ]

    print("\n[2/4] Executing Multi-Mode Query Benchmark...")
    for i, q in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: \"{q}\" ---")
        result = pipeline.run_query(q)
        print(f"  Routed Mode: {result['routing_mode']}")
        print(f"  Selected Concepts ({len(result['selected_nodes'])}):")
        for node in result["selected_nodes"]:
            print(f"    - [{node['node_id']}] {node['name']} (Depth {node['depth']}, Conf {node['confidence']:.2f})")
        print(f"  Total Context Tokens: {result['total_tokens']}")
        print(f"  Attribution Reward R: {result['attribution_reward']:.3f}")
        print(f"  Generated Answer:\n    {result['answer']}")

    print("\n[3/4] Verifying Forest Invariants (Post-Retrieval)...")
    pipeline.validate_invariants()

    print("\n[4/4] Pipeline Demonstration Complete. All systems operational.")
    print("=" * 70)


if __name__ == "__main__":
    main()
