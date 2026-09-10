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
from src.layer7_generation import ClaimAttributionVerifier, NaturalLanguageSynthesizer
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
        self.synthesizer = NaturalLanguageSynthesizer(self.verifier)
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

        # Multi-Turn Conversational Memory
        self.conversation_history: List[Dict[str, Any]] = []
        self.last_query: Optional[str] = None
        self.last_topic: Optional[str] = None
        self.last_selected_node_ids: List[str] = []

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

        # Infer doc_id from source blocks if not explicitly provided
        if not getattr(node, "doc_id", None) and node.source_block_ids:
            first_block = self.blocks.get(node.source_block_ids[0])
            if first_block and first_block.doc_id:
                node = node.model_copy(update={"doc_id": first_block.doc_id})

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

    def clear_forest(self) -> None:
        """Completely resets all documents, blocks, concept nodes, and edges."""
        self.nodes.clear()
        self.edges.clear()
        self.node_depth_map.clear()
        self.forest_parents_map.clear()
        self.forest_children_map.clear()
        self.edge_priors.clear()
        self.edge_id_to_nodes.clear()
        self.embeddings.clear()
        self.block_to_nodes.clear()
        self.documents.clear()
        self.blocks.clear()
        self.conversation_history.clear()
        self.last_query = None
        self.last_topic = None
        self.last_selected_node_ids.clear()
        logger.info("Knowledge Forest reset to clean empty slate.")

    def reset_conversation(self) -> None:
        """Resets conversational memory without clearing the knowledge forest."""
        self.conversation_history.clear()
        self.last_query = None
        self.last_topic = None
        self.last_selected_node_ids.clear()
        logger.info("Conversational memory reset to clean state.")

    def clear(self) -> None:
        """Alias for clear_forest."""
        self.clear_forest()

    def delete_document(self, doc_id: str) -> bool:
        """
        Removes a document and its dedicated concept tree completely from the forest.
        
        Cleans up:
          - DocumentNode from self.documents
          - TextBlocks belonging to doc_id from self.blocks
          - KnowledgeNodes belonging to doc_id from self.nodes
          - Associated edges, embeddings, parent/child maps, depth maps
        """
        if doc_id not in self.documents:
            return False

        # Identify blocks belonging to this document
        block_ids_to_remove = {bid for bid, b in self.blocks.items() if b.doc_id == doc_id}
        for bid in block_ids_to_remove:
            self.blocks.pop(bid, None)
            self.block_to_nodes.pop(bid, None)

        # Identify nodes belonging to this document
        node_ids_to_remove = set()
        for nid, n in self.nodes.items():
            if getattr(n, "doc_id", None) == doc_id:
                node_ids_to_remove.add(nid)
            elif any(bid in block_ids_to_remove for bid in n.source_block_ids):
                node_ids_to_remove.add(nid)

        # Remove edges connected to these nodes
        self.edges = [e for e in self.edges if e.source_id not in node_ids_to_remove and e.target_id not in node_ids_to_remove]
        self.edge_priors = {eid: p for eid, p in self.edge_priors.items() if eid in [e.edge_id for e in self.edges]}
        self.edge_id_to_nodes = {eid: p for eid, p in self.edge_id_to_nodes.items() if eid in [e.edge_id for e in self.edges]}

        # Remove nodes and metadata
        for nid in node_ids_to_remove:
            self.nodes.pop(nid, None)
            self.node_depth_map.pop(nid, None)
            self.embeddings.pop(nid, None)
            self.forest_parents_map.pop(nid, None)
            self.forest_children_map.pop(nid, None)

        # Clean remaining child/parent maps
        for nid, plist in list(self.forest_parents_map.items()):
            self.forest_parents_map[nid] = [p for p in plist if p not in node_ids_to_remove]
        for nid, clist in list(self.forest_children_map.items()):
            self.forest_children_map[nid] = [c for c in clist if c not in node_ids_to_remove]

        # Remove document record
        doc_filename = self.documents.pop(doc_id).filename

        # Invariant check and confidence propagation if nodes remain
        if self.nodes:
            self.validate_invariants()
            self.run_confidence_propagation()

        logger.info(f"Deleted document '{doc_filename}' ({doc_id}) and {len(node_ids_to_remove)} associated concept nodes.")
        return True

    def run_crosslink_discovery(self, allow_cross_document: bool = False) -> List[KnowledgeEdge]:
        """Runs Layer 5 Cross-Link Discovery to find semantic inter-tree relationships."""
        new_edges = self.cld.discover_crosslinks(
            nodes=self.nodes,
            existing_edges=self.edges,
            embeddings=self.embeddings,
            block_to_nodes=self.block_to_nodes,
            allow_cross_document=allow_cross_document,
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

    def ingest_pdf(
        self,
        pdf_path: str | Path,
        clear_existing: bool = False,
        allow_cross_document: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingests a real PDF document into Tier 1 TextBlocks, extracts Tier 2 concepts
        and hierarchical relations, integrates them into the Knowledge Forest,
        discovers semantic cross-links, propagates confidence, and validates structural invariants.

        Args:
            pdf_path: Filepath to the PDF document.
            clear_existing: If True, resets existing forest before ingesting to form a fresh standalone tree.
            allow_cross_document: If False, preserves strict intra-document isolation without linking across PDFs.

        Returns:
            Dictionary summarizing ingestion metrics and forest statistics.
        """
        if clear_existing:
            self.clear_forest()

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

        # Layer 5: Semantic cross-link discovery (intra-document by default)
        new_crosslinks = self.run_crosslink_discovery(allow_cross_document=allow_cross_document)

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

    def _get_node_ancestors(self, node_id: str) -> Set[str]:
        """Returns the set of all ancestor node IDs for a given node in the DAG."""
        ancestors: Set[str] = set()
        queue = list(self.forest_parents_map.get(node_id, []))
        while queue:
            p = queue.pop(0)
            if p not in ancestors:
                ancestors.add(p)
                queue.extend(self.forest_parents_map.get(p, []))
        return ancestors

    def _get_node_descendants(self, node_id: str) -> Set[str]:
        """Returns the set of all descendant node IDs for a given node in the DAG."""
        descendants: Set[str] = set()
        queue = list(self.forest_children_map.get(node_id, []))
        while queue:
            ch = queue.pop(0)
            if ch not in descendants:
                descendants.add(ch)
                queue.extend(self.forest_children_map.get(ch, []))
        return descendants

    def is_followup_query(self, query: str) -> bool:
        """Determines if query is an elaboration or follow-up to prior conversational context."""
        q_lower = query.lower().strip()
        followup_phrases = [
            "more content", "give me more", "tell me more", "explain more", "more details",
            "more detail", "can you expand", "expand on this", "expand this", "expand further",
            "elaborate", "explain in detail", "go deeper", "continue", "what else",
            "more examples", "explain further", "give more", "more info", "more information",
            "provide more", "in detail", "tell me about it", "how does it work",
            "what are its", "why does it", "tell more", "more on this", "can you give me more",
            "give additional", "additional content", "additional details", "elaborate on this",
            "can you explain more", "can you elaborate"
        ]
        if any(p in q_lower for p in followup_phrases):
            return True

        stopwords = {
            "what", "is", "the", "exact", "of", "and", "or", "in", "for", "to",
            "how", "does", "like", "which", "one", "give", "me", "an", "all",
            "about", "with", "this", "that", "these", "those", "can", "you",
            "tell", "explain", "paper", "document", "work", "study", "project",
            "please", "show", "describe", "detail", "details"
        }
        words = set(re.findall(r"\w+", q_lower))
        meaningful = {w for w in words if w not in stopwords and (len(w) > 2 or w.isdigit())}
        conversational_modifiers = {
            "more", "content", "expand", "elaborate", "further", "info", "information",
            "example", "examples", "deeper", "else", "types", "function", "functions", "additional"
        }
        if not meaningful or meaningful <= conversational_modifiers:
            return True
        return False

    def resolve_conversational_topic(
        self,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[Optional[str], List[str]]:
        """
        Extracts the active conversational topic and previously selected node IDs.
        """
        # 1. From chat_history if provided
        if chat_history:
            for msg in reversed(chat_history):
                if msg.get("role") == "user":
                    content = msg.get("content", "").strip()
                    if content and not self.is_followup_query(content):
                        return content, self.last_selected_node_ids

        # 2. From pipeline's recorded last_query / last_topic
        if self.last_query and not self.is_followup_query(self.last_query):
            return self.last_query, self.last_selected_node_ids

        if self.last_topic:
            return self.last_topic, self.last_selected_node_ids

        # 3. If previous selected nodes exist, extract their names
        if self.last_selected_node_ids:
            names = [
                self.nodes[nid].canonical_name
                for nid in self.last_selected_node_ids
                if nid in self.nodes and self.nodes[nid].depth > 0
            ]
            if names:
                return " ".join(names[:2]), self.last_selected_node_ids

        return None, []

    def _score_node_relevance(
        self,
        query: str,
        node: KnowledgeNode,
        mode: str,
        is_followup: bool = False,
        context_node_ids: Optional[List[str]] = None,
    ) -> float:
        """Computes query-node relevance based on textual similarity, domain synonyms, Tier 1 block evidence, and routing mode."""
        query_lower = query.lower()
        query_words = set(re.findall(r"\w+", query_lower))
        stopwords = {
            "what", "is", "the", "exact", "of", "and", "or", "in", "for", "to",
            "how", "does", "like", "which", "one", "give", "me", "an", "all",
            "about", "with", "this", "that", "these", "those", "can", "you",
            "tell", "explain", "paper", "document", "work", "study", "project",
            "please", "show", "describe", "detail", "details"
        }
        meaningful_query_words = {w for w in query_words if w not in stopwords and (len(w) > 2 or w.isdigit())}
        if not meaningful_query_words:
            meaningful_query_words = {w for w in query_words if len(w) > 2 or w.isdigit()} or query_words

        # Universal Morphological Stemming (suffix stripping)
        def get_stems(words: Set[str]) -> Set[str]:
            stems = set(words)
            for w in words:
                if len(w) > 4:
                    if w.endswith("ies"): stems.add(w[:-3] + "y")
                    elif w.endswith("sses"): stems.add(w[:-2])
                    elif w.endswith("ing"): stems.add(w[:-3])
                    elif w.endswith("tion") or w.endswith("sion"): stems.add(w[:-4])
                    elif w.endswith("ed"): stems.add(w[:-2])
                    elif w.endswith("ment"): stems.add(w[:-4])
                    elif w.endswith("able"): stems.add(w[:-4])
                    elif w.endswith("al"): stems.add(w[:-2])
                    elif w.endswith("s") and not w.endswith("ss"): stems.add(w[:-1])
            return stems

        expanded_query_words = get_stems(meaningful_query_words)

        # Universal Intent Classification
        # 1. Heading / Outline queries
        is_heading_q = any(t in query_words for t in (
            "heading", "headings", "section", "sections", "outline", "toc",
            "table of contents", "structure", "subheading", "subheadings", "index"
        ))
        if is_heading_q:
            is_sec_node = (
                bool(re.match(r"^(?:[1-9]\d*(?:\.\d+)*|[A-H]\.\d+|Appendix\s+[A-H]|SEC-)\b", node.canonical_name))
                or (node.depth in (1, 2) and not node.canonical_name.endswith("Details") and "Novelties" not in node.canonical_name and "Key Contributions" not in node.canonical_name)
            )
            if is_sec_node:
                return max(0.90, 0.98 - 0.04 * node.depth)
            else:
                return 0.05

        # 2. Novelty / Contribution queries
        is_novelty_q = any(t in query_words for t in (
            "novelty", "novelties", "novel", "contribution", "contributions", "innovations", "innovation"
        )) or any(p in query_lower for p in (
            "what is proposed", "what do they propose", "what is new", "key novelties",
            "main contributions", "what are the novelties", "list all novelties", "list all the novelties"
        ))

        # 3. Summary / Overview / Thesis queries
        is_summary_q = any(t in query_words for t in (
            "summary", "summarize", "overview", "abstract"
        )) or any(p in query_lower for p in (
            "what is this paper", "what is this document", "tell me about",
            "about this paper", "about the paper", "explain this paper",
            "explain the paper", "what does this paper do", "what does this document do",
            "what is the paper about", "what is the document about", "give me detail"
        ))

        # 4. Method / Architecture queries (only if asking about paper's approach in general)
        is_method_q = not is_novelty_q and not is_summary_q and (
            any(p in query_lower for p in (
                "what is the method", "what is the methodology", "explain the method",
                "explain the methodology", "proposed method", "proposed architecture",
                "system architecture", "overall architecture", "how does the method work",
                "how does the architecture work"
            ))
            or (query_words <= {"what", "is", "the", "method", "methodology", "architecture", "framework", "how", "does", "it", "work", "explain"})
        )

        # 5. Results / Evaluation queries
        is_eval_q = not is_novelty_q and not is_summary_q and not is_method_q and any(p in query_lower for p in (
            "what are the results", "experimental results", "evaluation results",
            "performance results", "benchmark results", "main findings", "what are the findings"
        ))

        # 6. Conclusion queries
        is_conclusion_q = not is_novelty_q and not is_summary_q and any(t in query_words for t in (
            "conclusion", "conclusions", "future", "limitation", "limitations"
        ))

        if is_novelty_q:
            expanded_query_words.update([
                "objective", "objectives", "propose", "proposes", "proposed", "scheme", "schemes",
                "contribution", "contributions", "novelties", "novelty", "innovations", "innovation",
                "conclusions", "conclusion", "abstract"
            ])
        elif is_summary_q:
            expanded_query_words.update([
                "abstract", "introduction", "overview", "proposed", "conclusion", "novelty", "contribution"
            ])
        elif is_method_q:
            expanded_query_words.update([
                "method", "methodology", "architecture", "mechanism", "algorithm",
                "pipeline", "framework", "workflow", "process", "procedure", "step", "steps"
            ])
        elif is_eval_q:
            expanded_query_words.update([
                "result", "results", "performance", "benchmark", "benchmarks",
                "accuracy", "metrics", "evaluation", "experiment", "experiments", "findings"
            ])

        # Stemmed term frequency match in canonical name, aliases, definition, evidence, and Tier 1 blocks
        name_words = get_stems({w for w in re.findall(r"\w+", node.canonical_name.lower()) if len(w) > 2})
        for alias in node.aliases:
            name_words.update(get_stems({w for w in re.findall(r"\w+", alias.lower()) if len(w) > 2}))
        def_words = get_stems({w for w in re.findall(r"\w+", node.definition_text.lower()) if len(w) > 2})
        ev_words = get_stems({w for w in re.findall(r"\w+", " ".join(node.evidence_texts).lower()) if len(w) > 2})

        overlap = len(expanded_query_words & name_words)
        query_coverage = overlap / max(1, len(expanded_query_words))
        name_coverage = overlap / max(1, len(name_words))
        name_match = max(query_coverage, name_coverage)

        # Check direct substring matching on canonical name and aliases
        clean_q_phrase = " ".join([w for w in re.findall(r"\w+", query_lower) if w not in stopwords and len(w) > 2])
        if clean_q_phrase and clean_q_phrase in node.canonical_name.lower():
            name_match = 1.0
        elif query_coverage >= 1.0:
            name_match = max(name_match, 0.95)
        elif query_coverage >= 0.5:
            name_match = max(name_match, 0.65)
        elif node.canonical_name.lower() in query_lower or any(w in node.canonical_name.lower() for w in expanded_query_words if len(w) > 3):
            name_match = max(name_match, 0.40)

        for alias in node.aliases:
            if clean_q_phrase and clean_q_phrase in alias.lower():
                name_match = 1.0
            elif alias.lower() in query_lower or any(w in alias.lower() for w in expanded_query_words if len(w) > 3):
                name_match = max(name_match, 0.60)

        # Inherent priority boost for contribution nodes on novelty queries
        if is_novelty_q and any(a in ("Novelties", "Novelty", "Key Novelties", "Contributions", "Contribution", "Innovations") for a in node.aliases):
            name_match = max(name_match, 0.95)

        def_match = len(expanded_query_words & def_words) / max(1, len(expanded_query_words))
        if clean_q_phrase and clean_q_phrase in node.definition_text.lower():
            def_match = max(def_match, 0.90)
        ev_match = len(expanded_query_words & ev_words) / max(1, len(expanded_query_words))

        # Full-text Tier 1 block evidence scoring
        block_match = 0.0
        if hasattr(self, "blocks") and node.source_block_ids:
            for bid in node.source_block_ids:
                blk = self.blocks.get(bid)
                if blk:
                    b_words = get_stems({w for w in re.findall(r"\w+", blk.text_content.lower()) if len(w) > 2})
                    b_overlap = len(expanded_query_words & b_words)
                    m = b_overlap / max(1, len(expanded_query_words))
                    if m > block_match:
                        block_match = m

        base_relevance = 0.45 * name_match + 0.20 * def_match + 0.15 * ev_match + 0.20 * block_match

        # Contextual boost for multi-turn follow-up queries (e.g. "more content", "expand", "elaborate")
        if is_followup and context_node_ids:
            if node.node_id in context_node_ids:
                base_relevance = max(base_relevance, 0.90)
            elif node.parent_id in context_node_ids:
                base_relevance = max(base_relevance, 0.96)
            elif any(anc in context_node_ids for anc in self._get_node_ancestors(node.node_id)):
                base_relevance = max(base_relevance, 0.93)
            elif any(
                (e.source_id in context_node_ids and e.target_id == node.node_id) or
                (e.target_id in context_node_ids and e.source_id == node.node_id)
                for e in self.edges
            ):
                base_relevance = max(base_relevance, 0.86)

        # Direct exact match boost: if node definition, name, or evidence matches all numerical query targets
        num_query_words = {w for w in meaningful_query_words if w.isdigit()}
        combined_node_text = f"{node.canonical_name} {node.definition_text} {' '.join(node.evidence_texts)}".lower()
        if num_query_words and all(nw in combined_node_text for nw in num_query_words):
            base_relevance = max(base_relevance, 0.92)
        elif meaningful_query_words and len(meaningful_query_words) >= 2 and all(w in combined_node_text for w in meaningful_query_words if len(w) > 3):
            base_relevance = max(base_relevance, 0.88)

        # Strict isolation for factual needle queries: unrelated nodes must return 0.0
        if not is_followup and mode == "FACTUAL" and name_match == 0 and def_match < 0.15 and ev_match < 0.15 and block_match < 0.15 and not (num_query_words and any(nw in combined_node_text for nw in num_query_words)):
            if node.depth == 0 and is_summary_q:
                return 0.50
            return 0.0

        # Intent-driven adaptive scoring (for THEMATIC, MULTIHOP, PARAMETRIC, summary or novelty queries)
        if is_novelty_q:
            c_name_lower = node.canonical_name.lower()
            if any(w in c_name_lower for w in (
                "future work", "future", "related work", "limitations", "limitation",
                "performance analysis", "running time", "memory footprint", "research status", "references"
            )):
                return 0.05
            if node.depth == 0:
                base_relevance = max(base_relevance, 0.92)
            elif any(w in c_name_lower for w in ("novelty", "novelties", "contribution", "contributions", "scheme", "proposed", "architecture", "method", "key", "agreement", "security", "initialization", "authentication", "membership")):
                base_relevance = max(base_relevance, 0.95)
            elif any(a.lower() in ("novelties", "novelty", "key novelties", "contributions", "contribution", "innovations") for a in node.aliases):
                base_relevance = max(base_relevance, 0.98)
            elif node.depth == 1:
                base_relevance = max(base_relevance, 0.75)
            elif node.depth == 2:
                base_relevance = max(base_relevance, 0.55)
        elif is_summary_q:
            if node.depth == 0:
                base_relevance = max(base_relevance, 0.96)
            elif any(w in node.canonical_name.lower() for w in ("abstract", "introduction", "conclusion", "contribution", "novelties")):
                base_relevance = max(base_relevance, 0.88)
            elif node.depth == 1:
                base_relevance = max(base_relevance, 0.72)
            elif node.depth == 2:
                base_relevance = max(base_relevance, 0.50)
        elif is_method_q and mode != "FACTUAL":
            if any(w in node.canonical_name.lower() for w in ("method", "architecture", "proposed", "protocol", "scheme", "system", "design", "model", "algorithm")):
                base_relevance = max(base_relevance, 0.92)
            elif node.depth == 1 and not any(w in node.canonical_name.lower() for w in ("references", "related")):
                base_relevance = max(base_relevance, 0.65)
        elif is_eval_q and mode != "FACTUAL":
            if any(w in node.canonical_name.lower() for w in ("experiment", "evaluation", "result", "performance", "benchmark", "analysis")):
                base_relevance = max(base_relevance, 0.92)
            elif node.depth == 1 and any(w in node.canonical_name.lower() for w in ("result", "experiment")):
                base_relevance = max(base_relevance, 0.75)
        elif is_conclusion_q and mode != "FACTUAL":
            if any(w in node.canonical_name.lower() for w in ("conclusion", "conclusions", "future", "limitation", "discussion")):
                base_relevance = max(base_relevance, 0.92)

        # Baseline for overview queries: guarantee root and major sections provide grounded context
        if is_summary_q or is_novelty_q or mode == "THEMATIC":
            if node.depth == 0:
                base_relevance = max(base_relevance, 0.60)
            elif node.depth == 1 and not any(w in node.canonical_name.lower() for w in ("references", "ref")):
                base_relevance = max(base_relevance, 0.35)

        if base_relevance <= 0.035:
            return 0.0

        if is_followup and context_node_ids and base_relevance >= 0.85:
            return base_relevance

        # Mode-dependent weighting
        if mode == "THEMATIC":
            # Favor abstract / high-level nodes (near root)
            relevance = base_relevance * (0.6 + 0.4 * node.abstraction_level)
            if node.depth <= 2:
                relevance += 0.25
        elif mode == "FACTUAL":
            # Favor deep, concrete evidence nodes
            relevance = base_relevance * (0.7 + 0.3 * (1.0 - node.abstraction_level))
            if ev_match > 0.10 or block_match > 0.10:
                relevance += 0.25
        elif mode == "MULTIHOP":
            # Balanced, boosted by confidence
            relevance = base_relevance * 0.8 + node.confidence * 0.2
        else:  # PARAMETRIC
            relevance = base_relevance * 0.9

        return min(1.0, max(0.0, relevance))

    def retrieve(
        self,
        query: str,
        token_budget: Optional[int] = None,
        doc_id: Optional[str] = None,
        is_followup: bool = False,
        context_node_ids: Optional[List[str]] = None,
    ) -> RetrievalResult:
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
        target_doc_id = doc_id
        if not target_doc_id and self.documents:
            is_cross_doc = any(w in query.lower() for w in [
                "compare", "comparison", "difference between", "both papers", "both documents",
                "all papers", "all documents", "across documents", "across papers"
            ])
            if not is_cross_doc:
                target_doc_id = list(self.documents.keys())[-1]

        candidate_nodes = self.nodes
        if target_doc_id:
            matching = {nid: n for nid, n in self.nodes.items() if getattr(n, "doc_id", None) == target_doc_id}
            if matching:
                candidate_nodes = matching
            else:
                doc_blocks = {bid for bid, b in self.blocks.items() if b.doc_id == target_doc_id}
                candidate_nodes = {nid: n for nid, n in self.nodes.items() if any(b in doc_blocks for b in n.source_block_ids)}

        is_heading_q = any(t in re.findall(r"\w+", query.lower()) for t in (
            "heading", "headings", "section", "sections", "outline", "toc",
            "table of contents", "structure", "subheading", "subheadings", "index"
        ))
        if is_heading_q and effective_budget < 2500:
            effective_budget = 2500
        if is_followup and effective_budget < 2500:
            effective_budget = 2500

        knapsack_items: Dict[str, KnapsackItem] = {}
        for nid, node in candidate_nodes.items():
            rel_score = self._score_node_relevance(
                query,
                node,
                mode,
                is_followup=is_followup,
                context_node_ids=context_node_ids,
            )
            # Parent prerequisites (scoped to available candidate nodes)
            parents = [p for p in self.forest_parents_map.get(nid, []) if p in candidate_nodes]
            # Token cost (definition + evidence)
            if is_heading_q and (
                bool(re.match(r"^(?:[1-9]\d*(?:\.\d+)*|[A-H]\.\d+|Appendix\s+[A-H]|SEC-)\b", node.canonical_name))
                or (node.depth in (1, 2) and not node.canonical_name.endswith("Details") and "Novelties" not in node.canonical_name)
            ):
                cost = 15
            else:
                cost = max(20, node.token_cost if node.token_cost > 0 else (len(node.definition_text.split()) + 25))

            knapsack_items[nid] = KnapsackItem(
                node_id=nid,
                relevance_score=rel_score,
                token_cost=cost,
                parent_ids=parents,
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
            doc_id=doc_id,
        )

    # ─────────────────────────────────────────────────────────────
    # Layer 7: Generation & Attribution Verification
    # ─────────────────────────────────────────────────────────────

    def generate_and_verify(
        self,
        retrieval_result: RetrievalResult,
        chat_history: Optional[List[Dict[str, str]]] = None,
        is_followup: bool = False,
        original_query: Optional[str] = None,
    ) -> GenerationResult:
        """
        Executes Layer 7:
          1. Assembles grounded synthesis from retrieved context.
          2. Explicitly tags citations [KN-xxxx] for all factual claims.
          3. Evaluates claim factual support via ClaimAttributionVerifier.
          4. Returns continuous attribution reward R in [0.0, 1.0].
        """
        selected_nodes = {nid: self.nodes[nid] for nid in retrieval_result.selected_node_ids}

        # Synthesize answer using selected nodes with explicit citations and pre-verification
        if not selected_nodes:
            answer = "I could not find sufficient grounded evidence in the Knowledge Forest to answer this query."
            reward = 0.0
            records = []
            formatted_answer = answer
        else:
            synth_out = self.synthesizer.synthesize_and_verify(
                query=retrieval_result.query,
                retrieval_res=retrieval_result,
                pipeline_ref=self,
                scoped_doc_id=getattr(retrieval_result, "doc_id", None),
                chat_history=chat_history,
                is_followup=is_followup,
                original_query=original_query,
            )
            answer = synth_out["plain_answer"]
            formatted_answer = synth_out.get("formatted_answer", answer)
            reward = synth_out["attribution_reward"]
            records = synth_out["attribution_records"]

        return GenerationResult(
            query=retrieval_result.query,
            answer=answer,
            formatted_answer=formatted_answer if 'formatted_answer' in locals() else answer,
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

    def run_query(
        self,
        query: str,
        token_budget: Optional[int] = None,
        doc_id: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Runs the complete SHIA-RAG 2.0 loop for a user query:
        Multi-turn Resolution -> Retrieve -> Generate & Verify -> Evolve Online.
        """
        is_follow = self.is_followup_query(query)
        effective_query = query
        context_node_ids = list(self.last_selected_node_ids)

        if is_follow:
            resolved_topic, hist_node_ids = self.resolve_conversational_topic(chat_history)
            if hist_node_ids:
                context_node_ids = hist_node_ids
            if resolved_topic:
                effective_query = f"{resolved_topic} {query} details mechanisms components architecture functions"
                logger.info(f"Follow-up query '{query}' resolved to context: '{resolved_topic}' (context nodes: {context_node_ids})")

        retrieval_res = self.retrieve(
            effective_query,
            token_budget=token_budget,
            doc_id=doc_id,
            is_followup=is_follow,
            context_node_ids=context_node_ids if is_follow else None,
        )
        gen_res = self.generate_and_verify(
            retrieval_res,
            chat_history=chat_history,
            is_followup=is_follow,
            original_query=query,
        )
        feedback_res = self.feedback_and_evolve(gen_res)

        # Update conversational state
        if retrieval_res.selected_node_ids:
            self.last_selected_node_ids = list(retrieval_res.selected_node_ids)
        if not is_follow:
            self.last_query = query
            self.last_topic = query

        self.conversation_history.append({
            "query": query,
            "effective_query": effective_query,
            "is_followup": is_follow,
            "answer": gen_res.answer,
            "selected_nodes": retrieval_res.selected_node_ids,
        })

        return {
            "query": query,
            "effective_query": effective_query,
            "is_followup": is_follow,
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
            "formatted_answer": getattr(gen_res, "formatted_answer", None) or gen_res.answer,
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
