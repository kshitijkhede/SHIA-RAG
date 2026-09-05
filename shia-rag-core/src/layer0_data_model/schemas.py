"""
SHIA-RAG Layer 0: Canonical Data Model Schemas
================================================
Pydantic v2 models defining the foundational data structures for
the Dual-Tier Heterogeneous Knowledge Forest (HKF).

These schemas serve as the single source of truth for all inter-layer
data contracts across the 9-layer pipeline.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── Enumerations ──

class NodeType(str, Enum):
    """Classification of knowledge node types in Tier 2."""
    CONCEPT = "CONCEPT"
    DEFINITION = "DEFINITION"
    EVIDENCE = "EVIDENCE"
    CLAIM = "CLAIM"
    PROCEDURE = "PROCEDURE"


class HierarchicalRelation(str, Enum):
    """Permitted hierarchical (acyclic) edge predicates."""
    IS_A = "IS_A"
    PART_OF = "PART_OF"
    INSTANCE_OF = "INSTANCE_OF"


class SemanticRelation(str, Enum):
    """Permitted semantic (cross-link) edge predicates."""
    USES = "USES"
    CAUSES = "CAUSES"
    COMPARED_TO = "COMPARED_TO"
    PREREQUISITE_OF = "PREREQUISITE_OF"
    CONTRADICTS = "CONTRADICTS"


class EdgeCategory(str, Enum):
    """Top-level edge category."""
    HIERARCHICAL = "HIERARCHICAL"
    SEMANTIC = "SEMANTIC"
    PROJECTION = "PROJECTION"


class DocumentMimeType(str, Enum):
    """Supported document input formats."""
    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    HTML = "text/html"
    MARKDOWN = "text/markdown"
    PLAIN = "text/plain"


# ── Tier 1: Syntactic Document Tree ──

class DocumentNode(BaseModel):
    """Represents a source document in Tier 1."""
    doc_id: str = Field(default_factory=lambda: f"DOC-{uuid.uuid4().hex[:8].upper()}")
    filename: str
    mime_type: DocumentMimeType
    sha256_hash: str = Field(..., min_length=64, max_length=64)
    total_pages: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_config = ConfigDict(frozen=True)


class TextBlock(BaseModel):
    """A structural text block within a document (paragraph, heading, table cell)."""
    block_id: str = Field(default_factory=lambda: f"BLK-{uuid.uuid4().hex[:8].upper()}")
    doc_id: str
    reading_order: int = Field(..., ge=0, description="Sequential position in document reading order")
    text_content: str = Field(..., min_length=1)
    section_path: str = Field(default="", description="Dot-separated section hierarchy, e.g., '2.3.1'")
    char_start: int = Field(default=0, ge=0)
    char_end: int = Field(default=0, ge=0)
    page_number: Optional[int] = None
    bbox_coords: Optional[List[float]] = Field(default=None, description="[x0, y0, x1, y1] bounding box")

    @field_validator("char_end")
    @classmethod
    def end_after_start(cls, v: int, info) -> int:
        start = info.data.get("char_start", 0)
        if v > 0 and v < start:
            raise ValueError(f"char_end ({v}) must be >= char_start ({start})")
        return v


# ── Tier 2: Semantic Concept Forest ──

class KnowledgeNode(BaseModel):
    """
    A canonical concept node in the Semantic Concept Forest (Tier 2).
    
    Each node represents a deduplicated, abstracted knowledge unit
    with provenance anchors back to Tier 1 text blocks.
    """
    node_id: str = Field(default_factory=lambda: f"KN-{uuid.uuid4().hex[:6].upper()}")
    canonical_name: str = Field(..., min_length=1, max_length=512)
    node_type: NodeType = NodeType.CONCEPT
    domain: str = Field(default="general", description="Domain classification, e.g., 'networking'")
    definition_text: str = Field(default="", description="Concise definition of this concept")
    evidence_texts: List[str] = Field(default_factory=list, description="Supporting evidence snippets")
    
    # Quantitative attributes
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="KCE confidence score")
    abstraction_level: float = Field(default=0.5, ge=0.0, le=1.0, description="0=leaf/concrete, 1=root/abstract")
    token_cost: int = Field(default=0, ge=0, description="BPE token footprint of definition + evidence")
    
    # Embedding (stored separately in Milvus, referenced by node_id)
    embedding_dim: int = Field(default=768)
    
    # Provenance anchors (E_proj pointers to Tier 1)
    source_block_ids: List[str] = Field(default_factory=list, description="Block IDs this concept was extracted from")
    
    # Hierarchy position (computed after forest construction)
    depth: int = Field(default=0, ge=0, description="Depth in the concept forest (0 = root)")
    parent_id: Optional[str] = Field(default=None, description="Hierarchical parent node_id")
    
    # Metadata
    aliases: List[str] = Field(default_factory=list, description="Alternative names / abbreviations")
    lsh_fingerprint: Optional[str] = Field(default=None, description="MinHash LSH signature for dedup")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="before")
    @classmethod
    def handle_confidence_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "confidence_score" in data and "confidence" not in data:
                data["confidence"] = data["confidence_score"]
        return data

    @property
    def confidence_score(self) -> float:
        """Alias for confidence."""
        return self.confidence

    @field_validator("node_id")
    @classmethod
    def validate_node_id_format(cls, v: str) -> str:
        if not v.startswith("KN-"):
            raise ValueError(f"node_id must start with 'KN-', got '{v}'")
        return v


class KnowledgeEdge(BaseModel):
    """
    An edge in the Knowledge Forest connecting two KnowledgeNodes.
    
    Hierarchical edges form the acyclic taxonomy DAG.
    Semantic edges form the cross-link web (may contain cycles).
    """
    edge_id: str = Field(default_factory=lambda: f"E-{uuid.uuid4().hex[:8].upper()}")
    source_id: str = Field(..., description="Source node_id")
    target_id: str = Field(..., description="Target node_id")
    category: EdgeCategory
    predicate: str = Field(..., description="Relation predicate, e.g., IS_A, USES, CAUSES")
    
    # Thompson Sampling Beta priors
    alpha: float = Field(default=1.0, ge=1.0, description="Beta prior successes (α)")
    beta: float = Field(default=1.0, ge=1.0, description="Beta prior failures (β)")
    
    # Confidence & provenance
    extraction_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    extraction_method: str = Field(default="llm", description="How this edge was discovered: llm | pattern | manual")
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def no_self_loops(self) -> "KnowledgeEdge":
        if self.source_id == self.target_id:
            raise ValueError(f"Self-loop detected: source_id == target_id == '{self.source_id}'")
        return self

    @field_validator("predicate")
    @classmethod
    def validate_predicate(cls, v: str) -> str:
        all_predicates = {e.value for e in HierarchicalRelation} | {e.value for e in SemanticRelation} | {"GROUNDED_IN"}
        if v not in all_predicates:
            raise ValueError(f"Unknown predicate '{v}'. Must be one of: {sorted(all_predicates)}")
        return v


# ── Retrieval & Context Assembly ──

class RetrievalRequest(BaseModel):
    """Input to the retrieval engine (Layer 6)."""
    query: str = Field(..., min_length=1)
    token_budget: int = Field(default=3072, ge=256)
    mode_override: Optional[str] = Field(default=None, description="Force a specific SRDR mode")
    session_id: Optional[str] = None


class RetrievalResult(BaseModel):
    """Output from the retrieval engine after DC-Knapsack optimization."""
    query: str
    mode: str = Field(..., description="SRDR routing mode used")
    selected_node_ids: List[str]
    total_tokens: int
    total_utility: float
    traversed_edge_ids: List[str] = Field(default_factory=list)
    context_text: str = Field(default="", description="Assembled context string for LLM prompt")


class GenerationResult(BaseModel):
    """Output from the generation and attribution verification layer (Layer 7)."""
    query: str
    answer: str
    attribution_reward: float = Field(ge=0.0, le=1.0, description="Fraction of claims verified")
    attribution_records: List[dict] = Field(default_factory=list)
    selected_node_ids: List[str]
    traversed_edge_ids: List[str]
    total_context_tokens: int
    generation_tokens: int = Field(default=0)
