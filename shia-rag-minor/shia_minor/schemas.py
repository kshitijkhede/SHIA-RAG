"""
Layer 0 --- Data model (Pydantic v2).

Two tiers, kept separate so physical layout is not mixed with semantic content:
  Tier 1 : BoundingBox, TextBlock   (physical page geometry)
  Tier 2 : ConceptNode, HierarchyEdge, CrossLinkEdge   (semantic hierarchy)

Reference: report Chapter 3 (Data Model).
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

MAX_DEPTH = 8  # D_max


class BoundingBox(BaseModel):
    """Physical rectangle on a page, top-left origin, in PDF points."""
    x0: float = Field(ge=0)
    y0: float = Field(ge=0)
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> "BoundingBox":
        if self.x1 <= self.x0:
            raise ValueError(f"x1 ({self.x1}) must be > x0 ({self.x0})")
        if self.y1 <= self.y0:
            raise ValueError(f"y1 ({self.y1}) must be > y0 ({self.y0})")
        return self

    @property
    def width(self) -> float:
        return self.x1 - self.x0


class TextBlock(BaseModel):
    """Tier 1: a physical text block with geometry and font attributes."""
    block_id: int
    page: int = Field(ge=1)
    text: str
    bbox: BoundingBox
    font_size: float = Field(gt=0)
    is_bold: bool = False
    column: int = 0                      # 0 = left / single, 1 = right, -1 = spanning
    reading_order: int = Field(default=0, ge=0)


class EvidenceSpan(BaseModel):
    """A grounding pointer from a Tier 2 concept back to a Tier 1 block."""
    page: int
    block_id: int
    bbox: BoundingBox
    text: str


class ConceptNode(BaseModel):
    """Tier 2: a concept (usually a heading) in the induced hierarchy."""
    concept_id: str
    name: str
    doc_id: str
    level: int = Field(ge=0, le=6)        # typographic heading level (0=title/root)
    section_number: Optional[str] = None  # e.g. "3.1" when the heading is numbered
    order: int = Field(ge=0)              # rank in reading order
    definition: str = ""
    is_bold: bool = False
    font_ratio: float = 1.0              # font size / body font size
    parent_id: Optional[str] = None
    depth: int = Field(default=0, ge=0)
    local_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: List[EvidenceSpan] = Field(default_factory=list)

    @field_validator("depth")
    @classmethod
    def _depth_bound(cls, v: int) -> int:
        if v > MAX_DEPTH:
            raise ValueError(f"depth {v} exceeds D_max={MAX_DEPTH}")
        return v


class HierarchyEdge(BaseModel):
    """A directed child -> parent edge in the acyclic hierarchy."""
    child_id: str
    parent_id: str

    @model_validator(mode="after")
    def _no_self_loop(self) -> "HierarchyEdge":
        if self.child_id == self.parent_id:
            raise ValueError("self-loop edge is not allowed")
        return self


class CrossLinkEdge(BaseModel):
    """A non-hierarchical semantic association between two concepts."""
    source_id: str
    target_id: str
    similarity: float = Field(ge=0.0, le=1.0)
