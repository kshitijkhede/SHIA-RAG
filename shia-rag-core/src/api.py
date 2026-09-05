"""
SHIA-RAG 2.0: FastAPI HTTP Service Endpoint
===========================================
Provides RESTful API endpoints for:
  - Health check: GET /health
  - System statistics: GET /stats
  - Query execution: POST /query
  - PDF document ingestion: POST /ingest
  - Forest invariant validation: GET /invariants

Reference: docker/Dockerfile.api
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, File, HTTPException, UploadFile
FASTAPI_AVAILABLE = True

from src.pipeline import SHIARAGPipeline

logger = logging.getLogger("SHIARAG_API")

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="SHIA-RAG 2.0 API",
        description="Semantic Hierarchy Induction Architecture for Retrieval-Augmented Generation REST API",
        version="2.0.0",
    )
    pipeline = SHIARAGPipeline()
    pipeline.load_sample_knowledge_base()

    class QueryRequest(BaseModel):
        query: str = Field(..., min_length=1, description="User question or query")
        token_budget: Optional[int] = Field(default=2048, ge=128, le=8192)

    @app.get("/health")
    def health_check() -> Dict[str, str]:
        return {"status": "healthy", "service": "shia-rag-2.0"}

    @app.get("/stats")
    def get_stats() -> Dict[str, Any]:
        return pipeline.get_forest_statistics()

    @app.get("/invariants")
    def check_invariants() -> Dict[str, str]:
        try:
            pipeline.validate_invariants()
            return {"status": "valid", "invariants": "All forest invariants hold."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/query")
    def execute_query(req: QueryRequest) -> Dict[str, Any]:
        try:
            return pipeline.run_query(req.query, token_budget=req.token_budget)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ingest")
    async def ingest_document(file: UploadFile = File(...)) -> Dict[str, Any]:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF documents are supported.")
        temp_path = Path("/tmp") / file.filename
        try:
            content = await file.read()
            with open(temp_path, "wb") as f:
                f.write(content)
            return pipeline.ingest_pdf(temp_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if temp_path.exists():
                temp_path.unlink()
else:
    app = None
