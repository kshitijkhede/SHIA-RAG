"""
SHIA-RAG 2.0: FastAPI HTTP & Web Service Endpoint
===================================================
Provides RESTful API endpoints and serves the ChatGPT & Claude AI interface:
  - UI Web App: GET /
  - Health check: GET /health and GET /api/health
  - System statistics: GET /stats and GET /api/stats
  - Query execution: POST /query and POST /api/query
  - PDF document ingestion: POST /ingest and POST /api/upload
  - Forest invariant validation: GET /invariants and GET /api/invariants
  - Forest graph visualization: GET /forest and GET /api/forest
  - Node detail & evidence inspection: GET /node/{node_id} and GET /api/node/{node_id}
  - Ingested documents list: GET /documents and GET /api/documents
  - Sample knowledge base preload: POST /preload_sample and POST /api/preload_sample
  - Forest reset: POST /reset and POST /api/reset
  - Sample queries for active document: GET /sample_queries and GET /api/sample_queries

Reference: Chapter 10 & 11, SHIA_RAG_Complete_Report.md
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.layer0_data_model.schemas import EdgeCategory, NodeType
from src.pipeline import SHIARAGPipeline

logger = logging.getLogger("SHIARAG_API")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

STATIC_DIR = _SRC_DIR / "web" / "static"
UPLOADS_DIR = _PROJECT_ROOT / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

def load_env_file():
    """Auto-loads environment variables from .env files without external dependencies."""
    for p in [_PROJECT_ROOT / ".env", _PROJECT_ROOT.parent / ".env", Path.home() / ".env"]:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip().strip("'\"")
                            if k and k not in os.environ:
                                os.environ[k] = v
            except Exception as e:
                logger.warning("Failed reading .env from %s: %s", p, e)

    if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GEMINI_ACTIVE_MODEL"):
        os.environ["GEMINI_ACTIVE_MODEL"] = "gemini-3.1-flash-lite"

load_env_file()

app = FastAPI(
    title="SHIA-RAG 2.0 Web Service",
    description="Dual-Tier Heterogeneous Knowledge Forest RAG Interface (ChatGPT & Claude AI Aesthetic)",
    version="2.0.0",
)

# Enable CORS for local cross-origin development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (starts clean with 0 documents and 0 nodes)
pipeline = SHIARAGPipeline()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question or query")
    token_budget: Optional[int] = Field(default=2048, ge=128, le=8192)
    doc_id: Optional[str] = Field(default=None, description="Optional document ID to scope retrieval to its dedicated tree")
    chat_history: Optional[List[Dict[str, str]]] = Field(default=None, description="Recent conversation turns for multi-turn coherence")


def _format_natural_grounded_answer(
    query: str,
    res: Any,
    pipeline_ref: SHIARAGPipeline,
    scoped_doc_id: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    is_followup: bool = False,
) -> str:
    """
    Formats the grounded retrieval results into a clean, rich conversational answer
    matching ChatGPT & Claude AI style, strictly sanitized of raw node IDs, LaTeX math noise, and dollar signs.
    """
    from src.layer7_generation.synthesizer import clean_answer_output
    effective_q = res.get("effective_query", query) if isinstance(res, dict) else query
    synth_out = pipeline_ref.synthesizer.synthesize_and_verify(
        query=effective_q,
        retrieval_res=res,
        pipeline_ref=pipeline_ref,
        scoped_doc_id=scoped_doc_id,
        chat_history=chat_history,
        is_followup=is_followup,
        original_query=query,
    )
    formatted = clean_answer_output(synth_out.get("formatted_answer", ""))
    plain = clean_answer_output(synth_out.get("plain_answer", ""))
    if isinstance(res, dict):
        res["attribution_reward"] = synth_out.get("attribution_reward", 1.0)
        res["attribution_records"] = synth_out.get("attribution_records", [])
        res["verified"] = synth_out.get("verified", True)
        res["plain_answer"] = plain
        res["formatted_answer"] = formatted
    return formatted



# ─────────────────────────────────────────────────────────────
# Core API Endpoints
# ─────────────────────────────────────────────────────────────

@app.post("/reset_chat")
@app.post("/api/reset_chat")
def reset_chat() -> Dict[str, str]:
    pipeline.reset_conversation()
    return {"status": "success", "message": "Conversational memory reset."}


@app.get("/health")
@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    stats = pipeline.get_forest_statistics()
    return {
        "status": "healthy",
        "service": "shia-rag-2.0",
        "total_nodes": stats["total_nodes"],
        "total_edges": stats["total_edges"],
        "max_depth": stats["max_depth"],
        "avg_confidence": stats["avg_confidence"],
    }


@app.get("/stats")
@app.get("/api/stats")
def get_stats() -> Dict[str, Any]:
    return pipeline.get_forest_statistics()


@app.get("/invariants")
@app.get("/api/invariants")
def check_invariants() -> Dict[str, str]:
    try:
        pipeline.validate_invariants()
        return {"status": "valid", "invariants": "All forest invariants hold: Acyclic DAG taxonomy maintained."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
@app.post("/api/query")
def execute_query(req: QueryRequest) -> Dict[str, Any]:
    try:
        target_doc = req.doc_id
        if (not target_doc or not str(target_doc).strip()) and pipeline.documents:
            is_cross_doc = any(w in req.query.lower() for w in [
                "compare", "comparison", "difference between", "both papers", "both documents",
                "all papers", "all documents", "across documents", "across papers"
            ])
            if not is_cross_doc:
                target_doc = list(pipeline.documents.keys())[-1]

        raw_res = pipeline.run_query(
            req.query,
            token_budget=req.token_budget,
            doc_id=target_doc,
            chat_history=req.chat_history,
        )
        natural_answer = _format_natural_grounded_answer(
            req.query,
            raw_res,
            pipeline,
            scoped_doc_id=target_doc,
            chat_history=req.chat_history,
            is_followup=raw_res.get("is_followup", False),
        )
        
        # Enrich selected node details for instant frontend display
        enriched_nodes = []
        for n_summary in raw_res.get("selected_nodes", []):
            nid = n_summary["node_id"]
            node_obj = pipeline.nodes.get(nid)
            if node_obj:
                enriched_nodes.append({
                    "node_id": nid,
                    "name": node_obj.canonical_name,
                    "node_type": node_obj.node_type.value if hasattr(node_obj.node_type, "value") else str(node_obj.node_type),
                    "depth": node_obj.depth,
                    "confidence": node_obj.confidence,
                    "definition": node_obj.definition_text,
                    "evidence": node_obj.evidence_texts[:2],
                    "parent_id": node_obj.parent_id,
                })
            else:
                enriched_nodes.append(n_summary)

        return {
            **raw_res,
            "doc_id": target_doc,
            "formatted_answer": natural_answer,
            "enriched_nodes": enriched_nodes,
        }
    except Exception as e:
        logger.exception("Error executing query")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
@app.post("/api/upload")
async def ingest_document(
    file: UploadFile = File(...),
    clear_existing: bool = Form(False),
    allow_cross_document: bool = Form(False),
) -> Dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")
    
    # Save file persistently in uploads directory
    target_path = UPLOADS_DIR / file.filename
    try:
        content = await file.read()
        with open(target_path, "wb") as f:
            f.write(content)
        
        ingest_result = pipeline.ingest_pdf(
            target_path,
            clear_existing=clear_existing,
            allow_cross_document=allow_cross_document,
        )
        return {
            "status": "success",
            "message": f"Successfully ingested '{file.filename}' into Knowledge Forest.",
            "data": ingest_result,
        }
    except Exception as e:
        logger.exception(f"Error ingesting PDF: {file.filename}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/document/{doc_id}")
@app.delete("/api/document/{doc_id}")
def delete_document(doc_id: str) -> Dict[str, Any]:
    success = pipeline.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return {
        "status": "success",
        "message": f"Document '{doc_id}' and its dedicated tree deleted.",
        "stats": pipeline.get_forest_statistics(),
    }


@app.get("/documents")
@app.get("/api/documents")
def list_documents() -> List[Dict[str, Any]]:
    docs = []
    for doc in pipeline.documents.values():
        doc_blocks = [b for b in pipeline.blocks.values() if b.doc_id == doc.doc_id]
        doc_nodes = [
            n for n in pipeline.nodes.values()
            if getattr(n, "doc_id", None) == doc.doc_id
            or any(b_id in [b.block_id for b in doc_blocks] for b_id in n.source_block_ids)
        ]
        # Find root node for this document
        root_nodes = [n for n in doc_nodes if n.parent_id is None]
        root_name = root_nodes[0].canonical_name if root_nodes else (doc_nodes[0].canonical_name if doc_nodes else "Taxonomy Root")
        root_id = root_nodes[0].node_id if root_nodes else (doc_nodes[0].node_id if doc_nodes else "")
        max_d = max([n.depth for n in doc_nodes], default=0)
        docs.append({
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "total_pages": doc.total_pages,
            "sha256": doc.sha256_hash[:12] + "..." if doc.sha256_hash else "",
            "blocks_count": len(doc_blocks),
            "concepts_count": len(doc_nodes),
            "root_id": root_id,
            "root_name": root_name,
            "max_depth": max_d,
        })
    return docs


@app.get("/forest")
@app.get("/api/forest")
def get_forest_graph(doc_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns forest graph structure for interactive visualization:
    nodes, hierarchical edges, semantic cross-links, and per-document tree segmentation.
    """
    # Filter nodes by document if specified
    if isinstance(doc_id, str) and doc_id.strip():
        matching = {nid: n for nid, n in pipeline.nodes.items() if getattr(n, "doc_id", None) == doc_id}
        if not matching:
            doc_blocks = {bid for bid, b in pipeline.blocks.items() if b.doc_id == doc_id}
            matching = {nid: n for nid, n in pipeline.nodes.items() if any(b in doc_blocks for b in n.source_block_ids)}
        target_nodes = matching
    else:
        target_nodes = pipeline.nodes

    target_node_ids = set(target_nodes.keys())
    doc_names = {d.doc_id: d.filename for d in pipeline.documents.values()}

    nodes_data = []
    for nid, node in target_nodes.items():
        n_doc_id = getattr(node, "doc_id", None)
        if not n_doc_id and node.source_block_ids:
            first_b = pipeline.blocks.get(node.source_block_ids[0])
            if first_b:
                n_doc_id = first_b.doc_id
        doc_name = doc_names.get(n_doc_id, "Document") if n_doc_id else "Document"

        nodes_data.append({
            "id": nid,
            "name": node.canonical_name,
            "type": node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type),
            "depth": node.depth,
            "confidence": round(node.confidence, 3),
            "definition": node.definition_text,
            "parent_id": node.parent_id,
            "aliases": node.aliases,
            "evidence_count": len(node.evidence_texts),
            "doc_id": n_doc_id,
            "doc_name": doc_name,
        })

    edges_data = []
    for edge in pipeline.edges:
        if edge.source_id in target_node_ids and edge.target_id in target_node_ids:
            is_hierarchical = (edge.category == EdgeCategory.HIERARCHICAL)
            edges_data.append({
                "id": edge.edge_id,
                "source": edge.source_id,
                "target": edge.target_id,
                "category": edge.category.value if hasattr(edge.category, "value") else str(edge.category),
                "predicate": edge.predicate,
                "confidence": round(edge.extraction_confidence, 3),
                "is_hierarchical": is_hierarchical,
                "alpha": round(edge.alpha, 2),
                "beta": round(edge.beta, 2),
            })

    # Structured document trees summary
    doc_trees = {}
    for d_id, d_obj in pipeline.documents.items():
        d_blocks = {bid for bid, b in pipeline.blocks.items() if b.doc_id == d_id}
        d_nodes = [
            n for n in pipeline.nodes.values()
            if getattr(n, "doc_id", None) == d_id or any(b in d_blocks for b in n.source_block_ids)
        ]
        d_roots = [n for n in d_nodes if n.parent_id is None]
        doc_trees[d_id] = {
            "doc_id": d_id,
            "filename": d_obj.filename,
            "root_id": d_roots[0].node_id if d_roots else "",
            "root_name": d_roots[0].canonical_name if d_roots else d_obj.filename,
            "total_nodes": len(d_nodes),
            "max_depth": max([n.depth for n in d_nodes], default=0),
        }

    stats = pipeline.get_forest_statistics()
    return {
        "nodes": nodes_data,
        "edges": edges_data,
        "stats": stats,
        "doc_trees": doc_trees,
        "active_doc_id": doc_id,
    }


@app.get("/node/{node_id}")
@app.get("/api/node/{node_id}")
def get_node_details(node_id: str) -> Dict[str, Any]:
    """Retrieves full details, evidence texts, and source blocks for a specific node."""
    node = pipeline.nodes.get(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"KnowledgeNode '{node_id}' not found.")

    # Find source blocks
    source_blocks = []
    for b_id in node.source_block_ids:
        b = pipeline.blocks.get(b_id)
        if b:
            source_blocks.append({
                "block_id": b.block_id,
                "doc_id": b.doc_id,
                "page_num": getattr(b, "page_number", None) or getattr(b, "page_num", 1) or 1,
                "section_path": b.section_path,
                "reading_order": b.reading_order,
                "text_content": b.text_content,
            })

    # Find parent and children
    parent_node = pipeline.nodes.get(node.parent_id) if node.parent_id else None
    child_nodes = [
        {"id": cid, "name": c.canonical_name, "depth": c.depth, "conf": round(c.confidence, 2)}
        for cid, c in pipeline.nodes.items()
        if c.parent_id == node_id
    ]

    # Find cross-links
    cross_links = []
    for e in pipeline.edges:
        if e.category == EdgeCategory.SEMANTIC:
            if e.source_id == node_id:
                tgt = pipeline.nodes.get(e.target_id)
                cross_links.append({
                    "direction": "outgoing",
                    "target_id": e.target_id,
                    "target_name": tgt.canonical_name if tgt else e.target_id,
                    "predicate": e.predicate,
                    "confidence": round(e.extraction_confidence, 2),
                })
            elif e.target_id == node_id:
                src = pipeline.nodes.get(e.source_id)
                cross_links.append({
                    "direction": "incoming",
                    "source_id": e.source_id,
                    "source_name": src.canonical_name if src else e.source_id,
                    "predicate": e.predicate,
                    "confidence": round(e.extraction_confidence, 2),
                })

    return {
        "node_id": node.node_id,
        "canonical_name": node.canonical_name,
        "node_type": node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type),
        "depth": node.depth,
        "confidence": round(node.confidence, 3),
        "abstraction_level": round(node.abstraction_level, 2),
        "token_cost": node.token_cost,
        "definition_text": node.definition_text,
        "aliases": node.aliases,
        "parent": {"id": parent_node.node_id, "name": parent_node.canonical_name} if parent_node else None,
        "children": child_nodes,
        "cross_links": cross_links,
        "evidence_texts": node.evidence_texts,
        "source_blocks": source_blocks,
    }


@app.post("/preload_sample")
@app.post("/api/preload_sample")
def preload_sample_kb() -> Dict[str, Any]:
    pipeline.load_sample_knowledge_base()
    stats = pipeline.get_forest_statistics()
    return {
        "status": "success",
        "message": "Computer Science & Artificial Intelligence sample Knowledge Forest reloaded.",
        "stats": stats,
    }


@app.post("/reset")
@app.post("/api/reset")
def reset_forest() -> Dict[str, Any]:
    global pipeline
    pipeline = SHIARAGPipeline()
    return {
        "status": "success",
        "message": "Knowledge Forest has been completely reset. Ready for new PDF ingestion.",
        "stats": pipeline.get_forest_statistics(),
    }


@app.get("/sample_queries")
@app.get("/api/sample_queries")
def get_sample_queries() -> List[Dict[str, str]]:
    """Generates suggested sample queries based on currently active knowledge forest."""
    forest_nodes = list(pipeline.nodes.values())
    roots = [n for n in forest_nodes if n.parent_id is None]
    if not forest_nodes:
        return [
            {"title": "Upload a Document", "query": "Please upload a PDF document first to begin exploring."},
        ]

    root_title = roots[0].canonical_name if roots else "System Overview"
    depth_1_nodes = [n for n in forest_nodes if n.depth == 1]
    deeper_nodes = [n for n in forest_nodes if n.depth >= 2]

    queries = []
    queries.append({
        "title": "Thematic Overview",
        "query": f"Summarize the architecture, purpose, and key principles of {root_title}.",
        "mode": "THEMATIC",
    })

    if deeper_nodes:
        c_needle = deeper_nodes[0].canonical_name
        queries.append({
            "title": "Factual Needle",
            "query": f"What is the exact definition and mechanism of {c_needle}?",
            "mode": "FACTUAL",
        })

    if len(depth_1_nodes) >= 2:
        c1 = depth_1_nodes[0].canonical_name
        c2 = depth_1_nodes[1].canonical_name
        queries.append({
            "title": "Multi-hop Comparison",
            "query": f"Compare {c1} versus {c2} and explain their structural relationships.",
            "mode": "MULTIHOP",
        })
    elif len(forest_nodes) >= 2:
        c1 = forest_nodes[0].canonical_name
        c2 = forest_nodes[1].canonical_name
        queries.append({
            "title": "Multi-hop Comparison",
            "query": f"How does {c1} relate to {c2}?",
            "mode": "MULTIHOP",
        })

    queries.append({
        "title": "Invariant Verification",
        "query": "Verify the forest health, depth bounds, and acyclicity invariants.",
        "mode": "PARAMETRIC",
    })

    return queries


# ─────────────────────────────────────────────────────────────
# Cloud LLM Acceleration Endpoints (Gemini / OpenAI)
# ─────────────────────────────────────────────────────────────

class LLMConfigureRequest(BaseModel):
    provider: str = Field(..., description="'gemini' or 'openai'")
    api_key: str = Field(default="", description="API Key string (empty to revert to local synthesizer)")


@app.get("/api/llm/status")
def get_llm_status() -> Dict[str, Any]:
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    active_model = os.environ.get("GEMINI_ACTIVE_MODEL", "gemini-3.6-flash")

    active_engine = "Local Semantic Synthesizer"
    if gemini_key:
        active_engine = f"Google Gemini ({active_model})"
    elif openai_key:
        active_engine = "OpenAI GPT-4o-mini (Accelerated)"

    return {
        "gemini_configured": bool(gemini_key),
        "openai_configured": bool(openai_key),
        "active_engine": active_engine,
        "active_model": active_model,
    }


def _verify_llm_key(provider: str, api_key: str) -> Tuple[bool, str]:
    """Tests the API key with a fast 1-token ping to ensure it works."""
    import urllib.request
    import urllib.error
    import json

    if provider in ("gemini", "google"):
        if api_key.startswith("sk-"):
            return False, "This is an OpenAI API key (starts with sk-), not a Google Gemini key. Please select OpenAI or get a free Gemini key from Google AI Studio (https://aistudio.google.com/app/apikey)."
        
        # Call ListModels to verify key and discover available model
        list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        try:
            req = urllib.request.Request(list_url, method="GET")
            with urllib.request.urlopen(req, timeout=8) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    models = data.get("models", [])
                    gen_models = [
                        m["name"].replace("models/", "")
                        for m in models
                        if "generateContent" in m.get("supportedGenerationMethods", [])
                    ]
                    logger.info("Discovered Gemini models for key: %s", gen_models[:6] if gen_models else "None")
                    if gen_models:
                        # Prioritize modern supported models that exist for user's key
                        priority = [
                            "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-flash",
                            "gemini-flash-latest", "gemini-3.1-flash-lite", "gemini-3.6-flash"
                        ]
                        preferred = None
                        for p in priority:
                            if p in gen_models:
                                preferred = p
                                break
                        if not preferred:
                            preferred = next(
                                (m for m in gen_models if "flash" in m and "preview" not in m),
                                gen_models[0]
                            )
                        os.environ["GEMINI_ACTIVE_MODEL"] = preferred
                        return True, f"Valid Google Gemini API Key (Using {preferred})"
                    return True, "Valid Google Gemini API Key"
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", str(e))
            except Exception:
                msg = str(e)
            return False, f"Google Gemini API error: {msg}"
        except Exception as e:
            return False, f"Connection to Google Gemini failed: {e}"

    elif provider == "openai":
        if api_key.startswith("AIzaSy") or api_key.startswith("AIza"):
            return False, "This is a Google Gemini key (starts with AIzaSy), not an OpenAI key. Please select Google Gemini as the provider."
        url = "https://api.openai.com/v1/chat/completions"
        payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                if resp.status == 200:
                    return True, "Valid OpenAI API Key"
        except urllib.error.HTTPError as e:
            try:
                err_dict = json.loads(e.read().decode("utf-8")).get("error", {})
                code = err_dict.get("code", "")
                msg = err_dict.get("message", str(e))
                if code == "credit_balance_exhausted" or "credit" in msg.lower() or "quota" in msg.lower():
                    return False, "Your OpenAI account has no credits remaining ($0.00 balance). Please add credits at https://platform.openai.com/settings/organization/billing/ or use a completely free Google Gemini key from https://aistudio.google.com/app/apikey"
            except Exception:
                msg = str(e)
            return False, f"OpenAI API error: {msg}"
        except Exception as e:
            return False, f"Connection to OpenAI failed: {e}"

    return False, f"Unsupported provider: {provider}"


@app.post("/api/llm/configure")
def configure_llm(req: LLMConfigureRequest) -> Dict[str, Any]:
    prov = req.provider.lower().strip()
    key = req.api_key.strip()
    env_file = _PROJECT_ROOT / ".env"

    if not key:
        # Clear configured key
        if prov in ("gemini", "google"):
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ.pop("GOOGLE_API_KEY", None)
            env_var = "GEMINI_API_KEY"
        else:
            os.environ.pop("OPENAI_API_KEY", None)
            env_var = "OPENAI_API_KEY"

        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                lines = [l for l in f if not l.startswith(f"{env_var}=")]
            with open(env_file, "w", encoding="utf-8") as f:
                f.writelines(lines)

        return {
            "status": "success",
            "message": "Cleared cloud LLM key. Switched to Local Synthesizer.",
            "provider": "local",
            "active_engine": "Local Semantic Synthesizer",
        }

    logger.info("configure_llm called: provider=%s, key_len=%d, prefix=%s", prov, len(key), key[:8] if key else "EMPTY")

    # Auto-detect mismatch
    if key.startswith("sk-") and prov in ("gemini", "google"):
        logger.info("Auto-switched provider from %s to openai based on sk- prefix", prov)
        prov = "openai"
    elif (key.startswith("AIzaSy") or key.startswith("AIza")) and prov == "openai":
        logger.info("Auto-switched provider from openai to gemini based on AIza prefix")
        prov = "gemini"

    # Verify key works before saving
    is_valid, validation_msg = _verify_llm_key(prov, key)
    logger.info("Key verification result: is_valid=%s, msg=%s", is_valid, validation_msg)
    if not is_valid:
        raise HTTPException(status_code=400, detail=validation_msg)

    if prov in ("gemini", "google"):
        os.environ["GEMINI_API_KEY"] = key
        env_var = "GEMINI_API_KEY"
    elif prov == "openai":
        os.environ["OPENAI_API_KEY"] = key
        env_var = "OPENAI_API_KEY"
    else:
        raise HTTPException(status_code=400, detail="Unsupported provider. Please specify 'gemini' or 'openai'.")

    # Persist key to project .env file
    try:
        existing_lines = []
        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                existing_lines = [line for line in f if not line.startswith(f"{env_var}=")]
        existing_lines.append(f"{env_var}={key}\n")
        with open(env_file, "w", encoding="utf-8") as f:
            f.writelines(existing_lines)
    except Exception as e:
        logger.warning("Could not persist API key to %s: %s", env_file, e)

    return {
        "status": "success",
        "message": f"Successfully verified and connected {prov.title()}!",
        "provider": prov,
        "active_engine": f"{prov.title()} Cloud Accelerated",
    }


# ─────────────────────────────────────────────────────────────
# Static Assets & Frontend Web Application
# ─────────────────────────────────────────────────────────────

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def serve_index() -> FileResponse:
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(
                index_file,
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
