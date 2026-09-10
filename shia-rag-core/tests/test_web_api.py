"""
Unit & Integration tests for SHIA-RAG 2.0 Web & REST API endpoints.
Tests endpoint handlers directly without requiring httpx.
"""

import sys
from pathlib import Path
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.api import (
    QueryRequest,
    check_invariants,
    execute_query,
    get_forest_graph,
    get_node_details,
    get_sample_queries,
    get_stats,
    health_check,
    list_documents,
    serve_index,
)


class TestWebAPI:
    @pytest.fixture(autouse=True)
    def setup_sample_knowledge_base(self):
        from src.api import preload_sample_kb
        preload_sample_kb()

    def test_health_check(self):
        data = health_check()
        assert data["status"] == "healthy"
        assert data["service"] == "shia-rag-2.0"
        assert data["total_nodes"] > 0

    def test_stats_endpoint(self):
        stats = get_stats()
        assert "total_nodes" in stats
        assert "total_edges" in stats
        assert "max_depth" in stats

    def test_invariants_endpoint(self):
        inv = check_invariants()
        assert inv["status"] == "valid"

    def test_sample_queries_endpoint(self):
        queries = get_sample_queries()
        assert len(queries) >= 2
        assert "title" in queries[0]
        assert "query" in queries[0]

    def test_query_execution(self):
        req = QueryRequest(
            query="What is Stochastic Gradient Descent?",
            token_budget=2048,
        )
        data = execute_query(req)
        assert "routing_mode" in data
        assert "answer" in data
        assert "formatted_answer" in data
        assert "selected_nodes" in data
        assert "attribution_reward" in data
        assert len(data["selected_nodes"]) > 0

    def test_forest_graph_endpoint(self):
        forest = get_forest_graph()
        assert "nodes" in forest
        assert "edges" in forest
        assert "stats" in forest
        assert len(forest["nodes"]) > 0

    def test_node_details_endpoint(self):
        forest = get_forest_graph()
        first_node_id = forest["nodes"][0]["id"]
        node = get_node_details(first_node_id)
        assert node["node_id"] == first_node_id
        assert "canonical_name" in node
        assert "definition_text" in node
        assert "depth" in node

    def test_documents_endpoint(self):
        docs = list_documents()
        assert isinstance(docs, list)

    def test_static_index_serving(self):
        res = serve_index()
        assert Path(res.path).exists()
        assert res.media_type in ("text/html", None)
