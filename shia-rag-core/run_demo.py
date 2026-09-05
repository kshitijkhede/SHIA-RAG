#!/usr/bin/env python3
"""
SHIA-RAG 2.0: Interactive & Benchmark Runner Script
===================================================
Run this script to test and verify the entire SHIA-RAG pipeline on
preloaded domain ontologies OR on any real uploaded PDF document.

Usage:
  # 1. Run on an actual uploaded PDF document:
  python run_demo.py --pdf "research_papers/TreeRAG Unleashing the Power of Hierarchical Storage for Enhanced.pdf"

  # 2. Run a specific single query against a PDF:
  python run_demo.py --pdf "research_papers/TreeRAG Unleashing the Power of Hierarchical Storage for Enhanced.pdf" -q "What is TreeRAG?"

  # 3. Start an interactive terminal chat session over the uploaded PDF:
  python run_demo.py --pdf "research_papers/TreeRAG Unleashing the Power of Hierarchical Storage for Enhanced.pdf" --interactive

  # 4. Run the default multi-mode benchmark on the sample CS/AI ontology:
  python run_demo.py

  # 5. Interactive query session on the sample ontology:
  python run_demo.py --interactive
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

# Add project root and src/ to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
for _p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), str(WORKSPACE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from src.pipeline import SHIARAGPipeline
except ImportError:
    from pipeline import SHIARAGPipeline


def resolve_pdf_path(pdf_arg: str) -> Path:
    """Finds the PDF file checking multiple candidate locations."""
    candidates = [
        Path(pdf_arg),
        PROJECT_ROOT / pdf_arg,
        WORKSPACE_ROOT / pdf_arg,
        WORKSPACE_ROOT / "research_papers" / pdf_arg,
        WORKSPACE_ROOT / "research_papers" / Path(pdf_arg).name,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()
    raise FileNotFoundError(
        f"PDF file not found: '{pdf_arg}'. Searched locations:\n"
        + "\n".join(f"  - {c.resolve()}" for c in candidates)
    )


def run_benchmark(pipeline: SHIARAGPipeline):
    print("\n" + "=" * 76)
    print("  RUNNING SHIA-RAG 2.0 BENCHMARK SUITE (Preloaded CS & AI Ontology)")
    print("=" * 76)

    queries = [
        ("Thematic Overview", "Summarize the architecture and overview of Artificial Intelligence and Machine Learning"),
        ("Factual / Needle", "What is the exact mechanism of Transmission Control Protocol (TCP) flow control?"),
        ("Multi-hop Comparison", "Compare TCP versus UDP in transport protocols, and explain how Transformer architectures utilize optimization algorithms like Stochastic Gradient Descent and Backpropagation."),
        ("Parametric Direct", "Define Stochastic Gradient Descent."),
    ]

    for title, query in queries:
        _execute_and_display_query(pipeline, title, query)

    _verify_and_display_forest(pipeline)


def run_pdf_benchmark(pipeline: SHIARAGPipeline, pdf_stats: Dict):
    print("\n" + "=" * 76)
    print(f"  RUNNING SHIA-RAG 2.0 BENCHMARK SUITE ON: {pdf_stats['document']}")
    print("=" * 76)

    doc_name = pdf_stats["document"].replace(".pdf", "")
    forest_nodes = list(pipeline.nodes.values())
    roots = [n for n in forest_nodes if n.parent_id is None]
    root_title = roots[0].canonical_name if roots else doc_name

    # Select representative nodes from different depths for targeted questions
    depth_1_nodes = [n for n in forest_nodes if n.depth == 1 and n.canonical_name != "Abstract"]
    deeper_nodes = [n for n in forest_nodes if n.depth >= 2]

    c1 = depth_1_nodes[0].canonical_name if depth_1_nodes else root_title
    c2 = deeper_nodes[0].canonical_name if deeper_nodes else (depth_1_nodes[1].canonical_name if len(depth_1_nodes) > 1 else c1)
    c3 = deeper_nodes[1].canonical_name if len(deeper_nodes) > 1 else c2

    queries = [
        ("Thematic Overview", f"Summarize the architecture, purpose, and key contributions of {root_title}."),
        ("Factual / Needle", f"What is the exact definition and mechanism of {c2}?"),
        ("Multi-hop Comparison", f"Compare {c1} versus {c3} in the context of this document."),
        ("Parametric Direct", f"Define {c2}."),
    ]

    for title, query in queries:
        _execute_and_display_query(pipeline, title, query)

    _verify_and_display_forest(pipeline)


def _execute_and_display_query(pipeline: SHIARAGPipeline, title: str, query: str):
    print(f"\n[{title.upper()}]")
    print(f"  Query: \"{query}\"")
    res = pipeline.run_query(query)

    print(f"  --> Routing Mode:         {res['routing_mode']}")
    print(f"  --> Total Context Tokens: {res['total_tokens']}")
    print(f"  --> Attribution Reward R: {res['attribution_reward']:.3f} (Continuous ∈ [0, 1])")
    print(f"  --> Selected Nodes ({len(res['selected_nodes'])}):")
    for n in res["selected_nodes"][:6]:
        print(f"        * [{n['node_id']}] {n['name']} (Depth: {n['depth']}, Conf: {n['confidence']:.2f})")
    if len(res["selected_nodes"]) > 6:
        print(f"        ... and {len(res['selected_nodes']) - 6} more concepts")
    print("  --> Generated Grounded Answer:")
    answer_preview = res['answer'][:250] + "..." if len(res['answer']) > 250 else res['answer']
    print(f"        {answer_preview}")
    if res['evolution_update']['traversed_edge_count'] > 0:
        print(f"  --> Thompson Sampling:   Updated {res['evolution_update']['traversed_edge_count']} traversed edge posteriors")


def _verify_and_display_forest(pipeline: SHIARAGPipeline):
    print("\n" + "=" * 76)
    print("  FOREST STRUCTURAL HEALTH & INVARIANT VERIFICATION")
    print("=" * 76)
    pipeline.validate_invariants()
    stats = pipeline.get_forest_statistics()
    print(f"  [OK] Acyclic Taxonomy:       0 cycles, {stats['hierarchical_edges']} hierarchical edges")
    print(f"  [OK] Cross-Link Web:         {stats['semantic_edges']} typed semantic cross-links")
    print(f"  [OK] Depth Bounds:           Max depth {stats['max_depth']} <= 8 limit")
    print(f"  [OK] Online Bandit Beliefs:  {stats['edge_priors_tracked']} edge Beta priors maintained")
    print("=" * 76 + "\n")


def run_interactive(pipeline: SHIARAGPipeline, doc_title: Optional[str] = None):
    print("\n" + "=" * 76)
    print("  SHIA-RAG 2.0 INTERACTIVE QUERY CONSOLE")
    if doc_title:
        print(f"  Active Document: {doc_title}")
    print("  Type your question and press Enter. Type 'exit' or 'quit' to stop.")
    print("=" * 76 + "\n")

    while True:
        try:
            query = input("shia-rag> ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting console. Goodbye!")
                break

            res = pipeline.run_query(query)
            print(f"\n  [Routing Mode]      {res['routing_mode']}")
            print(f"  [Context Tokens]    {res['total_tokens']}")
            print(f"  [Attribution R]     {res['attribution_reward']:.3f}")
            print(f"  [Selected Concepts] {', '.join(n['name'] for n in res['selected_nodes'][:6])}")
            print(f"  [Answer]\n  {res['answer']}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting console. Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="SHIA-RAG 2.0 Runner & Benchmark (Supports Preloaded Ontologies & Real PDFs)"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to an uploaded PDF file to ingest, extract hierarchy, and query",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default=None,
        help="Single query to run against the ingested knowledge forest",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Start interactive REPL session",
    )
    parser.add_argument(
        "-b",
        "--budget",
        type=int,
        default=2048,
        help="Token budget for context selection (default: 2048)",
    )
    parser.add_argument(
        "--preload-sample",
        action="store_true",
        help="Preload the Computer Science / AI sample knowledge base in addition to any PDF",
    )
    args = parser.parse_args()

    print("\n" + "=" * 76)
    print("  SHIA-RAG 2.0: Semantic Hierarchy Induction Architecture for RAG")
    print("=" * 76)

    pipeline = SHIARAGPipeline(token_budget=args.budget)
    pdf_stats = None

    if args.pdf:
        pdf_path = resolve_pdf_path(args.pdf)
        print(f"\n[Ingesting PDF Document] -> {pdf_path.name}")
        pdf_stats = pipeline.ingest_pdf(pdf_path)
        print(f"  --> Pages:               {pdf_stats['total_pages']}")
        print(f"  --> Text Blocks Ingested:{pdf_stats['blocks_ingested']}")
        print(f"  --> Concepts Extracted:  {pdf_stats['concepts_extracted']}")
        print(f"  --> Cross-links Found:   {pdf_stats['crosslinks_discovered']}")
        print(f"  --> Forest Max Depth:    {pdf_stats['forest_stats']['max_depth']}")
        print(f"  --> Forest Roots:        {pdf_stats['forest_stats']['roots']}")

    if args.preload_sample or not args.pdf:
        print("\n[Loading Sample CS/AI Knowledge Base]...")
        pipeline.load_sample_knowledge_base()

    if args.query:
        print(f"\n[Executing Query]: \"{args.query}\"")
        res = pipeline.run_query(args.query)
        print("\n[Result]")
        print(f"  Routing Mode:      {res['routing_mode']}")
        print(f"  Context Tokens:    {res['total_tokens']}")
        print(f"  Attribution R:     {res['attribution_reward']:.3f}")
        print(f"  Selected Concepts: {len(res['selected_nodes'])} concepts")
        for n in res['selected_nodes'][:6]:
            print(f"    - [{n['node_id']}] {n['name']} (depth {n['depth']}, conf {n['confidence']:.2f})")
        print(f"\n[Grounded Answer]:\n{res['answer']}\n")
    elif args.interactive:
        doc_title = pdf_stats["document"] if pdf_stats else "Sample CS/AI Knowledge Base"
        run_interactive(pipeline, doc_title=doc_title)
    elif args.pdf and pdf_stats:
        run_pdf_benchmark(pipeline, pdf_stats)
    else:
        run_benchmark(pipeline)


if __name__ == "__main__":
    main()
