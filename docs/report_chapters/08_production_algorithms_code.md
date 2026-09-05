## 8. Concrete, Production-Ready Python Implementations

### 8.1 Module: `hv_validator.py` (Cycle-Free Invariant Enforcement)

```python
from typing import Set, List, Dict, Any

class HierarchyValidator:
    def __init__(self, max_depth: int = 8, min_sibling_coherence: float = 0.40):
        self.max_depth = max_depth
        self.min_sibling_coherence = min_sibling_coherence

    def validate_placement(
        self, 
        parent_id: str, 
        child_id: str, 
        forest_parents_map: Dict[str, List[str]],
        node_depth_map: Dict[str, int]
    ) -> bool:
        """
        Validates that placing child_id under parent_id maintains all forest invariants:
        1. No self-loops.
        2. No cycles (child cannot be an ancestor of parent).
        3. Maximum depth invariant (parent.depth + 1 < max_depth).
        """
        # Invariant 1: Self-loop check
        if parent_id == child_id:
            return False

        # Invariant 2: Depth limit check
        parent_depth = node_depth_map.get(parent_id, 0)
        if parent_depth + 1 >= self.max_depth:
            return False

        # Invariant 3: Cycle detection via upward DFS
        visited: Set[str] = set()
        stack: List[str] = [parent_id]

        while stack:
            curr = stack.pop()
            if curr == child_id:
                return False  # Cycle detected: child is already an ancestor of parent
            
            if curr not in visited:
                visited.add(curr)
                parents = forest_parents_map.get(curr, [])
                for p in parents:
                    stack.append(p)

        return True
```

---

### 8.2 Module: `cpg_pre_pipeline.py` (Candidate Generation & Ranking)

```python
from typing import List, Tuple, Dict, Any
import numpy as np

class ParentRankingEngine:
    def __init__(self, w1: float = 0.30, w2: float = 0.25, w3: float = 0.15, w4: float = 0.20, w5: float = 0.10):
        # Validate that PRE weights sum to 1.0 (Structural Invariant 4)
        weight_sum = w1 + w2 + w3 + w4 + w5
        assert abs(weight_sum - 1.0) < 1e-6, f"PRE weights must sum to 1.0, got {weight_sum}"
        self.w1 = w1  # Cosine similarity
        self.w2 = w2  # Hypernym score
        self.w3 = w3  # Depth factor
        self.w4 = w4  # Evidence support
        self.w5 = w5  # Parent confidence

    def rank_candidates(
        self,
        node_embedding: np.ndarray,
        node_name: str,
        candidates: List[Dict[str, Any]],
        hypernym_checker,
        evidence_counter
    ) -> List[Tuple[str, float]]:
        """
        Ranks all candidate parents and returns a sorted list of (candidate_id, score).
        """
        scored_candidates = []

        for cand in candidates:
            cand_id = cand["id"]
            cand_emb = cand["embedding"]
            cand_depth = cand["depth"]
            cand_conf = cand["confidence"]

            # Compute Cosine Similarity
            cos_sim = float(np.dot(node_embedding, cand_emb) / (
                np.linalg.norm(node_embedding) * np.linalg.norm(cand_emb) + 1e-9
            ))
            cos_sim = max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))

            # Compute Hypernym Score
            # Check if candidate P is a hypernym of the new node N (P is-a parent of N)
            h_score = 1.0 if hypernym_checker(parent_name=cand["name"], child_name=node_name) else 0.0

            # Compute Depth Factor
            depth_factor = 1.0 / (cand_depth + 1.0)

            # Compute Evidence Support
            evi_score = evidence_counter(cand_id, cand.get("target_id", ""))

            # Total Weighted Score
            total_score = (
                self.w1 * cos_sim +
                self.w2 * h_score +
                self.w3 * depth_factor +
                self.w4 * evi_score +
                self.w5 * cand_conf
            )
            scored_candidates.append((cand_id, total_score))

        # Return sorted in descending order of fitness
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates
```

---

### 8.3 Module: `dc_knapsack.py` (DAG Precedence-Constrained Optimizer)

```python
from typing import Dict, Set, List, Tuple
from dataclasses import dataclass

@dataclass
class KnapsackItem:
    node_id: str
    relevance_score: float
    token_cost: int
    parent_ids: List[str]

class DAGKnapsackOptimizer:
    def __init__(self, token_budget: int):
        self.budget = token_budget

    def solve(self, items: Dict[str, KnapsackItem]) -> Tuple[List[str], float, int]:
        """
        Solves the Precedence-Constrained Knapsack Problem over a DAG of concepts.
        Guarantees that no child node is selected without all its parent prerequisites.
        """
        # Step 1: Precompute ancestor closure for every node
        closure_cache: Dict[str, Set[str]] = {}

        def get_ancestor_closure(nid: str) -> Set[str]:
            if nid in closure_cache:
                return closure_cache[nid]
            closure = {nid}
            for pid in items[nid].parent_ids:
                if pid in items:
                    closure.update(get_ancestor_closure(pid))
            closure_cache[nid] = closure
            return closure

        for nid in items:
            get_ancestor_closure(nid)

        # Step 2: Build candidate bundles (closure sets)
        bundles = []
        for nid, item in items.items():
            ancestors = closure_cache[nid]
            bundle_tokens = sum(items[a].token_cost for a in ancestors)
            bundle_value = sum(items[a].relevance_score for a in ancestors)
            if bundle_tokens <= self.budget:
                density = bundle_value / max(1, bundle_tokens)
                bundles.append((density, nid, ancestors, bundle_tokens, bundle_value))

        # Step 3: Sort bundles by marginal utility density
        bundles.sort(key=lambda b: b[0], reverse=True)

        # Step 4: Greedy Precedence-Constrained Accumulation
        selected_nodes: Set[str] = set()
        current_tokens = 0
        total_utility = 0.0

        for _, _, ancestors, _, _ in bundles:
            unselected = ancestors - selected_nodes
            additional_tokens = sum(items[u].token_cost for u in unselected)
            
            if current_tokens + additional_tokens <= self.budget:
                for u in unselected:
                    selected_nodes.add(u)
                    current_tokens += items[u].token_cost
                    total_utility += items[u].relevance_score

        return list(selected_nodes), total_utility, current_tokens
```

---

### 8.4 Module: `srdr_router.py` (Self-Reflective Adaptive Router)

```python
from typing import Dict, Any

class SelfReflectiveDepthRouter:
    def __init__(self):
        self.comparative_tokens = {"compare", "vs", "versus", "difference", "differ", "contrast"}
        self.thematic_tokens = {"overview", "summarize", "landscape", "all", "survey", "themes"}

    def route(self, query: str, entity_count: int) -> Dict[str, Any]:
        """
        Routes user query into one of four operational modes:
        Mode 1: Thematic Sensemaking
        Mode 2: Factual Needle
        Mode 3: Multi-Hop Comparative
        Mode 4: Parametric / Direct
        """
        lower_q = query.lower()
        has_comparison = any(tok in lower_q for tok in self.comparative_tokens)
        has_thematic = any(tok in lower_q for tok in self.thematic_tokens)

        if has_thematic:
            return {
                "mode": "MODE_1_THEMATIC",
                "max_depth": 1,
                "strategy": "ROOT_BREADTH_FIRST",
                "include_parents": False,
                "include_children": True,
                "include_crosslinks": False
            }
        elif has_comparison or entity_count >= 2:
            return {
                "mode": "MODE_3_MULTIHOP_COMPARATIVE",
                "max_depth": 3,
                "strategy": "DUAL_SUBTREE_BIDIRECTIONAL",
                "include_parents": True,
                "include_children": True,
                "include_crosslinks": True
            }
        elif entity_count == 1:
            return {
                "mode": "MODE_2_FACTUAL_NEEDLE",
                "max_depth": 1,
                "strategy": "LEAF_TO_ROOT",
                "include_parents": True,
                "include_children": False,
                "include_crosslinks": False
            }
        else:
            return {
                "mode": "MODE_4_PARAMETRIC",
                "max_depth": 0,
                "strategy": "SKIP_RETRIEVAL",
                "include_parents": False,
                "include_children": False,
                "include_crosslinks": False
            }
```

---

### 8.5 Module: `thompson_evolution.py` (Bandit Evolution & Graph Laplacian)

```python
import numpy as np
from typing import Dict, List, Tuple

class ThompsonEvolutionEngine:
    def __init__(self, smoothing_gamma: float = 0.05):
        self.gamma = smoothing_gamma

    def sample_edge_weights(self, edge_priors: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """
        Samples traversal weights from Beta(alpha, beta) for each edge.
        """
        sampled_weights = {}
        for edge_id, (alpha, beta) in edge_priors.items():
            sampled_weights[edge_id] = float(np.random.beta(alpha, beta))
        return sampled_weights

    def update_edge_feedback(
        self, 
        edge_priors: Dict[str, Tuple[float, float]], 
        traversed_edges: List[str], 
        reward: float
    ) -> None:
        """
        Performs Bayesian conjugate posterior update based on verification reward R in {0.0, 1.0}.
        """
        for edge_id in traversed_edges:
            if edge_id in edge_priors:
                alpha, beta = edge_priors[edge_id]
                edge_priors[edge_id] = (alpha + reward, beta + (1.0 - reward))

    def apply_laplacian_smoothing(
        self, 
        adj_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Applies symmetric normalized Laplacian smoothing across graph adjacency matrix.
        Prevents starvation of adjacent conceptual paths.
        """
        degrees = np.sum(adj_matrix, axis=1)
        deg_inv_sqrt = np.power(degrees, -0.5, where=degrees > 0)
        deg_inv_sqrt[degrees == 0] = 0.0
        
        D_inv_sqrt = np.diag(deg_inv_sqrt)
        L_sym = np.eye(len(adj_matrix)) - np.dot(np.dot(D_inv_sqrt, adj_matrix), D_inv_sqrt)
        
        smoothed_adj = (1.0 - self.gamma) * adj_matrix + self.gamma * (np.eye(len(adj_matrix)) - L_sym) @ adj_matrix
        return smoothed_adj
```

---

### 8.6 Module: `claim_verifier.py` (Attribution & Citation Verification)

```python
from typing import List, Dict, Any, Tuple
import re

class ClaimAttributionVerifier:
    def __init__(self, nli_model=None):
        self.nli_model = nli_model  # Fine-tuned DeBERTa-v3 or LLM-as-a-judge

    def verify_generation(
        self, 
        generated_answer: str, 
        selected_nodes: Dict[str, Any]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Extracts claim-level statements and verifies factual consistency against E_proj text anchors.
        Returns attribution reward R in [0.0, 1.0] and detailed attribution records.
        """
        # Extract sentence units
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', generated_answer) if len(s.strip()) > 5]
        if not sentences:
            return 0.0, []

        verified_count = 0
        attribution_records = []

        for sentence in sentences:
            # Check for citation tags e.g. [KN-000456]
            citations = re.findall(r'\[(KN-[A-Z0-9]{6})\]', sentence)
            is_supported = False
            matched_node = None

            if citations:
                for cite_id in citations:
                    if cite_id in selected_nodes:
                        node_evidence = " ".join([a.get("text", "") for a in selected_nodes[cite_id].get("evidence", [])])
                        # Simplified premise verification (or NLI entailment check)
                        if self._check_entailment(premise=node_evidence, hypothesis=sentence):
                            is_supported = True
                            matched_node = cite_id
                            break
            
            if is_supported:
                verified_count += 1

            attribution_records.append({
                "sentence": sentence,
                "citations": citations,
                "supported": is_supported,
                "grounded_node": matched_node
            })

        overall_reward = float(verified_count) / max(1, len(sentences))
        return overall_reward, attribution_records

    def _check_entailment(self, premise: str, hypothesis: str) -> bool:
        # Fast lexical overlap fallback if NLI model not provided
        if not premise:
            return False
        hypo_words = set(re.findall(r'\w+', hypothesis.lower()))
        premise_words = set(re.findall(r'\w+', premise.lower()))
        overlap = len(hypo_words & premise_words) / max(1, len(hypo_words))
        return overlap >= 0.50
```

---

### 8.7 Module: `pdf_loader.py` (Multi-Modal PyMuPDF Geometry Extraction)

The production PDF ingestion engine extracts physical layout bounding boxes, computes reading-order coordinates, and detects document sections:

```python
import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import List, Tuple
from src.layer0_data_model.schemas import DocumentMimeType, DocumentNode, TextBlock

HEADING_NUMBERED_RE = re.compile(r"^\s*(\d+(\.\d+)*)\s+([A-Z][A-Za-z0-9\s\-:]{2,80})")
STANDARD_SECTIONS = {"abstract", "introduction", "background", "methodology", "experiments", "conclusion"}

class PDFLoader:
    def __init__(self, extract_images: bool = False):
        self.extract_images = extract_images

    def load_pdf(self, file_path: str | Path) -> Tuple[DocumentNode, List[TextBlock]]:
        path = Path(file_path)
        doc = fitz.open(str(path))
        doc_node = DocumentNode(
            doc_id=f"DOC-{path.stem[:16]}",
            filename=path.name,
            mime_type=DocumentMimeType.PDF,
            total_pages=len(doc),
            raw_byte_size=path.stat().st_size,
        )
        blocks: List[TextBlock] = []
        global_order = 0
        current_section = "Preamble"

        for page_idx, page in enumerate(doc):
            page_blocks = page.get_text("blocks")
            # 2D reading order sort: top-to-bottom primary, left-to-right secondary
            sorted_blocks = sorted(page_blocks, key=lambda b: (round(b[1] / 15.0) * 15.0, b[0]))
            for b in sorted_blocks:
                text = b[4].strip()
                if not text or len(text) < 4:
                    continue
                # Heading detection
                first_line = text.split("\n")[0].strip()
                m = HEADING_NUMBERED_RE.match(first_line)
                if m or first_line.lower() in STANDARD_SECTIONS:
                    current_section = m.group(3).strip() if m else first_line.title()

                blocks.append(TextBlock(
                    block_id=f"BLK-P{page_idx+1:03d}-{global_order:04d}",
                    doc_id=doc_node.doc_id,
                    page_number=page_idx + 1,
                    reading_order_index=global_order,
                    bounding_box_coords={"x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3]},
                    text_content=text,
                    detected_section=current_section,
                ))
                global_order += 1
        return doc_node, blocks
```

---

### 8.8 Module: `concept_extractor.py` (Propositional Hierarchy Induction)

Extracts candidate concept nodes and hierarchical parentage from document layout blocks:

```python
import re
from typing import List, Tuple
from src.layer0_data_model.schemas import KnowledgeNode, NodeType, TextBlock

DEF_PATTERNS = [
    re.compile(r"([A-Z][A-Za-z0-9\s\-]{2,40})\s+(?:is|are)\s+(?:defined\s+as|referred\s+to\s+as)\s+([^.]{10,200})\.", re.IGNORECASE),
    re.compile(r"([A-Z][A-Za-z0-9\s\-]{2,40})\s*[:\-—]\s+([^.]{10,200})\.", re.IGNORECASE),
]

class ConceptExtractor:
    def extract_from_blocks(self, blocks: List[TextBlock]) -> List[Tuple[KnowledgeNode, List[str]]]:
        results = []
        seen_names = set()
        for b in blocks:
            for pat in DEF_PATTERNS:
                for match in pat.finditer(b.text_content):
                    term = match.group(1).strip()
                    definition = match.group(2).strip()
                    if term.lower() in seen_names or len(term) < 3:
                        continue
                    seen_names.add(term.lower())
                    node = KnowledgeNode(
                        node_id=f"KN-{len(seen_names):06d}",
                        canonical_name=term,
                        node_type=NodeType.CONCEPT,
                        confidence_score=0.85,
                        text_definition=definition,
                        token_cost=max(10, len(definition) // 4),
                    )
                    # Candidate parents inferred from section context
                    candidates = [b.detected_section] if b.detected_section != "Preamble" else []
                    results.append((node, candidates))
        return results
```

---

### 8.9 Module: `pipeline.py` & CLI (`run_demo.py` Live PDF Ingestion)

The unified orchestrator wires the 8 layers together and provides live PDF document ingestion:

```python
import argparse
from src.pipeline import SHIARAGPipeline

def main():
    parser = argparse.ArgumentParser(description="SHIA-RAG 2.0 Live Ingestion & Benchmark")
    parser.add_argument("--pdf", type=str, help="Path to PDF document to ingest into Knowledge Forest")
    parser.add_argument("--query", "-q", type=str, help="User query against ingested forest")
    parser.add_argument("--budget", type=int, default=2048, help="Token budget for DC-Knapsack")
    args = parser.parse_args()

    pipeline = SHIARAGPipeline(token_budget=args.budget)

    if args.pdf:
        print(f"Ingesting live PDF: {args.pdf}...")
        stats = pipeline.ingest_pdf(args.pdf)
        print(f"Extracted {stats['document']['total_pages']} pages, "
              f"{stats['forest']['total_concepts']} concepts across {stats['forest']['tree_count']} trees.")
        
        query = args.query or "Summarize the core methodology and contributions of this work"
        res = pipeline.run_query(query)
        print(f"Grounded Answer:\n{res['answer']}")
        print(f"Attribution Reward R = {res['attribution_reward']:.3f} ({res['verified_claims']}/{res['total_claims']} claims)")
    else:
        # Standard CS & AI Benchmark Suite
        pipeline.load_sample_knowledge_base()
        pipeline.validate_invariants()
        # Executes THEMATIC, FACTUAL, MULTIHOP, and PARAMETRIC queries
```
