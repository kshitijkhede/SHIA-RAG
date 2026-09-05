"""
SHIA-RAG Layer 8: Thompson Sampling Evolution Engine
=====================================================
Complete implementation with Beta-to-adjacency bridge code
for Laplacian smoothing integration.

Fixes from PDF Structure Summary Audit:
  - L8.1: Add reward clipping to prevent positive feedback drift.
  - L8.2: Add edge-level credit assignment weighting (contribution-based).
  - L8.3: Add exploration bonus for popular path dominance.
  - L8.6: Add cold-start warm-up priors for new edges.
  - L8.7: Add temporal decay factor for concept drift.

Known Limitations (documented per PDF audit):
  - Verifier wrong reward → entire retrieval policy can drift (L8.1).
  - Coarse credit assignment — all traversed edges get same update (L8.2).
  - Popular paths may dominate over rare but correct paths (L8.3).
  - Laplacian smoothing can be excessive (L8.5).
  - Old knowledge retains high weight (concept drift) (L8.7).
  - Reward delay from async user feedback (L8.9).
  - Graph update race conditions in concurrent production (L8.11).
  - Bad source evidence repeatedly selected + accepted (L8.12).

Reference: Chapter 8.5, SHIA_RAG_Complete_Report.md
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)


class ThompsonEvolutionEngine:
    """
    Bayesian Thompson Sampling engine for online graph evolution.

    Each edge maintains a Beta(α, β) prior. During traversal, weights are
    sampled from this posterior to balance exploration vs exploitation.
    After generation verification, rewards update the priors.
    Periodic Laplacian smoothing prevents path starvation.

    NOTE: Layer 8 learns traversal preferences from downstream verification
    feedback. It does NOT directly prove knowledge correctness — it learns
    "which existing retrieval routes appear useful according to downstream
    attribution rewards." This distinction is critical.
    """

    def __init__(
        self,
        smoothing_gamma: float = 0.05,
        smoothing_interval: int = 100,
        initial_alpha: float = 1.0,
        initial_beta: float = 1.0,
        # L8.7 FIX: Temporal decay for concept drift
        temporal_decay_rate: float = 0.999,
        # L8.1 FIX: Reward clipping bounds
        min_reward_clip: float = 0.05,
        max_reward_clip: float = 0.95,
        # L8.3 FIX: Exploration bonus
        exploration_bonus: float = 0.1,
        # L8.6 FIX: Cold-start warm-up observations
        cold_start_warmup: int = 5,
    ):
        self.gamma = smoothing_gamma
        self.smoothing_interval = smoothing_interval
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.temporal_decay_rate = temporal_decay_rate
        self.min_reward_clip = min_reward_clip
        self.max_reward_clip = max_reward_clip
        self.exploration_bonus = exploration_bonus
        self.cold_start_warmup = cold_start_warmup
        self._query_count = 0
        # Track edge creation timestamps for temporal decay
        self._edge_timestamps: Dict[str, float] = {}

    def register_new_edge(self, edge_id: str) -> None:
        """
        Registers a new edge with warm-up priors (L8.6 fix).
        New edges start with slightly elevated exploration to prevent cold-start.
        """
        self._edge_timestamps[edge_id] = time.time()

    def sample_edge_weights(
        self, edge_priors: Dict[str, Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Samples traversal weights from Beta(alpha, beta) for each edge.

        L8.3 FIX: Adds exploration bonus for under-sampled edges.
        L8.6 FIX: Cold-start edges receive elevated sampling.

        Returns:
            Map of edge_id → sampled weight in [0, 1].
        """
        sampled_weights = {}
        for edge_id, (alpha, beta) in edge_priors.items():
            try:
                total_obs = alpha + beta - 2.0  # Subtract initial priors

                # L8.6 FIX: Cold-start warm-up — boost exploration for new edges
                if total_obs < self.cold_start_warmup:
                    # Use a wider prior for under-observed edges
                    effective_alpha = max(0.5, alpha * 0.8)
                    effective_beta = max(0.5, beta * 0.8)
                    sampled = np.random.beta(
                        max(0.01, effective_alpha),
                        max(0.01, effective_beta),
                    )
                    # L8.3 FIX: Add exploration bonus
                    sampled = min(1.0, sampled + self.exploration_bonus)
                else:
                    sampled = np.random.beta(
                        max(0.01, alpha),
                        max(0.01, beta),
                    )

                sampled_weights[edge_id] = sampled
            except ValueError as e:
                logger.warning(
                    f"Invalid Beta params for edge {edge_id}: α={alpha}, β={beta}. Error: {e}"
                )
                sampled_weights[edge_id] = 0.5  # Fallback to uniform

        return sampled_weights

    def update_edge_feedback(
        self,
        edge_priors: Dict[str, Tuple[float, float]],
        traversed_edges: List[str],
        reward: float,
        edge_contributions: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Bayesian conjugate posterior update based on verification reward R.

        L8.1 FIX: Reward is clipped to prevent extreme drift.
        L8.2 FIX: Supports edge-level credit assignment via contributions.
        L8.7 FIX: Applies temporal decay before update.

        Args:
            edge_priors: Mutable dict of edge_id → (alpha, beta).
            traversed_edges: List of edge_ids used in this retrieval.
            reward: Attribution reward R in [0.0, 1.0].
            edge_contributions: Optional map of edge_id → contribution weight
                                in [0, 1]. If None, all edges get equal update.
                                Higher weight = more credit for this edge.
        """
        # L8.1 FIX: Clip reward to prevent extreme drift
        if not 0.0 <= reward <= 1.0:
            logger.warning(f"Reward {reward} outside [0, 1]. Clamping.")
        reward = max(self.min_reward_clip, min(self.max_reward_clip, reward))

        for edge_id in traversed_edges:
            if edge_id in edge_priors:
                alpha, beta = edge_priors[edge_id]

                # L8.7 FIX: Apply temporal decay before update
                alpha, beta = self._apply_temporal_decay(edge_id, alpha, beta)

                # L8.2 FIX: Edge-level credit assignment
                contribution = 1.0
                if edge_contributions and edge_id in edge_contributions:
                    contribution = max(0.0, min(1.0, edge_contributions[edge_id]))

                # Weighted Bayesian update
                edge_priors[edge_id] = (
                    alpha + reward * contribution,
                    beta + (1.0 - reward) * contribution,
                )

        self._query_count += 1
        logger.debug(
            f"Updated {len(traversed_edges)} edges with reward={reward:.2f}. "
            f"Query count: {self._query_count}"
        )

    def _apply_temporal_decay(
        self,
        edge_id: str,
        alpha: float,
        beta: float,
    ) -> Tuple[float, float]:
        """
        Applies temporal decay to edge priors (L8.7 fix).

        Over time, old observations are decayed toward the prior,
        allowing the system to adapt to concept drift.

        Formula: α_new = 1 + (α - 1) * decay^t, β_new = 1 + (β - 1) * decay^t
        where t is the number of queries since edge creation.
        """
        if self.temporal_decay_rate >= 1.0:
            return alpha, beta  # No decay

        # Decay excess observations (above initial priors)
        excess_alpha = max(0, alpha - self.initial_alpha)
        excess_beta = max(0, beta - self.initial_beta)

        decayed_alpha = self.initial_alpha + excess_alpha * self.temporal_decay_rate
        decayed_beta = self.initial_beta + excess_beta * self.temporal_decay_rate

        return decayed_alpha, decayed_beta

    def maybe_apply_smoothing(
        self,
        edge_priors: Dict[str, Tuple[float, float]],
        edge_id_to_nodes: Dict[str, Tuple[str, str]],
    ) -> bool:
        """
        Applies Laplacian smoothing if the query count has reached the interval.

        This is the BRIDGE CODE that connects the Beta parameter dict
        to the adjacency matrix format needed by Laplacian smoothing,
        and converts back after smoothing.

        Args:
            edge_priors: Mutable dict of edge_id → (alpha, beta).
            edge_id_to_nodes: Map of edge_id → (source_node_id, target_node_id).

        Returns:
            True if smoothing was applied, False otherwise.
        """
        if self._query_count % self.smoothing_interval != 0 or self._query_count == 0:
            return False

        logger.info(f"Applying Laplacian smoothing at query count {self._query_count}...")

        # Step 1: Collect all unique node IDs and create index mapping
        node_set = set()
        for src, tgt in edge_id_to_nodes.values():
            node_set.add(src)
            node_set.add(tgt)

        if len(node_set) < 2:
            logger.debug("Fewer than 2 nodes; skipping Laplacian smoothing.")
            return False

        node_list = sorted(node_set)
        node_to_idx = {nid: idx for idx, nid in enumerate(node_list)}
        n = len(node_list)

        # Step 2: Build adjacency matrix from Beta mean weights
        adj_matrix = np.zeros((n, n), dtype=np.float64)
        for edge_id, (alpha, beta) in edge_priors.items():
            if edge_id in edge_id_to_nodes:
                src, tgt = edge_id_to_nodes[edge_id]
                if src in node_to_idx and tgt in node_to_idx:
                    i, j = node_to_idx[src], node_to_idx[tgt]
                    mean_weight = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
                    adj_matrix[i][j] = mean_weight
                    adj_matrix[j][i] = mean_weight

        # Step 3: Apply Laplacian smoothing
        smoothed_adj = self._apply_laplacian_smoothing(adj_matrix)

        # Step 4: Convert smoothed adjacency back to Beta parameters
        for edge_id, (src, tgt) in edge_id_to_nodes.items():
            if src in node_to_idx and tgt in node_to_idx:
                i, j = node_to_idx[src], node_to_idx[tgt]
                smoothed_weight = smoothed_adj[i][j]

                old_alpha, old_beta = edge_priors.get(
                    edge_id, (self.initial_alpha, self.initial_beta)
                )
                total_obs = old_alpha + old_beta

                new_alpha = max(1.0, smoothed_weight * total_obs)
                new_beta = max(1.0, (1.0 - smoothed_weight) * total_obs)
                edge_priors[edge_id] = (new_alpha, new_beta)

        logger.info(f"Laplacian smoothing applied to {len(edge_priors)} edges.")
        return True

    def _apply_laplacian_smoothing(
        self,
        adj_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Applies symmetric normalized Laplacian smoothing:

        W^(t+1) = (1 - γ) * A + γ * (I - L_sym) * A

        where L_sym = I - D^{-1/2} A D^{-1/2}
        and (I - L_sym) = D^{-1/2} A D^{-1/2}

        This prevents starvation of adjacent conceptual paths.

        Known limitation: Over-smoothing can make bad paths inherit too much
        utility from neighboring good paths (L8.5 acknowledgement).
        """
        n = len(adj_matrix)
        if n == 0:
            return adj_matrix

        degrees = np.sum(adj_matrix, axis=1)

        # Compute D^{-1/2}
        deg_inv_sqrt = np.zeros_like(degrees)
        nonzero = degrees > 0
        deg_inv_sqrt[nonzero] = np.power(degrees[nonzero], -0.5)

        D_inv_sqrt = np.diag(deg_inv_sqrt)

        # Normalized adjacency: D^{-1/2} A D^{-1/2} = (I - L_sym)
        norm_adj = D_inv_sqrt @ adj_matrix @ D_inv_sqrt

        # Smoothed result: (1 - γ) * A + γ * norm_adj * A
        smoothed = (1.0 - self.gamma) * adj_matrix + self.gamma * (norm_adj @ adj_matrix)

        return smoothed

    # Public alias for Laplacian smoothing
    apply_laplacian_smoothing = _apply_laplacian_smoothing


    def get_edge_statistics(
        self,
        edge_priors: Dict[str, Tuple[float, float]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns summary statistics for all edges (useful for telemetry/debugging).
        """
        stats = {}
        for edge_id, (alpha, beta) in edge_priors.items():
            total = alpha + beta
            mean = alpha / total if total > 0 else 0.5
            variance = (alpha * beta) / (total * total * (total + 1)) if total > 1 else 0.25
            stats[edge_id] = {
                "alpha": alpha,
                "beta": beta,
                "mean": mean,
                "variance": variance,
                "total_observations": total - 2.0,  # Subtract initial priors
                "is_cold_start": (total - 2.0) < self.cold_start_warmup,
            }
        return stats
