"""
Unit tests for ThompsonEvolutionEngine (Layer 8 Operations).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.layer8_operations import ThompsonEvolutionEngine


@pytest.fixture
def engine():
    return ThompsonEvolutionEngine(
        smoothing_gamma=0.05,
        smoothing_interval=10,
        temporal_decay_rate=0.999,
        min_reward_clip=0.05,
        max_reward_clip=0.95,
    )


class TestThompsonSampling:
    def test_sample_edge_weights_bounds(self, engine):
        """All sampled weights must fall strictly within [0.0, 1.0]."""
        priors = {
            "E-1": (1.0, 1.0),
            "E-2": (10.0, 2.0),
            "E-3": (2.0, 10.0),
        }
        samples = engine.sample_edge_weights(priors)
        assert len(samples) == 3
        for weight in samples.values():
            assert 0.0 <= weight <= 1.0

    def test_posterior_update_positive_reward(self, engine):
        """Positive reward increases alpha."""
        priors = {"E-1": (2.0, 2.0)}
        engine.update_edge_feedback(priors, ["E-1"], reward=0.9)
        alpha, beta = priors["E-1"]
        # alpha should increase by clipped reward (~0.90)
        assert alpha > 2.8
        assert beta > 2.0

    def test_reward_clipping_bounds(self, engine):
        """Extreme rewards (0.0 and 1.0) must be clipped to [0.05, 0.95]."""
        priors = {"E-1": (1.0, 1.0)}
        # Reward 1.0 clipped to 0.95
        engine.update_edge_feedback(priors, ["E-1"], reward=1.0)
        alpha, beta = priors["E-1"]
        assert alpha == pytest.approx(1.95, abs=0.01)
        assert beta == pytest.approx(1.05, abs=0.01)

    def test_credit_assignment_weighting(self, engine):
        """Edges with higher contribution receive more of the update."""
        priors = {"E-A": (1.0, 1.0), "E-B": (1.0, 1.0)}
        engine.update_edge_feedback(
            priors,
            ["E-A", "E-B"],
            reward=0.8,
            edge_contributions={"E-A": 1.0, "E-B": 0.25},
        )
        # E-A gets full credit, E-B gets quarter credit
        diff_a = priors["E-A"][0] - 1.0
        diff_b = priors["E-B"][0] - 1.0
        assert diff_a > diff_b

    def test_laplacian_smoothing(self, engine):
        """Laplacian smoothing diffuses weights across adjacency matrix."""
        adj = np.array([
            [0.0, 0.8, 0.0],
            [0.8, 0.0, 0.1],
            [0.0, 0.1, 0.0],
        ], dtype=np.float64)
        smoothed = engine.apply_laplacian_smoothing(adj)
        assert smoothed.shape == (3, 3)
        assert not np.isnan(smoothed).any()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
