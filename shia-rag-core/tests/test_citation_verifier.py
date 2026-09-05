"""
Unit tests for ClaimAttributionVerifier (Layer 7).
"""

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.layer7_generation import ClaimAttributionVerifier


@pytest.fixture
def verifier():
    return ClaimAttributionVerifier(overlap_threshold=0.30)


class TestCitationVerifier:
    def test_empty_answer(self, verifier):
        reward, records = verifier.verify_generation("", {})
        assert reward == 0.0
        assert records == []

    def test_unsupported_claim(self, verifier):
        answer = "The quick brown fox jumps over the lazy dog."
        reward, records = verifier.verify_generation(answer, {})
        assert reward == 0.0
        assert len(records) == 1
        assert records[0]["supported"] is False

    def test_supported_claim_with_citation(self, verifier):
        answer = "OSPF is an interior gateway protocol [KN-000123]."
        selected_nodes = {
            "KN-000123": {
                "evidence": ["OSPF is an interior gateway protocol used in IP networks."]
            }
        }
        reward, records = verifier.verify_generation(answer, selected_nodes)
        assert reward == 1.0
        assert len(records) == 1
        assert records[0]["supported"] is True

    def test_partial_support(self, verifier):
        answer = (
            "OSPF is an interior gateway routing protocol [KN-000123]. "
            "Quantum computers process qubits instantly [KN-000999]."
        )
        selected_nodes = {
            "KN-000123": {
                "evidence": ["OSPF is an interior gateway routing protocol."]
            },
            "KN-000999": {
                "evidence": ["Unrelated text about biology and proteins."]
            }
        }
        reward, records = verifier.verify_generation(answer, selected_nodes)
        assert reward == 0.5
        assert len(records) == 2
        assert records[0]["supported"] is True
        assert records[1]["supported"] is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
