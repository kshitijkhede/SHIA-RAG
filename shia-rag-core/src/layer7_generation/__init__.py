"""SHIA-RAG Layer 7: Generation & Claim Attribution Verification"""

from .citation_verifier import ClaimAttributionVerifier

# Alias for backwards compatibility
CitationVerifier = ClaimAttributionVerifier

__all__ = ["ClaimAttributionVerifier", "CitationVerifier"]
