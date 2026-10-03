"""Deterministic cross-domain logical plan evaluator."""

from .cross_domain import CROSS_DOMAIN_API_VERSION, verify_cross_domain_files
from .kernel import VerificationResult, verify

__all__ = [
    "CROSS_DOMAIN_API_VERSION",
    "VerificationResult",
    "verify",
    "verify_cross_domain_files",
]

__version__ = "1.7.0"
