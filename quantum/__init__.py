"""
quantum — coherence-limited dicing for quantum-processor chips.

A qubit chip is diced under a constraint no logic die has: the cut must not spoil
quantum coherence. Subsurface damage, debris and wet contamination at the dicing
edge seed two-level-system (TLS) defects whose dielectric loss caps the qubit T1.

This package re-scores the dicing methods against a *coherence* budget instead of
strength/throughput, and shows the ranking invert: dry, low-damage stealth dicing
beats the wet, high-damage blade that is perfectly adequate for logic dies.

Public API:
    from quantum import (QubitSpec, SubstrateTLS, QUBIT_SUBSTRATES,
                         ProcessSignature, CANONICAL_SIGNATURES,
                         evaluate_coherence, rank_methods, signature_from_outcome)
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from quantum.coherence import (   # noqa: E402
    QubitSpec, SubstrateTLS, QUBIT_SUBSTRATES,
    ProcessSignature, CANONICAL_SIGNATURES, signature_from_outcome,
    CoherenceReport, evaluate_coherence, rank_methods,
)

__all__ = [
    "QubitSpec", "SubstrateTLS", "QUBIT_SUBSTRATES",
    "ProcessSignature", "CANONICAL_SIGNATURES", "signature_from_outcome",
    "CoherenceReport", "evaluate_coherence", "rank_methods",
]
