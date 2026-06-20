"""
Tests for the coherence-limited dicing model (quantum/).

The physics claim under test: for a qubit chip the dicing ranking inverts —
dry low-damage stealth beats the wet high-damage blade — and deeper subsurface
damage / shorter keep-out both lower T1.
"""
from __future__ import annotations

import math

import pytest

from quantum import (
    QubitSpec, QUBIT_SUBSTRATES, CANONICAL_SIGNATURES, ProcessSignature,
    evaluate_coherence, rank_methods, signature_from_outcome,
)


def _si():
    return QUBIT_SUBSTRATES["Si"]


def test_stealth_beats_blade_for_coherence():
    q = QubitSpec()
    blade = evaluate_coherence(CANONICAL_SIGNATURES["blade"], q, _si())
    stealth = evaluate_coherence(CANONICAL_SIGNATURES["stealth"], q, _si())
    assert stealth.T1_us > blade.T1_us            # ranking inverts vs logic dies


def test_ranking_orders_by_t1():
    ranked = rank_methods(QubitSpec(), _si())
    t1s = [rep.T1_us for _, rep in ranked]
    assert t1s == sorted(t1s, reverse=True)
    assert ranked[0][0].method in ("stealth", "plasma")
    assert ranked[-1][0].method == "blade"


def test_deeper_damage_lowers_t1():
    q = QubitSpec()
    shallow = evaluate_coherence(ProcessSignature("x", ssd_um=0.5, debris_um=0.0, wet=False), q, _si())
    deep = evaluate_coherence(ProcessSignature("x", ssd_um=20.0, debris_um=0.0, wet=False), q, _si())
    assert deep.T1_us < shallow.T1_us


def test_larger_keepout_raises_t1():
    sig = CANONICAL_SIGNATURES["blade"]
    near = evaluate_coherence(sig, QubitSpec(keepout_um=50.0), _si()).T1_us
    far = evaluate_coherence(sig, QubitSpec(keepout_um=400.0), _si()).T1_us
    assert far > near


def test_blade_needs_more_keepout_than_stealth():
    q = QubitSpec(T1_target_us=120.0)
    blade = evaluate_coherence(CANONICAL_SIGNATURES["blade"], q, _si())
    stealth = evaluate_coherence(CANONICAL_SIGNATURES["stealth"], q, _si())
    assert blade.min_keepout_um > stealth.min_keepout_um


def test_wet_process_adds_loss():
    q = QubitSpec()
    dry = evaluate_coherence(ProcessSignature("x", ssd_um=5.0, debris_um=0.0, wet=False), q, _si())
    wet = evaluate_coherence(ProcessSignature("x", ssd_um=5.0, debris_um=0.0, wet=True), q, _si())
    assert wet.T1_us < dry.T1_us


def test_debris_kills_regardless_of_t1():
    # huge keep-out → T1 fine, but a killer particle still fails the chip
    q = QubitSpec(keepout_um=1000.0, debris_kill_um=8.0)
    rep = evaluate_coherence(ProcessSignature("x", ssd_um=1.0, debris_um=12.0, wet=False), q, _si())
    assert rep.debris_fail and not rep.feasible


def test_min_keepout_meets_target():
    # placing the qubit exactly at the reported min keep-out should hit the budget
    q0 = QubitSpec(T1_target_us=120.0)
    sig = CANONICAL_SIGNATURES["blade"]
    L = evaluate_coherence(sig, q0, _si()).min_keepout_um
    assert math.isfinite(L)
    rep = evaluate_coherence(sig, QubitSpec(T1_target_us=120.0, keepout_um=L), _si())
    assert rep.T1_us == pytest.approx(120.0, rel=1e-3)


def test_signature_from_outcome():
    class _O:
        method = "blade"
        subsurface_crack_um = 9.0
        haz_um = 3.0
        chipping_um = 11.0
    sig = signature_from_outcome(_O())
    assert sig.method == "blade" and sig.wet is True
    assert sig.ssd_um == 9.0 and sig.debris_um == 11.0


def test_pristine_substrate_ceiling():
    # with essentially no edge coupling (huge keep-out, zero damage) T1 → bulk ceiling
    q = QubitSpec(keepout_um=5000.0)
    rep = evaluate_coherence(ProcessSignature("x", ssd_um=0.0, debris_um=0.0, wet=False), q, _si())
    s = _si()
    ceiling = 1.0 / (q.omega * s.p_bulk * s.tan_delta_bulk) * 1e6
    assert rep.T1_us == pytest.approx(ceiling, rel=1e-3)
