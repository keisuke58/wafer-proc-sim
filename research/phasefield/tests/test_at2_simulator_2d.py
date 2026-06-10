"""
Tests for at2_simulator_2d.py

Physics-based assertions:
  - Isotropic: crack angle phi ≈ 0° (Mode-I BC gives exact symmetry)
  - Anisotropy sign: beta<0 at theta=45° deflects UP; beta>0 deflects DOWN
  - Symmetric endpoints: phi ≈ 0° at theta=0° and theta=90° (diagonal A, no cross term)
  - Stronger |beta| → larger |phi| at theta=45°
"""
from __future__ import annotations
import sys, os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from research.phasefield.at2_simulator_2d import (
    PFParams2D, SimConfig2D, ForwardResult2D,
    run_forward_2d, crack_deflection_angle,
)

# ── Fast config: small grid, few steps, runs in ~1–2 s per call ──────────────
FAST = SimConfig2D(nx=40, ny=21, n_steps=50, u_max_factor=5.0)
BASE = PFParams2D(Gc=8e-4, ell=0.04, E=1.0, beta=0.0)


def test_isotropic_phi_zero():
    """Isotropic (beta=0) Mode-I produces zero crack deflection."""
    r = run_forward_2d(BASE, FAST)
    assert abs(r.crack_angle) < 0.5, (
        f"Isotropic crack angle should be ≈0°, got {r.crack_angle:.3f}°"
    )


def test_anisotropy_sign_flip_at_45():
    """beta<0 deflects crack upward at theta=45°; beta>0 deflects downward."""
    r_neg = run_forward_2d(PFParams2D(Gc=8e-4, ell=0.04, E=1.0,
                                      beta=-0.8, theta_cut=np.radians(45)), FAST)
    r_pos = run_forward_2d(PFParams2D(Gc=8e-4, ell=0.04, E=1.0,
                                      beta=+0.8, theta_cut=np.radians(45)), FAST)
    assert r_neg.crack_angle > 0.0, (
        f"beta<0 at 45° should deflect up, got {r_neg.crack_angle:.3f}°"
    )
    assert r_pos.crack_angle < 0.0, (
        f"beta>0 at 45° should deflect down, got {r_pos.crack_angle:.3f}°"
    )


def test_symmetric_endpoints_zero_phi():
    """theta=0° and theta=90° give diagonal A (no cross term): phi ≈ 0°."""
    for theta_deg in [0, 90]:
        r = run_forward_2d(
            PFParams2D(Gc=8e-4, ell=0.04, E=1.0, beta=-0.8,
                       theta_cut=np.radians(theta_deg)),
            FAST,
        )
        assert abs(r.crack_angle) < 1.0, (
            f"theta={theta_deg}° should give phi≈0°, got {r.crack_angle:.3f}°"
        )


def test_stronger_beta_larger_deflection():
    """Larger |beta| at theta=45° produces larger crack deflection."""
    r_weak = run_forward_2d(
        PFParams2D(Gc=8e-4, ell=0.04, E=1.0, beta=-0.4, theta_cut=np.radians(45)), FAST
    )
    r_strong = run_forward_2d(
        PFParams2D(Gc=8e-4, ell=0.04, E=1.0, beta=-0.8, theta_cut=np.radians(45)), FAST
    )
    assert abs(r_strong.crack_angle) >= abs(r_weak.crack_angle), (
        f"|beta|=0.8 should deflect more than |beta|=0.4: "
        f"{r_strong.crack_angle:.3f}° vs {r_weak.crack_angle:.3f}°"
    )


def test_damage_field_in_unit_interval():
    """Damage field stays in [0, 1] throughout simulation."""
    r = run_forward_2d(BASE, FAST)
    assert r.d_field.min() >= -1e-10, "Damage below 0"
    assert r.d_field.max() <= 1.0 + 1e-10, "Damage above 1"


def test_crack_angle_log_likelihood_import():
    """crack_angle_log_likelihood is importable and returns a finite float."""
    from research.phasefield.at2_simulator_2d import crack_angle_log_likelihood
    ll = crack_angle_log_likelihood(
        PFParams2D(Gc=8e-4, ell=0.04, E=1.0, beta=-0.4),
        obs=[(45.0, 1.5)],
        sigma_phi=2.0,
        cfg=FAST,
    )
    assert np.isfinite(ll), f"Log-likelihood should be finite, got {ll}"
    assert ll <= 0.0, f"Log-likelihood should be ≤ 0, got {ll}"
