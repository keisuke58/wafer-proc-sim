"""
Tests for at2_simulator_3d.py

Physics assertions (mirroring 2D tests, plus unique 3D checks):
  - Isotropic: both crack angles ≈ 0°
  - phi_xy sign flip between beta<0 / beta>0 at theta=45° (same as 2D)
  - Symmetric endpoints: phi_xy ≈ 0° at theta=0° and theta=90°
  - psi_cut ≠ 0 produces non-zero phi_xz (unique 3D observable)
  - phi_xz = 0 when psi_cut = 0 (no out-of-plane cleavage)
  - Damage stays in [0, 1]
  - Log-likelihood is finite and ≤ 0
"""
from __future__ import annotations
import sys, os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from research.phasefield.at2_simulator_3d import (
    PFParams3D, SimConfig3D, ForwardResult3D,
    run_forward_3d, crack_deflection_angles_3d,
)

# ── Fast config: tiny grid, few steps ────────────────────────────────────────
FAST = SimConfig3D(nx=15, ny=9, nz=9, n_steps=40, u_max_factor=5.0)
BASE = PFParams3D(Gc=8e-4, ell=0.04, E=1.0, beta=0.0)


def test_isotropic_both_angles_zero():
    """Isotropic (beta=0) gives phi_xy ≈ 0° and phi_xz ≈ 0°."""
    r = run_forward_3d(BASE, FAST)
    assert abs(r.crack_angle_xy) < 1.0, f"phi_xy={r.crack_angle_xy:.3f}°"
    assert abs(r.crack_angle_xz) < 1.0, f"phi_xz={r.crack_angle_xz:.3f}°"


def test_phi_xy_sign_flip_at_theta45():
    """beta<0 at theta=45° deflects crack up (phi_xy>0); beta>0 deflects down."""
    r_neg = run_forward_3d(
        PFParams3D(Gc=8e-4, ell=0.04, E=1.0, beta=-0.8,
                   theta_cut=np.radians(45)), FAST)
    r_pos = run_forward_3d(
        PFParams3D(Gc=8e-4, ell=0.04, E=1.0, beta=+0.8,
                   theta_cut=np.radians(45)), FAST)
    assert r_neg.crack_angle_xy > 0.0, f"beta<0 phi_xy={r_neg.crack_angle_xy:.3f}°"
    assert r_pos.crack_angle_xy < 0.0, f"beta>0 phi_xy={r_pos.crack_angle_xy:.3f}°"


def test_phi_xy_zero_at_endpoints():
    """theta=0° and theta=90° give diagonal A → phi_xy ≈ 0°."""
    for theta_deg in [0, 90]:
        r = run_forward_3d(
            PFParams3D(Gc=8e-4, ell=0.04, E=1.0, beta=-0.8,
                       theta_cut=np.radians(theta_deg)), FAST)
        assert abs(r.crack_angle_xy) < 1.5, (
            f"theta={theta_deg}° phi_xy={r.crack_angle_xy:.3f}°"
        )


def test_phi_xz_zero_without_psi():
    """With psi_cut=0 the crack stays in x-y plane → phi_xz ≈ 0°."""
    r = run_forward_3d(
        PFParams3D(Gc=8e-4, ell=0.04, E=1.0, beta=-0.8,
                   theta_cut=np.radians(45), psi_cut=0.0), FAST)
    assert abs(r.crack_angle_xz) < 1.0, f"phi_xz={r.crack_angle_xz:.3f}°"


def test_phi_xz_nonzero_with_psi():
    """psi_cut=45° introduces out-of-plane cleavage → phi_xz ≠ 0 (3D-only observable)."""
    r_psi0  = run_forward_3d(
        PFParams3D(Gc=8e-4, ell=0.04, E=1.0, beta=-0.8,
                   psi_cut=0.0), FAST)
    r_psi45 = run_forward_3d(
        PFParams3D(Gc=8e-4, ell=0.04, E=1.0, beta=-0.8,
                   psi_cut=np.radians(45)), FAST)
    assert abs(r_psi45.crack_angle_xz) > abs(r_psi0.crack_angle_xz), (
        f"psi=45° should twist more than psi=0°: "
        f"{r_psi45.crack_angle_xz:.3f}° vs {r_psi0.crack_angle_xz:.3f}°"
    )


def test_damage_in_unit_interval():
    """Damage field stays in [0, 1]."""
    r = run_forward_3d(BASE, FAST)
    assert r.d_field.min() >= -1e-10
    assert r.d_field.max() <= 1.0 + 1e-10


def test_log_likelihood_finite_and_nonpositive():
    """crack_angle_log_likelihood_3d returns a finite, non-positive float."""
    from research.phasefield.at2_simulator_3d import crack_angle_log_likelihood_3d
    ll = crack_angle_log_likelihood_3d(
        PFParams3D(Gc=8e-4, ell=0.04, E=1.0, beta=-0.4),
        obs=[(45.0, 0.0, 1.5, 0.0)],
        sigma_phi=2.0,
        cfg=FAST,
    )
    assert np.isfinite(ll), f"ll={ll}"
    assert ll <= 0.0, f"ll={ll}"
