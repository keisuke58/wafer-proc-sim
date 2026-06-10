"""
crack_metrics.py — Crack-geometry summary statistics for damage fields.

NOTE on the deflection-angle estimator
---------------------------------------
We tried to "improve" crack_deflection_angle (at2_simulator_2d) by restricting
the fit to the propagation region and edge-masking the loading zones, and also a
PCA-of-the-damage-cloud variant.  On SIMULATOR ground-truth fields both variants
were WORSE — non-monotonic in theta and noisy — because the deflection signal in
this regime is small (~1.7°→6.7° over theta∈[0,90]) and the whole-domain,
damage-weighted per-column line fit of the original estimator is the most stable
way to extract it.  Conclusion: the original crack_deflection_angle is adequate;
DDPM's poor phi(theta) tracking (corr=-0.6) is a REAL model behaviour (it emits a
compact initiation blob, not an extended directional crack), not an estimator
artifact.  Use crack_deflection_angle from at2_simulator_2d for the angle.

This module keeps only the summary statistics that ARE useful downstream
(e.g. as conditioning features for the FMPS inverse model).
"""

from __future__ import annotations

import numpy as np


def crack_extent_px(d_field: np.ndarray, threshold: float = 0.5) -> int:
    """Number of damaged pixels (d > threshold) — a crack-size summary stat."""
    return int((d_field > threshold).sum())


def crack_summary(d_field: np.ndarray, cfg, threshold: float = 0.5) -> np.ndarray:
    """
    Low-dimensional summary of a damage field for simulation-based inference:
      [ deflection_angle_deg, cracked_fraction, d_mean, centroid_x_frac ]

    deflection_angle uses the original (adequate) estimator from at2_simulator_2d.
    Robust to absent cracks (returns 0-angle, 0-extent).
    """
    from research.phasefield.at2_simulator_2d import crack_deflection_angle
    ny, nx = cfg.ny, cfg.nx
    cracked = d_field > threshold
    frac = float(cracked.mean())
    d_mean = float(d_field.mean())
    if cracked.sum() > 0:
        xs = np.linspace(0.0, 1.0, nx)
        cx = float(np.average(np.tile(xs, (ny, 1))[cracked]))
    else:
        cx = 0.0
    ang = crack_deflection_angle(d_field, cfg)
    return np.array([ang, frac, d_mean, cx], dtype=np.float32)
