"""
taiko/deflection.py — TAIKO® Process: Physics-Based Plate-Deflection Model

Physics complement to the data-driven modules
(`taiko/optimizer.py`, `ml/taiko_grinding_gp.py`): instead of mapping
grinding recipes to measured warpage with a GP, this module predicts the
*self-weight (gravity) sag* of a thinned 300 mm Si wafer from first
principles (Kirchhoff thin-plate theory), and quantifies why DISCO's
TAIKO® edge ring works.

Background (DISCO TAIKO® process)
---------------------------------
A 0.7 mm thick, 300 mm wafer is back-ground to <= 50 µm. A uniformly
thinned wafer of that size sags severely under its own weight
(w ∝ q a⁴ / D with D ∝ h³) and cracks during handling. TAIKO® leaves a
~3 mm wide ring at full thickness (≈700 µm) on the outer edge and thins
only the inside — the stiff ring acts as a built-in elastic clamp for
the thin membrane ("taiko drum" structure), so no carrier wafer is
needed.

Models
------
1. Uniformly thinned wafer — *analytical* Kirchhoff solution for a
   simply supported circular plate under uniform load q = ρ g h:

       w_max = (5+ν)/(64 (1+ν)) · q a⁴ / D,   D = E h³ / 12(1-ν²)

2. TAIKO® structure — axisymmetric *variable-thickness* circular plate
   (thin inner disk of thickness h_thin, edge ring of thickness h_ring
   and width w_ring), simply supported at the outer rim. Solved
   numerically as a linear first-order ODE system in (w, φ, M_r) with
   scipy.integrate.solve_bvp:

       w'   = φ
       φ'   = -M_r/D(r) - ν φ / r
       M_r' = Q_r(r) - (M_r - M_θ)/r,   M_θ = ν M_r - D(r)(1-ν²) φ/r
       Q_r(r) = -(1/r) ∫₀ʳ q(s) s ds   (statics, q(r) = ρ g h(r))

   BCs: φ(0) = 0 (symmetry), w(a) = 0, M_r(a) = 0 (simple support).
   The thickness step at r = a - w_ring is smoothed with a tanh fillet
   (grinding wheels leave a fillet anyway), which keeps the BVP smooth.

Sign convention: w and q positive *downward* (direction of gravity),
so w_max > 0 means the wafer sags.

Caveat: linear Kirchhoff theory is only quantitative while w ≲ h; for a
50 µm full wafer the predicted sag (≈25 mm) far exceeds the thickness,
i.e. membrane stiffening would limit it in reality. The model is used
*comparatively* (uniform vs TAIKO®), where the linear ratio is the
meaningful, conservative figure of merit.

Bending stress (crack risk): σ = 6 M / h(r)², compared against the Si
flexural strength with a handling-shock factor (g_factor) and a safety
factor — gives the "crack-risk boundary" in the parametric map.

Usage
-----
    from taiko.deflection import (
        uniform_disk_max_deflection, solve_taiko_deflection,
        taiko_advantage, parametric_study,
    )

    adv = taiko_advantage(h_thin=50e-6, w_ring=3e-3)   # ≈ 3–4×

Demo:  python taiko/deflection.py   → results/taiko_deflection.png

Dependencies: numpy, scipy, matplotlib only (no JAX).
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_bvp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from data.materials.material_properties import Si as _Si

    E_SI = _Si["E"]                       # 130 GPa, <100>
    NU_SI = _Si["nu"]                     # 0.28
    RHO_SI = _Si["density"]               # 2330 kg/m³
    SIGMA_FLEX_SI = _Si["sigma_flex"]     # 200 MPa flexural strength
except ImportError:  # pragma: no cover - fallback if run standalone
    E_SI, NU_SI, RHO_SI, SIGMA_FLEX_SI = 130e9, 0.28, 2330.0, 200e6

G_ACCEL = 9.81                  # m/s²
WAFER_RADIUS_M = 0.150          # 300 mm wafer
H_FULL_M = 700e-6               # full (incoming) wafer thickness ≈ 0.7 mm


# ── Analytical: uniform circular plate, simply supported ──────────────────────

def flexural_rigidity(h: float, E: float = E_SI, nu: float = NU_SI) -> float:
    """Plate flexural rigidity D = E h³ / 12(1-ν²)  [N·m]."""
    return E * h ** 3 / (12.0 * (1.0 - nu ** 2))


def uniform_disk_max_deflection(
    h: float,
    a: float = WAFER_RADIUS_M,
    E: float = E_SI,
    nu: float = NU_SI,
    rho: float = RHO_SI,
    q: float | None = None,
    g_factor: float = 1.0,
) -> float:
    """
    Max (center) deflection of a uniform, simply supported circular plate
    under uniform pressure q [N/m²] (Timoshenko & Woinowsky-Krieger, eq. 68):

        w_max = (5+ν)/(64(1+ν)) · q a⁴ / D

    If q is None it defaults to self-weight q = ρ g h · g_factor.
    Returns w_max [m], positive downward.
    """
    if q is None:
        q = rho * G_ACCEL * g_factor * h
    D = flexural_rigidity(h, E, nu)
    return (5.0 + nu) / (64.0 * (1.0 + nu)) * q * a ** 4 / D


def uniform_disk_profile(
    r: np.ndarray,
    h: float,
    a: float = WAFER_RADIUS_M,
    E: float = E_SI,
    nu: float = NU_SI,
    rho: float = RHO_SI,
    q: float | None = None,
    g_factor: float = 1.0,
) -> np.ndarray:
    """
    Full analytical deflection profile w(r) of the simply supported
    uniform circular plate under uniform load:

        w(r) = q (a²-r²) / (64 D) · [ (5+ν)/(1+ν) a² - r² ]
    """
    if q is None:
        q = rho * G_ACCEL * g_factor * h
    D = flexural_rigidity(h, E, nu)
    r = np.asarray(r, dtype=float)
    return q * (a ** 2 - r ** 2) / (64.0 * D) * (
        (5.0 + nu) / (1.0 + nu) * a ** 2 - r ** 2
    )


def uniform_disk_max_stress(
    h: float,
    a: float = WAFER_RADIUS_M,
    nu: float = NU_SI,
    rho: float = RHO_SI,
    q: float | None = None,
    g_factor: float = 1.0,
) -> float:
    """
    Max bending stress (center) of the simply supported uniform plate:
    M_max = (3+ν) q a² / 16,  σ = 6 M / h².  Returns σ_max [Pa].
    """
    if q is None:
        q = rho * G_ACCEL * g_factor * h
    M_max = (3.0 + nu) * q * a ** 2 / 16.0
    return 6.0 * M_max / h ** 2


# ── Numerical: TAIKO® stepped (variable-thickness) plate ──────────────────────

def solve_taiko_deflection(
    h_thin: float,
    w_ring: float,
    h_ring: float = H_FULL_M,
    a: float = WAFER_RADIUS_M,
    E: float = E_SI,
    nu: float = NU_SI,
    rho: float = RHO_SI,
    g_factor: float = 1.0,
    n_mesh: int = 600,
    smooth_w: float | None = None,
) -> dict:
    """
    Self-weight deflection of the TAIKO® wafer: thin inner disk
    (thickness h_thin, radius b = a - w_ring) + full-thickness edge ring
    (h_ring, width w_ring), simply supported at the outer rim r = a.

    Axisymmetric Kirchhoff bending with radially varying thickness h(r),
    solved as a linear BVP in y = (w, φ=dw/dr, M_r) — see module docstring.

    Parameters
    ----------
    h_thin   : ground (inner) thickness [m], e.g. 50e-6
    w_ring   : TAIKO ring width [m], e.g. 3e-3
    h_ring   : ring (original wafer) thickness [m], default 700 µm
    g_factor : load multiplier (handling shock, e.g. 10 for 10 g)
    smooth_w : tanh smoothing half-width of the thickness step [m];
               default min(50 µm, w_ring/4)

    Returns
    -------
    dict with keys:
        r, w, phi, M_r, M_t, h   : radial profiles (r in m, w in m, M in N)
        w_max                    : max deflection [m] (positive = sag)
        sigma_max                : max bending stress 6|M|/h² [Pa]
        b                        : inner (thin-region) radius [m]
    """
    if not (0.0 < h_thin <= h_ring):
        raise ValueError("require 0 < h_thin <= h_ring")
    if not (0.0 < w_ring < a):
        raise ValueError("require 0 < w_ring < a")

    b = a - w_ring
    if smooth_w is None:
        smooth_w = min(50e-6, w_ring / 4.0)

    def h_of(r):
        """Smoothed thickness step h_thin → h_ring at r = b."""
        return h_thin + (h_ring - h_thin) * 0.5 * (
            1.0 + np.tanh((r - b) / smooth_w)
        )

    # Shear from statics: Q_r(r) = -(1/r) ∫₀ʳ q(s) s ds, q = ρ g h(r)
    r_fine = np.linspace(0.0, a, 6000)
    q_fine = rho * G_ACCEL * g_factor * h_of(r_fine)
    cum = cumulative_trapezoid(q_fine * r_fine, r_fine, initial=0.0)

    def Q_of(r):
        return -np.interp(r, r_fine, cum) / np.maximum(r, 1e-12)

    def D_of(r):
        return E * h_of(r) ** 3 / (12.0 * (1.0 - nu ** 2))

    def rhs(r, y):
        w, phi, Mr = y
        D = D_of(r)
        dphi = -Mr / D - nu * phi / r
        Mt = nu * Mr - D * (1.0 - nu ** 2) * phi / r
        dMr = Q_of(r) - (Mr - Mt) / r
        return np.vstack([phi, dphi, dMr])

    def bc(y0, y1):
        # φ(r0)=0 (axis symmetry), w(a)=0, M_r(a)=0 (simple support)
        return np.array([y0[1], y1[0], y1[2]])

    r0 = a * 1e-4
    # mesh refined around the thickness step
    r_mesh = np.linspace(r0, a, n_mesh)
    step_lo = max(r0, b - 8.0 * smooth_w)
    step_hi = min(a, b + 8.0 * smooth_w)
    r_mesh = np.union1d(r_mesh, np.linspace(step_lo, step_hi, 200))

    # initial guess: analytical uniform-thin solution (BVP is linear,
    # so the guess only sets the mesh-adaptation starting point)
    w_guess = uniform_disk_profile(r_mesh, h_thin, a, E, nu, rho,
                                   g_factor=g_factor)
    y_guess = np.vstack([w_guess, np.zeros_like(r_mesh),
                         np.zeros_like(r_mesh)])

    sol = solve_bvp(rhs, bc, r_mesh, y_guess, max_nodes=200000, tol=1e-8)
    if not sol.success:  # pragma: no cover
        raise RuntimeError(f"solve_bvp failed: {sol.message}")

    r_out = np.linspace(r0, a, 800)
    w, phi, Mr = sol.sol(r_out)
    D = D_of(r_out)
    Mt = nu * Mr - D * (1.0 - nu ** 2) * phi / r_out
    h_out = h_of(r_out)
    sigma = 6.0 * np.maximum(np.abs(Mr), np.abs(Mt)) / h_out ** 2

    return {
        "r": r_out, "w": w, "phi": phi, "M_r": Mr, "M_t": Mt, "h": h_out,
        "w_max": float(np.max(w)),
        "sigma_max": float(np.max(sigma)),
        "b": b,
    }


# ── Comparison & parametric study ──────────────────────────────────────────────

def taiko_advantage(
    h_thin: float,
    w_ring: float,
    h_ring: float = H_FULL_M,
    a: float = WAFER_RADIUS_M,
    **kwargs,
) -> float:
    """
    Ratio of self-weight max deflection:

        advantage = w_max(uniformly thinned to h_thin)
                    --------------------------------------
                    w_max(TAIKO®: h_thin inside + h_ring ring)

    > 1 means the TAIKO® structure sags less. For the standard case
    (h_thin=50 µm, w_ring=3 mm, 300 mm wafer) the ring acts as a nearly
    built-in clamp, giving ≈ (5+ν)/(1+ν) ≈ 4× reduction.
    """
    w_uniform = uniform_disk_max_deflection(h_thin, a=a, **{
        k: v for k, v in kwargs.items()
        if k in ("E", "nu", "rho", "g_factor")
    })
    res = solve_taiko_deflection(h_thin, w_ring, h_ring=h_ring, a=a, **kwargs)
    return w_uniform / res["w_max"]


def parametric_study(
    ring_widths: np.ndarray | None = None,
    thin_thicknesses: np.ndarray | None = None,
    h_ring: float = H_FULL_M,
    a: float = WAFER_RADIUS_M,
    **kwargs,
) -> dict:
    """
    Deflection / stress map over ring width × remaining thickness.

    Defaults: ring widths 1–5 mm (9 pts) × h_thin 30–100 µm (8 pts).

    Returns dict:
        ring_widths [m], thin_thicknesses [m],
        w_max     [m]  shape (n_thick, n_width) — TAIKO max sag (1 g)
        sigma_max [Pa] shape (n_thick, n_width) — TAIKO max bending stress (1 g)
        advantage      shape (n_thick, n_width) — uniform/TAIKO sag ratio

    Stress scales linearly with g_factor, so handling-shock maps are
    obtained by multiplying sigma_max by the shock factor.
    """
    if ring_widths is None:
        ring_widths = np.linspace(1e-3, 5e-3, 9)
    if thin_thicknesses is None:
        thin_thicknesses = np.linspace(30e-6, 100e-6, 8)

    nw, nt = len(ring_widths), len(thin_thicknesses)
    w_max = np.zeros((nt, nw))
    sigma_max = np.zeros((nt, nw))
    advantage = np.zeros((nt, nw))

    for i, h_thin in enumerate(thin_thicknesses):
        w_uni = uniform_disk_max_deflection(h_thin, a=a)
        for j, wr in enumerate(ring_widths):
            res = solve_taiko_deflection(h_thin, wr, h_ring=h_ring, a=a,
                                         **kwargs)
            w_max[i, j] = res["w_max"]
            sigma_max[i, j] = res["sigma_max"]
            advantage[i, j] = w_uni / res["w_max"]

    return {
        "ring_widths": np.asarray(ring_widths),
        "thin_thicknesses": np.asarray(thin_thicknesses),
        "w_max": w_max,
        "sigma_max": sigma_max,
        "advantage": advantage,
    }


# ── Demo ───────────────────────────────────────────────────────────────────────

def _demo(out_png: str | None = None) -> str:
    """Generate the two-panel demo figure → results/taiko_deflection.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_png is None:
        out_png = os.path.join(os.path.dirname(__file__), "..",
                               "results", "taiko_deflection.png")
    out_png = os.path.normpath(out_png)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    h_thin, w_ring = 50e-6, 3e-3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # (a) cross-section: uniform 50 µm vs TAIKO 50 µm + 3 mm ring
    res = solve_taiko_deflection(h_thin, w_ring)
    r_mm = res["r"] * 1e3
    w_uni = uniform_disk_profile(res["r"], h_thin) * 1e3   # mm
    adv = taiko_advantage(h_thin, w_ring)

    ax1.plot(r_mm, -w_uni, "r--", lw=2,
             label=f"uniform 50 µm (w_max={np.max(w_uni):.1f} mm)")
    ax1.plot(r_mm, -res["w"] * 1e3, "b-", lw=2,
             label=f"TAIKO 50 µm + 3 mm ring "
                   f"(w_max={res['w_max']*1e3:.1f} mm)")
    ax1.axvline(res["b"] * 1e3, color="gray", ls=":", lw=1)
    ax1.text(res["b"] * 1e3 - 2, ax1.get_ylim()[0] * 0.05, "ring",
             color="gray", fontsize=8, ha="right")
    ax1.set_xlabel("radius r [mm]")
    ax1.set_ylabel("self-weight deflection w [mm] (down = negative)")
    ax1.set_title(f"(a) 300 mm Si wafer self-weight sag — "
                  f"TAIKO advantage ×{adv:.1f}\n"
                  "(linear Kirchhoff, edge simply supported; illustrative "
                  "for w > h)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # (b) parametric map: ring width × remaining thickness
    study = parametric_study()
    rw_mm = study["ring_widths"] * 1e3
    ht_um = study["thin_thicknesses"] * 1e6
    w_mm = study["w_max"] * 1e3

    G_SHOCK, SF = 10.0, 2.0          # 10 g handling shock, safety factor 2
    sigma_shock_MPa = study["sigma_max"] * G_SHOCK / 1e6
    sigma_allow_MPa = SIGMA_FLEX_SI / SF / 1e6   # 100 MPa

    pc = ax2.pcolormesh(rw_mm, ht_um, w_mm, shading="auto", cmap="viridis")
    fig.colorbar(pc, ax=ax2, label="TAIKO max sag w_max [mm] (1 g)")
    cs = ax2.contour(rw_mm, ht_um, sigma_shock_MPa,
                     levels=[sigma_allow_MPa], colors="red", linewidths=2)
    ax2.clabel(cs, fmt=lambda _: "crack-risk boundary\n"
               f"(σ@{G_SHOCK:.0f}g = {sigma_allow_MPa:.0f} MPa)",
               fontsize=8)
    ax2.plot(3.0, 50.0, "w*", ms=15, mec="k",
             label="TAIKO standard (3 mm, 50 µm)")
    ax2.set_xlabel("ring width w_ring [mm]")
    ax2.set_ylabel("remaining thickness h_thin [µm]")
    ax2.set_title("(b) TAIKO deflection map + crack-risk boundary\n"
                  f"(risk: bending σ under {G_SHOCK:.0f} g handling shock "
                  f"> σ_flex/{SF:.0f})")
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


if __name__ == "__main__":
    h_thin, w_ring = 50e-6, 3e-3
    w_uni = uniform_disk_max_deflection(h_thin)
    res = solve_taiko_deflection(h_thin, w_ring)
    adv = taiko_advantage(h_thin, w_ring)
    print("TAIKO® deflection model — 300 mm Si wafer, self-weight (1 g)")
    print(f"  uniform 50 µm        : w_max = {w_uni*1e3:8.2f} mm, "
          f"sigma_max = {uniform_disk_max_stress(h_thin)/1e6:6.1f} MPa")
    print(f"  TAIKO 50 µm + 3 mm   : w_max = {res['w_max']*1e3:8.2f} mm, "
          f"sigma_max = {res['sigma_max']/1e6:6.1f} MPa")
    print(f"  TAIKO advantage      : ×{adv:.2f} less sag")
    png = _demo()
    print(f"  figure saved         : {png}")
