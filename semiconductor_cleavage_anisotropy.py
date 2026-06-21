"""semiconductor_cleavage_anisotropy.py — anisotropic-cleavage fracture for SiC/GaN dicing [#2].

The semiconductor fracture-method ranking pointed at PHASE-FIELD with an anisotropic cleavage energy
as the frontier for SiC/GaN dicing. The wafer-proc-sim 2-D anisotropic AT2 model
(research/phasefield/at2_simulator_2d.py) already encodes that anisotropy as
    Gc_eff(n) = Gc · (1 + β (n·c)²),   c = cleavage-plane normal,  n = crack-propagation direction.
This module makes the DICING-relevant consequences explicit and fast (analytic), then links them to
the GNN edge-conditioning recipe.

For a kerf cut at angle θ_cut to the cleavage plane (c = (−sinθ, cosθ)), a crack going in direction
n=(cosφ, sinφ) has  n·c = sin(φ−θ), so
    Gc_eff(φ) = Gc · (1 + β sin²(φ−θ_cut)).
Two dicing quantities follow:
  * KERF toughness (crack along the cut, φ=0):  Gc_eff = Gc(1 + β sin²θ_cut)  — the cutting effort.
  * DEFLECTION driving force at the kerf:  |dGc_eff/dφ|_{φ=0} = Gc|β sin(2θ_cut)| — the tendency of the
    crack to LEAVE the kerf toward the cleavage plane = kerf-deviation / chipping. Max at θ_cut=45°,
    ZERO at θ_cut=0° or 90° (cut aligned with cleavage → clean, straight kerf).

=> Dicing guide: orient the cut ALONG a cleavage plane (θ_cut→0/90°) to null the deflection force and
   minimise chipping; SiC's strong anisotropy (β≈−0.51) makes orientation matter far more than for Si.

β provenance: SiC β≈−0.51 is the wafer-proc-sim MAP-fit value; Si/GaN here are ILLUSTRATIVE model
anisotropies (to be calibrated to fab cleavage data). This is the analytic core; a full crack-growth
deflection demo uses run_forward_2d at the calibrated grid.

Run:  python3 semiconductor_cleavage_anisotropy.py   (writes semiconductor_cleavage_anisotropy.png)
"""
from __future__ import annotations

import numpy as np

# model cleavage-anisotropy β (SiC from wafer-proc-sim fit; Si/GaN illustrative)
MAT = {"Si": dict(beta=-0.12, color="#1f6f6f"),
       "GaN": dict(beta=-0.30, color="#c0392b"),
       "SiC": dict(beta=-0.51, color="#b8860b")}


def gc_eff(phi, theta, beta):
    """Gc_eff/Gc for crack direction phi, cut angle theta, anisotropy beta."""
    return 1.0 + beta * np.sin(phi - theta) ** 2


def kerf_toughness(theta, beta):
    """Gc_eff/Gc for a crack running along the kerf (phi=0)."""
    return 1.0 + beta * np.sin(theta) ** 2


def deflection_force(theta, beta):
    """|dGc_eff/dphi| at phi=0 = |beta sin(2 theta)| — tendency to deviate from the kerf."""
    return np.abs(beta * np.sin(2.0 * theta))


def main():
    th = np.linspace(0, np.pi / 2, 91)
    th_deg = np.rad2deg(th)
    print("=== dicing anisotropy (Gc_eff/Gc) ===")
    for n, m in MAT.items():
        kmin, kmax = kerf_toughness(th, m["beta"]).min(), kerf_toughness(th, m["beta"]).max()
        dmax = deflection_force(th, m["beta"]).max()
        print(f"  {n:4s} β={m['beta']:+.2f}  kerf Gc_eff/Gc∈[{kmin:.2f},{kmax:.2f}]  "
              f"max deflection-force={dmax:.2f} at θ_cut=45°")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14.5, 4.6))

    # (a) polar Gc_eff(phi) at a fixed cut (theta=45deg) — the toughness anisotropy
    ax = fig.add_subplot(1, 3, 1, projection="polar")
    phi = np.linspace(0, 2 * np.pi, 361)
    for n, m in MAT.items():
        ax.plot(phi, gc_eff(phi, np.pi / 4, m["beta"]), lw=2, color=m["color"], label=f"{n} (β={m['beta']:+.2f})")
    ax.set_title("(a) Toughness anisotropy Gc_eff(φ)\n(cut θ=45°; dip = cleavage soft direction)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.25, 1.1))

    # (b) kerf cutting effort vs cut orientation
    ax = fig.add_subplot(1, 3, 2)
    for n, m in MAT.items():
        ax.plot(th_deg, kerf_toughness(th, m["beta"]), lw=2, color=m["color"], label=n)
    ax.set(xlabel="cut angle θ_cut to cleavage [deg]", ylabel="kerf Gc_eff / Gc",
           title="(b) Cutting effort along the kerf\nlower = easier split (cut across cleavage)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    # (c) deflection driving force = chipping/kerf-deviation tendency
    ax = fig.add_subplot(1, 3, 3)
    for n, m in MAT.items():
        ax.plot(th_deg, deflection_force(th, m["beta"]), lw=2, color=m["color"], label=n)
    ax.axvline(45, color="gray", ls=":", lw=1)
    ax.set(xlabel="cut angle θ_cut to cleavage [deg]", ylabel="|dGc_eff/dφ|  (deflection force)",
           title="(c) Kerf-deviation / chipping tendency\nZERO at θ=0/90° → dice ALONG cleavage")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    fig.suptitle("Anisotropic-cleavage fracture for SiC/GaN dicing: cut ALONG a cleavage plane to "
                 "null the deflection force and minimise chipping (DISCO)", fontweight="bold", fontsize=12)
    fig.tight_layout()
    out = "/home/nishioka/git/wafer-proc-sim/semiconductor_cleavage_anisotropy.png"
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
