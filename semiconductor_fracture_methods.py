"""semiconductor_fracture_methods.py — the CFRP fracture-method ranking, ported to wafer dicing.

The same fracture-mechanics method ranking that orders CFRP analysis (strength criteria → LEFM/VCCT
→ CZM → phase-field) maps directly onto brittle single-crystal wafers (Si / SiC / GaN), where DISCO's
core value is controlling CHIPPING and subsurface damage during dicing/grinding. This module computes,
on literature-typical properties, the machining-fracture quantities that actually drive dicing:

  1. DUCTILE–BRITTLE transition depth  d_c = β (E/H)(K_Ic/H)²  (Bifano): the max depth of cut that
     stays chip-free. Below d_c → ductile-regime machining (no chipping); above → brittle (chipping).
  2. MEDIAN/RADIAL crack depth (subsurface damage) vs dicing load — Lawn–Evans–Marshall indentation
     fracture: c = α (E/H)^(2/3) (P / K_Ic)^(2/3). The depth of latent cracks left under the kerf.
  3. CHIPPING susceptibility index ~ (H/K_Ic)² · E (brittleness × stiffness): higher = chips more.
  4. METHOD–CAPABILITY matrix for DICING (strength / LEFM-indentation / CZM / phase-field /
     peridynamics) × {chipping, subsurface damage, anisotropic cleavage, laser/stealth multiphysics,
     ductile–brittle, differentiable/ML}.  Phase-field (anisotropic cleavage) stands out for SiC/GaN.

Properties are literature-typical, order-of-magnitude — this is a prototype to be calibrated against
DISCO/fab dicing data (KABRA laser, blade DBG). Strategic link: wafer-proc-sim, startup_plan,
disco_research.

Run:  python3 semiconductor_fracture_methods.py   (writes semiconductor_fracture_methods.png)
"""
from __future__ import annotations

import numpy as np

# E [GPa], H hardness [GPa], K_Ic [MPa·m^0.5], cleavage note. (literature-typical single crystal)
MAT = {
    "Si":  dict(E=170.0, H=11.0, KIc=0.90, cleave="{111} cubic", color="#1f6f6f"),
    "SiC": dict(E=430.0, H=26.0, KIc=3.30, cleave="basal/prism (hex)", color="#b8860b"),
    "GaN": dict(E=300.0, H=15.0, KIc=0.95, cleave="m-plane (hex)", color="#c0392b"),
}
BETA_BIFANO = 0.15      # Bifano ductile-brittle prefactor
ALPHA_LEMS = 0.027      # Lawn-Evans-Marshall median-crack prefactor (Vickers, order-of-mag)


def d_critical_nm(m):
    """Bifano ductile-brittle transition depth [nm]."""
    E, H, K = m["E"] * 1e9, m["H"] * 1e9, m["KIc"] * 1e6
    return BETA_BIFANO * (E / H) * (K / H) ** 2 * 1e9


def median_crack_um(m, P_N):
    """Lawn-Evans-Marshall median/radial crack depth [µm] for indentation load P [N]."""
    E, H, K = m["E"] * 1e9, m["H"] * 1e9, m["KIc"] * 1e6
    c = ALPHA_LEMS * (E / H) ** (2 / 3) * (np.asarray(P_N) / K) ** (2 / 3)   # [m]
    return c * 1e6


def chipping_index(m):
    """Relative chipping susceptibility ~ brittleness²·stiffness (higher = chips more)."""
    return (m["H"] / m["KIc"]) ** 2 * m["E"]


METHODS = ["Strength", "LEFM /\nindentation-FM", "CZM", "Phase-field\n(anisotropic)", "Peridyn."]
CAPS = ["chipping", "subsurface\ndamage", "anisotropic\ncleavage", "laser/stealth\nmultiphysics",
        "ductile–\nbrittle", "differen-\ntiable/ML"]
SCORE = np.array([
    # chip subsurf cleav laser dbt  diff
    [1, 0, 0, 0, 1, 1],   # Strength
    [1, 2, 1, 0, 1, 1],   # LEFM / indentation-FM (Lawn) — the dicing-damage workhorse
    [2, 1, 1, 1, 1, 1],   # CZM
    [2, 2, 2, 2, 2, 2],   # Phase-field anisotropic  <-- cleavage + branching + multiphysics + diff
    [2, 1, 1, 1, 1, 0],   # Peridynamics (fragmentation)
])


def main():
    print("=== wafer machining-fracture metrics (literature-typical) ===")
    print(f"{'material':6s}{'E[GPa]':>8}{'H[GPa]':>8}{'KIc':>7}{'d_c[nm]':>9}{'chip-idx':>10}  cleavage")
    cidx = {n: chipping_index(m) for n, m in MAT.items()}
    cmax = max(cidx.values())
    for n, m in MAT.items():
        print(f"{n:6s}{m['E']:>8.0f}{m['H']:>8.0f}{m['KIc']:>7.2f}{d_critical_nm(m):>9.1f}"
              f"{cidx[n]/cmax:>10.2f}  {m['cleave']}")

    P = np.logspace(-2, 0.5, 60)   # dicing/indentation load 0.01 .. ~3 N

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 8.2))

    # (a) ductile-brittle transition depth (the key dicing parameter)
    ax = fig.add_subplot(2, 3, 1)
    names = list(MAT); dcs = [d_critical_nm(MAT[n]) for n in names]
    ax.bar(names, dcs, color=[MAT[n]["color"] for n in names], ec="k")
    for i, v in enumerate(dcs):
        ax.annotate(f"{v:.0f} nm", (i, v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("ductile–brittle depth d_c [nm]")
    ax.set_title("(a) Chip-free machining limit d_c (Bifano)\nlarger = easier to dice ductile")
    ax.grid(alpha=.3, axis="y")

    # (b) median crack (subsurface damage) vs load
    ax = fig.add_subplot(2, 3, 2)
    for n in names:
        ax.loglog(P, median_crack_um(MAT[n], P), lw=2, color=MAT[n]["color"], label=n)
    ax.set(xlabel="dicing/indentation load P [N]", ylabel="median crack depth [µm]",
           title="(b) Subsurface damage (Lawn–Evans–Marshall)\nlatent crack depth under the kerf")
    ax.legend(fontsize=9); ax.grid(alpha=.3, which="both")

    # (c) chipping susceptibility index
    ax = fig.add_subplot(2, 3, 3)
    ci = [cidx[n] / cmax for n in names]
    ax.bar(names, ci, color=[MAT[n]["color"] for n in names], ec="k")
    for i, v in enumerate(ci):
        ax.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("relative chipping index  (H/K_Ic)²·E")
    ax.set_title("(c) Chipping susceptibility (higher = worse)\nSiC hard+tough; GaN brittle")
    ax.grid(alpha=.3, axis="y")

    # (d) capability matrix = the ranking (for dicing)
    ax = fig.add_subplot(2, 1, 2)
    im = ax.imshow(SCORE, cmap="RdYlGn", vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(CAPS))); ax.set_xticklabels(CAPS, fontsize=8)
    ax.set_yticks(range(len(METHODS))); ax.set_yticklabels(METHODS, fontsize=9)
    for i in range(len(METHODS)):
        for j in range(len(CAPS)):
            ax.text(j, i, ["–", "△", "●"][SCORE[i, j]], ha="center", va="center", fontsize=11)
    ax.set_title("(d) Method capability for WAFER DICING (● yes / △ partial / – no). "
                 "Anisotropic phase-field uniquely covers cleavage + branching + laser multiphysics "
                 "+ differentiability — the DISCO-relevant frontier (SiC/GaN).", fontsize=9.5)
    fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01, ticks=[0, 1, 2])

    fig.suptitle("Fracture-mechanics method ranking ported to Si/SiC/GaN wafer dicing "
                 "(DISCO chipping/subsurface-damage control)", fontweight="bold", fontsize=13)
    fig.tight_layout()
    out = "/home/nishioka/git/wafer-proc-sim/semiconductor_fracture_methods.png"
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
