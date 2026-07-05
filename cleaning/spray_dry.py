#!/usr/bin/env python3
"""spray_dry.py — two-fluid spray cleaning and Marangoni drying.

Companion to spin_clean.py, covering the two remaining acts of a modern
single-wafer cleaner:

  SPRAY (two-fluid jet) — droplets hitting the wafer generate a water-hammer
  impulse (P ≈ 0.2 ρ c v) and a lateral micro-jet (~3 v). The dynamic
  pressure removes particles megasonic can't reach — but the same impulse
  bends fine pattern lines. Removal needs velocity; damage forbids it. The
  OPERATING WINDOW [v_removal, v_damage] is the product spec, and it CLOSES
  as patterns shrink — the quantified reason "damage-free spray" is the
  industry's arms race.

  MARANGONI / IPA DRY — a drying front with an IPA concentration gradient
  has a surface-tension gradient that pulls the water film toward the bulk
  (Marangoni flow). If the film ruptures by evaporation before the front
  passes, dissolved silica precipitates: a WATERMARK. The model compares the
  front transit time against the evaporative rupture time per site and
  yields watermark density vs IPA supply — the knob a process engineer
  actually turns.

Pure numpy + matplotlib.  Run:  python3 cleaning/spray_dry.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cleaning.spin_clean import (adhesion_force, A_CONTACT,      # noqa: E402
                                 SIGMA_LN, E_SI)

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "spray_dry_results.json")

RHO = 1000.0
C_WATER = 1480.0          # speed of sound [m/s]
K_WH = 0.2                # water-hammer compliance factor
JET_GAIN = 3.0            # lateral micro-jet velocity / impact velocity
SI_STRENGTH = 2.0e9       # fracture strength of nano-Si lines [Pa]
LINE_H = 120e-9           # pattern line height [m] (roughly node-invariant,
                          # which is exactly why AR and damage risk GROW as
                          # the half-pitch shrinks)
BL_JET = 1.0e-6           # lateral-jet viscous sublayer thickness [m]: a
                          # particle smaller than this sees only the fraction
                          # (d/δ) of the jet velocity → dynamic pressure
                          # × (d/δ)² — why sprays ALSO struggle below ~50 nm


def impact_pressures(v):
    """(water-hammer pressure, lateral-jet dynamic pressure) [Pa]."""
    p_wh = K_WH * RHO * C_WATER * np.asarray(v, dtype=float)
    p_dyn = 0.5 * RHO * (JET_GAIN * np.asarray(v, dtype=float)) ** 2
    return p_wh, p_dyn


def spray_removal_prob(d_nm, v, n_mc=20000, seed=0):
    """P(remove) for particle diameter d under droplet impact at velocity v:
    lateral-jet dynamic pressure on the particle cross-section, rolling
    moment vs lognormal-scattered adhesion (same criterion as spin_clean)."""
    rng = np.random.default_rng(seed)
    d = float(d_nm) * 1e-9
    _, p_dyn = impact_pressures(v)
    shield = min(1.0, d / BL_JET) ** 2        # viscous-sublayer shielding
    f_rem = shield * p_dyn * np.pi * (d / 2.0) ** 2
    fa = adhesion_force(d) * rng.lognormal(0.0, SIGMA_LN, n_mc)
    return float(np.mean(f_rem * 1.4 * d / 2.0 > fa * A_CONTACT))


def v_removal(d_nm, target=0.95):
    """Velocity needed for P(remove) >= target (bisection)."""
    lo, hi = 0.5, 300.0
    if spray_removal_prob(d_nm, hi) < target:
        return np.inf
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if spray_removal_prob(d_nm, mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def v_damage(hp_nm):
    """Velocity at which the lateral jet snaps a pattern line.

    Line width w = half-pitch, height H = LINE_H (node-invariant). Jet
    dynamic pressure loads one side; root bending stress of a cantilever
    wall sigma = 3 p_dyn (H/w)^2. Damage when sigma > SI_STRENGTH →
    p_dyn* = S w² / (3 H²), so the ceiling drops LINEARLY with half-pitch."""
    ar = LINE_H / (hp_nm * 1e-9)
    p_star = SI_STRENGTH / (3.0 * ar ** 2)
    v_jet = np.sqrt(2.0 * p_star / RHO)
    return float(v_jet / JET_GAIN)


def spray_window(hp_list=(64.0, 32.0, 16.0), d_list=(100.0, 50.0, 30.0)):
    """Operating windows: for each pattern node, [v_removal(d), v_damage]."""
    out = {}
    for hp in hp_list:
        vd = v_damage(hp)
        out[f"hp{int(hp)}nm"] = {
            "v_damage_mps": round(vd, 1),
            "windows": {f"{int(d)}nm": {
                "v_removal_mps": round(v_removal(d), 1),
                "open": bool(v_removal(d) < vd)} for d in d_list},
        }
    return out


# ── Marangoni / IPA drying ──────────────────────────────────────────────────
MU = 0.9e-3
DSIGMA_IPA = 0.049        # sigma(water) - sigma(IPA-rich) [N/m]
FILM_H = 0.8e-6           # film thickness at the drying front [m]
FRONT_LEN = 2.0e-3        # Marangoni gradient length at the front [m]
E_EVAP0 = 0.25e-6         # evaporation thinning rate [m/s] at 25 C
H_RUPTURE = 0.15e-6       # film ruptures (watermark seed) below this
SITE_L = 5.0e-3           # characteristic site spacing the front must cross


def marangoni_velocity(ipa_frac):
    """Front retreat velocity from the surface-tension gradient:
    u ~ dSigma * h / (2 mu L), scaled by delivered IPA fraction."""
    return DSIGMA_IPA * ipa_frac * FILM_H / (2.0 * MU * FRONT_LEN)


def watermark_density(ipa_frac, temp_C=25.0, n_sites=1.0e4, seed=1):
    """Sites/wafer where evaporation ruptures the film before the front
    passes. Evaporation doubles every ~12 C; site-to-site lognormal
    variation in local film thickness."""
    rng = np.random.default_rng(seed)
    u = marangoni_velocity(ipa_frac) + 1e-9
    t_pass = SITE_L / u
    e_rate = E_EVAP0 * 2.0 ** ((temp_C - 25.0) / 12.0)
    h_local = FILM_H * rng.lognormal(0.0, 0.35, int(n_sites))
    ruptured = (h_local - e_rate * t_pass) < H_RUPTURE
    return float(ruptured.mean() * n_sites)


def run():
    res = {"spray": spray_window(),
           "marangoni": {
               "u_front_mm_s_at_full_ipa": round(
                   marangoni_velocity(1.0) * 1000, 2),
               "watermarks_per_wafer": {
                   "spin_only(ipa=0)": round(watermark_density(0.0), 0),
                   "ipa_25pct": round(watermark_density(0.25), 0),
                   "ipa_100pct": round(watermark_density(1.0), 0)}}}
    return res


def _plot(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    ax = axes[0]
    hps = np.linspace(10, 90, 60)
    vdam = np.array([v_damage(h) for h in hps])
    ax.plot(hps, vdam, color="#c44", lw=2,
            label="damage limit (120 nm tall lines)")
    for d, c in ((100.0, "#26a"), (50.0, "#b4740a"), (30.0, "#7b3fb8")):
        vr = v_removal(d)
        ax.axhline(vr, color=c, lw=1.6, ls="--",
                   label=f"removal of {int(d)} nm needs {vr:.0f} m/s")
    ax.fill_between(hps, 0, vdam, color="#2a7", alpha=0.08)
    ax.set_xlabel("pattern half-pitch [nm]")
    ax.set_ylabel("droplet velocity [m/s]")
    ax.set_ylim(0, 130)
    ax.set_title("Spray operating window: removal floor vs damage ceiling\n"
                 "(the window CLOSES as patterns shrink)")
    ax.legend(fontsize=8)

    ax = axes[1]
    q = np.linspace(0.0, 1.0, 40)
    for T, c in ((25.0, "#26a"), (40.0, "#c44")):
        wm = [watermark_density(x, T) + 0.5 for x in q]   # +0.5 for log axis
        ax.semilogy(q * 100, wm, color=c, lw=1.8,
                    label=f"wafer at {T:.0f} C")
    ax.set_xlabel("IPA delivery [% of full]")
    ax.set_ylabel("watermarks per wafer (log)")
    ax.set_title("Marangoni drying: watermarks vs IPA supply")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    res = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "spray_dry.png"))
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
