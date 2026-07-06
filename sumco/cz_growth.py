#!/usr/bin/env python3
"""cz_growth.py — Czochralski crystal growth: how fast you may pull, and
the v/G window where "perfect silicon" lives.

Every module in this repo starts from a wafer; this one is about where the
wafer comes from. Two governing results of CZ growth, from the energy and
point-defect balances up:

  PULL-RATE LIMIT   the latent heat released at the melt-crystal interface
               must be conducted up the crystal and radiated away:
                   v_max = k_s (dT/dz)_s / (rho L)
               with the crystal-side gradient set by the radiation-cooled
               fin balance, (dT/dz)_s ~ sqrt(2 eps sigma T_m^5 k_s^-1 R^-1)
               -> the classic v_max ~ 1/sqrt(R) law: BIGGER CRYSTALS MUST
               GROW SLOWER. Verified: the numerically swept exponent of
               v_max(R) equals -0.5 exactly.

  VORONKOV v/G  which intrinsic point defect survives is set not by v or
               G alone but by their RATIO at the interface:
                   v/G > xi_crit  -> vacancy-rich   (COP voids)
                   v/G < xi_crit  -> interstitial-rich (dislocation loops)
               with xi_crit ~ 1.34e-3 cm^2 min^-1 K^-1. G falls from edge
               to center (radiation cools the rim), so ONE pull rate can
               put the center in vacancy mode and the rim in interstitial
               mode — the radial defect bands of real ingots. The
               "perfect-silicon" recipe is the narrow pull-rate window
               where the whole radius stays within the neutral band; the
               window SHRINKS as G flattens less at large R (quantified:
               300 mm window is a fraction of the 150 mm one).

Pure numpy + matplotlib.  Run:  python3 sumco/cz_growth.py
"""
import json
import os

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "cz_growth_results.json")

K_S = 22.0               # solid Si conductivity near melting [W/mK]
RHO = 2330.0             # density [kg/m^3]
LATENT = 1.8e6           # latent heat [J/kg]
T_M = 1685.0             # melting point [K]
EPS = 0.55               # emissivity
SIGMA = 5.67e-8
XI_CRIT = 1.34e-3        # Voronkov criterion [cm^2 / (min K)]
DXI_NEUTRAL = 0.10       # +/-10% band around xi_crit treated as "perfect"
G300_KCM = 40.0          # interface gradient at crystal center, 300 mm
                         # crystal [K/cm] (hot-zone calibrated scale)


def grad_s(r_m):
    """Crystal-side axial gradient from the radiating-fin balance [K/m]."""
    return np.sqrt(2.0 * EPS * SIGMA * T_M ** 5 / (K_S * r_m))


def v_max_mmh(r_m):
    """Heat-balance pull-rate limit [mm/h]."""
    v = K_S * grad_s(r_m) / (RHO * LATENT)      # m/s
    return v * 3.6e6


def g_ratio(r_m):
    """Edge/center gradient ratio: radiation cools the rim harder the
    bigger the crystal (the reason 300 mm perfect-Si is hard)."""
    return 1.0 + 1.2 * r_m


def g_profile(r_m, n=100):
    """Interface axial gradient vs radius [K/cm]: rim cools by radiation,
    so G rises from center to edge (quadratic; center value scales with
    the fin-balance gradient, calibrated to G300_KCM at 300 mm)."""
    rr = np.linspace(0, r_m, n)
    g_c = G300_KCM * grad_s(r_m) / grad_s(0.150)
    return rr, g_c * (1 + (g_ratio(r_m) - 1) * (rr / r_m) ** 2)


def defect_mode(v_mmh, g_kcm):
    """Voronkov: sign of v/G - xi_crit with the neutral band."""
    v_cm_min = v_mmh / 10.0 / 60.0
    xi = v_cm_min / g_kcm
    lo, hi = XI_CRIT * (1 - DXI_NEUTRAL), XI_CRIT * (1 + DXI_NEUTRAL)
    return np.where(xi > hi, 1, np.where(xi < lo, -1, 0))   # v / neutral / i


def perfect_window(r_m):
    """Pull-rate interval [mm/h] where the ENTIRE radius is neutral."""
    rr, g = g_profile(r_m)
    # neutral requires lo*g <= v <= hi*g for all r -> intersect intervals
    lo = XI_CRIT * (1 - DXI_NEUTRAL) * g.max() * 600.0   # cm/min -> mm/h
    hi = XI_CRIT * (1 + DXI_NEUTRAL) * g.min() * 600.0
    return (lo, hi) if hi > lo else (float("nan"), float("nan"))


def vmax_exponent():
    rs = np.array([0.050, 0.075, 0.100, 0.150])
    vs = np.array([v_max_mmh(r) for r in rs])
    return float(np.polyfit(np.log(rs), np.log(vs), 1)[0])


def run():
    radii = {"150mm": 0.075, "200mm": 0.100, "300mm": 0.150}
    vmax = {k: round(float(v_max_mmh(r)), 1) for k, r in radii.items()}
    windows = {}
    for k, r in radii.items():
        lo, hi = perfect_window(r)
        windows[k] = {"lo_mmh": round(lo, 2), "hi_mmh": round(hi, 2),
                      "width_mmh": round(hi - lo, 2)}
    return {
        "heat_balance": {
            "v_max_mmh": vmax,
            "swept_exponent_vs_R": round(vmax_exponent(), 3),
            "theory_exponent": -0.5},
        "voronkov": {
            "xi_crit_cm2_minK": XI_CRIT,
            "neutral_band_pct": 100 * DXI_NEUTRAL,
            "G_edge_over_center_300mm": round(g_ratio(0.150), 2),
            "perfect_window": windows,
            "window_shrink_300_vs_150":
                round(windows["300mm"]["width_mmh"]
                      / windows["150mm"]["width_mmh"], 2)},
    }


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    ax = axes[0]
    rs = np.linspace(0.03, 0.20, 100)
    ax.plot(rs * 2000, [v_max_mmh(r) for r in rs], lw=2, color="#0f766e")
    for k, r in (("150mm", 0.075), ("200mm", 0.100), ("300mm", 0.150)):
        v = v_max_mmh(r)
        ax.plot(r * 2000, v, "o", color="#c0392b", ms=6)
        ax.annotate(f"{k}: {v:.0f} mm/h", (r * 2000, v),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xlabel("crystal diameter [mm]")
    ax.set_ylabel("max pull rate [mm/h]")
    e = res["heat_balance"]["swept_exponent_vs_R"]
    ax.set_title(f"Heat-balance pull limit: v_max ~ R^{e:.2f}\n(theory "
                 "-1/2: bigger crystals must grow slower)", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1]
    r_m = 0.150
    rr, g = g_profile(r_m)
    vv = np.linspace(20.0, 60.0, 400)        # mm/h
    RR, VV = np.meshgrid(rr * 100, vv)
    GG = np.tile(g, (len(vv), 1))
    mode = defect_mode(VV, GG)
    ax.contourf(RR, VV, mode, levels=[-1.5, -0.5, 0.5, 1.5],
                colors=["#d4e6f1", "#d5f5e3", "#fadbd8"], alpha=0.9)
    lo, hi = perfect_window(r_m)
    ax.axhspan(lo, hi, color="#1a7a4a", alpha=0.25)
    ax.annotate("interstitial-rich (loops)", (2, 24), fontsize=9,
                color="#21618c")
    ax.annotate("vacancy-rich (COP voids)", (2, 55), fontsize=9,
                color="#a93226")
    ax.annotate(f"perfect-Si window {lo:.1f}-{hi:.1f} mm/h\n(whole radius "
                "neutral)", (4, hi + 2), fontsize=9, color="#145a32")
    ax.set_xlabel("radius position [cm] (300 mm crystal)")
    ax.set_ylabel("pull rate v [mm/h]")
    w = res["voronkov"]["window_shrink_300_vs_150"]
    ax.set_title("Voronkov v/G map: one pull rate, two defect regimes\n"
                 f"across the radius; 300 mm window = {w:.2f}x the 150 mm "
                 "one", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    res = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "sumco_cz.png"), res)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
