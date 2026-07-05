#!/usr/bin/env python3
"""cmp_pattern.py — pattern-density CMP: why layout, not the pad, sets uniformity.

The C++ suite already covers the CMP unit process (Preston removal,
run-to-run time control, dishing/erosion metrics, friction endpoint —
see vision/cpp/src/cmp.cpp). This module adds the DIE-SCALE model used
for incoming-layout prediction (Stine / Ouma, MIT):

  PATTERN DENSITY — oxide over a patterned metal level polishes at the
  blanket Preston rate R only where the pad touches. Early in the polish
  the pad rides on the raised features, so the LOCAL up-area rate is
      R_up(x) = R / rho_eff(x)
  where rho_eff is the layout density AVERAGED OVER THE PLANARIZATION
  LENGTH L_pl (~1-2 mm): the pad is stiff enough to bridge features but
  bends over millimetres, so it responds to the smoothed density, not
  the drawn one. rho_eff = conv(rho_layout, Gaussian(L_pl)).

  STEP CLOSURE — the local step height falls at R_up until the pad lands
  (step = 0); from then on the surface polishes at the blanket rate R.
  Time-to-clear therefore maps the effective density:
      t_clear(x) = rho_eff(x) * h_step / R  (+ h_clear/R for the rest)

  The consequence: at the moment the SPARSEST region clears, dense
  regions still carry oxide; polishing until the DENSEST region clears
  over-polishes the sparse ones. The post-CMP thickness RANGE across the
  die is set by the density CONTRAST seen through the L_pl filter —
  a layout/dummy-fill decision, not a pad or slurry knob.

Pure numpy + matplotlib.  Run:  python3 ebara/cmp_pattern.py
"""
import json
import os

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "cmp_pattern_results.json")

DX_MM = 0.02             # spatial step [mm]
DIE_MM = 20.0            # die strip length [mm]
L_PL_MM = 1.5            # planarization length [mm]
R_BLANKET = 250.0        # blanket Preston rate [nm/min]
H_STEP = 600.0           # as-deposited step height [nm]
H_OVER = 300.0           # oxide to remove after landing [nm]
DENSITY_BLOCKS = [0.15, 0.90, 0.30, 0.70, 0.50,
                  0.85, 0.20, 0.60, 0.40, 0.75]   # 2 mm layout blocks


def layout_density(blocks=None, die_mm=DIE_MM, dx_mm=DX_MM):
    """Drawn pattern density rho(x) as piecewise-constant blocks."""
    blocks = DENSITY_BLOCKS if blocks is None else blocks
    n = int(die_mm / dx_mm)
    x = np.arange(n) * dx_mm
    rho = np.empty(n)
    w = die_mm / len(blocks)
    for i, b in enumerate(blocks):
        rho[(x >= i * w) & (x < (i + 1) * w)] = b
    return x, rho


def effective_density(rho, l_pl_mm=L_PL_MM, dx_mm=DX_MM):
    """Pad-averaged density: Gaussian convolution over L_pl (reflect pad)."""
    if l_pl_mm <= 0:
        return rho.copy()
    sig = l_pl_mm / dx_mm / 2.0
    half = int(4 * sig)
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sig) ** 2)
    k /= k.sum()
    pad = np.pad(rho, half, mode="reflect")
    return np.convolve(pad, k, "valid")


def polish(rho_eff, t_min, r=R_BLANKET, h_step=H_STEP):
    """Removed up-area oxide [nm] at each x after polishing t_min minutes.

    Up-area rate r/rho_eff while the step is open, blanket rate r after
    the pad lands. Closed form per point (no time stepping needed)."""
    t_land = rho_eff * h_step / r
    early = np.minimum(t_min, t_land) * r / rho_eff
    late = np.maximum(t_min - t_land, 0.0) * r
    return early + late


def clear_times(rho_eff, r=R_BLANKET, h_step=H_STEP, h_over=H_OVER):
    """Time [min] at which each point has removed step + overburden."""
    return rho_eff * h_step / r + h_over / r


def run():
    x, rho = layout_density()
    reff = effective_density(rho)
    tcl = clear_times(reff)
    t_stop = float(tcl.max())            # polish until the last point clears
    removed = polish(reff, t_stop)
    over = removed - (H_STEP + H_OVER)   # extra oxide lost vs target
    rng_drawn = float(np.ptp(polish(rho, float(clear_times(rho).max()))
                             - (H_STEP + H_OVER)))
    sweep = {}
    for lpl in (0.5, 1.5, 3.0):
        re = effective_density(rho, lpl)
        t = float(clear_times(re).max())
        sweep[f"L_pl_{lpl}mm"] = round(float(np.ptp(polish(re, t))), 1)
    return {
        "model": {"R_blanket_nm_min": R_BLANKET, "h_step_nm": H_STEP,
                  "L_pl_mm": L_PL_MM},
        "eff_density": {"min": round(float(reff.min()), 3),
                        "max": round(float(reff.max()), 3),
                        "drawn_min": float(np.min(rho)),
                        "drawn_max": float(np.max(rho))},
        "clear_time_min": {"first": round(float(tcl.min()), 2),
                           "last": round(t_stop, 2)},
        "overpolish_range_nm": round(float(np.ptp(over)), 1),
        "overpolish_range_no_pad_avg_nm": round(rng_drawn, 1),
        "range_vs_Lpl_nm": sweep,
    }


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, rho = layout_density()
    reff = effective_density(rho)
    tcl = clear_times(reff)
    t_stop = float(tcl.max())
    over = polish(reff, t_stop) - (H_STEP + H_OVER)

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.4), sharex=True)
    ax = axes[0]
    ax.step(x, rho, color="0.5", lw=1.2, where="mid",
            label="drawn layout density")
    ax.plot(x, reff, color="#1f618d", lw=2,
            label=f"effective density (pad-averaged, L_pl={L_PL_MM} mm)")
    ax.set_ylabel("density")
    ax.legend(fontsize=9)
    ax.set_title("Pattern-density CMP (Stine/Ouma): the pad polishes the "
                 "SMOOTHED layout", fontsize=10)
    ax = axes[1]
    ax.plot(x, tcl, color="#b4740a", lw=2)
    ax.axhline(t_stop, color="#c44", ls="--", lw=1)
    ax.annotate("polish stops when the densest\nregion finally clears",
                (x[int(np.argmax(tcl))], t_stop),
                textcoords="offset points", xytext=(10, -26), fontsize=9,
                color="#c44")
    ax.set_ylabel("time to clear [min]")
    ax = axes[2]
    ax.plot(x, over, color="#26a", lw=2)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel("die position [mm]")
    ax.set_ylabel("over-polish [nm]")
    ax.set_title(f"Sparse regions pay for dense ones: over-polish range "
                 f"{res['overpolish_range_nm']:.0f} nm "
                 f"(a LAYOUT/dummy-fill problem, not a pad knob)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    res = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "eb_cmp.png"), res)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
