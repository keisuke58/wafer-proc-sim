#!/usr/bin/env python3
"""photon_budget.py — sensitivity vs throughput: an inspector is a photon
counter.

Every mask-inspection spec sheet is secretly one equation. A pixel that
dwells t seconds under photon flux Φ collects N = Φ·t photons; a defect that
changes the local reflectivity by contrast c produces a signal c·N over shot
noise √N, so the single-pixel SNR is

        SNR = c · √(Φ·t)

Everything the customer sees follows from it: minimum detectable defect size
(c grows ~ with defect area), scan time per mask (t per pixel × pixels /
TDI parallelism), and the false-alarm budget (the detection threshold k is
set so that ~10 false counts survive PER MASK over ~10^13 pixels → k ≈ 7σ,
which is why inspection thresholds look absurdly high to ML people).

This module computes the honest trade-off curves:
  - capture probability vs defect size at several masks/day operating points
  - the sensitivity-throughput frontier (min size at 90 % capture vs
    masks/day), for source power ×1 and ×2 — why EUV source power is the
    industry bottleneck for actinic inspection.

Contrast-vs-size is anchored to the aerial-image detection scores computed
by inspection/euv_printability.py (c ∝ size² small-defect scaling, matched
at 20 nm).

Pure numpy + matplotlib.  Run:  python3 inspection/photon_budget.py
"""
import json
import os

import numpy as np
from scipy.stats import norm

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "photon_budget_results.json")

# ── instrument constants ────────────────────────────────────────────────────
PIX_NM = 20.0               # inspection pixel [nm]
MASK_MM = 104.0             # inspected area side [mm]
N_TDI = 4096                # parallel TDI pixels
PHI_1X = 5.0e8              # photons/s/pixel at source x1 (at the sensor)
FALSE_PER_MASK = 10.0       # false-count budget per mask scan
C20 = 0.25                  # defect contrast at 20 nm (from euv_printability
                            # detection scores, amplitude defects)
OVERHEAD_H = 0.5            # load/align/unload per mask [h]

N_PIX = (MASK_MM * 1e6 / PIX_NM) ** 2          # ~2.7e13 pixels


def contrast(size_nm):
    """Small-defect contrast ~ area scaling, anchored at 20 nm."""
    return C20 * (np.asarray(size_nm, dtype=float) / 20.0) ** 2


def threshold_sigma():
    """k such that N_PIX * Q(k) = FALSE_PER_MASK."""
    return float(norm.isf(FALSE_PER_MASK / N_PIX))


def scan_hours(dwell_s):
    return N_PIX * dwell_s / N_TDI / 3600.0


def dwell_for_masks_per_day(mpd):
    """Invert masks/day -> pixel dwell (scan time = 24/mpd - overhead)."""
    scan_h = 24.0 / mpd - OVERHEAD_H
    if scan_h <= 0:
        return None
    return scan_h * 3600.0 * N_TDI / N_PIX


def capture_prob(size_nm, dwell_s, phi=PHI_1X):
    """P(detect) for a defect of given size at the deployed threshold."""
    k = threshold_sigma()
    snr = contrast(size_nm) * np.sqrt(phi * dwell_s)
    return norm.sf(k - snr)


def min_detectable_size(dwell_s, phi=PHI_1X, p_target=0.90):
    """Smallest size with capture >= p_target (solve c·√N = k + z_p)."""
    k = threshold_sigma()
    need = k + norm.isf(1.0 - p_target)        # = k - z_{1-p}
    c_need = need / np.sqrt(phi * dwell_s)
    return float(20.0 * np.sqrt(c_need / C20))


def run():
    k = threshold_sigma()
    ops = {}
    for mpd in (1.0, 2.0, 4.0):
        dw = dwell_for_masks_per_day(mpd)
        ops[f"{mpd:g}_masks_per_day"] = {
            "dwell_us": round(dw * 1e6, 3),
            "photons_per_pixel": round(PHI_1X * dw, 1),
            "capture_16nm": round(float(capture_prob(16.0, dw)), 4),
            "capture_20nm": round(float(capture_prob(20.0, dw)), 4),
            "min_size_90pct_nm": round(min_detectable_size(dw), 2),
        }
    return {"threshold_sigma": round(k, 2),
            "n_pixels": f"{N_PIX:.2e}",
            "false_per_mask": FALSE_PER_MASK,
            "operating_points": ops}


def _plot(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    ax = axes[0]
    sizes = np.linspace(8, 30, 120)
    for mpd, c in ((1.0, "#26a"), (2.0, "#b4740a"), (4.0, "#c44")):
        dw = dwell_for_masks_per_day(mpd)
        ax.plot(sizes, capture_prob(sizes, dw) * 100.0, color=c, lw=1.8,
                label=f"{mpd:g} masks/day (dwell {dw*1e6:.1f} us)")
    ax.axhline(90, color="k", ls=":", lw=1)
    ax.set_xlabel("defect size [nm]")
    ax.set_ylabel("capture probability [%]")
    ax.set_title(f"Capture vs size at fixed {FALSE_PER_MASK:.0f} false/mask "
                 f"(k = {threshold_sigma():.1f} sigma)")
    ax.legend(fontsize=8)

    ax = axes[1]
    mpds = np.linspace(0.75, 8.0, 100)
    for phi, c, lab in ((PHI_1X, "#26a", "source x1"),
                        (2 * PHI_1X, "#2a7", "source x2")):
        ms = []
        for m in mpds:
            dw = dwell_for_masks_per_day(m)
            ms.append(min_detectable_size(dw, phi) if dw else np.nan)
        ax.plot(mpds, ms, color=c, lw=1.8, label=lab)
    ax.set_xlabel("throughput [masks/day]")
    ax.set_ylabel("min size at 90% capture [nm]")
    ax.set_title("The sensitivity-throughput frontier:\n"
                 "source power buys speed at equal sensitivity")
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
    _plot(os.path.join(ASSETS, "photon_budget.png"))
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
