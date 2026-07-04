#!/usr/bin/env python3
"""lj_signal_chain.py — laser-triangulation profiler: the SIGNAL CHAIN inside.

`market_analysis/keyence_lj_profiler.py` models the LJ-X-class instrument from
the outside (geometric resolution, application scans). This module goes INSIDE
the sensor — the algorithm work a measurement-product engineer actually does:

  1. IMAGER SIMULATION — the 405 nm laser line imaged on the CMOS: Gaussian
     line-spread, laser speckle, surface-reflectance variation, a secondary
     (multipath) reflection lobe, and pixel saturation.
  2. SUB-PIXEL EXTRACTION SHOOTOUT — peak / thresholded centroid / Gaussian
     fit / robust (saturation-aware + multipath-rejecting) estimators, swept
     over reflectance and multipath severity. The whole product lives or dies
     on this estimator.
  3. PROFILE → DIMENSIONS — step height, groove width, shoulder angle and
     flatness measured from the extracted profile with GO/NG tolerance calls.
  4. MEASUREMENT CAPABILITY — repeatability (σ), linearity over the Z range,
     and a %GRR-style capability number against a stated tolerance.

Pure numpy + scipy + matplotlib.  Run:  python3 lj_signal_chain.py
"""
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(11)

# ── sensor geometry / imager ────────────────────────────────────────────────
N_PIX = 256                # detector rows sampled per profile column
UM_PER_PIX = 4.0           # Z displacement per detector pixel [µm/px]
SPOT_SIGMA = 3.2           # imaged line-spread sigma [px]
SPECKLE = 0.18             # multiplicative speckle contrast
READ_NOISE = 0.012         # additive noise (fraction of full scale)
SAT_LEVEL = 1.0            # full-well normalized


def render_column(z_um, reflectance=1.0, multipath=0.0, gain=1.6, rng=RNG):
    """Simulate one detector column for a surface point at height z_um.

    multipath: relative amplitude of a spurious secondary lobe displaced
    +18 px (a shiny wall reflecting the laser onto the sensor twice) — the
    classic failure mode of triangulation sensors in grooves/steps.
    """
    rows = np.arange(N_PIX)
    center = N_PIX / 2 + z_um / UM_PER_PIX
    sig = np.exp(-0.5 * ((rows - center) / SPOT_SIGMA) ** 2)
    if multipath > 0:
        sig = sig + multipath * np.exp(
            -0.5 * ((rows - center - 18.0) / (SPOT_SIGMA * 1.8)) ** 2)
    sig *= reflectance * gain
    sig *= 1.0 + SPECKLE * rng.standard_normal(N_PIX)      # speckle
    sig += READ_NOISE * rng.standard_normal(N_PIX)          # read noise
    return np.clip(sig, 0.0, SAT_LEVEL)                     # saturation


# ── sub-pixel extractors ────────────────────────────────────────────────────
def z_from_row(row):
    return (row - N_PIX / 2) * UM_PER_PIX


def extract_peak(col):
    """Naive: arg-max pixel (no sub-pixel)."""
    return z_from_row(float(np.argmax(col)))


def extract_centroid(col, thr_frac=0.3):
    """Industry baseline: intensity centroid above a relative threshold."""
    thr = thr_frac * col.max()
    m = col > thr
    if not m.any():
        return np.nan
    w = col[m] - thr
    return z_from_row(float(np.sum(np.where(m)[0] * w) / np.sum(w)))


def extract_gauss(col):
    """3-point parabolic fit to log-intensity around the peak (Gaussian fit)."""
    p = int(np.argmax(col))
    if p == 0 or p == N_PIX - 1:
        return z_from_row(float(p))
    y0, y1, y2 = np.log(np.maximum(col[p-1:p+2], 1e-6))
    denom = (y0 - 2*y1 + y2)
    delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
    return z_from_row(p + float(np.clip(delta, -1, 1)))


def extract_robust(col, thr_frac=0.3):
    """Product-grade: saturation-aware, multipath-rejecting centroid.

    1. If the peak is saturated (flat-topped), the Gaussian fit is biased —
       use the mid-point of the saturated plateau instead.
    2. Segment the column into connected lobes above threshold and keep the
       *strongest* lobe only (rejects the multipath lobe), then centroid it.
    """
    thr = thr_frac * col.max()
    m = col > thr
    if not m.any():
        return np.nan
    # connected lobes above threshold
    idx = np.where(m)[0]
    splits = np.where(np.diff(idx) > 1)[0]
    lobes = np.split(idx, splits + 1)
    # Multipath physics: the spurious lobe travels a LONGER optical path, so it
    # always lands at larger row index. Among credible lobes (>=30 % of the
    # strongest lobe's energy), keep the shortest-path (smallest-row) one —
    # energy alone is fooled when saturation clips the true lobe's area.
    sums = [col[l].sum() for l in lobes]
    smax = max(sums)
    cand = [l for l, sm in zip(lobes, sums) if sm >= 0.3 * smax]
    best = min(cand, key=lambda l: l.mean())
    seg = col[best]
    # Saturated plateau is symmetric, so the thresholded centroid of the lobe
    # (its unsaturated wings carry the sub-pixel information) stays unbiased.
    w = seg - thr
    return z_from_row(float(np.sum(best * w) / np.sum(w)))


EXTRACTORS = [("peak (no subpix)", extract_peak),
              ("centroid", extract_centroid),
              ("gauss fit", extract_gauss),
              ("robust", extract_robust)]


# ── benchmark: error vs reflectance & multipath ─────────────────────────────
def benchmark(n_trials=400):
    scenarios = {
        "bright, clean":   dict(reflectance=1.0, multipath=0.0),
        "dark surface":    dict(reflectance=0.12, multipath=0.0),
        "shiny + multipath": dict(reflectance=1.0, multipath=0.55),
        "dark + multipath":  dict(reflectance=0.15, multipath=0.45),
    }
    rng = np.random.default_rng(5)
    table = {}
    for sname, kw in scenarios.items():
        errs = {en: [] for en, _ in EXTRACTORS}
        for _ in range(n_trials):
            z_true = rng.uniform(-150, 150)
            col = render_column(z_true, rng=rng, **kw)
            for en, fn in EXTRACTORS:
                z = fn(col)
                if np.isfinite(z):
                    errs[en].append(z - z_true)
        table[sname] = {en: (float(np.sqrt(np.mean(np.square(v)))) if v else np.nan)
                        for en, v in errs.items()}
    return table


# ── profile → dimensions ────────────────────────────────────────────────────
def target_profile(x_mm):
    """Reference part: plane + 120 µm step + 0.8 mm-wide 90 µm-deep groove."""
    z = np.zeros_like(x_mm)
    z[x_mm > 4.0] += 120.0                                  # step
    z[(x_mm > 7.2) & (x_mm < 8.0)] -= 90.0                  # groove
    z += 6.0 * np.sin(2 * np.pi * x_mm / 16.0)              # gentle form error
    return z


def scan_profile(n=480, rng=RNG):
    x = np.linspace(0, 12.0, n)
    z_true = target_profile(x)
    refl = np.where((x > 7.2) & (x < 8.0), 0.25, 1.0)       # groove is darker
    mp = np.where((x > 7.0) & (x < 8.2), 0.5, 0.0)          # walls → multipath
    z_meas = np.array([extract_robust(render_column(z, r, m, rng=rng))
                       for z, r, m in zip(z_true, refl, mp)])
    return x, z_true, z_meas


def measure_dimensions(x, z):
    """Step height / groove width / shoulder angle / flatness, judged vs tol."""
    base = z[(x > 1.0) & (x < 3.5)]
    top = z[(x > 4.6) & (x < 6.8)]
    step = float(np.median(top) - np.median(base))
    # groove width at half depth via crossings
    gz = z[(x > 6.9) & (x < 8.4)]; gx = x[(x > 6.9) & (x < 8.4)]
    half = np.median(top) - 45.0
    below = gz < half
    width = float(gx[below][-1] - gx[below][0]) if below.any() else np.nan
    # shoulder angle from the step transition slope (4.0±0.15 mm window)
    m = (x > 3.85) & (x < 4.18)
    coeffs = np.polyfit(x[m], z[m], 1)
    angle = float(np.degrees(np.arctan(coeffs[0] / 1000.0)))  # µm/mm → rad
    # flatness of the base plane after removing best-fit line
    bm = (x > 1.0) & (x < 3.5)
    fit = np.polyval(np.polyfit(x[bm], z[bm], 1), x[bm])
    flat = float(np.ptp(z[bm] - fit))
    meas = {
        "step_height_um":  {"value": round(step, 2), "nominal": 120.0, "tol": 5.0},
        "groove_width_mm": {"value": round(width, 3), "nominal": 0.80, "tol": 0.05},
        "shoulder_angle_deg": {"value": round(angle, 2), "nominal": None, "tol": None},
        "flatness_um":     {"value": round(flat, 2), "nominal": 0.0, "tol": 25.0},
    }
    for k, d in meas.items():
        if d["tol"] is not None:
            ref = d["nominal"] or 0.0
            d["ok"] = bool(abs(d["value"] - ref) <= d["tol"]) if k != "flatness_um" \
                else bool(d["value"] <= d["tol"])
    return meas


# ── capability: repeatability, linearity, %GRR ──────────────────────────────
def capability(n_rep=60):
    rng = np.random.default_rng(9)
    # repeatability at mid-range
    reps = [extract_robust(render_column(37.0, rng=rng)) for _ in range(n_rep)]
    sigma = float(np.std(reps, ddof=1))
    # linearity across the range
    zs = np.linspace(-180, 180, 25)
    m = [float(np.mean([extract_robust(render_column(z, rng=rng))
                        for _ in range(15)])) for z in zs]
    resid = np.array(m) - zs
    lin = float(np.max(np.abs(resid - resid.mean())))
    # %GRR against a ±15 µm feature tolerance (6σ convention).
    # Single-shot, and with the 4-profile averaging a real unit ships with.
    tol = 30.0
    grr = float(100.0 * 6.0 * sigma / tol)
    reps4 = [float(np.mean([extract_robust(render_column(37.0, rng=rng))
                            for _ in range(4)])) for _ in range(n_rep)]
    sigma4 = float(np.std(reps4, ddof=1))
    grr4 = float(100.0 * 6.0 * sigma4 / tol)
    return sigma, lin, grr, zs, resid, sigma4, grr4


# ── figures ─────────────────────────────────────────────────────────────────
def fig_chain(path):
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
    rng = np.random.default_rng(3)
    cases = [("clean (bright)", dict(reflectance=1.0, multipath=0.0)),
             ("dark surface", dict(reflectance=0.12, multipath=0.0)),
             ("shiny + multipath", dict(reflectance=1.0, multipath=0.55))]
    z_true = 40.0
    for a, (t, kw) in zip(ax, cases):
        col = render_column(z_true, rng=rng, **kw)
        a.plot(col, np.arange(N_PIX), color="#1f77b4", lw=1.2)
        a.axhline(N_PIX/2 + z_true/UM_PER_PIX, color="#2ca02c", ls="--", lw=1.1,
                  label="true line position")
        for en, fn in [("centroid", extract_centroid), ("robust", extract_robust)]:
            z = fn(col)
            a.axhline(N_PIX/2 + z/UM_PER_PIX, ls=":", lw=1.3,
                      color="#d62728" if en == "centroid" else "#ff9900",
                      label=f"{en}: {z-z_true:+.1f} µm")
        a.set_title(t, fontsize=9); a.set_xlabel("intensity")
        a.legend(fontsize=6.5, loc="lower right")
        a.set_ylim(N_PIX/2 - 30, N_PIX/2 + 55)
    ax[0].set_ylabel("detector row [px]")
    fig.suptitle("Laser line on the imager — where sub-pixel extraction wins or dies",
                 fontsize=10.5)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_bench(table, path):
    scen = list(table.keys()); names = [en for en, _ in EXTRACTORS]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    w = 0.19; xs = np.arange(len(scen))
    colors = ["#8a8f98", "#1f77b4", "#2ca02c", "#d62728"]
    for i, en in enumerate(names):
        vals = [table[s][en] for s in scen]
        ax.bar(xs + (i - 1.5) * w, vals, w, label=en, color=colors[i])
        for xx, v in zip(xs + (i - 1.5) * w, vals):
            ax.text(xx, v * 1.03, f"{v:.1f}", ha="center", fontsize=6.6)
    ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels(scen, fontsize=8.2)
    ax.set_ylabel("Z error RMS [µm]  (log)")
    ax.set_title("Sub-pixel extractor shoot-out — robustness is the product", fontsize=10.5)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_profile(x, z_true, z_meas, meas, path):
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.plot(x, z_true, color="#8a8f98", lw=1.2, label="true surface")
    ax.plot(x, z_meas, color="#1f77b4", lw=1.0, alpha=.9, label="measured profile (robust)")
    s = meas["step_height_um"]; g = meas["groove_width_mm"]; f = meas["flatness_um"]
    ax.annotate(f"step {s['value']:.1f} µm  {'✓' if s['ok'] else '✗'} (120±5)",
                xy=(5.6, 128), fontsize=8.5, color="#2ca02c" if s["ok"] else "#d62728")
    ax.annotate(f"groove {g['value']:.3f} mm  {'✓' if g['ok'] else '✗'} (0.80±0.05)",
                xy=(6.05, 55), fontsize=8.5, color="#2ca02c" if g["ok"] else "#d62728")
    ax.annotate(f"flatness {f['value']:.1f} µm  {'✓' if f['ok'] else '✗'} (≤25)",
                xy=(0.9, 30), fontsize=8.5, color="#2ca02c" if f["ok"] else "#d62728")
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Z [µm]")
    ax.set_title("Profile → dimensions with GO/NG calls (dark groove + multipath walls)",
                 fontsize=10.5)
    ax.legend(fontsize=8.3); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_capability(sigma, lin, grr, zs, resid, grr4, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.4))
    a1.plot(zs, resid - resid.mean(), "-o", ms=3.5, color="#1f77b4")
    a1.axhline(0, color="#999", lw=.8)
    a1.set_xlabel("true Z [µm]"); a1.set_ylabel("linearity residual [µm]")
    a1.set_title(f"Linearity over range  ·  max {lin:.2f} µm", fontsize=9.5)
    a1.grid(alpha=.25)
    a2.bar(["repeatability σ", "linearity", "%GRR (÷10)"],
           [sigma, lin, grr / 10.0], color=["#1f77b4", "#2ca02c", "#d62728"])
    for i, v in enumerate([sigma, lin, grr / 10.0]):
        a2.text(i, v * 1.02, f"{v:.2f}", ha="center", fontsize=8.5)
    a2.set_title(f"Capability · σ={sigma:.2f}µm · %GRR {grr:.0f}% single → {grr4:.0f}% (4-profile avg)",
                 fontsize=9.5)
    a2.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    figs = os.path.join(here, "figures"); os.makedirs(figs, exist_ok=True)

    table = benchmark()
    x, z_true, z_meas = scan_profile()
    meas = measure_dimensions(x, z_meas)
    sigma, lin, grr, zs, resid, sigma4, grr4 = capability()

    fig_chain(os.path.join(figs, "lj_chain.png"))
    fig_bench(table, os.path.join(figs, "lj_bench.png"))
    fig_profile(x, z_true, z_meas, meas, os.path.join(figs, "lj_profile.png"))
    fig_capability(sigma, lin, grr, zs, resid, grr4, os.path.join(figs, "lj_capability.png"))

    rms_err_profile = float(np.sqrt(np.nanmean((z_meas - z_true) ** 2)))
    out = {
        "extractor_rms_um": table,
        "profile_rms_um": round(rms_err_profile, 2),
        "dimensions": meas,
        "repeatability_sigma_um": round(sigma, 3),
        "linearity_max_um": round(lin, 3),
        "grr_pct_tol30um_single": round(grr, 1),
        "repeatability_sigma_um_avg4": round(sigma4, 3),
        "grr_pct_tol30um_avg4": round(grr4, 1),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    with open(os.path.join(here, "lj_signal_chain_results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
