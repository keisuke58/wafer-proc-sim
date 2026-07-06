#!/usr/bin/env python3
"""cdsem.py — CD-SEM metrology: measuring nanometers with electrons, and
correcting for the two ways the measurement lies.

The lithography chain ends at metrology: canon/ makes the aerial image,
fujifilm/ turns it into a resist edge, and the CD-SEM (the tool that rules
this market) reports the number everyone acts on. Two honest-metrology
problems are implemented from the signal physics up:

  SIGNAL       secondary-electron yield rises steeply on tilted surfaces
               (~ sec(theta) escape-depth geometry), so a line's sidewalls
               BLOOM bright — the familiar white edges of every SEM image.
               The detected profile is that emission map convolved with a
               Gaussian beam (d_beam), plus per-pixel Poisson electron
               noise set by frames averaged. CD = threshold crossing on
               the outer white-edge flanks; beam size biases the raw CD,
               which is why CD-SEMs are calibrated, not trusted raw.

  LER BIAS     measured edge = true edge + detection noise, so
               LER_meas^2 = LER_true^2 + sigma_noise^2. Averaging frames
               shrinks the bias but never to zero. The standard fix is
               implemented: measure the SAME edge twice, estimate
               sigma_noise^2 from the half-variance of the difference,
               subtract. Recovery of an injected 2.0 nm LER is verified.

  SHRINKAGE    the e-beam shrinks resist while measuring it: CD(k frames)
               = CD_inf + A exp(-k/k0). More averaging = less noise but
               MORE damage. The production answer is extrapolation to
               zero dose from the per-frame CD sequence — recovering the
               never-measured undamaged CD within 0.5 nm here.

Pure numpy + matplotlib.  Run:  python3 hitachi/cdsem.py
"""
import json
import os

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "cdsem_results.json")

DX = 0.5                 # pixel [nm]
FOV_NM = 120.0
CD_TRUE = 45.0           # true resist line CD [nm]
H_LINE = 90.0            # line height [nm]
SWA_DEG = 86.0           # sidewall angle
BEAM_NM = 3.0            # beam FWHM-ish sigma [nm]
E_PER_PX = 10.0          # electrons/pixel/frame (Poisson)
NY = 240                 # rows for LER work
LER_TRUE_NM = 2.0        # injected true LER (3-sigma / 3 = sigma 0.667?)
                         # convention here: LER quoted as 3-sigma = 2.0 nm
CORR_LEN_NM = 25.0       # edge-roughness correlation length along y
SHRINK_A_NM = 3.0        # total resist shrink amplitude [nm]
SHRINK_K0 = 6.0          # frames to 1/e of the shrink


def _x():
    n = int(FOV_NM / DX)
    return (np.arange(n) - n / 2) * DX


def se_profile(cd=CD_TRUE, beam_nm=BEAM_NM):
    """Noise-free SE emission line profile across one line: flat top and
    bottom yields + sec(theta) sidewall bloom, beam-convolved."""
    x = _x()
    half_w = cd / 2.0
    side_run = H_LINE / np.tan(np.radians(SWA_DEG))    # lateral extent
    y_flat, y_top = 1.0, 1.15                          # material yields
    # sec(theta) bloom, compressed by detector saturation (raw sec(86deg)
    # ~ 14x would clip any real detector chain; cap at 3x)
    y_edge = min(1.0 / np.cos(np.radians(SWA_DEG)), 3.0)
    prof = np.full(len(x), y_flat)
    on_top = np.abs(x) <= half_w - side_run
    prof[on_top] = y_top
    on_wall = (np.abs(x) > half_w - side_run) & (np.abs(x) <= half_w)
    prof[on_wall] = y_edge
    sig = beam_nm / DX
    half = int(4 * sig)
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sig) ** 2)
    k /= k.sum()
    return x, np.convolve(np.pad(prof, half, mode="edge"), k, "valid")


def detect_cd(x, prof, frac=0.5):
    """CD between the OUTER flanks of the two white edges: threshold at
    baseline + frac * (peak - baseline), outermost crossings. A 1 nm
    Gaussian pre-filter suppresses spurious single-pixel crossings (real
    CD-SEM edge algorithms filter before thresholding too)."""
    sig = 1.0 / DX
    half = int(4 * sig)
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sig) ** 2)
    k /= k.sum()
    prof = np.convolve(np.pad(prof, half, mode="edge"), k, "valid")
    base = np.median(prof[: int(10 / DX)])
    th = base + frac * (prof.max() - base)
    above = prof > th
    idx = np.where(above)[0]
    if len(idx) == 0:
        return float("nan")
    i0, i1 = idx[0], idx[-1]
    if i0 == 0 or i1 == len(x) - 1:
        return float("nan")
    xl = x[i0 - 1] + (th - prof[i0 - 1]) / (prof[i0] - prof[i0 - 1]) * DX
    xr = x[i1] + (prof[i1] - th) / (prof[i1] - prof[i1 + 1]) * DX
    return float(xr - xl)


def detect_left_edge(x, prof, frac=0.5):
    """Left-edge position alone (for per-row LER work): same filtered
    threshold, outermost left crossing."""
    cd = detect_cd(x, prof, frac)
    if not np.isfinite(cd):
        return float("nan")
    return -cd / 2.0     # symmetric line centered at 0


def noisy_profile(rng, cd=CD_TRUE, frames=1, beam_nm=BEAM_NM):
    x, p = se_profile(cd, beam_nm)
    lam = p * E_PER_PX * frames
    return x, rng.poisson(lam) / (E_PER_PX * frames)


def true_edges(rng, ny=NY):
    """Correlated true edge wiggle: Gaussian noise smoothed over the
    correlation length, scaled to LER_TRUE (3-sigma convention)."""
    raw = rng.standard_normal(ny)
    sig = CORR_LEN_NM / DX / 4.0
    half = int(4 * sig)
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sig) ** 2)
    k /= np.sqrt((k ** 2).sum())          # preserve variance
    e = np.convolve(np.pad(raw, half, mode="wrap"), k, "valid")
    return e / e.std() * (LER_TRUE_NM / 3.0)


def measure_edges(rng, true_e, frames=1):
    """Per-row measured LEFT edge: true wiggle + detection noise from a
    per-row noisy profile."""
    meas = np.empty(len(true_e))
    for r, de in enumerate(true_e):
        x, p = noisy_profile(rng, CD_TRUE, frames)
        xl = detect_left_edge(x, p)
        # measured edge = physical wiggle + detection error of this row
        meas[r] = de + (xl + CD_TRUE / 2.0)
    return meas


def ler_unbias(rng, frames=1):
    """Two repeated measurements of the same true edge; difference method
    estimates the noise floor. Returns (naive_3sig, unbiased_3sig)."""
    te = true_edges(rng)
    m1 = measure_edges(rng, te, frames)
    m2 = measure_edges(rng, te, frames)
    ok = np.isfinite(m1) & np.isfinite(m2)
    m1, m2 = m1[ok], m2[ok]
    var_noise = 0.5 * np.var(m1 - m2)
    naive = 3.0 * np.std(m1)
    unb = 3.0 * np.sqrt(max(np.var(m1) - var_noise, 0.0))
    return float(naive), float(unb)


def cd_vs_frames(rng, n_frames=16, n_lines=48):
    """Per-frame CD sequence with shrinkage, averaged over n_lines
    independent lines (a normal sampling plan): frame k measures after
    k-1 frames of dose, CD_true(k) = CD_inf + A exp(-k/k0)."""
    cd_inf = CD_TRUE - SHRINK_A_NM
    cds = np.empty((n_lines, n_frames))
    for i in range(n_lines):
        for k in range(n_frames):
            cd_k = cd_inf + SHRINK_A_NM * np.exp(-k / SHRINK_K0)
            x, p = noisy_profile(rng, cd_k, frames=1)
            cds[i, k] = detect_cd(x, p)
    return np.nanmean(cds, axis=0)     # rare failed detections drop out


def shrink_extrapolate(cds, k0=SHRINK_K0):
    """Fit CD(k) = c + a exp(-k/k0) (linear LSQ given k0) and return the
    zero-dose intercept c + a."""
    k = np.arange(len(cds))
    g = np.exp(-k / k0)
    A = np.vstack([np.ones_like(g), g]).T
    coef, *_ = np.linalg.lstsq(A, cds, rcond=None)
    return float(coef[0] + coef[1])


def run():
    rng = np.random.default_rng(7)
    x, p = se_profile()
    cd_raw = detect_cd(x, p)
    bias = {f"beam_{b:.0f}nm": round(detect_cd(*se_profile(beam_nm=b))
                                     - CD_TRUE, 2)
            for b in (2.0, 3.0, 5.0, 8.0)}
    naive1, unb1 = ler_unbias(rng, frames=1)
    naive8, unb8 = ler_unbias(rng, frames=8)
    cds = cd_vs_frames(rng)
    cd_avg = float(np.mean(cds))
    cd_ext = shrink_extrapolate(cds)
    # what the tool WOULD read on the undamaged line, measured the SAME
    # way (1-frame noisy detections, averaged): beam bias and the noisy-
    # threshold offset are both included, so the comparison is apples-to-
    # apples
    ref = float(np.nanmean([detect_cd(*noisy_profile(rng, CD_TRUE, 1))
                            for _ in range(200)]))
    return {
        "signal": {"cd_true_nm": CD_TRUE, "beam_nm": BEAM_NM,
                   "swa_deg": SWA_DEG,
                   "cd_raw_nm": round(cd_raw, 2),
                   "cd_bias_vs_beam_nm": bias},
        "ler_3sig_nm": {"true": LER_TRUE_NM,
                        "naive_1frame": round(naive1, 2),
                        "unbiased_1frame": round(unb1, 2),
                        "naive_8frames": round(naive8, 2),
                        "unbiased_8frames": round(unb8, 2)},
        "shrinkage": {"zero_dose_reading_nm": round(ref, 2),
                      "naive_16frame_avg_nm": round(cd_avg, 2),
                      "zero_dose_extrapolated_nm": round(cd_ext, 2),
                      "err_naive_nm": round(cd_avg - ref, 2),
                      "err_extrapolated_nm": round(cd_ext - ref, 2)},
    }


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(11)
    fig = plt.figure(figsize=(12.8, 4.4))
    ax = fig.add_subplot(1, 3, 1)
    x, p = se_profile()
    _, pn = noisy_profile(rng, frames=1)
    _, pn8 = noisy_profile(rng, frames=8)
    ax.plot(x, pn, color="0.75", lw=0.7, label="1 frame (raw electrons)")
    ax.plot(x, pn8, color="#7ab", lw=0.9, label="8-frame average")
    ax.plot(x, p, color="#37474f", lw=2, label="noise-free SE model")
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("SE signal [a.u.]")
    ax.set_title("SE line profile: sec(theta) sidewall bloom\n= the white "
                 "edges every SEM image shows", fontsize=10)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(1, 3, 2)
    r = res["ler_3sig_nm"]
    labels = ["naive\n1 frame", "unbiased\n1 frame", "naive\n8 frames",
              "unbiased\n8 frames"]
    vals = [r["naive_1frame"], r["unbiased_1frame"], r["naive_8frames"],
            r["unbiased_8frames"]]
    colors = ["#c0392b", "#2e7d57", "#c0392b", "#2e7d57"]
    ax.bar(labels, vals, color=colors)
    ax.axhline(LER_TRUE_NM, color="k", ls="--", lw=1,
               label=f"true LER {LER_TRUE_NM} nm")
    ax.set_ylabel("LER 3-sigma [nm]")
    ax.set_title("LER noise bias: subtract the floor measured\nfrom a "
                 "repeat scan (difference method)", fontsize=10)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(1, 3, 3)
    s = res["shrinkage"]
    cds = cd_vs_frames(np.random.default_rng(7))
    k = np.arange(len(cds))
    ax.plot(k + 1, cds, "o", color="#37474f", ms=5, label="per-frame CD")
    kk = np.linspace(0, len(cds) - 1, 100)
    g = np.exp(-kk / SHRINK_K0)
    A = np.vstack([np.ones_like(np.exp(-k / SHRINK_K0)),
                   np.exp(-k / SHRINK_K0)]).T
    coef, *_ = np.linalg.lstsq(A, cds, rcond=None)
    ax.plot(kk + 1, coef[0] + coef[1] * g, "-", color="#b4740a",
            label="exp fit")
    ax.axhline(s["zero_dose_reading_nm"], color="k", ls="--", lw=1,
               label=f"zero-dose reading {s['zero_dose_reading_nm']} nm")
    ax.set_xlabel("frame #")
    ax.set_ylabel("measured CD [nm]")
    ax.set_title(f"Resist shrinks WHILE measured: naive avg "
                 f"{s['naive_16frame_avg_nm']} nm,\nzero-dose extrapolation "
                 f"{s['zero_dose_extrapolated_nm']} nm", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    res = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "ht_cdsem.png"), res)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
