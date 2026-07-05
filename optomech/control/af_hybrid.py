#!/usr/bin/env python3
"""af_hybrid.py — hybrid autofocus: PDAF coarse jump + contrast fine search.

servo_analysis.py designs the VCM loop itself; this module sits one level up
and simulates the FOCUS STRATEGY a camera actually runs:

  CONTRAST-ONLY  hill-climb on a sharpness metric. Robust, needs no special
                 pixels — but every probe costs a frame (33 ms) plus a lens
                 move, and the search direction is unknown at the start.
  PDAF           masked half-pupil pixels see the scene through the left /
                 right pupil halves; defocus shifts the two images apart by
                 a disparity ∝ defocus. ONE frame gives magnitude AND sign
                 of the error — but the estimate degrades in low light
                 (photon noise) and with heavy blur (correlation washout).
  HYBRID         PDAF jump → contrast verify/refine near focus. The point of
                 the simulation: quantify when the hybrid wins and how the
                 crossover moves with light level.

Both image formation paths are simulated for real (texture scene, defocus
blur, fractional-pixel disparity, Poisson photon noise; NCC + parabolic
sub-pixel disparity estimate; gradient-energy sharpness metric) — the AF
loops run on the measured signals, not on ground truth.

Pure numpy + matplotlib.  Run:  python3 optomech/control/af_hybrid.py
"""
import json
import os

import numpy as np

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "af_hybrid_results.json")

# ── optics / sensor / actuator constants ────────────────────────────────────
F_NUM = 2.8               # working f/# (matches mtf_wavefront)
WAVELEN_UM = 0.55
PITCH_UM = 1.0            # pixel pitch
N_PIX = 384               # 1-D AF window length
MARGIN = 260              # world extends past the window (shift + blur reach)
ROWS = 32                 # AF window height: metric/PDAF average over rows
                          # (a real AF window is 2-D; SNR scales with sqrt N)
DOF_UM = 2.0 * WAVELEN_UM * F_NUM ** 2       # ± depth of focus ≈ 8.6 µm
K_DISP = 0.85 / (2.0 * F_NUM)  # PDAF disparity gain [px shift per µm defocus
                               # ... in µm at the sensor] (half-pupil centroid)
SIGMA0_UM = 0.7           # in-focus PSF sigma (diffraction + pixel aperture)

T_FRAME_MS = 33.0         # sensor frame time
T_MOVE_BASE_MS = 6.0      # VCM settle for a small move (2nd-order, z=0.9)
T_MOVE_PER_UM = 0.02      # slew contribution [ms/µm]

FULL_SCALE_PH = {1000.0: 2000.0, 10.0: 20.0}   # photons/px at each lux level


def _scene(rng, n=N_PIX + 2 * MARGIN):
    """Natural-statistics WORLD strip: step edges (1/f spectrum) + texture.

    The 1/f low-frequency content matters: it is what survives heavy defocus
    blur and lets PDAF correlate at large defocus. A purely band-limited
    texture would (unrealistically) go featureless under blur. The strip is
    wider than the sensor window — the window sees a CROP of the world, so
    image shift and blur never manufacture edge artifacts."""
    s = np.zeros(n)
    for _ in range(10):
        s[int(rng.integers(10, n - 10)):] += rng.uniform(-1.0, 1.0)
    k = np.exp(-0.5 * (np.arange(-12, 13) / 4.0) ** 2)
    s += 0.35 * np.convolve(rng.standard_normal(n), k / k.sum(), "same")
    s = (s - s.min()) / max(np.ptp(s), 1e-9) + 0.15
    return s


def _blur_sigma_um(dz_um):
    """Geometric defocus blur (radius dz/2N) combined with in-focus PSF."""
    return np.hypot(SIGMA0_UM, abs(dz_um) / (2.0 * F_NUM) / 1.5)


def _image(scene, dz_um, shift_um, photons, rng):
    """Form the 1-D sensor-window image: fractional shift and defocus blur
    happen in the wide world strip, then the window crops the center —
    photon noise applies per pixel of the crop."""
    n = len(scene)
    xs = np.arange(n) - shift_um / PITCH_UM
    sh = np.interp(xs, np.arange(n), scene, left=scene[0], right=scene[-1])
    s_px = _blur_sigma_um(dz_um) / PITCH_UM
    half = max(int(np.ceil(3.0 * s_px)), 3)      # don't truncate wide PSFs
    kx = np.arange(-half, half + 1)
    k = np.exp(-0.5 * (kx / max(s_px, 1e-3)) ** 2)
    img = np.convolve(sh, k / k.sum(), "same")
    img = np.clip(img[MARGIN:MARGIN + N_PIX], 0, None)
    return rng.poisson(img * photons, size=(ROWS, N_PIX)) / photons


def pdaf_measure(scene, dz_um, photons, rng):
    """One PDAF read: left/right half-pupil images, NCC disparity with
    parabolic sub-pixel refinement → defocus estimate [µm]."""
    disp_um = K_DISP * dz_um * 2.0        # total L-R separation
    # each masked pixel sees HALF the pupil, so its blur extent is half the
    # full-aperture blur (pass dz/2 through the same geometric blur model)
    li = _image(scene, dz_um / 2.0, -disp_um / 2.0, photons, rng).mean(axis=0)
    ri = _image(scene, dz_um / 2.0, +disp_um / 2.0, photons, rng).mean(axis=0)
    li = li - li.mean()
    ri = ri - ri.mean()
    n = len(li)
    lags = np.arange(-180, 181)
    cc = np.empty(len(lags))
    for j, l in enumerate(lags):          # windowed, non-wrapping true NCC
        if l >= 0:
            a, b = li[l:], ri[:n - l]
        else:
            a, b = li[:n + l], ri[-l:]
        a = a - a.mean()
        b = b - b.mean()
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        cc[j] = np.dot(a, b) / norm if norm > 1e-12 else 0.0
    i = int(np.argmax(cc))
    if 0 < i < len(cc) - 1:
        a, b, c = cc[i - 1], cc[i], cc[i + 1]
        denom = a - 2 * b + c
        di = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
    else:
        di = 0.0
    # ri leads li by the total separation, so the matching lag is -disp
    disp_hat_px = -(lags[i] + di)
    dz_hat = disp_hat_px * PITCH_UM / (2.0 * K_DISP)
    return dz_hat, float(cc[i])           # (estimate, NCC confidence)


def contrast_metric(scene, dz_um, photons, rng):
    """Gradient energy with the Poisson noise floor subtracted.

    E[Σ diff²] = (true gradient energy) + 2·Σ var(pixel); for Poisson pixels
    var = img/photons, so subtracting 2·Σ img/photons makes the metric an
    unbiased estimate of scene sharpness — without this the noise floor
    (constant in dz) buries the focus peak at low light."""
    img = _image(scene, dz_um, 0.0, photons, rng)
    noise_floor = 2.0 * float(np.mean(np.sum(img, axis=1))) / photons
    return float(np.mean(np.sum(np.diff(img, axis=1) ** 2, axis=1))) \
        - noise_floor


def _move_time_ms(step_um):
    return T_MOVE_BASE_MS + T_MOVE_PER_UM * abs(step_um)


def contrast_af(scene, dz0_um, photons, rng, span_um=560.0, coarse_um=40.0,
                fine_um=6.0):
    """Contrast-only AF the way production firmware does it: a swept coarse
    scan across the stroke, then a fine scan around the peak.

    A local hill-climb is NOT viable from large defocus — the gradient-energy
    metric is nearly flat there and photon noise swamps its slope (this
    simulation demonstrates exactly that if you try) — which is precisely why
    contrast-only cameras visibly "sweep" before locking.
    Returns (t_ms, final_dz_um, trace[(t, dz)])."""
    t = 0.0
    trace = [(0.0, dz0_um)]
    # move to the near end of the stroke, sweep across
    positions = np.arange(-span_um, span_um + 1e-9, coarse_um)
    if dz0_um > 0:
        positions = positions[::-1]
    t += _move_time_ms(abs(positions[0] - dz0_um))
    best_dz, best_m = positions[0], -np.inf
    for p in positions:
        m = contrast_metric(scene, p, photons, rng)
        t += _move_time_ms(coarse_um) + T_FRAME_MS
        trace.append((t, p))
        if m > best_m:
            best_dz, best_m = p, m
    # fine scan around the coarse peak
    dz = best_dz
    t += _move_time_ms(abs(positions[-1] - dz))
    for fdz in (dz - 2 * fine_um, dz - fine_um, dz + fine_um,
                dz + 2 * fine_um):
        m = contrast_metric(scene, fdz, photons, rng)
        t += _move_time_ms(fine_um) + T_FRAME_MS
        trace.append((t, fdz))
        if m > best_m:
            best_dz, best_m = fdz, m
    t += _move_time_ms(abs(best_dz - dz))
    trace.append((t, best_dz))
    return t, best_dz, trace


def hybrid_af(scene, dz0_um, photons, rng, max_pdaf=3, max_refine=4,
              conf_min=0.6):
    """Iterated PDAF jumps (each one frame) until the estimate is small,
    then a short contrast verify/refine around the landing point. If the
    PDAF correlation confidence is too low (dark / featureless), fall back
    to the contrast sweep — exactly what production hybrid AF does."""
    dz, t = dz0_um, 0.0
    trace = [(0.0, dz)]
    for it in range(max_pdaf):
        dz_hat, conf = pdaf_measure(scene, dz, photons, rng)
        t += T_FRAME_MS
        if conf < conf_min:
            if it == 0:                   # no usable PDAF at all → sweep
                t_c, dz_c, tr_c = contrast_af(scene, dz, photons, rng)
                return t + t_c, dz_c, trace + [(t + tt, zz)
                                               for tt, zz in tr_c]
            break                         # keep the coarse landing we have
        dz = dz - dz_hat
        t += _move_time_ms(dz_hat)
        trace.append((t, dz))
        if abs(dz_hat) < 25.0:
            break
    # contrast verify / refine with small steps
    fine = 6.0
    m_here = contrast_metric(scene, dz, photons, rng)
    t += T_FRAME_MS
    for _ in range(max_refine):
        m_l = contrast_metric(scene, dz - fine, photons, rng)
        t += _move_time_ms(fine) + T_FRAME_MS
        m_r = contrast_metric(scene, dz + fine, photons, rng)
        t += _move_time_ms(2 * fine) + T_FRAME_MS
        if m_here >= m_l and m_here >= m_r:
            t += _move_time_ms(fine)          # return to center
            break
        if m_l > m_r:
            dz, m_here = dz - fine, m_l
        else:
            dz, m_here = dz + fine, m_r
        trace.append((t, dz))
    trace.append((t, dz))
    return t, dz, trace


def benchmark(n_trials=60, seed=5):
    rng = np.random.default_rng(seed)
    out = {}
    for lux, ph in FULL_SCALE_PH.items():
        rows = {"contrast": [], "hybrid": []}
        for _ in range(n_trials):
            scene = _scene(rng)
            dz0 = float(rng.uniform(60, 500)) * float(rng.choice([-1, 1]))
            t_c, e_c, _ = contrast_af(scene, dz0, ph, rng)
            t_h, e_h, _ = hybrid_af(scene, dz0, ph, rng)
            rows["contrast"].append((t_c, abs(e_c)))
            rows["hybrid"].append((t_h, abs(e_h)))
        stats = {}
        for k, v in rows.items():
            ts = np.array([r[0] for r in v])
            es = np.array([r[1] for r in v])
            stats[k] = {
                "mean_ms": round(float(ts.mean()), 1),
                "p95_ms": round(float(np.percentile(ts, 95)), 1),
                "in_dof_rate": round(float(np.mean(es <= DOF_UM)), 3),
            }
        out[f"{lux:.0f}lux"] = stats
    return out


# ── figure ──────────────────────────────────────────────────────────────────
def _plot(path, seed=11):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))

    # (a) example traces
    ax = axes[0]
    scene = _scene(rng)
    dz0 = 400.0
    ph = FULL_SCALE_PH[1000.0]
    _, _, tr_c = contrast_af(scene, dz0, ph, rng)
    _, _, tr_h = hybrid_af(scene, dz0, ph, rng)
    ax.plot(*zip(*tr_c), "o-", ms=3, color="#c86", label="contrast-only")
    ax.plot(*zip(*tr_h), "o-", ms=3, color="#26a", label="hybrid PDAF")
    ax.axhspan(-DOF_UM, DOF_UM, color="0.88", zorder=0)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("defocus [um]")
    ax.set_title("Focus sequence from 400 um defocus\n(gray band = DOF)")
    ax.legend(fontsize=9)

    # (b) PDAF calibration / noise
    ax = axes[1]
    for lux, color in ((1000.0, "#26a"), (10.0, "#c86")):
        ph = FULL_SCALE_PH[lux]
        dzs = np.linspace(-350, 350, 40)
        est = [pdaf_measure(_scene(rng), d, ph, rng)[0] for d in dzs]
        ax.plot(dzs, est, "o", ms=3.5, color=color, label=f"{lux:.0f} lux")
    lim = np.array([-350, 350])
    ax.plot(lim, lim, "k--", lw=1, label="ideal")
    ax.set_xlabel("true defocus [um]")
    ax.set_ylabel("PDAF estimate [um]")
    ax.set_title("PDAF: one-frame defocus estimate")
    ax.legend(fontsize=9)

    # (c) benchmark bars
    ax = axes[2]
    bench = benchmark()
    labels, means, p95s, colors = [], [], [], []
    for lux in ("1000lux", "10lux"):
        for m, c in (("contrast", "#c86"), ("hybrid", "#26a")):
            labels.append(f"{m}\n{lux}")
            means.append(bench[lux][m]["mean_ms"])
            p95s.append(bench[lux][m]["p95_ms"])
            colors.append(c)
    x = np.arange(len(labels))
    ax.bar(x, means, color=colors)
    ax.errorbar(x, means, yerr=[np.zeros(len(x)),
                                np.array(p95s) - np.array(means)],
                fmt="none", ecolor="k", capsize=4, label="p95")
    for i, (m, r) in enumerate(zip(
            means, [bench["1000lux"]["contrast"], bench["1000lux"]["hybrid"],
                    bench["10lux"]["contrast"], bench["10lux"]["hybrid"]])):
        ax.text(i, m + 8, f"{r['in_dof_rate']*100:.0f}%", ha="center",
                fontsize=8)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("time to focus [ms]")
    ax.set_title("Time to focus, mean (bar) / p95 (whisker)\n"
                 "labels = in-DOF success rate")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return bench


def main():
    bench = benchmark()
    summary = {"dof_um": round(DOF_UM, 1), **bench}
    b, h = bench["1000lux"]["contrast"], bench["1000lux"]["hybrid"]
    summary["speedup_1000lux"] = round(b["mean_ms"] / h["mean_ms"], 2)
    with open(RESULTS_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(ASSETS, exist_ok=True)
    for d in (FIGDIR, ASSETS):
        _plot(os.path.join(d, "af_hybrid.png"))
    print("figure ->", FIGDIR, "and", ASSETS)


if __name__ == "__main__":
    main()
