#!/usr/bin/env python3
"""product_suite.py — three more Keyence product lines, one core algorithm each.

The LJ profiler module covers displacement sensing. This module adds the
algorithmic heart of three other flagship product families:

  1. FIBER / PHOTOELECTRIC SENSOR (FS/PS class)
     Modulated-light detection with SYNCHRONOUS (lock-in) DEMODULATION: the
     receiver sees ambient sunlight + 100 Hz flicker + impulse noise at 100x
     the signal amplitude; carrier modulation + phase-locked demodulation +
     low-pass recovers a clean detection margin. Plus hysteresis design —
     the anti-chatter detail every real sensor ships with.

  2. LASER MARKER (MD class)
     2-axis GALVO TRAJECTORY GENERATION: a marking path executed by a galvo
     with finite bandwidth. Naive constant-speed corners overshoot and round
     the mark; corner-lookahead speed planning + tracking-delay compensation
     keeps the beam on the vector at production speed.

  3. DIGITAL MICROSCOPE (VHX class)
     DEPTH-FROM-FOCUS COMPOSITION (focus stacking): sweep focus through a
     3-D scene, pick the sharpest frame per pixel (local-Laplacian focus
     measure), fuse an all-in-focus image AND a calibrated height map — the
     signature VHX capability.

Pure numpy + scipy + matplotlib.  Run:  python3 product_suite.py
"""
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from scipy import signal as sps

RNG = np.random.default_rng(21)
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")


# ═════════════════════ 1. fiber sensor: lock-in detection ═══════════════════
def fiber_sensor():
    fs = 2_000_000            # ADC rate [Hz]
    f_c = 100_000             # carrier (LED modulation) [Hz]
    dur = 0.06
    t = np.arange(int(fs * dur)) / fs

    # target passes through the beam: transmittance 1 -> 0.35 -> 1
    trans = np.ones_like(t)
    trans[(t > 0.025) & (t < 0.040)] = 0.35

    sig_amp = 1.0
    carrier = 0.5 * (1 + sps.square(2 * np.pi * f_c * t))      # 0/1 LED
    received = sig_amp * trans * carrier

    # ambient: DC sunlight 100x + mains flicker + impulses
    ambient = 100.0 + 20.0 * np.sin(2 * np.pi * 100 * t)
    impulses = np.zeros_like(t)
    idx = RNG.choice(t.size, 120, replace=False)
    impulses[idx] = RNG.uniform(20, 60, idx.size)
    noise = 0.8 * RNG.standard_normal(t.size)
    raw = received + ambient + ndi.gaussian_filter1d(impulses, 3) + noise

    # naive: low-pass the raw photocurrent (what a DC sensor would do)
    b, a = sps.butter(2, 500 / (fs / 2))
    naive = sps.filtfilt(b, a, raw)

    # lock-in: mix with the carrier reference, then low-pass
    ref = carrier - carrier.mean()
    mixed = (raw - raw.mean()) * ref
    lockin = sps.filtfilt(b, a, mixed)
    lockin /= np.median(lockin[t < 0.005])                     # normalize open-beam

    # detection margins (blocked vs open separation / noise)
    # windows span >=1 full mains-flicker cycle so in-band flicker counts as
    # noise (short windows would let the naive path win by lucky DC offset)
    def margin(x):
        open_lvl = x[(t > 0.005) & (t < 0.020)]
        blk_lvl = x[(t > 0.028) & (t < 0.038)]
        sep = abs(np.median(open_lvl) - np.median(blk_lvl))
        s = max(np.std(open_lvl), np.std(blk_lvl))
        return sep / (6 * s + 1e-12)                           # margin in 6-sigma units
    m_naive, m_lock = margin(naive), margin(lockin)

    # hysteresis demo on the lock-in output near threshold
    thr_on, thr_off = 0.80, 0.55                               # with hysteresis
    noisy = lockin + 0.03 * RNG.standard_normal(t.size)
    single = (noisy < 0.675).astype(int)  # same mid-band point                       # single threshold
    hyst = np.zeros_like(single)
    state = 0
    for i, v in enumerate(noisy):
        if state == 0 and v < thr_off:
            state = 1
        elif state == 1 and v > thr_on:
            state = 0
        hyst[i] = state
    chatter_single = int(np.abs(np.diff(single)).sum())
    chatter_hyst = int(np.abs(np.diff(hyst)).sum())

    # figure
    fig, ax = plt.subplots(3, 1, figsize=(8.6, 6.4), sharex=True)
    ax[0].plot(t * 1e3, raw, lw=.4, color="#8a8f98")
    ax[0].set_ylabel("raw [a.u.]")
    ax[0].set_title(f"raw photocurrent — ambient is 100x the signal (DC + 100 Hz flicker + impulses)",
                    fontsize=9)
    ax[1].plot(t * 1e3, naive, lw=.8, color="#d62728")
    ax[1].set_ylabel("naive LPF")
    ax[1].set_title(f"naive low-pass — detection margin {m_naive:.2f} (6-sigma units): buried in ambient", fontsize=9)
    ax[2].plot(t * 1e3, lockin, lw=.8, color="#1f77b4")
    ax[2].axhline(thr_on, color="#2ca02c", ls=":", lw=1)
    ax[2].axhline(thr_off, color="#2ca02c", ls=":", lw=1)
    ax[2].set_ylabel("lock-in")
    ax[2].set_xlabel("time [ms]")
    ax[2].set_title(f"lock-in demodulation — margin {m_lock:.1f}: "
                    f"chatter {chatter_single}x (single threshold) -> {chatter_hyst}x (hysteresis)",
                    fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "ps_lockin.png"), dpi=130)
    plt.close(fig)

    return {"margin_naive_6sigma": round(float(m_naive), 2),
            "margin_lockin_6sigma": round(float(m_lock), 1),
            "chatter_single_threshold": chatter_single,
            "chatter_with_hysteresis": chatter_hyst}


# ═════════════════════ 2. laser marker: galvo trajectory ════════════════════
def _galvo_track(cmd_x, cmd_y, fs, fn=1200.0, zeta=0.75):
    """2nd-order galvo response per axis."""
    wn = 2 * np.pi * fn
    sysd = sps.cont2discrete(([wn * wn], [1, 2 * zeta * wn, wn * wn]), 1 / fs)
    b, a = sysd[0].ravel(), sysd[1]
    return sps.lfilter(b, a, cmd_x), sps.lfilter(b, a, cmd_y)


def _path_K(scale=1.0):
    """Vector strokes of the letter 'K' (mm) — sharp corners on purpose."""
    strokes = [
        [(0, 0), (0, 10)],
        [(0, 4.5), (6, 10)],
        [(2.4, 6.8), (7, 0)],
    ]
    return [np.array(s, float) * scale for s in strokes]


def _plan(strokes, v_mark, fs, corner_lookahead=False, v_jump=6000.0,
          settle_s=0.0015):
    """Sample strokes at marking speed; insert laser-OFF jumps between strokes
    (at v_jump) plus a settle dwell before switching the laser on. Returns
    x, y command arrays and an ON mask (True while marking)."""
    xs, ys, on = [], [], []
    pos = strokes[0][0]
    for s in strokes:
        # jump (laser off) from current pos to stroke start
        jump = np.hypot(*(s[0] - pos))
        if jump > 1e-9:
            n = max(int(jump / v_jump * fs), 2)
            ts = (1 - np.cos(np.pi * np.linspace(0, 1, n))) / 2
            xs.append(pos[0] + (s[0][0] - pos[0]) * ts)
            ys.append(pos[1] + (s[0][1] - pos[1]) * ts)
            on.append(np.zeros(n, bool))
        nd = max(int(settle_s * fs), 1)          # settle dwell, laser still off
        xs.append(np.full(nd, s[0][0])); ys.append(np.full(nd, s[0][1]))
        on.append(np.zeros(nd, bool))
        for p0, p1 in zip(s[:-1], s[1:]):        # marking segments (laser on)
            seg = np.hypot(*(p1 - p0))
            n = max(int(seg / v_mark * fs), 2)
            ts = np.linspace(0, 1, n)
            if corner_lookahead:
                ts = (1 - np.cos(np.pi * ts)) / 2
            xs.append(p0[0] + (p1[0] - p0[0]) * ts)
            ys.append(p0[1] + (p1[1] - p0[1]) * ts)
            on.append(np.ones(n, bool))
        pos = s[-1]
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(on)


def laser_marker():
    fs = 200_000
    strokes = _path_K()
    v_fast = 2000.0  # mm/s marking speed
    delay = 1.0 / (np.pi * 1200.0) * 0.85        # galvo group-delay estimate

    def run(lookahead, comp):
        cx, cy, on = _plan(strokes, v_fast, fs, corner_lookahead=lookahead)
        ax_, ay_ = _galvo_track(cx, cy, fs)
        if comp:                                  # advance command by the delay
            k = int(delay * fs)
            ax_, ay_ = ax_[k:], ay_[k:]
            cx, cy, on = cx[:ax_.size], cy[:ay_.size], on[:ax_.size]
        err = np.hypot(ax_ - cx, ay_ - cy)
        # error only while the laser is ON (jumps/settle don't mark the part)
        e_on = float(np.max(err[on])) * 1e3       # µm
        t_total = cx.size / fs * 1e3              # ms (incl. jumps)
        return cx, cy, ax_, ay_, on, e_on, t_total

    cx, cy, axn, ayn, on_n, err_naive, t_naive = run(False, False)
    cx2, cy2, axp, ayp, on_p, err_plan, t_plan = run(True, True)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 4.4))
    for a, (X, Y, AX, AY, M, ttl, err, tt) in zip(
        (a1, a2),
        [(cx, cy, axn, ayn, on_n, "naive (const speed, no delay comp)", err_naive, t_naive),
         (cx2, cy2, axp, ayp, on_p, "corner slow-down + delay comp", err_plan, t_plan)]):
        a.plot(X[M], Y[M], ".", color="#8a8f98", ms=1.0, label="command (laser ON)")
        a.plot(AX[M], AY[M], ".", color="#d62728" if err > 100 else "#1f77b4",
               ms=1.0, label="galvo actual (laser ON)")
        a.set_title(f"{ttl}\nmax error while marking {err:.0f} um - total {tt:.1f} ms",
                    fontsize=9.5)
        a.set_aspect("equal"); a.legend(fontsize=8, markerscale=6); a.grid(alpha=.25)
        a.set_xlabel("X [mm]"); a.set_ylabel("Y [mm]")
    fig.suptitle("laser marker - 2-axis galvo trajectory (1.2 kHz bandwidth, letter 'K', 2000 mm/s)",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "md_galvo.png"), dpi=130)
    plt.close(fig)

    return {"galvo_bandwidth_hz": 1200.0, "mark_speed_mm_s": v_fast,
            "max_error_naive_um": round(err_naive, 1),
            "max_error_planned_um": round(err_plan, 1),
            "mark_time_naive_ms": round(t_naive, 2),
            "mark_time_planned_ms": round(t_plan, 2)}


# ═════════════════════ 3. microscope: focus stacking ════════════════════════
def microscope():
    n = 160
    yy, xx = np.mgrid[0:n, 0:n] / n

    # 3-D scene: tilted plane + a tall cylindrical component + a sphere cap
    z_true = 40.0 * xx + 8.0 * np.sin(8 * np.pi * yy)               # µm
    cyl = (xx - 0.3) ** 2 + (yy - 0.62) ** 2 < 0.02
    z_true[cyl] += 120.0
    dome = np.clip(0.03 - ((xx - 0.72) ** 2 + (yy - 0.3) ** 2), 0, None)
    z_true += 2600.0 * dome

    # texture (what the camera sees when in focus)
    tex = 0.55 + 0.25 * np.sin(40 * np.pi * xx) * np.sin(40 * np.pi * yy)
    tex += 0.15 * RNG.standard_normal((n, n))
    tex[cyl] *= 0.75

    # focus sweep
    z_min, z_max, n_steps = -10.0, 190.0, 21
    zs = np.linspace(z_min, z_max, n_steps)
    dof = 12.0                                                   # depth of field [µm]
    stack, sharp = [], []
    for zf in zs:
        blur_px = np.abs(z_true - zf) / dof                       # blur radius map
        # approximate spatially-varying blur with 3 quantized levels
        img = np.zeros_like(tex)
        levels = [0.5, 2.0, 5.0]
        masks = [blur_px < 1.0, (blur_px >= 1.0) & (blur_px < 3.5), blur_px >= 3.5]
        for lv, mk in zip(levels, masks):
            img[mk] = ndi.gaussian_filter(tex, lv)[mk]
        img += 0.02 * RNG.standard_normal((n, n))
        stack.append(img)
        lap = ndi.gaussian_laplace(img, 1.2) ** 2
        sharp.append(ndi.uniform_filter(lap, 7))
    stack = np.array(stack); sharp = np.array(sharp)

    best = np.argmax(sharp, axis=0)                              # per-pixel frame
    fused = np.take_along_axis(stack, best[None], axis=0)[0]
    # sub-step depth: parabolic fit around the sharpness peak
    z_map = zs[best].astype(float)
    for (i, j) in [(0, 0)]:
        pass
    b0 = np.clip(best, 1, n_steps - 2)
    s_m = np.take_along_axis(sharp, (b0 - 1)[None], 0)[0]
    s_0 = np.take_along_axis(sharp, b0[None], 0)[0]
    s_p = np.take_along_axis(sharp, (b0 + 1)[None], 0)[0]
    denom = s_m - 2 * s_0 + s_p
    delta = np.where(np.abs(denom) > 1e-12, 0.5 * (s_m - s_p) / denom, 0.0)
    z_map = zs[b0] + np.clip(delta, -1, 1) * (zs[1] - zs[0])

    depth_rmse = float(np.sqrt(np.mean((z_map - z_true) ** 2)))
    mid = stack[n_steps // 2]

    fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.5))
    ax[0].imshow(mid, cmap="gray"); ax[0].set_title("single-focus frame (mid sweep)\nnear and far both blurred", fontsize=9)
    ax[1].imshow(fused, cmap="gray"); ax[1].set_title("focus-stacked composite\nsharp everywhere", fontsize=9)
    im = ax[2].imshow(z_map, cmap="viridis")
    ax[2].set_title(f"depth map (3-D measurement)\nRMSE {depth_rmse:.1f} um / 200 um range", fontsize=9)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=ax[2], label="height [µm]", fraction=.046)
    fig.suptitle("digital microscope - depth-from-focus composition (21-frame sweep)",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "vhx_stack.png"), dpi=130)
    plt.close(fig)

    return {"stack_frames": n_steps, "z_range_um": float(z_max - z_min),
            "depth_of_field_um": dof, "depth_rmse_um": round(depth_rmse, 2)}


def main():
    os.makedirs(FIGS, exist_ok=True)
    out = {
        "fiber_sensor_lockin": fiber_sensor(),
        "laser_marker_galvo": laser_marker(),
        "microscope_focus_stack": microscope(),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    with open(os.path.join(HERE, "product_suite_results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
