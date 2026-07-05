#!/usr/bin/env python3
"""jitter_bathtub.py — eye diagrams, bathtub curves, and RJ/DJ separation.

The core deliverable of a high-speed digital tester is a JITTER NUMBER with
a defensible extrapolation: you can only measure bit errors down to ~1e-5
in test time, but the spec lives at 1e-12. The industry bridge is the
dual-Dirac model — split jitter into RANDOM (Gaussian, unbounded: sets the
extrapolation slope) and DETERMINISTIC (bounded: ISI + periodic + duty-cycle
distortion), fit the measured bathtub tails on a Q-scale, and extrapolate.

This module builds the whole chain and CLOSES THE LOOP against ground truth:

  1. LINK SIMULATION — PRBS-7 at 10 Gb/s through a band-limited channel
     (ISI = data-dependent DJ), with INJECTED jitter of known size:
     RJ sigma, sinusoidal PJ, and DCD. The truth is known by construction.
  2. EYE DIAGRAM — folded persistence eye at the receiver.
  3. BATHTUB — measured BER vs sampling phase from actual error counting
     (1e6 bits, floor ~1e-6), like a tester does.
  4. DUAL-DIRAC FIT — Q-scale linear fit of both tails → estimated RJ sigma
     and DJ(dd); compare against the injected truth, then extrapolate
     TJ @ 1e-12 and the open eye width.

Pure numpy + matplotlib.  Run:  python3 ate/jitter_bathtub.py
"""
import json
import os

import numpy as np
from scipy.special import erfc, erfcinv

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "jitter_bathtub_results.json")

UI_PS = 100.0            # 10 Gb/s
OSR = 64                 # samples per UI
N_BITS = 1_000_000
RJ_PS = 1.5              # injected random jitter sigma (TRUTH)
PJ_PS = 2.0              # sinusoidal periodic jitter amplitude
PJ_PERIOD = 211          # bits per PJ cycle (incommensurate with PRBS)
DCD_PS = 1.5             # duty-cycle distortion (early rise, late fall)
CH_BW_UI = 0.65          # channel bandwidth x bitrate (ISI strength)
EDGE_RISE_PS = 25.0      # 20-80 edge time before channel


def prbs7(n, seed=0x5A):
    reg = seed & 0x7F or 0x7F
    out = np.empty(n, dtype=np.int8)
    for i in range(n):
        fb = ((reg >> 6) ^ (reg >> 5)) & 1
        reg = ((reg << 1) | fb) & 0x7F
        out[i] = reg & 1
    return out


def synthesize(n_bits=N_BITS, seed=1):
    """Waveform with injected edge jitter, then a 1-pole channel (ISI)."""
    rng = np.random.default_rng(seed)
    bits = prbs7(n_bits)
    dt = UI_PS / OSR
    # nominal edge positions with jitter applied per transition
    lev = 2.0 * bits - 1.0
    n_s = n_bits * OSR
    wave = np.repeat(lev, OSR).astype(np.float64)
    # apply timing jitter by shifting transition instants: build a time-
    # shifted square wave via edge list -> sampled comparator
    trans = np.nonzero(np.diff(bits))[0]              # bit index of edge
    edge_t = (trans + 1.0) * UI_PS                    # ideal edge time [ps]
    rj = RJ_PS * rng.standard_normal(len(trans))
    pj = PJ_PS * np.sin(2 * np.pi * (trans / PJ_PERIOD))
    rising = lev[trans + 1] > 0
    dcd = np.where(rising, -DCD_PS / 2.0, +DCD_PS / 2.0)
    edge_t = edge_t + rj + pj + dcd
    # sample the jittered NRZ: for each sample time, level = value after
    # the most recent edge
    ts = np.arange(n_s) * dt
    idx = np.searchsorted(edge_t, ts, side="right")
    start_lev = lev[0]
    seq_lev = np.empty(len(trans) + 1)
    seq_lev[0] = start_lev
    seq_lev[1:] = lev[trans + 1]
    wave = seq_lev[idx]
    # finite edge rate + channel: two cascaded 1-pole filters
    for tau_ps in (EDGE_RISE_PS / 2.2,
                   UI_PS / (2 * np.pi * CH_BW_UI)):
        a = dt / (tau_ps + dt)
        wave = _lp1(wave, a)
    return bits, wave


def _lp1(x, a):
    from scipy.signal import lfilter
    return lfilter([a], [1.0, -(1.0 - a)], x, zi=[x[0] * (1.0 - a)])[0]


def eye_matrix(wave, n_ui=2):
    """Fold the waveform into (n_traces, n_ui*OSR) for the persistence eye."""
    n = len(wave) // (n_ui * OSR)
    return wave[:n * n_ui * OSR].reshape(n, n_ui * OSR)


def bathtub(bits, wave):
    """Measured BER vs sampling phase over TWO UI (error counting at each
    offset, best integer-bit alignment — absorbs the channel group delay
    so the full valley is visible wherever the eye center lands)."""
    n_bits = len(bits)
    b = bits.astype(bool)
    ber = np.empty(2 * OSR)
    for ph in range(2 * OSR):
        samp = wave[ph::OSR]
        best = 0.5
        for shift in (0, 1, 2):
            n = min(len(samp), n_bits - shift)
            e = np.mean((samp[:n] > 0) != b[shift:shift + n])
            best = min(best, e)
        ber[ph] = max(best, 0.5 / n_bits)
    return ber


def qscale_fit(phases_ps, ber, side, i_min, floor=2e-6, ceil=2e-3):
    """Dual-Dirac: Q(BER) = (t - mu)/sigma is linear in t on each tail.

    Only the MONOTONE flank adjacent to the eye center (index i_min) is
    fitted — walking outward stops as soon as BER stops increasing, which
    excludes the far side of the crossing bump (the next UI's tail)."""
    q = np.sqrt(2.0) * erfcinv(2.0 * np.clip(ber, 1e-300, 0.5))
    idx = []
    step = -1 if side == "left" else 1
    j = i_min + step
    prev = ber[i_min]
    while 0 <= j < len(ber):
        if ber[j] < prev - 1e-12:
            break
        if floor < ber[j] < ceil:
            idx.append(j)
        prev = max(prev, ber[j])
        if ber[j] >= ceil:
            break
        j += step
    idx = np.asarray(idx)
    t = phases_ps[idx]
    slope, icpt = np.polyfit(t, q[idx], 1)
    sigma = abs(1.0 / slope)
    mu = -icpt / slope
    return sigma, mu


def analyze(seed=1):
    bits, wave = synthesize(seed=seed)
    ber_full = bathtub(bits, wave)
    ph_full = np.arange(2 * OSR) * (UI_PS / OSR)
    # locate the eye: last high-BER crossing on the left, first on the
    # right, then center the one-UI window on the valley midpoint
    high = np.nonzero(ber_full > 0.05)[0]
    gaps = np.diff(high)
    g = int(np.argmax(gaps))                 # widest quiet valley
    i_l, i_r = high[g], high[g + 1]
    c = (i_l + i_r) // 2
    sel = slice(max(c - OSR // 2, 0), min(c + OSR // 2, 2 * OSR))
    ber = ber_full[sel]
    ph = ph_full[sel]
    i_min = int(np.argmin(np.abs(np.arange(len(ber))
                                 - (c - sel.start))))

    sig_l, mu_l = qscale_fit(ph, ber, "left", i_min)
    sig_r, mu_r = qscale_fit(ph, ber, "right", i_min)
    rj_hat = 0.5 * (sig_l + sig_r)
    dj_dd = (mu_r - mu_l) - 0.0                      # crossing separation
    # TJ at 1e-12 and open eye
    q12 = np.sqrt(2.0) * erfcinv(2.0e-12)
    tj_12 = 2.0 * q12 * rj_hat + (UI_PS - (mu_r - mu_l))
    eye_12 = UI_PS - tj_12
    return {
        "bits": int(len(bits)),
        "injected": {"rj_ps": RJ_PS, "pj_ps": PJ_PS, "dcd_ps": DCD_PS},
        "rj_hat_ps": round(float(rj_hat), 3),
        "rj_error_pct": round(100 * abs(rj_hat - RJ_PS) / RJ_PS, 1),
        "eye_crossing_sep_ps": round(float(mu_r - mu_l), 2),
        "tj_1e12_ps": round(float(tj_12), 2),
        "eye_open_1e12_ps": round(float(eye_12), 2),
    }, (bits, wave, ph, ber, (sig_l, mu_l, sig_r, mu_r))


def _plot(path, ctx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bits, wave, ph, ber, (sig_l, mu_l, sig_r, mu_r) = ctx
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    eye = eye_matrix(wave)
    tt = np.arange(eye.shape[1]) * (UI_PS / OSR)
    for row in eye[:400]:
        ax.plot(tt, row, color="#26a", alpha=0.03, lw=0.8)
    ax.set_xlabel("time [ps]")
    ax.set_ylabel("amplitude")
    ax.set_title(f"Persistence eye - 10 Gb/s PRBS-7, ISI channel "
                 f"(BW {CH_BW_UI}/UI)\ninjected RJ {RJ_PS} ps + PJ {PJ_PS} "
                 f"ps + DCD {DCD_PS} ps")

    ax = axes[1]
    ax.semilogy(ph, ber, "o-", ms=3, color="#26a", label="measured BER")
    q12 = np.sqrt(2.0) * erfcinv(2.0e-12)
    tfine = np.linspace(0, UI_PS, 300)
    bl = 0.5 * erfc((tfine - mu_l) / (sig_l * np.sqrt(2))) \
        + 0.5 * erfc(-(tfine - mu_l) / (sig_l * np.sqrt(2))) * 0
    ax.semilogy(tfine, np.clip(
        0.5 * erfc(-(tfine - mu_l) / (sig_l * np.sqrt(2))), 1e-14, 1),
        "--", color="#c86", lw=1.2, label="dual-Dirac fit (left)")
    ax.semilogy(tfine, np.clip(
        0.5 * erfc((tfine - mu_r) / (sig_r * np.sqrt(2))), 1e-14, 1),
        "--", color="#a7c", lw=1.2, label="dual-Dirac fit (right)")
    ax.axhline(1e-12, color="k", ls=":", lw=1, label="BER 1e-12 spec")
    ax.set_ylim(1e-14, 1)
    ax.set_xlabel("sampling phase [ps]")
    ax.set_ylabel("BER")
    ax.set_title("Bathtub: measured to ~1e-6, extrapolated to 1e-12")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    res, ctx = analyze()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "ate_eye.png"), ctx)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
