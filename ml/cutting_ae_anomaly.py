#!/usr/bin/env python3
"""cutting_ae_anomaly.py — self-supervised AI on the cutting sound itself.

`ml/anomaly_detection.py` monitors TABULAR process data (GP / IsolationForest
/ Shewhart). This module goes one level deeper — the raw acoustic-emission /
vibration WAVEFORM of every cut, which is where blade trouble appears first:

  NORMAL       grit-engagement broadband (18-38 kHz) + spindle harmonics +
               pink background; slow sensor-gain drift across cuts (the
               nuisance that kills fixed thresholds)
  CHIPPING     a handful of sharp exponential-decay impulses (grit fracture /
               edge chipping events) riding on the normal signature
  GLAZING      the grit band fades and low-frequency rubbing rises — no
               impulses, just a spectral SHAPE change (invisible to RMS)
  CRACK        one large burst ringing at the wafer resonance

The detector is SELF-SUPERVISED — a spectrum autoencoder trained on normal
cuts only (labels for failures don't exist in volume on a real line; normal
data is free). Anomaly score = reconstruction error of the log spectrum,
per-cut. Baseline for honesty: the classic Shewhart chart on waveform RMS.

Evaluated the way a fab would: AUROC per failure mode, false-alarm rate at
the deployed threshold, and DETECTION DELAY on a progressive-glazing stream
(how many cuts pass between onset and alarm).

torch (CPU) + numpy + matplotlib.  Run:  python3 ml/cutting_ae_anomaly.py
"""
import json
import os

import numpy as np

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "results")
ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "cutting_ae_anomaly_results.json")

FS = 100_000          # sample rate [Hz]
N_SAMP = 2048         # samples per analysis window (one cut segment)
N_SPEC = 256          # log-spectrum bins fed to the autoencoder


# ── waveform synthesis ──────────────────────────────────────────────────────
def _pink(n, rng):
    w = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / FS)
    w[1:] /= np.sqrt(f[1:] / f[1])
    return np.fft.irfft(w, n)


def synth_cut(kind="normal", gain=1.0, severity=1.0, rng=None):
    """One cut's AE/vibration segment. kind in {normal, chipping, glazing,
    crack}. severity in (0,1] scales how developed the fault is."""
    if rng is None:
        rng = np.random.default_rng(0)
    t = np.arange(N_SAMP) / FS
    x = 0.25 * _pink(N_SAMP, rng)

    # spindle harmonics (30 krpm -> 500 Hz) with jitter
    for k in (1, 2, 3):
        x += 0.10 / k * np.sin(2 * np.pi * 500 * k * t
                               + rng.uniform(0, 2 * np.pi))

    # grit-engagement band 18-38 kHz: band-passed noise with envelope
    band = rng.standard_normal(N_SAMP)
    B = np.fft.rfft(band)
    f = np.fft.rfftfreq(N_SAMP, 1 / FS)
    B[(f < 18e3) | (f > 38e3)] = 0.0
    band = np.fft.irfft(B, N_SAMP)
    band /= band.std() + 1e-12
    grit_amp = 0.9
    rub = 0.0
    if kind == "glazing":                 # grit band fades, rubbing rises
        grit_amp *= (1.0 - 0.55 * severity)
        rub = 0.5 * severity
    x += grit_amp * band * (1.0 + 0.25 * np.sin(2 * np.pi * 173 * t))
    if rub > 0.0:                          # low-frequency friction rumble
        R = np.fft.rfft(rng.standard_normal(N_SAMP))
        R[(f < 2e3) | (f > 9e3)] = 0.0
        r = np.fft.irfft(R, N_SAMP)
        x += rub * r / (r.std() + 1e-12)

    if kind == "chipping":
        for _ in range(rng.integers(3, 8)):
            i0 = int(rng.integers(0, N_SAMP - 220))
            dur = int(rng.integers(60, 200))
            imp = np.exp(-np.arange(dur) / (dur / 5.0))
            x[i0:i0 + dur] += severity * rng.uniform(2.2, 4.0) * imp \
                * np.sin(2 * np.pi * rng.uniform(25e3, 45e3)
                         * np.arange(dur) / FS)
    if kind == "crack":
        i0 = int(rng.integers(200, N_SAMP - 900))
        dur = 800
        imp = np.exp(-np.arange(dur) / 220.0)
        x[i0:i0 + dur] += severity * rng.uniform(5.0, 8.0) * imp \
            * np.sin(2 * np.pi * 31e3 * np.arange(dur) / FS)

    return gain * x


def log_spectrum(x):
    """N_SPEC-bin log power spectrum, per-window energy normalized — the
    normalization removes sensor-gain drift so the model sees SHAPE."""
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    edges = np.linspace(0, len(p), N_SPEC + 1).astype(int)
    b = np.array([p[a:c].mean() if c > a else p[a]
                  for a, c in zip(edges[:-1], edges[1:])])
    b = b / (b.sum() + 1e-20)
    return np.log10(b + 1e-12).astype(np.float32)


def make_sets(n_train=400, n_eval=150, seed=0):
    rng = np.random.default_rng(seed)

    def gain():
        return float(10 ** rng.uniform(-0.15, 0.15))   # ±1.5 dB drift

    train = np.stack([log_spectrum(synth_cut("normal", gain(), rng=rng))
                      for _ in range(n_train)])
    ev = {}
    for kind in ("normal", "chipping", "glazing", "crack"):
        ev[kind] = np.stack([
            log_spectrum(synth_cut(kind, gain(),
                                   severity=float(rng.uniform(0.15, 0.7)),
                                   rng=rng))
            for _ in range(n_eval)])
    return train, ev


# ── model ───────────────────────────────────────────────────────────────────
def train_autoencoder(train_spec, epochs=200, seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    mu = train_spec.mean(axis=0)
    sd = train_spec.std(axis=0) + 1e-6
    X = torch.tensor((train_spec - mu) / sd)
    model = nn.Sequential(
        nn.Linear(N_SPEC, 64), nn.ReLU(),
        nn.Linear(64, 12), nn.ReLU(),
        nn.Linear(12, 64), nn.ReLU(),
        nn.Linear(64, N_SPEC),
    )
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(epochs):
        opt.zero_grad()
        out = model(X)
        loss = ((out - X) ** 2).mean()
        loss.backward()
        opt.step()
    model.eval()
    return model, mu, sd


def ae_score(model, mu, sd, specs):
    import torch
    with torch.no_grad():
        X = torch.tensor((specs - mu) / sd)
        rec = model(X)
        return ((rec - X) ** 2).mean(dim=1).numpy()


def rms_score(kind_waves):
    return np.array([np.sqrt(np.mean(w ** 2)) for w in kind_waves])


def auroc(neg, pos):
    """Mann-Whitney with midranks (same convention as inspection modules)."""
    s = np.concatenate([neg, pos])
    r = np.empty(len(s))
    order = np.argsort(s, kind="mergesort")
    sr = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sr[j + 1] == sr[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    rp = r[len(neg):].sum()
    n0, n1 = len(neg), len(pos)
    return float((rp - n1 * (n1 + 1) / 2) / (n0 * n1))


# ── progressive-glazing stream: detection delay ─────────────────────────────
def drift_stream(model, mu, sd, thr_ae, thr_rms, n=300, onset=150, seed=3):
    """Glazing develops linearly from `onset`; return alarm indices (first
    of 2 consecutive scores above threshold — standard debounce)."""
    rng = np.random.default_rng(seed)
    ae_s, rms_s = [], []
    for i in range(n):
        sev = 0.0 if i < onset else min(1.0, (i - onset) / 80.0)
        kind = "normal" if sev <= 0.0 else "glazing"
        g = float(10 ** rng.uniform(-0.15, 0.15))
        w = synth_cut(kind, g, severity=max(sev, 1e-3), rng=rng)
        ae_s.append(float(ae_score(model, mu, sd,
                                   log_spectrum(w)[None, :])[0]))
        rms_s.append(float(np.sqrt(np.mean(w ** 2))))
    ae_s, rms_s = np.asarray(ae_s), np.asarray(rms_s)

    def first_alarm(scores, thr):
        hits = scores > thr
        for i in range(1, n):
            if hits[i] and hits[i - 1]:
                return i
        return None

    return ae_s, rms_s, first_alarm(ae_s, thr_ae), first_alarm(rms_s, thr_rms)


def run(seed=0, n_train=400, n_eval=150, epochs=200):
    rng = np.random.default_rng(seed + 100)
    train, ev = make_sets(n_train, n_eval, seed)
    model, mu, sd = train_autoencoder(train, epochs=epochs, seed=seed)

    scores = {k: ae_score(model, mu, sd, v) for k, v in ev.items()}
    # RMS baseline needs waveforms — regenerate matched sets
    waves = {}
    for kind in ev:
        waves[kind] = [synth_cut(kind, float(10 ** rng.uniform(-.15, .15)),
                                 severity=float(rng.uniform(0.15, 0.7)),
                                 rng=rng) for _ in range(n_eval)]
    rms = {k: rms_score(v) for k, v in waves.items()}

    res = {"auroc_ae": {}, "auroc_rms": {}}
    for kind in ("chipping", "glazing", "crack"):
        res["auroc_ae"][kind] = round(auroc(scores["normal"],
                                            scores[kind]), 3)
        res["auroc_rms"][kind] = round(auroc(rms["normal"], rms[kind]), 3)

    # deployed threshold: 99.5th percentile of normal (0.5 % FAR target)
    thr_ae = float(np.quantile(scores["normal"], 0.995))
    thr_rms = float(np.quantile(rms["normal"], 0.995))
    ae_s, rms_s, a_ae, a_rms = drift_stream(model, mu, sd, thr_ae, thr_rms)
    res["glazing_stream"] = {
        "onset_cut": 150,
        "alarm_ae": a_ae, "alarm_rms": a_rms,
        "delay_ae_cuts": None if a_ae is None else a_ae - 150,
        "delay_rms_cuts": None if a_rms is None else a_rms - 150,
    }
    return res, (train, ev, scores, rms, thr_ae, thr_rms,
                 ae_s, rms_s, a_ae, a_rms, model, mu, sd)


# ── figure ──────────────────────────────────────────────────────────────────
def _plot(path, res, ctx, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (train, ev, scores, rms, thr_ae, thr_rms,
     ae_s, rms_s, a_ae, a_rms, model, mu, sd) = ctx
    rng = np.random.default_rng(seed)
    colors = {"normal": "#26a", "chipping": "#c44",
              "glazing": "#b4740a", "crack": "#7b3fb8"}

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))

    ax = axes[0, 0]
    f = np.linspace(0, FS / 2 / 1000, N_SPEC)
    for kind, c in colors.items():
        ax.plot(f, ev[kind].mean(axis=0), color=c, lw=1.4, label=kind)
    ax.set_xlabel("frequency [kHz]")
    ax.set_ylabel("log power (normalized)")
    ax.set_title("Mean log spectra: glazing is a SHAPE change, not a level")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for kind, c in colors.items():
        ax.hist(scores[kind], bins=40, color=c, alpha=0.55, label=kind)
    ax.axvline(thr_ae, color="k", ls="--", lw=1, label="threshold (0.5% FAR)")
    ax.set_xlabel("autoencoder reconstruction error")
    ax.set_ylabel("cuts")
    ax.set_title("Self-supervised anomaly score")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    kinds = ("chipping", "glazing", "crack")
    x = np.arange(3)
    ax.bar(x - 0.18, [res["auroc_ae"][k] for k in kinds], width=0.36,
           color="#26a", label="spectrum AE (self-supervised)")
    ax.bar(x + 0.18, [res["auroc_rms"][k] for k in kinds], width=0.36,
           color="#c86", label="RMS + Shewhart (classic)")
    ax.set_xticks(x, kinds)
    ax.axhline(1.0, color="0.7", lw=0.8)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("AUROC")
    ax.set_title("Per-failure-mode detection")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    n = len(ae_s)
    ax.plot(np.arange(n), ae_s / thr_ae, color="#26a", lw=1,
            label="AE score / threshold")
    ax.plot(np.arange(n), rms_s / thr_rms, color="#c86", lw=1,
            label="RMS / threshold")
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.axvline(150, color="0.5", ls=":", label="glazing onset")
    if a_ae is not None:
        ax.axvline(a_ae, color="#26a", ls="--", lw=1.4,
                   label=f"AE alarm (+{a_ae-150} cuts)")
    if a_rms is not None:
        ax.axvline(a_rms, color="#c86", ls="--", lw=1.4,
                   label=f"RMS alarm (+{a_rms-150} cuts)")
    ax.set_xlabel("cut index")
    ax.set_ylabel("score / threshold")
    ax.set_title("Progressive glazing: who alarms first?")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    res, ctx = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    for d in (FIGDIR, ASSETS):
        _plot(os.path.join(d, "ae_anomaly.png"), res, ctx)
    print("figure ->", FIGDIR, "and", ASSETS)


if __name__ == "__main__":
    main()
