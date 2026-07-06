#!/usr/bin/env python3
"""drift_monitoring.py — the model that was right on launch day and wrong by
Q3: detecting covariate drift before the labels tell you.

An AI architect's job does not end when the model ships. In production the
input distribution moves (seasonality, new segments, upstream changes) and
accuracy silently rots. Labels — the only thing that proves the rot — arrive
late or never. So drift must be caught from the FEATURES alone, unsupervised.
This module builds that monitor from the statistics up:

  PSI          the Population Stability Index between a reference window and
               the live window, per feature, binned on the reference
               quantiles:  PSI = sum (p_live - p_ref) * ln(p_live / p_ref).
               Industry thresholds: <0.1 stable, 0.1-0.25 warning, >0.25
               shift. Closed loop: PSI = 0 for identical distributions and
               rises monotonically with the shift magnitude (verified).

  KS           the two-sample Kolmogorov-Smirnov distance as an independent,
               binning-free drift signal (max gap between empirical CDFs).

  LEAD TIME    a data stream that drifts progressively; the label-based
               accuracy (computable only where labels exist) is overlaid on
               the label-FREE PSI alarm. The alarm fires while accuracy is
               still nominal — the whole point: you intervene on the feature
               signal, not on the lagging accuracy you can't yet see.

Pure numpy + matplotlib.  Run:  python3 accenture/drift_monitoring.py
"""
import json
import os

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "drift_monitoring_results.json")

N_REF = 6000
N_WIN = 1500
N_BINS = 10
PSI_WARN = 0.10
PSI_SHIFT = 0.25


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def psi(ref, live, n_bins=N_BINS):
    """Population Stability Index of `live` vs `ref` (reference quantile
    bins). Laplace-smoothed so empty bins don't blow up the log."""
    edges = np.quantile(ref, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    pr = np.histogram(ref, edges)[0] + 0.5
    pl = np.histogram(live, edges)[0] + 0.5
    pr = pr / pr.sum()
    pl = pl / pl.sum()
    return float(np.sum((pl - pr) * np.log(pl / pr)))


def ks_stat(ref, live):
    """Two-sample KS distance: max gap between the empirical CDFs."""
    allv = np.sort(np.concatenate([ref, live]))
    cr = np.searchsorted(np.sort(ref), allv, side="right") / len(ref)
    cl = np.searchsorted(np.sort(live), allv, side="right") / len(live)
    return float(np.max(np.abs(cr - cl)))


def _true_p(x):
    """True label prob: a CURVED decision boundary y ~ x1 > 0.8 x0^2. A
    linear model can approximate it near the origin but not far out — so
    covariate drift in x0 (into the curved region) rots a linear model."""
    return _sigmoid(4.0 * (x[:, 1] - 0.8 * x[:, 0] ** 2 + 0.5))


def make_stream(n_windows=12, shift_per_window=0.16, seed=0):
    """Reference sample + a sequence of live windows whose mean drifts.
    The deployed model is LINEAR (misspecified for the curved boundary):
    it is fine near the reference and degrades as x0 drifts into curvature.
    Accuracy is recorded to overlay against the unsupervised alarm."""
    rng = np.random.default_rng(seed)
    xr = rng.normal(size=(N_REF, 2))
    yr = (rng.random(N_REF) < _true_p(xr)).astype(float)
    from_ref = _fit_logistic(xr, yr)          # frozen linear deployed model

    windows = []
    for k in range(n_windows):
        shift = shift_per_window * k
        xw = rng.normal(loc=[shift, 0.0], size=(N_WIN, 2))
        yw = (rng.random(N_WIN) < _true_p(xw)).astype(float)
        pred = (_predict(from_ref, xw) > 0.5).astype(float)
        acc = float((pred == yw).mean())
        windows.append({"x": xw, "acc": acc,
                        "psi": max(psi(xr[:, 0], xw[:, 0]),
                                   psi(xr[:, 1], xw[:, 1])),
                        "ks": max(ks_stat(xr[:, 0], xw[:, 0]),
                                  ks_stat(xr[:, 1], xw[:, 1]))})
    return xr, windows


def _fit_logistic(x, y, iters=25, l2=1e-3):
    n, d = x.shape
    xb = np.hstack([np.ones((n, 1)), x])
    w = np.zeros(d + 1)
    for _ in range(iters):
        p = _sigmoid(xb @ w)
        g = xb.T @ (p - y) + l2 * w
        s = p * (1 - p) + 1e-6
        h = xb.T @ (xb * s[:, None]) + l2 * np.eye(d + 1)
        w -= np.linalg.solve(h, g)
    return w


def _predict(w, x):
    return _sigmoid(np.hstack([np.ones((len(x), 1)), x]) @ w)


def run():
    xr, windows = make_stream()
    psis = [w["psi"] for w in windows]
    accs = [w["acc"] for w in windows]
    # first window whose PSI crosses the (label-free) shift threshold
    alarm = next((k for k, p in enumerate(psis) if p > PSI_SHIFT), None)
    warn = next((k for k, p in enumerate(psis) if p > PSI_WARN), None)
    # first window with a clearly meaningful accuracy drop (>5 pts vs the
    # settled launch accuracy) — the lagging, label-dependent signal
    base_acc = float(np.mean(accs[:2]))
    acc_drop = next((k for k, a in enumerate(accs)
                     if a < base_acc - 0.05), None)
    ident = psi(xr[:, 0], xr[:len(xr) // 2, 0])   # ref vs itself
    return {
        "config": {"n_ref": N_REF, "n_window": N_WIN, "bins": N_BINS,
                   "psi_shift_threshold": PSI_SHIFT},
        "psi_identical_dist": round(ident, 4),
        "psi_monotone": bool(np.all(np.diff(psis) >= -1e-9)),
        "warn_window_psi": warn,
        "alarm_window_psi": alarm,
        "acc_drop_window": acc_drop,
        "lead_time_windows": (acc_drop - alarm
                              if alarm is not None and acc_drop is not None
                              else None),
        "psi_series": [round(p, 3) for p in psis],
        "acc_series": [round(a, 3) for a in accs],
    }


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xr, windows = make_stream()
    psis = [w["psi"] for w in windows]
    accs = [w["acc"] for w in windows]
    ks = [w["ks"] for w in windows]
    k = np.arange(len(windows))

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    ax = axes[0]
    ax.hist(xr[:, 0], bins=40, density=True, alpha=0.5, color="0.6",
            label="reference (launch)")
    ax.hist(windows[-1]["x"][:, 0], bins=40, density=True, alpha=0.5,
            color="#d81b60", label="live window (drifted)")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("density")
    ax.set_title("Covariate drift: the input the model sees\nmoves away "
                 "from what it was trained on", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(k, psis, "-o", color="#d81b60", lw=1.8, ms=4,
            label="PSI (label-free)")
    ax.plot(k, ks, "-s", color="#7500c0", lw=1.2, ms=3, alpha=0.7,
            label="KS distance")
    ax.axhline(PSI_SHIFT, color="0.4", ls="--", lw=1)
    ax.annotate("PSI shift alarm 0.25", (0.2, PSI_SHIFT + 0.01),
                fontsize=8, color="0.4")
    if res["alarm_window_psi"] is not None:
        ax.axvline(res["alarm_window_psi"], color="#d81b60", ls=":", lw=1)
    ax2 = ax.twinx()
    ax2.plot(k, accs, "-^", color="#1a7a4a", lw=1.8, ms=4,
             label="accuracy (needs labels)")
    ax2.set_ylabel("model accuracy", color="#1a7a4a")
    ax.set_xlabel("production window")
    ax.set_ylabel("drift metric")
    lt = res["lead_time_windows"]
    ax.set_title(f"Alarm fires {lt} windows before accuracy drops\n"
                 f"(catch drift on features, not on lagging labels)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    res = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "acn_drift.png"), res)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
