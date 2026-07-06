#!/usr/bin/env python3
"""fairness_mitigation.py — trustworthy AI: measuring and removing disparate
impact, from the definitions and the reweighing math up.

A model that is accurate can still be unfair: if a protected group receives
positive predictions at a lower rate, the model has DISPARATE IMPACT even
when no protected attribute is used as a feature (proxies leak it). IBM's
open-source line (AIF360) formalizes this; here the metrics and one
canonical pre-processing mitigation are implemented from scratch:

  METRICS      * Disparate Impact ratio DIR = P(hat y=1 | A=unpriv) /
                 P(hat y=1 | A=priv)  (the "80% rule": fair if >= 0.8)
               * Demographic-parity difference
               * Equalized-odds difference (gap in TPR and FPR across groups)

  REWEIGHING   the AIF360 pre-processing view: weight each sample so the
               protected attribute A and the label Y become independent,
                   w(a, y) = P(A=a) P(Y=y) / P(A=a, Y=y).
               Closed loop: the reweighting makes the weighted positive-
               label rate EXACTLY equal across groups (verified to <1e-9).
               But reweighing addresses LABEL bias, not the proxy leakage
               in the features — so prediction disparity barely moves. That
               honest negative result motivates the next lever.

  POST-PROCESS the fix that actually closes prediction disparity: GROUP-
               AWARE thresholds (AIF360 reject-option family). Pick a
               per-group decision threshold so each group's positive rate
               equals a common target -> demographic parity by construction
               (DIR -> 1.0). Fairness is not free: the DIR-vs-accuracy
               frontier shows exactly how many accuracy points parity costs
               — the number a review board argues over.

Pure numpy + matplotlib.  Run:  python3 ibm/fairness_mitigation.py
"""
import json
import os

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "fairness_mitigation_results.json")

N = 6000


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def make_biased_data(n=N, seed=0):
    """Protected attribute A in {0=unpriv,1=priv}; features x correlated
    with A (proxies); true label y depends on x AND carries historical bias
    against A=0. Returns x (no A column), a, y."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, size=n)
    # two proxy features whose mean shifts with A (A leaks even if dropped)
    x0 = rng.normal(0.8 * a, 1.0)
    x1 = rng.normal(-0.5 * a, 1.0)
    # historical label: qualified score + a bias term favouring A=1
    score = 1.1 * x0 - 0.6 * x1 + 1.2 * a - 0.3
    y = (rng.random(n) < _sigmoid(score)).astype(float)
    return np.column_stack([x0, x1]), a, y


def fit_logistic(x, y, weights=None, iters=30, l2=1e-3):
    n, d = x.shape
    xb = np.hstack([np.ones((n, 1)), x])
    w = np.zeros(d + 1)
    sw = np.ones(n) if weights is None else weights
    for _ in range(iters):
        p = _sigmoid(xb @ w)
        g = xb.T @ (sw * (p - y)) + l2 * w
        s = sw * p * (1 - p) + 1e-6
        h = xb.T @ (xb * s[:, None]) + l2 * np.eye(d + 1)
        w -= np.linalg.solve(h, g)
    return w


def predict(w, x):
    return _sigmoid(np.hstack([np.ones((len(x), 1)), x]) @ w)


def reweigh(a, y):
    """AIF360 reweighing weights w(a,y) = P(a)P(y)/P(a,y)."""
    n = len(a)
    w = np.ones(n)
    for av in (0, 1):
        for yv in (0, 1):
            m = (a == av) & (y == yv)
            p_a = (a == av).mean()
            p_y = (y == yv).mean()
            p_ay = m.mean()
            if p_ay > 0:
                w[m] = p_a * p_y / p_ay
    return w


def fairness_metrics(yhat, a, y=None):
    """DIR, demographic-parity diff, and (if y given) equalized-odds diff."""
    r_unpriv = yhat[a == 0].mean()
    r_priv = yhat[a == 1].mean()
    dir_ = r_unpriv / r_priv if r_priv > 0 else float("nan")
    dp = r_priv - r_unpriv
    out = {"DIR": float(dir_), "dem_parity_diff": float(dp)}
    if y is not None:
        def tpr_fpr(mask):
            yy, hh = y[mask], yhat[mask]
            tpr = hh[yy == 1].mean() if (yy == 1).any() else 0.0
            fpr = hh[yy == 0].mean() if (yy == 0).any() else 0.0
            return tpr, fpr
        t0, f0 = tpr_fpr(a == 0)
        t1, f1 = tpr_fpr(a == 1)
        out["eq_odds_diff"] = float(max(abs(t0 - t1), abs(f0 - f1)))
    return out


def group_threshold_predict(p, a, target_rate):
    """Per-group threshold so each group's positive rate == target_rate
    (demographic parity by construction). Returns yhat and (t0, t1)."""
    yhat = np.zeros(len(p))
    ths = {}
    for av in (0, 1):
        pa = p[a == av]
        t = np.quantile(pa, 1.0 - target_rate)   # top target_rate fraction
        yhat[a == av] = (pa > t).astype(float)
        ths[av] = float(t)
    return yhat, ths


def run():
    x, a, y = make_biased_data()
    w_base = fit_logistic(x, y)                 # A dropped; proxies leak
    p = predict(w_base, x)
    yhat_base = (p > 0.5).astype(float)
    acc_base = float((yhat_base == y).mean())
    m_base = fairness_metrics(yhat_base, a, y)

    # reweighing closed loop: weighted positive-label rate equal by group
    rw = reweigh(a, y)
    wpr0 = np.sum(rw[a == 0] * y[a == 0]) / np.sum(rw[a == 0])
    wpr1 = np.sum(rw[a == 1] * y[a == 1]) / np.sum(rw[a == 1])
    w_rw = fit_logistic(x, y, weights=rw)
    yhat_rw = (predict(w_rw, x) > 0.5).astype(float)
    m_rw = fairness_metrics(yhat_rw, a, y)

    # post-processing: group-aware thresholds to the overall base rate
    target = float(yhat_base.mean())
    yhat_pp, ths = group_threshold_predict(p, a, target)
    acc_pp = float((yhat_pp == y).mean())
    m_pp = fairness_metrics(yhat_pp, a, y)

    return {
        "data": {"n": N, "groups": "A=0 unprivileged / A=1 privileged",
                 "note": "protected attribute NOT a feature; proxies leak"},
        "reweigh_closed_loop": {
            "weighted_pos_rate_A0": round(float(wpr0), 6),
            "weighted_pos_rate_A1": round(float(wpr1), 6),
            "abs_diff": round(abs(float(wpr0 - wpr1)), 9)},
        "baseline_thresh0.5": {
            "acc": round(acc_base, 3), "DIR": round(m_base["DIR"], 3),
            "dem_parity_diff": round(m_base["dem_parity_diff"], 3),
            "eq_odds_diff": round(m_base["eq_odds_diff"], 3)},
        "reweighed_model": {
            "DIR": round(m_rw["DIR"], 3),
            "note": "label reweighing barely moves prediction disparity"},
        "group_thresholds": {
            "acc": round(acc_pp, 3), "DIR": round(m_pp["DIR"], 3),
            "dem_parity_diff": round(m_pp["dem_parity_diff"], 4),
            "t_unpriv": round(ths[0], 3), "t_priv": round(ths[1], 3)},
        "parity_accuracy_cost_pts": round(100 * (acc_base - acc_pp), 2),
    }


def _frontier(x, a, y):
    """Sweep the parity target: for each common positive rate, group-aware
    thresholds give DIR ~ 1; interpolating single- vs group-threshold traces
    the fairness-accuracy frontier."""
    w = fit_logistic(x, y)
    p = predict(w, x)
    dirs, accs = [], []
    for t in np.linspace(0.2, 0.8, 30):
        yhat = (p > t).astype(float)              # single global threshold
        dirs.append(fairness_metrics(yhat, a)["DIR"])
        accs.append((yhat == y).mean())
    # the group-threshold operating point (parity)
    yhat_pp, _ = group_threshold_predict(p, a, float((p > 0.5).mean()))
    return (np.array(dirs), np.array(accs),
            fairness_metrics(yhat_pp, a)["DIR"], (yhat_pp == y).mean())


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, a, y = make_biased_data()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    ax = axes[0]
    labels = ["baseline\n(thr 0.5)", "reweighed\nmodel", "group\nthresholds"]
    dirs = [res["baseline_thresh0.5"]["DIR"], res["reweighed_model"]["DIR"],
            res["group_thresholds"]["DIR"]]
    xb = np.arange(3)
    colors = ["#888", "#6fa8d8", "#1f70c1"]
    ax.bar(xb, dirs, 0.55, color=colors)
    ax.axhline(0.8, color="#c0392b", ls="--", lw=1)
    ax.annotate("80% rule (fair >= 0.8)", (-0.4, 0.82), fontsize=8,
                color="#c0392b")
    for i, d in enumerate(dirs):
        ax.annotate(f"{d:.2f}", (i, d), ha="center", va="bottom",
                    fontsize=9)
    ax.set_xticks(xb)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Disparate Impact ratio")
    ax.set_title("Label reweighing barely moves prediction disparity;\n"
                 "group-aware thresholds reach parity", fontsize=10)

    ax = axes[1]
    d, ac, dir_pp, acc_pp = _frontier(x, a, y)
    ax.plot(ac, d, "-o", ms=3, color="#888",
            label="single global threshold")
    ax.scatter([acc_pp], [dir_pp], s=80, color="#1f70c1", zorder=5,
               label="group thresholds (parity)")
    ax.axhline(0.8, color="#c0392b", ls="--", lw=1)
    ax.set_xlabel("accuracy")
    ax.set_ylabel("Disparate Impact ratio")
    ax.set_title(f"Fairness-accuracy frontier\nparity costs "
                 f"{res['parity_accuracy_cost_pts']:.1f} accuracy points",
                 fontsize=10)
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
    _plot(os.path.join(ASSETS, "ibm_fairness.png"), res)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
