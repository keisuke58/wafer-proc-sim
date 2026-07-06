#!/usr/bin/env python3
"""uplift_modeling.py — causal targeting: who to act on, not who will convert.

A classifier ranks customers by "will they convert?". A business decision
needs a different, harder question: "who converts BECAUSE of the campaign?"
— the causal uplift (individual treatment effect). The two rankings are not
the same, and confusing them wastes the whole marketing budget on the
SURE-THINGS (would have converted anyway) and the LOST-CAUSES (never
convert). This module quantifies the gap, from a randomized trial up.

  DATA         a randomized experiment (propensity 0.5): features x,
               treatment t in {0,1}, binary outcome y. The TRUE per-person
               uplift tau(x) = P(y=1|do t=1, x) - P(y=1|do t=0, x) is known
               (synthetic), so recovery can be scored. Populations mix:
               persuadables (tau>0), sure-things (high baseline, tau~0),
               lost-causes (low baseline, tau~0), sleeping-dogs (tau<0 —
               the campaign back-fires).

  MODEL        the T-LEARNER: fit one response model on the treated arm and
               one on the control arm (logistic regression by IRLS, from
               scratch), uplift_hat(x) = p1_hat(x) - p0_hat(x).

  EVALUATION   the QINI CURVE — order the population by predicted uplift and
               plot the cumulative incremental conversions
                 Q(k) = responders_t(k) - responders_c(k) * n_t/n_c.
               Area under Qini vs the diagonal = value of the targeting.
               Targeting by uplift is compared against targeting by
               RESPONSE probability (the classifier's answer) and random —
               the money plot showing they rank different people.

Closed loops: recovered uplift correlates with true tau; on A/A data the
Qini area collapses to ~0; Qini(uplift) > Qini(response) > random.

Pure numpy + matplotlib.  Run:  python3 accenture/uplift_modeling.py
"""
import json
import os

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "uplift_modeling_results.json")

N = 8000
D = 4


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def make_trial(n=N, seed=0, aa=False):
    """Randomized trial. Returns x, t, y, tau_true (per-person uplift).
    aa=True makes treatment have zero effect (A/A test)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, D))
    t = rng.integers(0, 2, size=n)
    base = _sigmoid(0.9 * x[:, 0] - 0.7 * x[:, 1] - 0.3)     # baseline p0
    # uplift depends on x2, x3: a persuadable subgroup, some sleeping dogs
    lift = 0.0 if aa else (0.35 * _sigmoid(2.0 * x[:, 2])
                           - 0.10 * _sigmoid(2.0 * x[:, 3]))
    p0 = base
    p1 = np.clip(base + lift, 0.001, 0.999)
    tau = p1 - p0
    p = np.where(t == 1, p1, p0)
    y = (rng.random(n) < p).astype(float)
    return x, t, y, tau


def fit_logistic(x, y, iters=25, l2=1e-3):
    """Logistic regression by IRLS (Newton) with an intercept."""
    n, d = x.shape
    xb = np.hstack([np.ones((n, 1)), x])
    w = np.zeros(d + 1)
    for _ in range(iters):
        p = _sigmoid(xb @ w)
        grad = xb.T @ (p - y) + l2 * w
        s = p * (1 - p) + 1e-6
        h = xb.T @ (xb * s[:, None]) + l2 * np.eye(d + 1)
        w -= np.linalg.solve(h, grad)
    return w


def predict_logistic(w, x):
    xb = np.hstack([np.ones((len(x), 1)), x])
    return _sigmoid(xb @ w)


def t_learner(x, t, y):
    """Two-model uplift: p1 - p0 fit on the treated / control arms."""
    w1 = fit_logistic(x[t == 1], y[t == 1])
    w0 = fit_logistic(x[t == 0], y[t == 0])
    return predict_logistic(w1, x) - predict_logistic(w0, x), (w1, w0)


def qini_curve(score, t, y, n_pts=100):
    """Cumulative incremental conversions vs fraction targeted, ranking by
    `score` (desc). Returns (fractions, qini)."""
    order = np.argsort(-score)
    t, y = t[order], y[order]
    n = len(t)
    fr = np.linspace(0, 1, n_pts + 1)
    q = np.zeros(n_pts + 1)
    for i, f in enumerate(fr):
        k = int(f * n)
        if k == 0:
            continue
        tt, yy = t[:k], y[:k]
        n_t = max(tt.sum(), 1)
        n_c = max((1 - tt).sum(), 1)
        rt = yy[tt == 1].sum()
        rc = yy[tt == 0].sum()
        q[i] = rt - rc * n_t / n_c
    return fr, q


def qini_area(fr, q):
    """Area between the Qini curve and the random diagonal (normalized)."""
    diag = q[-1] * fr
    d = q - diag
    return float(np.sum((d[:-1] + d[1:]) / 2.0 * np.diff(fr)))


def _split(n, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = n // 2
    return idx[:cut], idx[cut:]


def _eval_split(x, t, y, tau=None):
    """Fit the T-learner on the train half, score Qini on the held-out half
    (honest: no in-sample optimism). Returns (fr, q_up, q_re, q_rand, corr)."""
    tr, te = _split(len(t))
    w1 = fit_logistic(x[tr][t[tr] == 1], y[tr][t[tr] == 1])
    w0 = fit_logistic(x[tr][t[tr] == 0], y[tr][t[tr] == 0])
    up_te = predict_logistic(w1, x[te]) - predict_logistic(w0, x[te])
    resp_te = predict_logistic(fit_logistic(x[tr], y[tr]), x[te])
    fr, q_up = qini_curve(up_te, t[te], y[te])
    _, q_re = qini_curve(resp_te, t[te], y[te])
    rng = np.random.default_rng(1)
    _, q_rand = qini_curve(rng.random(len(te)), t[te], y[te])
    corr = (float(np.corrcoef(up_te, tau[te])[0, 1])
            if tau is not None else None)
    return fr, q_up, q_re, q_rand, corr


def run():
    x, t, y, tau = make_trial()
    fr, q_up, q_re, q_rand, corr = _eval_split(x, t, y, tau)

    # A/A sanity (held-out too)
    xa, ta, ya, _ = make_trial(seed=2, aa=True)
    fra, qa, _, _, _ = _eval_split(xa, ta, ya)

    return {
        "trial": {"n": N, "propensity": 0.5, "features": D},
        "uplift_recovery_corr_vs_true": round(corr, 3),
        "qini_area": {"uplift": round(qini_area(fr, q_up), 1),
                      "response_model": round(qini_area(fr, q_re), 1),
                      "random": round(qini_area(fr, q_rand), 1)},
        "qini_area_AA_should_be_zero": round(qini_area(fra, qa), 1),
        "incremental_at_30pct": {
            "uplift": int(q_up[30]),
            "response_model": int(q_re[30])},
    }


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, t, y, tau = make_trial()
    up, _ = t_learner(x, t, y)
    fr, q_up, q_re, q_rand, _ = _eval_split(x, t, y, tau)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    ax = axes[0]
    ax.scatter(tau, up, s=4, alpha=0.25, color="#7500c0")
    lims = [min(tau.min(), up.min()), max(tau.max(), up.max())]
    ax.plot(lims, lims, "--", color="0.5", lw=1)
    ax.set_xlabel("true uplift tau(x)")
    ax.set_ylabel("T-learner predicted uplift")
    ax.set_title(f"Recovering the causal effect\n(corr = "
                 f"{res['uplift_recovery_corr_vs_true']:.2f} vs ground "
                 f"truth)", fontsize=10)

    ax = axes[1]
    ax.plot(fr, q_up, lw=2, color="#7500c0", label="target by UPLIFT")
    ax.plot(fr, q_re, lw=1.8, color="#b4740a",
            label="target by response prob.")
    ax.plot(fr, q_rand, "--", lw=1.2, color="0.5", label="random")
    ax.plot([0, 1], [0, q_up[-1]], ":", color="0.7", lw=1)
    ax.set_xlabel("fraction of population targeted")
    ax.set_ylabel("cumulative incremental conversions")
    qa = res["qini_area"]
    ax.set_title(f"Qini curve: uplift beats the classifier\narea "
                 f"{qa['uplift']} vs {qa['response_model']} vs "
                 f"{qa['random']} (random)", fontsize=10)
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
    _plot(os.path.join(ASSETS, "acn_uplift.png"), res)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
