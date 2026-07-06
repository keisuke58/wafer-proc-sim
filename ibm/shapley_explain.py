#!/usr/bin/env python3
"""shapley_explain.py — why did the model say that? EXACT Shapley attribution
with the efficiency axiom as a closed-loop proof of correctness.

"Trustworthy AI" needs more than an accurate model — it needs to decompose
any single prediction into per-feature contributions a human (or an auditor)
can check. Shapley values, from cooperative game theory, are the unique
attribution satisfying efficiency, symmetry, null-player and additivity. For
a handful of features they can be computed EXACTLY by averaging each
feature's marginal contribution over all coalitions, and that exactness lets
us prove the implementation is right rather than trust it:

  VALUE FN     v(S) = E_background[ f(x_S, X_-S) ] — the model's expected
               output when only the features in S are fixed to the query
               instance and the rest are marginalized over a background set.

  EXACT SHAP   phi_i = sum_{S !contains i} |S|!(n-|S|-1)!/n! [v(S+i) - v(S)],
               enumerated over all 2^(n-1) coalitions (n small).

  EFFICIENCY   the axiom that makes the explanation trustworthy:
                   sum_i phi_i = f(x) - E[f]  (base value).
               Verified here to < 1e-10 — the attribution literally adds up
               to the prediction, no residue, no hand-waving.

  APPROX       the permutation/Monte-Carlo estimator (what scales past ~15
               features) is shown converging to the exact values as samples
               grow — quantifying the error you accept when exact is too
               expensive.

Pure numpy + matplotlib.  Run:  python3 ibm/shapley_explain.py
"""
import json
import os
from itertools import combinations, permutations

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "shapley_explain_results.json")

N_FEAT = 6
N_BG = 200          # background sample for the value function
FEATS = ["income", "debt_ratio", "age", "n_accounts", "late_pays",
         "utilization"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def make_model(seed=0):
    """A nonlinear scoring model f(x) in [0,1] (credit-approval flavour) and
    a background dataset to marginalize over. Nonlinear so Shapley != the
    linear weights."""
    rng = np.random.default_rng(seed)
    bg = rng.normal(size=(N_BG, N_FEAT))
    w = np.array([1.4, -1.2, 0.3, 0.5, -1.6, -0.9])

    def f(X):
        X = np.atleast_2d(X)
        lin = X @ w
        inter = 0.8 * X[:, 0] * X[:, 1] - 0.6 * X[:, 4] * X[:, 5]
        return _sigmoid(lin + inter - 0.2)

    return f, bg, w


def value_fn(f, bg, x, S):
    """v(S) = mean over background of f with features in S set to x."""
    if len(S) == 0:
        return float(f(bg).mean())
    Xrep = bg.copy()
    for i in S:
        Xrep[:, i] = x[i]
    return float(f(Xrep).mean())


def exact_shapley(f, bg, x):
    """Exact Shapley values by coalition enumeration (2^(n-1) per feature)."""
    n = len(x)
    from math import factorial
    phi = np.zeros(n)
    others = list(range(n))
    for i in range(n):
        rest = [j for j in others if j != i]
        for k in range(len(rest) + 1):
            for S in combinations(rest, k):
                S = list(S)
                weight = factorial(len(S)) * factorial(n - len(S) - 1) \
                    / factorial(n)
                phi[i] += weight * (value_fn(f, bg, x, S + [i])
                                    - value_fn(f, bg, x, S))
    return phi


def permutation_shapley(f, bg, x, n_perm=200, seed=1):
    """Monte-Carlo estimator: average marginal contribution over random
    feature orderings."""
    rng = np.random.default_rng(seed)
    n = len(x)
    phi = np.zeros(n)
    for _ in range(n_perm):
        perm = rng.permutation(n)
        S = []
        v_prev = value_fn(f, bg, x, S)
        for i in perm:
            S2 = S + [i]
            v_cur = value_fn(f, bg, x, S2)
            phi[i] += v_cur - v_prev
            v_prev = v_cur
            S = S2
    return phi / n_perm


def run():
    f, bg, w = make_model()
    rng = np.random.default_rng(3)
    x = rng.normal(size=N_FEAT)
    phi = exact_shapley(f, bg, x)
    base = float(f(bg).mean())
    pred = float(f(x)[0])
    efficiency_residual = abs(phi.sum() - (pred - base))

    # permutation convergence (averaged over seeds to smooth MC noise)
    conv = {}
    for m in (25, 100, 400, 1600):
        errs = [np.max(np.abs(permutation_shapley(f, bg, x, n_perm=m,
                                                  seed=s) - phi))
                for s in range(5)]
        conv[str(m)] = round(float(np.mean(errs)), 4)

    order = np.argsort(-np.abs(phi))
    return {
        "model": {"n_features": N_FEAT, "background": N_BG,
                  "nonlinear": True},
        "instance": {"prediction": round(pred, 4),
                     "base_value": round(base, 4)},
        "efficiency_axiom": {
            "sum_shapley": round(float(phi.sum()), 6),
            "pred_minus_base": round(pred - base, 6),
            "residual": round(efficiency_residual, 12)},
        "shapley_values": {FEATS[i]: round(float(phi[i]), 4)
                           for i in order},
        "top_driver": FEATS[order[0]],
        "permutation_max_err_vs_exact": conv,
    }


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f, bg, w = make_model()
    rng = np.random.default_rng(3)
    x = rng.normal(size=N_FEAT)
    phi = exact_shapley(f, bg, x)
    base = float(f(bg).mean())
    pred = float(f(x)[0])
    order = np.argsort(phi)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    ax = axes[0]
    colors = ["#c0392b" if v < 0 else "#1a7a4a" for v in phi[order]]
    ax.barh([FEATS[i] for i in order], phi[order], color=colors)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("Shapley contribution to prediction")
    ax.set_title(f"Exact attribution: base {base:.2f} + contributions "
                 f"= pred {pred:.2f}\n(sum of bars = pred - base, "
                 f"residual {res['efficiency_axiom']['residual']:.0e})",
                 fontsize=10)

    ax = axes[1]
    ms = [25, 100, 400, 1600]
    errs = [res["permutation_max_err_vs_exact"][str(m)] for m in ms]
    ax.loglog(ms, errs, "-o", color="#1f70c1", lw=1.8, ms=5,
              label="max |perm - exact|")
    ax.loglog(ms, errs[0] * (np.array(ms) / ms[0]) ** -0.5, "--",
              color="0.5", lw=1, label="1/sqrt(N) reference")
    ax.set_xlabel("permutation samples")
    ax.set_ylabel("max error vs exact Shapley")
    ax.set_title("Monte-Carlo estimator converges to exact\n(the price of "
                 "scaling past enumeration)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    res = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "ibm_shapley.png"), res)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
