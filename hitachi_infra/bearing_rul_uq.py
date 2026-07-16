#!/usr/bin/env python3
"""bearing_rul_uq.py — remaining-useful-life prognostics for infrastructure
equipment, WITH calibrated Bayesian uncertainty.

For a railway, a power plant, a factory drive (Hitachi's world), the useful
question is not "when will this bearing fail?" but "give me a failure time I
can trust, WITH how sure you are" — because the credible interval is what
sets the safety margin and the maintenance window. A point estimate that is
silently overconfident is worse than useless. This module does prognostics
the way a social-infrastructure operator needs it: a distribution, and proof
the distribution is honest.

  DEGRADATION    each unit's health degrades stochastically: log-health
      y(t) = a + b t + noise, with unit-to-unit random effects (a, b) drawn
      from a population — the standard exponential-degradation prognostics
      model. Failure = health crossing a threshold.

  BAYESIAN RUL   observing a unit's noisy health up to now, a conjugate
      Bayesian linear regression gives the posterior over (a, b); Monte-Carlo
      propagation through the threshold-crossing time yields a full posterior
      over remaining useful life — median + a 90% credible interval, not a
      bare number. The population fit supplies the prior (empirical Bayes).

  CALIBRATION    the result that matters for safety: the 90% interval is
      checked for EMPIRICAL COVERAGE across many held-out units (does it
      actually contain the truth ~90% of the time?), and its width is shown
      to SHRINK as failure approaches. A naive least-squares point estimate
      is the foil — accurate on average but with no honest uncertainty.

numpy + scipy + matplotlib.  Run:  python3 hitachi_infra/bearing_rul_uq.py
"""
import json
import os

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "bearing_rul_uq_results.json")

# population of the exponential-degradation model (log-health = a + b t)
MU_A, TAU_A = 0.0, 0.15          # initial log-health spread
MU_B, TAU_B = 0.020, 0.006       # degradation rate spread [1/step], b>0
SIGMA = 0.08                     # measurement noise on log-health
Y_FAIL = np.log(3.0)             # failure threshold on health (=> logH)
DT = 1.0                         # observation interval [time step]


def simulate_unit(rng):
    """One run-to-failure unit: return (a, b, T_fail)."""
    a = rng.normal(MU_A, TAU_A)
    b = rng.normal(MU_B, TAU_B)
    b = max(b, 0.004)
    T = (Y_FAIL - a) / b
    return a, b, T


def observe(a, b, t_now, rng):
    """Noisy log-health observations from 0..t_now."""
    ts = np.arange(0.0, t_now + 1e-9, DT)
    if len(ts) < 2:
        ts = np.array([0.0, t_now])
    y = a + b * ts + rng.normal(0, SIGMA, len(ts))
    return ts, y


def fit_prior(n_train=120, seed=0):
    """Empirical-Bayes prior on theta=[a,b] from training run-to-failure
    units (population mean + covariance of the fitted parameters)."""
    rng = np.random.default_rng(seed)
    thetas = []
    for _ in range(n_train):
        a, b, T = simulate_unit(rng)
        ts, y = observe(a, b, T, rng)
        X = np.column_stack([np.ones_like(ts), ts])
        theta = np.linalg.lstsq(X, y, rcond=None)[0]
        thetas.append(theta)
    thetas = np.array(thetas)
    m0 = thetas.mean(0)
    S0 = np.cov(thetas.T) + np.eye(2) * 1e-6
    return m0, S0


def bayes_rul(ts, y, t_now, m0, S0, n_mc=4000, rng=None):
    """Conjugate Bayesian linear regression posterior over (a,b), then MC
    propagate to the RUL posterior. Returns (median, lo5, hi95, samples)."""
    if rng is None:
        rng = np.random.default_rng(0)
    X = np.column_stack([np.ones_like(ts), ts])
    S0i = np.linalg.inv(S0)
    Sn = np.linalg.inv(S0i + X.T @ X / SIGMA ** 2)
    mn = Sn @ (S0i @ m0 + X.T @ y / SIGMA ** 2)
    th = rng.multivariate_normal(mn, Sn, n_mc)
    a_s, b_s = th[:, 0], th[:, 1]
    ok = b_s > 1e-4
    T = (Y_FAIL - a_s[ok]) / b_s[ok]
    rul = T - t_now
    rul = rul[rul > -5]                       # drop degenerate draws
    med = float(np.median(rul))
    lo, hi = np.percentile(rul, [5, 95])
    return med, float(lo), float(hi), rul


def naive_rul(ts, y, t_now):
    """Least-squares point RUL — accurate-ish, but no uncertainty."""
    X = np.column_stack([np.ones_like(ts), ts])
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    b = max(b, 1e-4)
    return (Y_FAIL - a) / b - t_now


def run():
    m0, S0 = fit_prior()
    rng = np.random.default_rng(7)

    fracs = [0.4, 0.6, 0.8]
    per_frac = {}
    n_test = 240
    for f in fracs:
        cov = 0; widths = []; b_err = []; n_err = []; n_ok = 0
        for _ in range(n_test):
            a, b, T = simulate_unit(rng)
            t_now = f * T
            ts, y = observe(a, b, t_now, rng)
            true_rul = T - t_now
            med, lo, hi, _ = bayes_rul(ts, y, t_now, m0, S0, rng=rng)
            nv = naive_rul(ts, y, t_now)
            if lo <= true_rul <= hi:
                cov += 1
            widths.append(hi - lo)
            b_err.append((med - true_rul) ** 2)
            n_err.append((nv - true_rul) ** 2)
            n_ok += 1
        per_frac[f"life_{int(f*100)}pct"] = {
            "coverage_90ci": round(cov / n_ok, 3),
            "mean_ci_width": round(float(np.mean(widths)), 2),
            "bayes_rmse": round(float(np.sqrt(np.mean(b_err))), 2),
            "naive_rmse": round(float(np.sqrt(np.mean(n_err))), 2)}

    return {
        "model": "exponential degradation, conjugate Bayesian linear "
                 "regression + MC threshold crossing",
        "population": {"mu_b": MU_B, "sigma_noise": SIGMA,
                       "fail_threshold_logH": round(float(Y_FAIL), 3)},
        "n_test_per_point": n_test,
        "by_life_fraction": per_frac,
        "headline": {
            "coverage_at_60pct_life": per_frac["life_60pct"]["coverage_90ci"],
            "ci_width_shrinks":
                per_frac["life_40pct"]["mean_ci_width"] >
                per_frac["life_80pct"]["mean_ci_width"]},
    }


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m0, S0 = fit_prior()
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3))

    # 1: degradation trajectories to threshold
    ax = axes[0]
    for _ in range(14):
        a, b, T = simulate_unit(rng)
        ts = np.arange(0, T, DT)
        ax.plot(ts, np.exp(a + b * ts), color="#0e6ba8", alpha=0.5, lw=1)
    ax.axhline(np.exp(Y_FAIL), color="#c0392b", ls="--", lw=1.5)
    ax.annotate("failure threshold", (2, np.exp(Y_FAIL) * 1.03),
                color="#c0392b", fontsize=9)
    ax.set_xlabel("time [steps]"); ax.set_ylabel("health index")
    ax.set_title("Stochastic degradation to failure\n(unit-to-unit spread)",
                 fontsize=10)
    ax.grid(alpha=0.3)

    # 2: one unit's RUL posterior, narrowing with observation
    ax = axes[1]
    a, b, T = simulate_unit(np.random.default_rng(11))
    cols = ["#f0a500", "#e0670a", "#8a2be2"]
    for f, c in zip((0.4, 0.6, 0.8), cols):
        t_now = f * T
        ts, y = observe(a, b, t_now, np.random.default_rng(11))
        _, lo, hi, rul = bayes_rul(ts, y, t_now, m0, S0,
                                   rng=np.random.default_rng(5))
        ax.hist(rul, bins=40, density=True, color=c, alpha=0.5,
                label=f"observed {int(f*100)}% of life")
    ax.axvline(0, color="0.4", lw=1)
    ax.axvline(T - 0.8 * T, color="#8a2be2", ls=":", lw=1)
    ax.set_xlabel("remaining useful life [steps]")
    ax.set_ylabel("posterior density")
    ax.set_title("RUL posterior sharpens as\nfailure approaches", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.set_xlim(-5, None)

    # 3: calibration + width
    ax = axes[2]
    fr = [40, 60, 80]
    cov = [res["by_life_fraction"][f"life_{f}pct"]["coverage_90ci"] for f in fr]
    wid = [res["by_life_fraction"][f"life_{f}pct"]["mean_ci_width"] for f in fr]
    ax.plot(fr, cov, "-o", color="#1a7a4a", label="90% CI coverage")
    ax.axhline(0.90, color="0.5", ls="--", lw=1)
    ax.set_ylim(0.7, 1.0)
    ax.set_xlabel("% of life observed")
    ax.set_ylabel("empirical coverage", color="#1a7a4a")
    ax.set_title("Calibrated: ~90% coverage,\ninterval width shrinks",
                 fontsize=10)
    ax2 = ax.twinx()
    ax2.plot(fr, wid, "-s", color="#0e6ba8", label="CI width")
    ax2.set_ylabel("90% CI width [steps]", color="#0e6ba8")
    ax.grid(alpha=0.3)

    fig.suptitle("Infrastructure RUL prognostics with calibrated Bayesian "
                 "uncertainty", fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=115)
    plt.close(fig)


def main():
    res = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "hitachi_rul_uq.png"), res)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
