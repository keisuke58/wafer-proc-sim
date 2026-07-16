#!/usr/bin/env python3
"""blade_rul.py — a dicing-blade predictive-maintenance digital twin: wear
physics + acoustic-emission health + ML remaining-useful-life + the
replace-at-the-right-time economics.

The blade is DISCO's razor-blade business, and replacing it at the RIGHT
moment is worth real money: too early wastes a good blade, too late lets
chipping scrap dies. This ties three ideas into one twin —

  WEAR PHYSICS   the diamond blade wears as it cuts: exposed diameter shrinks,
      the edge dulls, so cutting force and edge chipping rise with cut count.
      End-of-life (EOL) is the cut count at which chipping crosses the die
      spec.

  ACOUSTIC HEALTH a dulling blade radiates a rising acoustic-emission (AE)
      signature; the twin models AE-feature drift vs wear (with sensor noise),
      the observable a real tool actually has.

  ML RUL         a regressor learns (AE feature, cut count) -> remaining
      useful life. Trained on many noisy simulated blade lives, held-out
      accuracy (R^2 / MAE) is reported — the surrogate that lets the tool
      predict RUL online instead of running to failure.

  R2R ECONOMICS  a cost-per-die curve over the replace-at-cut-count decision:
      blade cost amortized + chipping-scrap yield loss. The optimum is the
      run-to-run replacement point; the twin quantifies the savings vs
      fixed-interval or run-to-failure.

Pure numpy + scikit-learn + matplotlib.  Run:  python3 disco/blade_rul.py
"""
import json
import os

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "blade_rul_results.json")

# ── blade wear / chipping model ─────────────────────────────────────────────
CHIP0_UM = 3.0           # fresh-blade chipping [um]
CHIP_SPEC_UM = 8.0       # die-edge chipping spec (EOL threshold)
WEAR_P = 2.5             # super-linear dulling exponent
BLADE_COST = 120.0       # one blade [arb. $]
SCRAP_COST_PER_DIE = 0.8  # cost of one chipped/scrapped die
DIES_PER_CUT = 40        # dies produced per cut pass
LIFE0 = 6000             # nominal cuts to EOL for a nominal blade


def chipping_vs_cuts(cuts, life=LIFE0):
    """Edge chipping [um] vs cut count: fresh value rising super-linearly as
    the blade dulls toward end-of-life."""
    u = np.asarray(cuts, float) / life                 # normalized wear
    # super-linear dulling: CHIP0 at u=0, exactly SPEC at u=1, rises beyond
    return CHIP0_UM + (CHIP_SPEC_UM - CHIP0_UM) * u ** WEAR_P


def eol_cuts(life=LIFE0):
    """Cut count where chipping first crosses spec."""
    c = np.arange(1, int(life * 1.3))
    ch = chipping_vs_cuts(c, life)
    idx = np.where(ch >= CHIP_SPEC_UM)[0]
    return int(c[idx[0]]) if idx.size else int(life)


def ae_feature(cuts, life=LIFE0, rng=None):
    """Acoustic-emission RMS feature vs wear (rises as blade dulls) + sensor
    noise — the online observable."""
    u = np.asarray(cuts, float) / life
    base = 1.0 + 1.8 * u + 0.9 * u ** 2
    if rng is not None:
        base = base + rng.normal(0, 0.06, np.shape(base))
    return base


def simulate_life(life, rng):
    """One blade life: cut grid -> (features, RUL labels)."""
    eol = eol_cuts(life)
    cuts = np.linspace(1, eol, 40)
    ae = ae_feature(cuts, life, rng)
    rul = eol - cuts                                   # remaining useful life
    X = np.column_stack([ae, cuts])
    return X, rul, eol


def build_dataset(n_blades=60, seed=0):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for _ in range(n_blades):
        life = LIFE0 * rng.uniform(0.8, 1.25)          # blade-to-blade spread
        X, y, _ = simulate_life(life, rng)
        Xs.append(X); ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def train_rul(Xtr, ytr):
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(n_estimators=120, max_depth=9, random_state=0)
    m.fit(Xtr, ytr)
    return m


def cost_per_die(replace_at, life=LIFE0):
    """R2R economics: amortized blade cost + chipping-scrap yield loss for a
    policy that replaces the blade at cut-count `replace_at`."""
    replace_at = int(np.clip(replace_at, 1, int(life * 1.3)))
    cuts = np.arange(1, replace_at + 1)
    ch = chipping_vs_cuts(cuts, life)
    # scrap fraction rises once chipping exceeds spec (soft ramp)
    scrap_frac = 1.0 / (1.0 + np.exp(-(ch - CHIP_SPEC_UM) / 0.4))
    dies = replace_at * DIES_PER_CUT
    scrap = float(np.sum(scrap_frac) * DIES_PER_CUT)
    cost = BLADE_COST + scrap * SCRAP_COST_PER_DIE
    good = max(dies - scrap, 1.0)
    return cost / good


def run():
    eol = eol_cuts()
    X, y = build_dataset()
    ntr = int(len(X) * 0.8)
    idx = np.random.default_rng(1).permutation(len(X))
    tr, te = idx[:ntr], idx[ntr:]
    model = train_rul(X[tr], y[tr])
    pred = model.predict(X[te])
    ss = np.sum((y[te] - pred) ** 2)
    tot = np.sum((y[te] - y[te].mean()) ** 2)
    r2 = 1.0 - ss / tot
    mae = float(np.mean(np.abs(y[te] - pred)))

    # replacement-policy economics
    grid = np.arange(int(eol * 0.5), int(eol * 1.25), 50)
    costs = np.array([cost_per_die(r) for r in grid])
    opt_i = int(np.argmin(costs))
    opt_at = int(grid[opt_i])
    c_opt = float(costs[opt_i])
    c_fail = cost_per_die(int(eol * 1.2))       # run-to-failure
    c_early = cost_per_die(int(eol * 0.6))      # replace too early

    return {
        "eol_cuts": eol,
        "chipping": {"fresh_um": CHIP0_UM, "spec_um": CHIP_SPEC_UM},
        "rul_model": {"model": "RandomForest(120)", "features": ["AE_rms",
                      "cut_count"], "test_R2": round(float(r2), 4),
                      "test_MAE_cuts": round(mae, 1)},
        "replacement_economics": {
            "optimal_replace_at_cuts": opt_at,
            "optimal_cost_per_good_die": round(c_opt, 4),
            "run_to_failure_cost": round(c_fail, 4),
            "replace_too_early_cost": round(c_early, 4),
            "savings_vs_failure_pct": round(100 * (c_fail - c_opt) / c_fail, 2),
            "savings_vs_early_pct": round(100 * (c_early - c_opt) / c_early,
                                          2)},
    }


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))
    eol = res["eol_cuts"]

    # 1: chipping + AE vs cuts
    ax = axes[0]
    cuts = np.linspace(1, eol * 1.15, 200)
    ax.plot(cuts, chipping_vs_cuts(cuts), color="#c0392b", lw=2,
            label="edge chipping [um]")
    ax.axhline(CHIP_SPEC_UM, color="0.5", ls=":", lw=1)
    ax.axvline(eol, color="#c0392b", ls="--", lw=1)
    ax.annotate("EOL", (eol, CHIP0_UM), fontsize=9, color="#c0392b")
    ax2 = ax.twinx()
    ax2.plot(cuts, ae_feature(cuts), color="#1f6f8b", lw=1.6, alpha=0.9,
             label="AE RMS")
    ax.set_xlabel("cut count"); ax.set_ylabel("chipping [um]")
    ax2.set_ylabel("AE RMS [arb.]", color="#1f6f8b")
    ax.set_title("Blade wear: chipping & AE\nrise toward end-of-life",
                 fontsize=10)
    ax.grid(alpha=0.3)

    # 2: RUL parity
    ax = axes[1]
    X, y = build_dataset(seed=0)
    ntr = int(len(X) * 0.8)
    idx = np.random.default_rng(1).permutation(len(X))
    m = train_rul(X[idx[:ntr]], y[idx[:ntr]])
    pr = m.predict(X[idx[ntr:]])
    ax.scatter(y[idx[ntr:]], pr, s=10, alpha=0.5, color="#6a1b9a")
    lo, hi = y.min(), y.max()
    ax.plot([lo, hi], [lo, hi], "--", color="0.5", lw=1)
    r2 = res["rul_model"]["test_R2"]
    ax.set_xlabel("true RUL [cuts]"); ax.set_ylabel("predicted RUL [cuts]")
    ax.set_title(f"ML remaining-useful-life\nR^2={r2:.2f} from AE + cut count",
                 fontsize=10)
    ax.grid(alpha=0.3)

    # 3: cost vs replacement point
    ax = axes[2]
    grid = np.arange(int(eol * 0.5), int(eol * 1.25), 50)
    costs = [cost_per_die(r) for r in grid]
    ax.plot(grid, costs, color="#12a150", lw=2)
    re = res["replacement_economics"]
    ax.axvline(re["optimal_replace_at_cuts"], color="#12a150", ls="--", lw=1)
    ax.axvline(eol, color="0.5", ls=":", lw=1)
    ax.plot(re["optimal_replace_at_cuts"], re["optimal_cost_per_good_die"],
            "o", color="#c0392b", ms=7)
    ax.set_xlabel("replace blade at cut count")
    ax.set_ylabel("cost per good die [arb.]")
    ax.set_title(f"Replace at the right time:\n-{re['savings_vs_failure_pct']:.0f}% "
                 "vs run-to-failure", fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    res = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "disco_blade_rul.png"), res)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
