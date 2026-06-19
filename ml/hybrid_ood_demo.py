"""
ml/hybrid_ood_demo.py — out-of-distribution material generalization on the
hybrid 2nm dicing dataset.

Trains regressors to predict a dicing KPI (die strength / chipping) from the
RECIPE + GEOMETRY + MATERIAL-DESCRIPTOR features, then tests generalization to a
HELD-OUT material (leave-one-material-out). Because each material carries its
fracture/thermal descriptors (K_Ic, H, E, k, σ_flex) as features, the model can
in principle extrapolate to an unseen material — the OOD test measures whether it
actually does.

Models: RandomForest (fast, full data) and a Gaussian Process (subsampled — GP is
O(n³)). Reports R²/MAE for an in-domain random split vs the OOD material split.

Run:
    python ml/hybrid_ood_demo.py                      # die strength, hold out Ga2O3
    python ml/hybrid_ood_demo.py --target total_chipping_um --holdout GaN --plot
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sic.hybrid.dataset import (                       # noqa: E402
    FEATURE_COLS, GEOMETRY_COLS, MATERIAL_COLS, build_dataset, write_dataset,
)

from sklearn.ensemble import RandomForestRegressor      # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel  # noqa: E402
from sklearn.preprocessing import StandardScaler        # noqa: E402
from sklearn.metrics import r2_score, mean_absolute_error  # noqa: E402
from sklearn.model_selection import train_test_split    # noqa: E402

CSV = os.path.join(ROOT, "data", "hybrid_dicing", "hybrid_2nm_dataset.csv")
INPUT_COLS = FEATURE_COLS + GEOMETRY_COLS + MATERIAL_COLS


def load_or_build(csv: str = CSV, n: int = 150, seed: int = 0) -> pd.DataFrame:
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        if set(MATERIAL_COLS).issubset(df.columns) and "street_width_um" in df.columns:
            return df
        print("  (existing CSV lacks new columns → regenerating)")
    from sic.hybrid import HybridDicer, Constraints2nm
    df = build_dataset(n_per_cell=n, seed=seed,
                       dicer=HybridDicer(constraints=Constraints2nm(strength_min_frac=0.6)))
    write_dataset(df, os.path.dirname(csv))
    return df


def _design(df: pd.DataFrame) -> pd.DataFrame:
    """Input design matrix: numeric features + one-hot route."""
    X = df[INPUT_COLS].copy()
    routes = pd.get_dummies(df["route"], prefix="route").astype(float)
    return pd.concat([X, routes], axis=1)


def _gp():
    kern = (ConstantKernel(1.0) * RBF(length_scale=np.ones(1))
            + WhiteKernel(noise_level=1.0))
    return GaussianProcessRegressor(kernel=kern, normalize_y=True,
                                    n_restarts_optimizer=0, alpha=1e-6)


def _fit_eval(Xtr, ytr, Xte, yte, model, scale=False, gp_cap=600, seed=0):
    if scale:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if isinstance(model, GaussianProcessRegressor) and len(Xtr) > gp_cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(Xtr), gp_cap, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return r2_score(yte, pred), mean_absolute_error(yte, pred), pred


def run(df: pd.DataFrame, target: str, holdout: str, seed: int = 0):
    Xall = _design(df).values.astype(float)
    yall = df[target].values.astype(float)
    mat = df["material"].values

    # ── in-domain: random 75/25 split ──
    Xtr, Xte, ytr, yte = train_test_split(Xall, yall, test_size=0.25, random_state=seed)
    # ── OOD: leave-one-material-out ──
    oo = mat == holdout
    Xtr_o, ytr_o = Xall[~oo], yall[~oo]
    Xte_o, yte_o = Xall[oo], yall[oo]

    # die strength is σ_flex-capped → near-constant within a material; the
    # holdout set then has negligible spread vs the overall target and R²
    # becomes unstable (tiny SS_tot). Flag and report MAE instead.
    degenerate = np.std(yte_o) < 0.05 * (np.std(yall) + 1e-12)

    results = {}
    for name, mk, scale in [("RandomForest",
                             lambda: RandomForestRegressor(
                                 n_estimators=300, max_depth=None,
                                 random_state=seed, n_jobs=-1), False),
                            ("GaussianProcess", _gp, True)]:
        r2_id, mae_id, _ = _fit_eval(Xtr, ytr, Xte, yte, mk(), scale, seed=seed)
        r2_od, mae_od, pred_od = _fit_eval(Xtr_o, ytr_o, Xte_o, yte_o, mk(), scale, seed=seed)
        if degenerate:
            r2_od = float("nan")     # R² meaningless when held-out target is constant
        results[name] = dict(r2_id=r2_id, mae_id=mae_id, r2_od=r2_od,
                             mae_od=mae_od, pred_od=pred_od, true_od=yte_o,
                             degenerate=degenerate)
    return results


def plot(results, target, holdout, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    colors = {"RandomForest": "#1b7f79", "GaussianProcess": "#b5374a"}

    for name, r in results.items():
        axL.scatter(r["true_od"], r["pred_od"], s=14, alpha=.45,
                    color=colors[name], label=f"{name}  R²={r['r2_od']:.2f}")
    lo = min(min(r["true_od"].min(), r["pred_od"].min()) for r in results.values())
    hi = max(max(r["true_od"].max(), r["pred_od"].max()) for r in results.values())
    axL.plot([lo, hi], [lo, hi], "k--", lw=1)
    axL.set(xlabel=f"true {target}", ylabel=f"predicted {target}",
            title=f"(a) OOD prediction on held-out '{holdout}'")
    axL.legend(frameon=False, fontsize=9)
    axL.grid(alpha=.3)

    names = list(results)
    x = np.arange(len(names)); w = 0.38
    axR.bar(x - w/2, [results[n]["r2_id"] for n in names], w, label="in-domain (random)",
            color="#9ecae1", edgecolor="k")
    axR.bar(x + w/2, [results[n]["r2_od"] for n in names], w, label=f"OOD ({holdout})",
            color="#fc9272", edgecolor="k")
    axR.set_xticks(x); axR.set_xticklabels(names)
    axR.axhline(0, color="k", lw=.8)
    axR.set(ylabel="R²", title="(b) in-domain vs OOD generalization")
    axR.legend(frameon=False, fontsize=9); axR.grid(axis="y", alpha=.3)

    fig.suptitle(f"Hybrid 2nm dicing — OOD material generalization  (target: {target})",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description="OOD material generalization demo")
    ap.add_argument("--target", default="total_chipping_um",
                    help="total_chipping_um | residual_stress_MPa | throughput_wph | "
                         "die_strength_MPa (note: ~constant per material → degenerate OOD R²)")
    ap.add_argument("--holdout", default="Ga2O3", help="held-out material")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    df = load_or_build()
    print(f"dataset: {len(df)} rows, materials={sorted(df.material.unique())}")
    print(f"target={args.target}  holdout={args.holdout}\n")
    res = run(df, args.target, args.holdout, args.seed)

    if next(iter(res.values()))["degenerate"]:
        print(f"  ⚠ '{args.target}' is ~constant within '{args.holdout}' "
              "(σ_flex-capped) → OOD R² undefined; read MAE instead.\n")
    print(f"  {'model':16}{'R² in-dom':>11}{'MAE in-dom':>12}{'R² OOD':>9}{'MAE OOD':>10}")
    for name, r in res.items():
        r2od = "  n/a" if r["r2_od"] != r["r2_od"] else f"{r['r2_od']:>9.3f}"
        print(f"  {name:16}{r['r2_id']:>11.3f}{r['mae_id']:>12.3f}"
              f"{r2od}{r['mae_od']:>10.3f}")

    if args.plot:
        out = plot(res, args.target, args.holdout,
                   os.path.join(ROOT, "results", "hybrid_2nm_ood.png"))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
