"""
Multi-seed transfer evaluation — does the WM-811K-pretrained backbone help
the kerf chipping-grade CNN, beyond single-seed noise?

For each seed: build the synthetic kerf dataset, train the 3-class CNN twice
(cold = random init, warm = WM-811K backbone), and record the multi-metric
test scores.  Reports mean ± SD per condition plus the paired (warm − cold)
delta, which is the honest effect size since both share the same data split.

Usage
-----
    python vision/transfer_eval_multiseed.py                       # seeds 0-4
    python vision/transfer_eval_multiseed.py --seeds 0 1 2 3 4 5 6
    python vision/transfer_eval_multiseed.py --backbone results/wm811k_backbone.pt
Output
------
    results/transfer_multiseed.csv
    results/transfer_multiseed.png
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision.kerf_quality_classifier import (  # noqa: E402
    make_grade_dataset, train_cnn,
)

METRICS = ["accuracy", "macro_f1", "f1_grade_a", "f1_grade_b",
           "f1_grade_c", "defect_f1", "detection_recall", "false_positive_rate"]


def run_one(seed: int, epochs: int, n_per_class: int, backbone: str | None):
    """Train cold + warm on the same split; return (cold_metrics, warm_metrics)."""
    from sklearn.model_selection import train_test_split
    X, y = make_grade_dataset(n_per_class=n_per_class, H=128, W=128, seed=seed)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed)
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        _, m_cold, _ = train_cnn(Xtr, ytr, Xte, yte, epochs=epochs, seed=seed)
        _, m_warm, _ = train_cnn(Xtr, ytr, Xte, yte, epochs=epochs, seed=seed,
                                 pretrained=backbone)
    return m_cold, m_warm


def main():
    ap = argparse.ArgumentParser(description="Multi-seed WM-811K transfer eval")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-per-class", type=int, default=300)
    ap.add_argument("--backbone", default=os.path.join(
        os.path.dirname(__file__), "..", "results", "wm811k_backbone.pt"))
    args = ap.parse_args()

    if not os.path.exists(args.backbone):
        sys.exit(f"backbone not found: {args.backbone}\n"
                 "run: python vision/pretrain_wm811k.py --trust")

    cold_rows, warm_rows = [], []
    for s in args.seeds:
        print(f"[seed {s}] training cold + warm ...", flush=True)
        mc, mw = run_one(s, args.epochs, args.n_per_class, args.backbone)
        cold_rows.append([mc[k] for k in METRICS])
        warm_rows.append([mw[k] for k in METRICS])
        print(f"          acc cold={mc['accuracy']:.3f}  warm={mw['accuracy']:.3f}"
              f"  (Δ={mw['accuracy']-mc['accuracy']:+.3f})", flush=True)

    cold = np.array(cold_rows)   # (n_seeds, n_metrics)
    warm = np.array(warm_rows)
    delta = warm - cold

    # ── report ───────────────────────────────────────────────────────────────
    n = len(args.seeds)
    print(f"\n=== Transfer eval over {n} seeds {args.seeds} "
          f"(mean ± SD) ===")
    print(f"  {'metric':>20}  {'COLD':>14}  {'WARM':>14}  {'Δ(warm-cold)':>16}")
    for j, k in enumerate(METRICS):
        c_m, c_s = cold[:, j].mean(), cold[:, j].std(ddof=1)
        w_m, w_s = warm[:, j].mean(), warm[:, j].std(ddof=1)
        d_m, d_s = delta[:, j].mean(), delta[:, j].std(ddof=1)
        # paired t-stat (cheap effect-size sanity, not a formal claim)
        t = d_m / (d_s / np.sqrt(n)) if d_s > 1e-9 else float("nan")
        flag = "  *" if abs(t) >= 2.776 else ""  # |t|>t_{.975,df=4}
        print(f"  {k:>20}  {c_m:.3f}±{c_s:.3f}  {w_m:.3f}±{w_s:.3f}  "
              f"{d_m:+.3f}±{d_s:.3f} (t={t:+.2f}){flag}")
    print("  * = paired |t| exceeds 2.776 (two-sided .95, df=4) — suggestive only")

    # ── save csv ───────────────────────────────────────────────────────────────
    import csv
    out_csv = os.path.join(os.path.dirname(args.backbone), "transfer_multiseed.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "condition"] + METRICS)
        for i, s in enumerate(args.seeds):
            w.writerow([s, "cold"] + list(cold[i]))
            w.writerow([s, "warm"] + list(warm[i]))
    print(f"\n  csv → {out_csv}")

    # ── figure: paired deltas ──────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(METRICS))
        ax.bar(x, delta.mean(0), yerr=delta.std(0, ddof=1), capsize=4,
               color=["tab:green" if d > 0 else "tab:red" for d in delta.mean(0)])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(METRICS, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Δ (warm − cold)")
        ax.set_title(f"WM-811K transfer effect on kerf CNN — paired Δ over {n} seeds")
        out_png = os.path.join(os.path.dirname(args.backbone), "transfer_multiseed.png")
        fig.tight_layout(); fig.savefig(out_png, dpi=120)
        print(f"  figure → {out_png}")
    except Exception as e:  # noqa: BLE001
        print(f"  [skip figure] {e}")


if __name__ == "__main__":
    main()
