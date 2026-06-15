"""
Fast image triage + physics metrology — the complementary-fusion system.

Realises the pitch from docs/intelligent_inspection.md as an actual two-stage
decision pipeline and quantifies it:

  STAGE 1 (fast, every die): the kerf CNN grades the image and emits a softmax
          confidence. Confident calls are accepted as-is (line speed).
  STAGE 2 (slow, few dies): low-confidence dies are routed to physics metrology,
          which measures BOTH chipping AND subsurface crack depth and returns
          the correct grade.

Why fusion beats either alone — the honest core:
  An optical image encodes *chipping* but NOT *subsurface crack depth*. So a die
  can look clean (image → accept) yet be a true reject because of a deep
  subsurface crack the camera cannot see. A pure-image system lets those
  "subsurface escapes" through. Routing the uncertain fraction to metrology
  recovers them — at a fraction of the metrology cost of inspecting everything.

We sweep the confidence threshold τ and report, at each τ:
  - fast-path fraction (throughput; 1 = no metrology, 0 = metrology-all)
  - combined reject-recall (rejects caught by CNN-confident or by metrology)
  - subsurface-escape rate (crack-only rejects that slipped through)

Usage
-----
    python vision/inspection_triage.py            # full (~40-70s CPU)
    python vision/inspection_triage.py --quick     # fast smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision.intelligent_inspection import (
    GRADE_B_MAX,  # 5.0 µm chipping reject threshold
    PROCESSES,
    build_dataset,
)

CRACK_REJECT_UM = 2.0  # INSPECTION_SPEC crack_um_max — subsurface reject limit


def sample_crack_um(procs: np.ndarray, seed: int = 0) -> np.ndarray:
    """Per-die subsurface crack depth [µm] from the process crack mean × noise.

    Crack depth is *not* encoded in the kerf image — it is the metrology-only
    signal. blade (3.0) and laser_ns (2.0) processes produce crack-driven
    rejects invisible to the camera.
    """
    from ml.lasertec_inspection import PROCESS_PARAMS
    rng = np.random.default_rng(seed)
    means = np.array([PROCESS_PARAMS[PROCESSES[p]]["crack"] for p in procs])
    # lognormal multiplicative noise (median = mean), clipped non-negative
    return np.clip(means * rng.lognormal(0.0, 0.35, size=len(procs)), 0.0, None)


# ── vision arm with softmax confidence ───────────────────────────────────────

def train_with_confidence(
    X_tr, y_tr, X_te, epochs: int = 15, seed: int = 0,
    pretrained: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train the CNN on chipping grade; return (pred grade, max softmax) on test."""
    import torch
    import torch.nn.functional as F

    from vision.wafer_backbone import WaferClassifier, train_classifier

    torch.manual_seed(seed)
    clf = WaferClassifier(n_classes=3)
    if pretrained and os.path.exists(pretrained):
        clf.backbone.load(pretrained, map_location="cpu")
        print(f"    warm-started backbone from {pretrained}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_tr))
    n_val = max(1, int(0.15 * len(X_tr)))
    va, tr = perm[:n_val], perm[n_val:]
    to_t = lambda A: torch.from_numpy(A[:, None, :, :] / 255.0).float()  # noqa: E731

    train_classifier(clf, to_t(X_tr[tr]), torch.from_numpy(y_tr[tr]),
                     to_t(X_tr[va]), torch.from_numpy(y_tr[va]),
                     epochs=epochs, batch_size=64, lr=2e-3, device="cpu",
                     verbose=True)
    clf.eval()
    with torch.no_grad():
        probs = F.softmax(clf(to_t(X_te)), dim=1).numpy()
    return probs.argmax(1), probs.max(1)


# ── triage simulation ─────────────────────────────────────────────────────────

def simulate_triage(cnn_pred: np.ndarray, cnn_conf: np.ndarray,
                    true_reject: np.ndarray, tau: float) -> Dict[str, float]:
    """
    Two-stage decision at confidence threshold ``tau``.

    Fast path (conf ≥ τ): flag reject iff CNN grade == C (2).
    Slow path (conf < τ): metrology returns the correct reject decision.
    """
    routed = cnn_conf < tau                       # → metrology
    fast = ~routed
    # final reject decision per die
    flagged = np.where(routed, true_reject, cnn_pred == 2)

    n_rej = int(true_reject.sum())
    caught = int((flagged & true_reject).sum())
    escapes = true_reject & ~flagged              # missed rejects
    return {
        "tau": float(tau),
        "fast_fraction": float(fast.mean()),
        "routed_fraction": float(routed.mean()),
        "reject_recall": (caught / n_rej) if n_rej else 1.0,
        "escape_rate": float(escapes.sum() / max(1, n_rej)),
        "n_escapes": int(escapes.sum()),
    }


# ── figure ───────────────────────────────────────────────────────────────────

def plot_triage(sweep, op, baselines, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    taus = [s["tau"] for s in sweep]
    fast = [s["fast_fraction"] for s in sweep]
    recall = [s["reject_recall"] for s in sweep]
    escape = [s["escape_rate"] for s in sweep]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))

    # (a) tradeoff vs threshold
    a = ax[0]
    a.plot(taus, fast, "o-", color="tab:blue", label="fast-path fraction (throughput)")
    a.plot(taus, recall, "s-", color="tab:green", label="combined reject-recall")
    a.plot(taus, escape, "^-", color="tab:red", label="subsurface-escape rate")
    a.axvline(op["tau"], color="k", ls=":", lw=1.2,
              label=f"operating τ={op['tau']:.2f}")
    a.set_xlabel("CNN confidence threshold τ")
    a.set_ylabel("fraction")
    a.set_title("検査トリアージ — throughput / recall / escape vs τ", fontsize=11)
    a.legend(fontsize=8); a.grid(alpha=0.2); a.set_ylim(-0.02, 1.02)

    # (b) bar comparison: CNN-only vs fusion vs metrology-all
    b = ax[1]
    names = ["CNN only\n(image)", f"Fusion\n(τ={op['tau']:.2f})", "Metrology\nall"]
    fast_b = [baselines["cnn_only"]["fast_fraction"], op["fast_fraction"],
              baselines["metrology_all"]["fast_fraction"]]
    rec_b = [baselines["cnn_only"]["reject_recall"], op["reject_recall"],
             baselines["metrology_all"]["reject_recall"]]
    esc_b = [baselines["cnn_only"]["escape_rate"], op["escape_rate"],
             baselines["metrology_all"]["escape_rate"]]
    x = np.arange(3); w = 0.25
    b.bar(x - w, fast_b, w, label="fast fraction", color="tab:blue")
    b.bar(x, rec_b, w, label="reject-recall", color="tab:green")
    b.bar(x + w, esc_b, w, label="escape rate", color="tab:red")
    b.set_xticks(x); b.set_xticklabels(names, fontsize=9)
    b.set_title("画像のみ vs 併用 vs 全数計測", fontsize=11)
    b.legend(fontsize=8); b.set_ylim(0, 1.05)
    for xi, v in zip(x - w, fast_b):
        b.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    for xi, v in zip(x + w, esc_b):
        b.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)

    fig.suptitle("高速な画像トリアージ + 物理計測の併用 — fast image triage + physics metrology",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure saved -> {out_path}")


# ── demo entry point ─────────────────────────────────────────────────────────

def run_demo(n_per_process: int = 150, epochs: int = 15, seed: int = 0,
             escape_budget: float = 0.01, pretrained: Optional[str] = None,
             out_path: Optional[str] = None) -> Dict[str, object]:
    from sklearn.model_selection import train_test_split

    t0 = time.time()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = out_path or os.path.join(repo_root, "results",
                                        "inspection_triage.png")

    print(f"[1/4] Building dies + subsurface cracks "
          f"({len(PROCESSES)} x {n_per_process}) ...")
    X, y_chip, chips, procs = build_dataset(n_per_process=n_per_process, seed=seed)
    cracks = sample_crack_um(procs, seed=seed)
    true_reject = (chips >= GRADE_B_MAX) | (cracks >= CRACK_REJECT_UM)
    crack_only = (chips < GRADE_B_MAX) & (cracks >= CRACK_REJECT_UM)
    print(f"      {len(X)} dies | true rejects {int(true_reject.sum())} "
          f"(of which {int(crack_only.sum())} crack-only = image-invisible)")

    idx = np.arange(len(X))
    tr_i, te_i = train_test_split(idx, test_size=0.30, stratify=y_chip,
                                  random_state=seed)
    print("[2/4] Training vision arm (CNN) with confidence ...")
    cnn_pred, cnn_conf = train_with_confidence(
        X[tr_i], y_chip[tr_i], X[te_i], epochs=epochs, seed=seed,
        pretrained=pretrained)
    tr_te = true_reject[te_i]

    print("[3/4] Sweeping confidence threshold τ ...")
    sweep = [simulate_triage(cnn_pred, cnn_conf, tr_te, t)
             for t in np.linspace(0.0, 1.0, 21)]
    cnn_only = simulate_triage(cnn_pred, cnn_conf, tr_te, 0.0)       # never route
    metro_all = simulate_triage(cnn_pred, cnn_conf, tr_te, 1.01)     # route all
    # operating point: smallest τ whose escape ≤ budget (max throughput within budget)
    feasible = [s for s in sweep if s["escape_rate"] <= escape_budget]
    op = min(feasible, key=lambda s: s["routed_fraction"]) if feasible else metro_all

    print(f"      CNN-only : fast={cnn_only['fast_fraction']:.2f} "
          f"recall={cnn_only['reject_recall']:.2f} escape={cnn_only['escape_rate']:.3f}")
    print(f"      Fusion   : τ={op['tau']:.2f} fast={op['fast_fraction']:.2f} "
          f"recall={op['reject_recall']:.2f} escape={op['escape_rate']:.3f} "
          f"(metrology load {op['routed_fraction']*100:.0f}%)")
    print(f"      Metro-all: fast={metro_all['fast_fraction']:.2f} "
          f"recall={metro_all['reject_recall']:.2f} escape={metro_all['escape_rate']:.3f}")

    print("[4/4] Plotting ...")
    baselines = {"cnn_only": cnn_only, "metrology_all": metro_all}
    plot_triage(sweep, op, baselines, out_path)

    stats = {"operating_point": op, "cnn_only": cnn_only,
             "metrology_all": metro_all, "escape_budget": escape_budget,
             "n_true_reject": int(true_reject.sum()),
             "n_crack_only": int(crack_only.sum())}
    with open(os.path.join(os.path.dirname(out_path),
                           "inspection_triage_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=float)
    print(f"Done in {time.time() - t0:.1f}s")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Fast image triage + metrology fusion")
    p.add_argument("--n-per-process", type=int, default=150)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--escape-budget", type=float, default=0.01,
                   help="max tolerated subsurface-escape rate at the operating point")
    p.add_argument("--pretrained", type=str, default=None)
    p.add_argument("--quick", action="store_true", help="fast smoke run")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    if args.quick:
        args.n_per_process, args.epochs = 60, 8
    run_demo(n_per_process=args.n_per_process, epochs=args.epochs, seed=args.seed,
             escape_budget=args.escape_budget, pretrained=args.pretrained,
             out_path=args.out)


if __name__ == "__main__":
    main()
