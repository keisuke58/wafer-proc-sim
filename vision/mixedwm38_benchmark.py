"""
MixedWM38 — mixed-type wafer defect benchmark (8-way multilabel) + OOD probe.

MixedWM38 has 38,015 wafer maps (52×52) with an 8-dim multi-hot label over
the basic defect types {Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full,
Scratch, Random}.  Real wafers carry *combinations* of these (29 of the 38
patterns are mixtures), so this is a genuine multilabel problem — harder than
WM-811K single-label and a good stress test for our wafer feature backbone.

Two evaluation modes:
  * ``--split random``  — i.i.d. 80/20 split (standard benchmark)
  * ``--split ood``     — train on PURE single-defect maps only, test on
                          MIXED (multi-defect) maps.  Measures whether features
                          learned on isolated defects generalise to co-occurring
                          defects — the realistic deployment gap.

Optionally warm-start from the WM-811K-pretrained backbone (--init-backbone)
to quantify transfer gain.

Data
----
    data/external/MixedWM38/Wafer_Map_Datasets.npz
    arr_0: (38015, 52, 52) int32 maps {0,1,2};  arr_1: (38015, 8) one-hot/multi-hot

Usage
-----
    python vision/mixedwm38_benchmark.py                       # random split
    python vision/mixedwm38_benchmark.py --split ood           # pure→mixed
    python vision/mixedwm38_benchmark.py --init-backbone results/wm811k_backbone.pt
    python vision/mixedwm38_benchmark.py --quick
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision.wafer_backbone import (  # noqa: E402
    WaferClassifier, to_grayscale_tensor, train_classifier, predict,
)

NPZ_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "external", "MixedWM38",
    "Wafer_Map_Datasets.npz",
)
DEFECT_NAMES = ["Center", "Donut", "Edge-Loc", "Edge-Ring",
                "Loc", "Near-full", "Scratch", "Random"]


def load_mixedwm38(path: str = NPZ_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"MixedWM38 not found at {path}\n"
            "Download (gdown): 1M59pX-lPqL9APBIbp2AKQRTvngeUK8Va"
        )
    d = np.load(path)
    X = d["arr_0"].astype(np.float32)          # (N, 52, 52), values {0,1,2}
    Y = d["arr_1"].astype(np.float32)          # (N, 8) multi-hot
    return X, Y


def evaluate_multilabel(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Multi-metric multilabel eval (micro/macro/per-class F1, exact-match)."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    return {
        "exact_match": float((y_true == y_pred).all(axis=1).mean()),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "per_class_f1": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
    }


def make_split(Y: np.ndarray, kind: str, seed: int = 0):
    """Return (train_idx, test_idx) for 'random' or 'ood' (pure→mixed)."""
    n = len(Y)
    rng = np.random.default_rng(seed)
    if kind == "random":
        perm = rng.permutation(n)
        cut = int(0.8 * n)
        return perm[:cut], perm[cut:]
    if kind == "ood":
        ndef = Y.sum(axis=1)
        pure = np.where(ndef <= 1)[0]    # normal (0) + single-defect (1)
        mixed = np.where(ndef >= 2)[0]   # genuine mixtures
        rng.shuffle(pure)
        return pure, mixed
    raise ValueError(f"unknown split: {kind}")


def main():
    ap = argparse.ArgumentParser(description="MixedWM38 multilabel benchmark")
    ap.add_argument("--npz", default=NPZ_PATH)
    ap.add_argument("--split", choices=["random", "ood"], default="random")
    ap.add_argument("--init-backbone", default=None,
                    help="path to wm811k_backbone.pt for warm start")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="subsample to 6k")
    args = ap.parse_args()

    print(f"[1/4] loading MixedWM38 from {args.npz} ...")
    X, Y = load_mixedwm38(args.npz)
    print(f"      {len(X)} maps, label matrix {Y.shape}")
    if args.quick and len(X) > 6000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), 6000, replace=False)
        X, Y = X[idx], Y[idx]
        print(f"      --quick: subsampled to {len(X)}")

    tr, te = make_split(Y, args.split)
    print(f"[2/4] split={args.split}: {len(tr)} train / {len(te)} test")
    if args.split == "ood":
        print("      (train = pure/single-defect, test = mixed multi-defect)")

    Xt = to_grayscale_tensor(X, size=None, vmax=2.0)
    import torch
    Yt = torch.from_numpy(Y)

    # validation slice from the train pool
    rng = np.random.default_rng(1)
    trp = rng.permutation(tr)
    n_val = int(0.15 * len(trp))
    va, trn = trp[:n_val], trp[n_val:]

    model = WaferClassifier(n_classes=8, multilabel=True)
    if args.init_backbone:
        if os.path.exists(args.init_backbone):
            model.backbone.load(args.init_backbone)
            print(f"      warm-started backbone from {args.init_backbone}")
        else:
            print(f"      [warn] backbone {args.init_backbone} missing — cold start")

    # per-class positive weighting for imbalance
    pos = Y[trn].sum(axis=0)
    neg = len(trn) - pos
    pos_weight = torch.tensor((neg / np.maximum(pos, 1)).astype(np.float32))

    print(f"[3/4] training (multilabel, {len(trn)} train) ...")
    train_classifier(
        model, Xt[trn], Yt[trn], Xt[va], Yt[va],
        epochs=args.epochs, batch_size=256, class_weight=pos_weight,
    )

    print("[4/4] test evaluation")
    y_pred = predict(model, Xt[te])
    m = evaluate_multilabel(Y[te].astype(int), y_pred.astype(int))
    print(f"\n=== MixedWM38 [{args.split}] "
          f"{'(warm)' if args.init_backbone else '(cold)'} ===")
    print(f"  exact-match     : {m['exact_match']:.3f}")
    print(f"  micro-F1        : {m['micro_f1']:.3f}")
    print(f"  macro-F1        : {m['macro_f1']:.3f}")
    print(f"  micro P / R     : {m['micro_precision']:.3f} / {m['micro_recall']:.3f}")
    print("  per-class F1:")
    for name, f1 in zip(DEFECT_NAMES, m["per_class_f1"]):
        print(f"    {name:>10}: {f1:.3f}")


if __name__ == "__main__":
    main()
