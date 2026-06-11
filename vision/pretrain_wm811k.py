"""
Pretrain the shared wafer CNN backbone on WM-811K (real-fab wafer maps).

WM-811K = 811,457 real wafer maps; ~172k are expert-labelled with one of
9 failure patterns (Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full,
Random, Scratch, none).  We train the 9-class classifier, then export the
*convolutional backbone only* so downstream dicing tasks (kerf chipping
grade, MixedWM38 multilabel) can warm-start from real-data features instead
of cold-starting on synthetic images.

Data
----
    data/external/WM-811K/MIR-WM811K/Python/WM811K.pkl   (pandas DataFrame)
    columns: 'waferMap' (2-D int array {0,1,2}), 'failureType', 'trianTestLabel'

SECURITY: WM811K.pkl is a Python pickle from an external mirror — loading it
executes arbitrary code.  Run this only in a trusted environment.  Pass
``--trust`` to acknowledge (the script refuses to unpickle otherwise).

Usage
-----
    python vision/pretrain_wm811k.py --trust                  # full pretrain
    python vision/pretrain_wm811k.py --trust --quick          # 8k-sample smoke
    python vision/pretrain_wm811k.py --trust --epochs 15 --size 64
Output
------
    results/wm811k_backbone.pt   (state_dict of WaferBackbone)
    results/wm811k_pretrain.png  (confusion matrix + learning curve)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision.wafer_backbone import (  # noqa: E402
    WaferClassifier, to_grayscale_tensor, train_classifier, predict,
    DEFAULT_BACKBONE_PATH,
)

PKL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "external", "WM-811K",
    "MIR-WM811K", "Python", "WM811K.pkl",
)
CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
           "Near-full", "Random", "Scratch", "none"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def _flatten_label(v) -> str:
    """failureType cells are stored as str, [], or nested np arrays."""
    while isinstance(v, (list, tuple, np.ndarray)):
        if len(v) == 0:
            return ""
        v = v[0]
    return str(v).strip()


def load_wm811k(path: str = PKL_PATH, trust: bool = False):
    """Return (maps: list[np.ndarray], labels: np.ndarray[int]) for labelled rows."""
    if not trust:
        raise PermissionError(
            "Refusing to unpickle WM811K.pkl without --trust "
            "(pickle executes arbitrary code; source is an external mirror)."
        )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"WM811K.pkl not found at {path}\n"
            "Download: http://mirlab.org/dataset/public/MIR-WM811K.zip"
        )
    import pandas as pd
    df = pd.read_pickle(path)  # noqa: S301 — gated behind --trust above
    labels_raw = df["failureType"].map(_flatten_label)
    keep = labels_raw.isin(CLASS_TO_IDX)
    df = df[keep]
    maps = list(df["waferMap"].values)
    y = labels_raw[keep].map(CLASS_TO_IDX).to_numpy(dtype=np.int64)
    return maps, y


def stack_resized(maps, size: int) -> np.ndarray:
    """Resize each variable-sized wafer map to size×size via the tensor helper."""
    # to_grayscale_tensor needs uniform shape, so resize one-by-one then stack.
    import torch
    import torch.nn.functional as F
    out = np.empty((len(maps), size, size), dtype=np.float32)
    for i, m in enumerate(maps):
        t = torch.from_numpy(np.ascontiguousarray(m)).float()[None, None]
        t = F.interpolate(t, size=(size, size), mode="nearest")
        out[i] = t[0, 0].numpy()
    return out


def main():
    ap = argparse.ArgumentParser(description="Pretrain wafer backbone on WM-811K")
    ap.add_argument("--trust", action="store_true",
                    help="acknowledge that WM811K.pkl will be unpickled")
    ap.add_argument("--pkl", default=PKL_PATH)
    ap.add_argument("--size", type=int, default=64, help="resize wafer maps to NxN")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--quick", action="store_true", help="subsample to 8k for smoke test")
    ap.add_argument("--out", default=DEFAULT_BACKBONE_PATH)
    args = ap.parse_args()

    print(f"[1/4] loading WM-811K from {args.pkl} ...")
    maps, y = load_wm811k(args.pkl, trust=args.trust)
    print(f"      {len(y)} labelled wafer maps")
    if args.quick and len(y) > 8000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y), 8000, replace=False)
        maps, y = [maps[i] for i in idx], y[idx]
        print(f"      --quick: subsampled to {len(y)}")

    # report class balance (heavily imbalanced — 'none' dominates)
    counts = np.bincount(y, minlength=len(CLASSES))
    for c, n in zip(CLASSES, counts):
        print(f"        {c:>10}: {n}")

    print(f"[2/4] resizing to {args.size}×{args.size} ...")
    X = stack_resized(maps, args.size)
    Xt = to_grayscale_tensor(X, size=None, vmax=2.0)  # wafer maps are {0,1,2}
    import torch
    yt = torch.from_numpy(y)

    # stratified-ish split: shuffle then 85/15
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    n_val = int(0.15 * len(y))
    va, tr = perm[:n_val], perm[n_val:]

    # inverse-frequency class weights — defect classes must not be drowned out
    w = torch.tensor((counts.sum() / np.maximum(counts, 1)).astype(np.float32))
    w = w / w.mean()

    print(f"[3/4] training 9-class classifier ({len(tr)} train / {len(va)} val) ...")
    model = WaferClassifier(n_classes=len(CLASSES))
    history = train_classifier(
        model, Xt[tr], yt[tr], Xt[va], yt[va],
        epochs=args.epochs, batch_size=256, class_weight=w,
    )

    # multi-metric eval (never macro-F1 alone, per repo policy)
    from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
    y_pred = predict(model, Xt[va])
    y_true = y[va]
    defect_mask_true = y_true != CLASS_TO_IDX["none"]
    defect_mask_pred = y_pred != CLASS_TO_IDX["none"]
    print("\n=== WM-811K validation ===")
    print(f"  macro-F1            : {f1_score(y_true, y_pred, average='macro'):.3f}")
    print(f"  balanced-acc        : {balanced_accuracy_score(y_true, y_pred):.3f}")
    print(f"  defect-vs-none recall: "
          f"{defect_mask_pred[defect_mask_true].mean():.3f}")
    print(f"  false-positive rate : "
          f"{defect_mask_pred[~defect_mask_true].mean():.3f}")

    path = model.backbone.save(args.out)
    print(f"\n[4/4] backbone → {path}")
    print("      warm-start downstream: WaferClassifier(...).backbone.load(...)")

    # figure (confusion matrix + learning curve)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        im = ax[0].imshow(cm, cmap="Blues")
        ax[0].set_xticks(range(len(CLASSES))); ax[0].set_yticks(range(len(CLASSES)))
        ax[0].set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7)
        ax[0].set_yticklabels(CLASSES, fontsize=7)
        ax[0].set_title("WM-811K confusion (val)"); fig.colorbar(im, ax=ax[0])
        ax[1].plot(history["epoch"], history["train_loss"], "-o", label="train loss")
        ax[1].plot(history["epoch"], history["val_metric"], "-s", label="val acc")
        ax[1].legend(); ax[1].set_xlabel("epoch"); ax[1].set_title("learning curve")
        out_png = os.path.join(os.path.dirname(args.out), "wm811k_pretrain.png")
        fig.tight_layout(); fig.savefig(out_png, dpi=120)
        print(f"      figure → {out_png}")
    except Exception as e:  # noqa: BLE001
        print(f"      [skip figure] {e}")


if __name__ == "__main__":
    main()
