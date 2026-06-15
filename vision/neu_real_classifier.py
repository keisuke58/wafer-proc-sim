"""
Real-image defect classifier — NEU-CLS steel-surface photographs.

Unlike kerf_quality_classifier.py (synthetic dicing images), this trains and
evaluates the *same* BatchNorm CNN backbone on **real photographs**: the NEU
Surface Defect Database — 1,800 real grayscale images (200×200), 6 defect
classes × 300. This is the honest "we ran the CNN on real images, not just
synthetic" demo.

Each NEU class is a real-image analog of a wafer/kerf inspection defect
(see vision.neu_steel_adapter.CATEGORY_MAPPING): scratches → kerf-edge linear
chipping, crazing → distributed chipping, pitted_surface → point chipping, etc.

Backbone & transfer
-------------------
Uses vision.wafer_backbone.WaferClassifier (global-avg-pool ⇒ input-size
agnostic), so a WM-811K-pretrained backbone warm-starts here unchanged via
``--pretrained results/wm811k_backbone.pt``.

Evaluation follows the repo multi-metric policy (never accuracy alone):
top-1 accuracy, macro-F1, balanced accuracy, per-class F1, confusion matrix,
plus a RandomForest-on-pixels baseline for honesty.

Usage
-----
    python vision/neu_real_classifier.py                       # full (~1-2 min CPU)
    python vision/neu_real_classifier.py --quick               # fast smoke
    python vision/neu_real_classifier.py --pretrained results/wm811k_backbone.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision.neu_steel_adapter import CATEGORY_MAPPING, NeuSteelAdapter

DEFAULT_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "data", "external", "NEU-CLS")
# fixed class order for reproducible labels / confusion matrices
CLASSES = ["crazing", "inclusion", "patches",
           "pitted_surface", "rolled_in_scale", "scratches"]


# ── data ──────────────────────────────────────────────────────────────────────

def load_neu_dataset(
    root: str = DEFAULT_ROOT,
    size: int = 128,
    max_per_class: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load NEU-CLS as (X, y, class_names).

    X : float32 (N, size, size) in [0, 255], nearest-neighbour resized.
    y : int64 (N,) label index into CLASSES.
    """
    adapter = NeuSteelAdapter(root)
    avail = set(adapter.available_categories)
    images: List[np.ndarray] = []
    labels: List[int] = []
    for idx, cls in enumerate(CLASSES):
        if cls not in avail:  # adapter normalises rolled-in_scale → rolled_in_scale
            raise KeyError(f"NEU category '{cls}' missing. Found: {sorted(avail)}")
        imgs, _ = adapter.load_category(cls, max_images=max_per_class)
        for img in imgs:
            images.append(_resize(img, size))
            labels.append(idx)
    X = np.stack(images).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return X, y, CLASSES


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize a single (H, W) image to (size, size)."""
    h, w = img.shape
    ri = (np.linspace(0, h - 1, size)).round().astype(int)
    ci = (np.linspace(0, w - 1, size)).round().astype(int)
    return img[np.ix_(ri, ci)]


# ── evaluation (multi-metric, per repo policy) ───────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             class_names: List[str], name: str = "model") -> Dict[str, object]:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score)
    labels = list(range(len(class_names)))
    per_class = f1_score(y_true, y_pred, average=None, labels=labels,
                         zero_division=0)
    return {
        "name": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   labels=labels, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class_f1": {c: float(v) for c, v in zip(class_names, per_class)},
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def print_metrics(m: Dict[str, object]) -> None:
    print(f"  [{m['name']}]")
    print(f"    accuracy      : {m['accuracy']:.3f}")
    print(f"    macro-F1      : {m['macro_f1']:.3f}")
    print(f"    balanced-acc  : {m['balanced_acc']:.3f}")
    worst = min(m["per_class_f1"].items(), key=lambda kv: kv[1])
    print(f"    weakest class : {worst[0]} (F1={worst[1]:.3f})")


# ── baseline: RandomForest on downsampled pixels ─────────────────────────────

def train_pixel_baseline(X_tr, y_tr, X_te, y_te, class_names, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    ds = lambda A: A[:, ::8, ::8].reshape(len(A), -1) / 255.0  # noqa: E731
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    rf.fit(ds(X_tr), y_tr)
    return evaluate(y_te, rf.predict(ds(X_te)), class_names, "RandomForest(pixels)")


# ── CNN (BatchNorm backbone, optional WM-811K warm start) ────────────────────

def train_cnn(
    X_tr, y_tr, X_te, y_te, class_names,
    epochs: int = 20, batch_size: int = 64, lr: float = 2e-3,
    seed: int = 0, pretrained: Optional[str] = None,
) -> Tuple[Dict[str, object], Optional[Dict[str, list]]]:
    try:
        import torch
    except ImportError:
        from sklearn.neural_network import MLPClassifier
        ds = lambda A: A[:, ::4, ::4].reshape(len(A), -1) / 255.0  # noqa: E731
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                            random_state=seed)
        mlp.fit(ds(X_tr), y_tr)
        return evaluate(y_te, mlp.predict(ds(X_te)), class_names,
                        "MLP(sklearn)"), None

    import torch
    from vision.wafer_backbone import WaferClassifier

    torch.manual_seed(seed)
    device = "cpu"
    clf = WaferClassifier(n_classes=len(class_names))
    if pretrained:
        if os.path.exists(pretrained):
            clf.backbone.load(pretrained, map_location="cpu")
            print(f"    warm-started backbone from {pretrained}")
        else:
            print(f"    [warn] pretrained backbone {pretrained} missing — cold start")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_tr))
    n_val = max(1, int(0.15 * len(X_tr)))
    va, tr = perm[:n_val], perm[n_val:]

    def to_t(A):
        return torch.from_numpy(A[:, None, :, :] / 255.0).float()

    from vision.wafer_backbone import train_classifier, predict
    history = train_classifier(
        clf, to_t(X_tr[tr]), torch.from_numpy(y_tr[tr]),
        to_t(X_tr[va]), torch.from_numpy(y_tr[va]),
        epochs=epochs, batch_size=batch_size, lr=lr, device=device, verbose=True)
    y_pred = predict(clf, to_t(X_te), device=device)
    return evaluate(y_te, y_pred, class_names, "CNN(BatchNorm)"), history


# ── figure ───────────────────────────────────────────────────────────────────

def plot_results(X, y, class_names, cm, history, out_path, title_suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cls = len(class_names)
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(3, n_cls, height_ratios=[1.0, 1.3, 0.1],
                          hspace=0.4, wspace=0.25)

    # (a) one real sample per class
    for j, cls in enumerate(class_names):
        ax = fig.add_subplot(gs[0, j])
        idx = int(np.where(y == j)[0][0])
        ax.imshow(X[idx], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{cls}\n→ {CATEGORY_MAPPING.get(cls, '?')}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    # (b) confusion matrix
    ax = fig.add_subplot(gs[1, :n_cls // 2 + 1])
    cm = np.asarray(cm)
    im = ax.imshow(cm, cmap="Blues")
    for i in range(n_cls):
        for k in range(n_cls):
            ax.text(k, i, str(cm[i, k]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, k] > cm.max() / 2 else "black")
    ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (c) learning curve
    ax = fig.add_subplot(gs[1, n_cls // 2 + 1:])
    if history is not None:
        ax.plot(history["epoch"], history["train_loss"], "o-",
                color="tab:blue", label="train loss")
        ax.set_xlabel("epoch"); ax.set_ylabel("train loss", color="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(history["epoch"], history["val_metric"], "s-",
                 color="tab:green", label="val acc")
        ax2.set_ylabel("val accuracy", color="tab:green"); ax2.set_ylim(0, 1.02)
        ax.set_title("CNN learning curve", fontsize=10)
    else:
        ax.text(0.5, 0.5, "PyTorch unavailable\n(sklearn fallback)",
                ha="center", va="center"); ax.axis("off")

    fig.suptitle(f"NEU-CLS real-image defect classifier{title_suffix}", fontsize=13)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure saved -> {out_path}")


# ── demo entry point ─────────────────────────────────────────────────────────

def run_demo(root: str = DEFAULT_ROOT, size: int = 128, epochs: int = 20,
             max_per_class: Optional[int] = None, seed: int = 0,
             pretrained: Optional[str] = None,
             out_path: Optional[str] = None) -> Dict[str, object]:
    from sklearn.model_selection import train_test_split

    t0 = time.time()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = out_path or os.path.join(repo_root, "results", "neu_real_classifier.png")

    print(f"[1/4] Loading NEU-CLS ({size}x{size}) ...")
    X, y, names = load_neu_dataset(root, size=size, max_per_class=max_per_class)
    print(f"      {len(X)} real images, {len(names)} classes")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed)

    print("[2/4] RandomForest(pixels) baseline ...")
    m_rf = train_pixel_baseline(X_tr, y_tr, X_te, y_te, names, seed=seed)
    print_metrics(m_rf)

    print("[3/4] CNN (BatchNorm backbone) ...")
    m_cnn, history = train_cnn(X_tr, y_tr, X_te, y_te, names, epochs=epochs,
                               seed=seed, pretrained=pretrained)
    print_metrics(m_cnn)

    print("[4/4] Plotting ...")
    suffix = " (WM-811K warm start)" if pretrained else ""
    plot_results(X, y, names, m_cnn["confusion"], history, out_path, suffix)

    stats_path = os.path.join(os.path.dirname(out_path), "neu_real_stats.json")
    with open(stats_path, "w") as f:
        json.dump({"random_forest": m_rf, "cnn": m_cnn,
                   "n_images": len(X), "size": size, "epochs": epochs}, f, indent=2)
    print(f"  stats saved -> {stats_path}")
    print(f"Done in {time.time() - t0:.1f}s")
    return {"random_forest": m_rf, "cnn": m_cnn}


def main() -> None:
    p = argparse.ArgumentParser(description="NEU-CLS real-image defect classifier")
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--pretrained", type=str, default=None,
                   help="WM-811K backbone (results/wm811k_backbone.pt) to warm-start")
    p.add_argument("--quick", action="store_true",
                   help="fast smoke run (100 img/class, 64px, 8 epochs)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.quick:
        args.max_per_class, args.size, args.epochs = 100, 64, 8
    run_demo(root=args.root, size=args.size, epochs=args.epochs,
             max_per_class=args.max_per_class, seed=args.seed,
             pretrained=args.pretrained, out_path=args.out)


if __name__ == "__main__":
    main()
