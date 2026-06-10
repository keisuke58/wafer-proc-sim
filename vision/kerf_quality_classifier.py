"""
Kerf quality AI classifier demo — post-dicing chipping grade classification.

Motivation
----------
DISCO's PIM improvement cases include "wafer microscope image map UI" work and
the product line has an *inspection* category (optical kerf check). This demo
shows how kerf-check images can be auto-graded with ML — "intelligent
inspection" — using the same problem class as defect localization on CFRP
structures (graph/CNN classifiers on physics-grounded synthetic data).

Pipeline
--------
1. Generate 3-class synthetic dataset with ``vision.generate_synthetic_kerf``:
     Grade A — light chipping  (accept)
     Grade B — moderate chipping (rework / monitor)
     Grade C — heavy chipping  (reject)
2. Hand-crafted feature baseline (edge roughness, chipping-area ratio,
   brightness stats) + LogisticRegression / RandomForest.
3. Small 3-layer CNN (PyTorch if available, sklearn MLP fallback).
4. Multi-metric evaluation: per-class F1, macro-F1, defect(C)-only F1,
   detection recall, false-positive rate, confusion matrix.
5. Figure -> results/kerf_classifier.png

Usage
-----
    python vision/kerf_quality_classifier.py            # full demo (~1 min CPU)
    python vision/kerf_quality_classifier.py --quick    # fast smoke run
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision.generate_synthetic_kerf import KerfImageConfig, generate_kerf_image

GRADE_NAMES = ["Grade A", "Grade B", "Grade C"]

# Chipping parameter ranges per grade (severity, lateral extent in px).
# Grade C = reject level (heavy chipping along kerf edges).
GRADE_SPECS: Dict[int, Dict[str, Tuple[float, float]]] = {
    0: {"severity": (0.00, 0.08), "extent_px": (3.0, 5.0)},   # A: light
    1: {"severity": (0.15, 0.40), "extent_px": (5.0, 8.0)},   # B: moderate
    2: {"severity": (0.50, 0.90), "extent_px": (8.0, 14.0)},  # C: heavy
}

FEATURE_NAMES = [
    "edge_dark_frac",      # chipping area ratio in bands outside kerf walls
    "edge_roughness",      # row-wise std of dark-pixel count along the edge band
    "edge_mean",           # mean brightness of edge band
    "edge_std",            # brightness std of edge band
    "edge_min_row_mean",   # darkest row mean in edge band (worst local chip)
    "grad_mean",           # mean |horizontal gradient|
    "grad_p95",            # 95th percentile |horizontal gradient|
    "img_mean",            # global brightness mean
    "img_std",             # global brightness std
    "dark_frac_global",    # global dark-pixel fraction
]


# ── dataset ──────────────────────────────────────────────────────────────────

def make_grade_dataset(
    n_per_class: int = 300,
    H: int = 128,
    W: int = 128,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a balanced 3-class kerf-quality dataset.

    Returns
    -------
    X : float32 array (3*n_per_class, H, W), pixel range [0, 255]
    y : int64 array (3*n_per_class,), labels 0=Grade A, 1=B, 2=C
    """
    rng = np.random.default_rng(seed)
    images: List[np.ndarray] = []
    labels: List[int] = []
    for grade, spec in GRADE_SPECS.items():
        for _ in range(n_per_class):
            cfg = KerfImageConfig(
                H=H, W=W,
                kerf_width_mm=float(rng.uniform(1.5, 3.0)),
                pixel_per_mm=10.0,
                kerf_center_frac=float(rng.uniform(0.40, 0.60)),
                chipping_severity=float(rng.uniform(*spec["severity"])),
                chipping_extent_px=float(rng.uniform(*spec["extent_px"])),
                blade_wear=float(rng.uniform(0.0, 0.6)),
                include_blade_tip=False,
                seed=int(rng.integers(0, 2**31)),
            )
            img, _ = generate_kerf_image(cfg)
            images.append(img)
            labels.append(grade)
    X = np.stack(images).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


# ── hand-crafted features ────────────────────────────────────────────────────

def _locate_kerf_walls(img: np.ndarray) -> Tuple[int, int]:
    """Estimate kerf wall columns from the column-mean profile (dark void run)."""
    W = img.shape[1]
    col_mean = img.mean(axis=0)
    # smooth lightly to suppress chip-induced dips
    kernel = np.ones(5, dtype=np.float32) / 5.0
    smooth = np.convolve(col_mean, kernel, mode="same")
    thresh = 0.5 * (smooth.min() + np.median(smooth))
    void_idx = np.where(smooth < thresh)[0]
    if len(void_idx) >= 2:
        return int(void_idx[0]), int(void_idx[-1])
    return W // 3, 2 * W // 3  # fallback: assume centred kerf


def extract_features(img: np.ndarray, band_px: int = 12,
                     dark_thresh: float = 30.0) -> np.ndarray:
    """Extract 10 hand-crafted kerf-quality features (see FEATURE_NAMES)."""
    H, W = img.shape
    left, right = _locate_kerf_walls(img)

    lo = max(0, left - band_px)
    hi = min(W, right + 1 + band_px)
    left_band = img[:, lo:max(lo + 1, left)]
    right_band = img[:, min(right + 1, hi - 1):hi]
    edge = np.concatenate([left_band, right_band], axis=1)
    if edge.size == 0:  # degenerate wall estimate
        edge = img

    dark = edge < dark_thresh
    edge_dark_frac = float(dark.mean())
    edge_roughness = float(dark.sum(axis=1).std())
    edge_mean = float(edge.mean())
    edge_std = float(edge.std())
    edge_min_row_mean = float(edge.mean(axis=1).min())

    gx = np.abs(np.diff(img, axis=1))
    grad_mean = float(gx.mean())
    grad_p95 = float(np.percentile(gx, 95))

    img_mean = float(img.mean())
    img_std = float(img.std())
    dark_frac_global = float((img < dark_thresh).mean())

    return np.array([
        edge_dark_frac, edge_roughness, edge_mean, edge_std, edge_min_row_mean,
        grad_mean, grad_p95, img_mean, img_std, dark_frac_global,
    ], dtype=np.float64)


def extract_feature_matrix(X: np.ndarray) -> np.ndarray:
    return np.stack([extract_features(img) for img in X])


# ── evaluation (multi-metric, never macro-F1 alone) ──────────────────────────

def evaluate_multiclass(y_true: np.ndarray, y_pred: np.ndarray,
                        name: str = "model") -> Dict[str, float]:
    """
    Multi-metric evaluation. Grade C (label 2) is the reject/defect class.

    Returns accuracy, macro-F1, per-class F1, defect-only F1,
    detection recall (recall of C) and false-positive rate
    (non-C samples flagged as C).
    """
    from sklearn.metrics import accuracy_score, f1_score

    per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    defect_true = (y_true == 2).astype(int)
    defect_pred = (y_pred == 2).astype(int)
    defect_f1 = f1_score(defect_true, defect_pred, zero_division=0)
    det_recall = float(defect_pred[defect_true == 1].mean()) if (defect_true == 1).any() else 0.0
    fpr = float(defect_pred[defect_true == 0].mean()) if (defect_true == 0).any() else 0.0

    metrics = {
        "name": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "f1_grade_a": float(per_class[0]),
        "f1_grade_b": float(per_class[1]),
        "f1_grade_c": float(per_class[2]),
        "defect_f1": float(defect_f1),
        "detection_recall": det_recall,
        "false_positive_rate": fpr,
    }
    return metrics


def print_metrics(m: Dict[str, float]) -> None:
    print(f"  [{m['name']}]")
    print(f"    accuracy            : {m['accuracy']:.3f}")
    print(f"    macro-F1            : {m['macro_f1']:.3f}")
    print(f"    F1 (A / B / C)      : {m['f1_grade_a']:.3f} / "
          f"{m['f1_grade_b']:.3f} / {m['f1_grade_c']:.3f}")
    print(f"    defect-only F1 (C)  : {m['defect_f1']:.3f}")
    print(f"    detection recall (C): {m['detection_recall']:.3f}")
    print(f"    false-positive rate : {m['false_positive_rate']:.3f}")


# ── baselines: logistic regression + random forest ──────────────────────────

def train_feature_baselines(
    F_train: np.ndarray, y_train: np.ndarray,
    F_test: np.ndarray, y_test: np.ndarray,
    seed: int = 0,
):
    """Train LogisticRegression + RandomForest on hand-crafted features."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    )
    logreg.fit(F_train, y_train)
    m_lr = evaluate_multiclass(y_test, logreg.predict(F_test), "LogisticRegression")

    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    rf.fit(F_train, y_train)
    m_rf = evaluate_multiclass(y_test, rf.predict(F_test), "RandomForest")

    return logreg, m_lr, rf, m_rf


# ── small CNN (PyTorch, sklearn MLP fallback) ────────────────────────────────

def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def train_cnn(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    epochs: int = 12, batch_size: int = 64, seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, float], Optional[Dict[str, list]]]:
    """
    Train a small image classifier.

    PyTorch path: 3-layer CNN (8-16-32 ch) + global average pool + linear.
    Fallback    : sklearn MLPClassifier on flattened 32x32 downsampled images.

    Returns (y_pred_test, metrics, history-or-None).
    """
    if not _torch_available():
        from sklearn.neural_network import MLPClassifier
        # downsample 4x to keep MLP small
        Xtr = X_train[:, ::4, ::4].reshape(len(X_train), -1) / 255.0
        Xte = X_test[:, ::4, ::4].reshape(len(X_test), -1) / 255.0
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                            random_state=seed)
        mlp.fit(Xtr, y_train)
        y_pred = mlp.predict(Xte)
        return y_pred, evaluate_multiclass(y_test, y_pred, "MLP(sklearn)"), None

    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    device = torch.device("cpu")  # demo must run on CPU in <2 min

    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 64
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(32, 3),
    ).to(device)

    # hold out 15% of train as validation for the learning curve
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_train))
    n_val = max(1, int(0.15 * len(X_train)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    def to_tensor(X: np.ndarray) -> "torch.Tensor":
        return torch.from_numpy(X[:, None, :, :] / 255.0).float()

    Xtr, ytr = to_tensor(X_train[tr_idx]), torch.from_numpy(y_train[tr_idx])
    Xva, yva = to_tensor(X_train[val_idx]), torch.from_numpy(y_train[val_idx])
    Xte = to_tensor(X_test)

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()
    history = {"epoch": [], "train_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(Xtr))
        losses = []
        for i in range(0, len(Xtr), batch_size):
            idx = order[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_acc = float((model(Xva).argmax(1) == yva).float().mean())
        history["epoch"].append(epoch)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_acc"].append(val_acc)
        print(f"    epoch {epoch:2d}: train_loss={np.mean(losses):.4f} "
              f"val_acc={val_acc:.3f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(Xte).argmax(1).numpy()
    return y_pred, evaluate_multiclass(y_test, y_pred, "CNN(3-layer)"), history


# ── figure ───────────────────────────────────────────────────────────────────

def plot_results(
    X: np.ndarray, y: np.ndarray,
    cm: np.ndarray, cm_title: str,
    rf_importances: np.ndarray,
    history: Optional[Dict[str, list]],
    out_path: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.35, wspace=0.3)

    # (a) one sample image per grade
    for grade in range(3):
        ax = fig.add_subplot(gs[0, grade])
        idx = int(np.where(y == grade)[0][0])
        ax.imshow(X[idx], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"(a{grade + 1}) {GRADE_NAMES[grade]}", fontsize=11)
        ax.axis("off")

    # (b) confusion matrix
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3), ["A", "B", "C"])
    ax.set_yticks(range(3), ["A", "B", "C"])
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("True grade")
    ax.set_title(f"(b) Confusion matrix — {cm_title}", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (c) RF feature importances
    ax = fig.add_subplot(gs[1, 1])
    order = np.argsort(rf_importances)
    ax.barh(range(len(order)), rf_importances[order], color="tab:orange")
    ax.set_yticks(range(len(order)), [FEATURE_NAMES[i] for i in order], fontsize=8)
    ax.set_xlabel("RF importance")
    ax.set_title("(c) Feature importances (RandomForest)", fontsize=11)

    # (d) CNN learning curve (or note if fallback)
    ax = fig.add_subplot(gs[1, 2])
    if history is not None:
        ax.plot(history["epoch"], history["train_loss"], "o-",
                color="tab:blue", label="train loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train loss", color="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(history["epoch"], history["val_acc"], "s-",
                 color="tab:green", label="val acc")
        ax2.set_ylabel("Val accuracy", color="tab:green")
        ax2.set_ylim(0, 1.02)
        ax.set_title("(d) CNN learning curve", fontsize=11)
    else:
        ax.text(0.5, 0.5, "PyTorch unavailable\n(sklearn MLP fallback used)",
                ha="center", va="center")
        ax.axis("off")

    fig.suptitle("Kerf quality AI classifier — synthetic dicing inspection demo",
                 fontsize=13)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure saved -> {out_path}")


# ── demo entry point ─────────────────────────────────────────────────────────

def run_demo(n_per_class: int = 300, epochs: int = 12, seed: int = 0,
             out_path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split

    t0 = time.time()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = out_path or os.path.join(repo_root, "results", "kerf_classifier.png")

    print(f"[1/4] Generating dataset: 3 x {n_per_class} images (128x128) ...")
    X, y = make_grade_dataset(n_per_class=n_per_class, H=128, W=128, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed)
    print(f"      train={len(X_train)}, test={len(X_test)} "
          f"({time.time() - t0:.1f}s)")

    print("[2/4] Feature baselines (LogReg / RandomForest) ...")
    F_train = extract_feature_matrix(X_train)
    F_test = extract_feature_matrix(X_test)
    logreg, m_lr, rf, m_rf = train_feature_baselines(
        F_train, y_train, F_test, y_test, seed=seed)
    print_metrics(m_lr)
    print_metrics(m_rf)

    print(f"[3/4] CNN ({'PyTorch' if _torch_available() else 'sklearn MLP'}) ...")
    y_pred_cnn, m_cnn, history = train_cnn(
        X_train, y_train, X_test, y_test, epochs=epochs, seed=seed)
    print_metrics(m_cnn)

    print("[4/4] Plotting ...")
    cm = confusion_matrix(y_test, y_pred_cnn, labels=[0, 1, 2])
    plot_results(X, y, cm, m_cnn["name"],
                 rf.feature_importances_, history, out_path)

    print(f"Done in {time.time() - t0:.1f}s")
    return {"logreg": m_lr, "random_forest": m_rf, "cnn": m_cnn}


def main() -> None:
    p = argparse.ArgumentParser(description="Kerf quality classifier demo")
    p.add_argument("--n-per-class", type=int, default=300)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="fast smoke run (60 images/class, 3 epochs)")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    if args.quick:
        args.n_per_class, args.epochs = 60, 3
    run_demo(n_per_class=args.n_per_class, epochs=args.epochs,
             seed=args.seed, out_path=args.out)


if __name__ == "__main__":
    main()
