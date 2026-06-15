"""
Intelligent inspection — one story from optical metrology to learned vision.

This module ties three pieces of the repo into a single "intelligent
inspection" pipeline for post-dicing kerf quality:

  1. METROLOGY ARM (Lasertec-style, physics)
     ml.lasertec_inspection measures chipping size per die (confocal / Weibull
     model) and grades it against INSPECTION_SPEC: A (accept) / B (rework) /
     C (reject). This is the slow, instrument-based ground truth.

  2. VISION ARM (this repo's CNN)
     We render a kerf image whose chipping severity is calibrated to each die's
     measured chipping (vision.generate_synthetic_kerf) and train the shared
     BatchNorm backbone (vision.wafer_backbone) to predict the SAME grade from
     the image alone — i.e. the CNN learns to replicate the optical grader from
     pixels, with no metrology at inference time.

  3. REAL-IMAGE GENERALIZATION (cross-reference, not retrained here)
     The same backbone classifies real defect photographs (NEU-CLS) and, warm
     started from real-fab WM-811K wafer maps, reaches 0.96 — evidence the
     learned-inspection approach transfers to real images.
     See vision/neu_real_classifier.py and docs/neu_real_classifier.md.

Honest scope: the vision arm sees *chipping* (a visual defect). The metrology
arm additionally measures subsurface crack depth and PL defect density, which
an optical image cannot see. So the two are **complementary** — the CNN
replaces routine chipping grading at speed; metrology stays for what pixels
can't show. That fusion is the pitch, not "CNN replaces the instrument".

Usage
-----
    python vision/intelligent_inspection.py            # full (~30-60s CPU)
    python vision/intelligent_inspection.py --quick    # fast smoke
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

from vision.generate_synthetic_kerf import KerfImageConfig, generate_kerf_image
from vision.kerf_quality_classifier import evaluate_multiclass

# processes spanning the chipping range so grades span A/B/C
PROCESSES = ["stealth", "laser_fs", "plasma", "laser_ps", "laser_ns", "blade"]
GRADE_NAMES = ["A (accept)", "B (rework)", "C (reject)"]

# chipping thresholds [µm] — aligned with INSPECTION_SPEC chip_um_max = 5.0
# and the 2nm-node target of 0.5 µm.
GRADE_A_MAX = 1.0   # < 1.0 µm  → accept
GRADE_B_MAX = 5.0   # 1.0–5.0 µm → rework (within spec but elevated); ≥5 reject


def metrology_grade(chip_um: float) -> int:
    """Lasertec-style optical chipping grade: 0=A, 1=B, 2=C(reject)."""
    if chip_um < GRADE_A_MAX:
        return 0
    if chip_um < GRADE_B_MAX:
        return 1
    return 2


def _chip_to_image_params(chip_um: float) -> Tuple[float, float]:
    """Map a measured chipping size [µm] to kerf-image (severity, extent_px)."""
    severity = float(np.clip((chip_um - 0.2) / 9.0, 0.0, 0.92))
    extent_px = float(np.clip(chip_um * 1.4 + 3.0, 3.0, 14.0))
    return severity, extent_px


# ── dataset: metrology measures chipping → grade + matching image ────────────

def build_dataset(
    n_per_process: int = 120, H: int = 128, W: int = 128, seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each process, draw per-die chipping sizes from the Lasertec Weibull
    model, grade each die (metrology), and render a kerf image calibrated to
    that chipping. Returns (X images, y grades, chip_um, proc_idx).
    """
    from ml.lasertec_inspection import chipping_measurement

    rng = np.random.default_rng(seed)
    images: List[np.ndarray] = []
    grades: List[int] = []
    chips: List[float] = []
    procs: List[int] = []
    for pi, proc in enumerate(PROCESSES):
        meas = chipping_measurement(proc, n_sites=n_per_process,
                                    seed=int(rng.integers(0, 2**31)))
        for chip_um in meas["chips_um"]:
            severity, extent_px = _chip_to_image_params(float(chip_um))
            cfg = KerfImageConfig(
                H=H, W=W,
                kerf_width_mm=float(rng.uniform(1.5, 3.0)),
                pixel_per_mm=10.0,
                kerf_center_frac=float(rng.uniform(0.40, 0.60)),
                chipping_severity=severity,
                chipping_extent_px=extent_px,
                blade_wear=float(rng.uniform(0.0, 0.6)),
                include_blade_tip=False,
                seed=int(rng.integers(0, 2**31)),
            )
            img, _ = generate_kerf_image(cfg)
            images.append(img)
            grades.append(metrology_grade(float(chip_um)))
            chips.append(float(chip_um))
            procs.append(pi)
    return (np.stack(images).astype(np.float32),
            np.asarray(grades, dtype=np.int64),
            np.asarray(chips, dtype=np.float64),
            np.asarray(procs, dtype=np.int64))


# ── vision arm: CNN predicts the metrology grade from the image ──────────────

def train_vision_grader(
    X_tr, y_tr, X_te, y_te, epochs: int = 15, seed: int = 0,
    pretrained: Optional[str] = None,
):
    try:
        import torch
    except ImportError:
        from sklearn.neural_network import MLPClassifier
        ds = lambda A: A[:, ::4, ::4].reshape(len(A), -1) / 255.0  # noqa: E731
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                            random_state=seed).fit(ds(X_tr), y_tr)
        return (mlp.predict(ds(X_te)),
                evaluate_multiclass(y_te, mlp.predict(ds(X_te)), "MLP(sklearn)"),
                None)

    import torch
    from vision.wafer_backbone import WaferClassifier, train_classifier, predict

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

    history = train_classifier(
        clf, to_t(X_tr[tr]), torch.from_numpy(y_tr[tr]),
        to_t(X_tr[va]), torch.from_numpy(y_tr[va]),
        epochs=epochs, batch_size=64, lr=2e-3, device="cpu", verbose=True)
    y_pred = predict(clf, to_t(X_te), device="cpu")
    return y_pred, evaluate_multiclass(y_te, y_pred, "CNN(vision arm)"), history


# ── figure ───────────────────────────────────────────────────────────────────

def plot_story(X, y, chips, procs, y_test, y_pred, metrics, history,
               out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1.1, 1.0],
                          hspace=0.45, wspace=0.3)

    # (a) sample image per grade with its measured chipping
    for g in range(3):
        ax = fig.add_subplot(gs[0, g])
        idx = int(np.where(y == g)[0][0])
        ax.imshow(X[idx], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{GRADE_NAMES[g]}\nchipping ≈ {chips[idx]:.2f} µm",
                     fontsize=10)
        ax.axis("off")

    # (b) chipping µm → grade mapping (the metrology rule the CNN must learn)
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(chips, y + np.random.default_rng(0).normal(0, 0.05, len(y)),
               s=4, alpha=0.2, color="tab:blue")
    ax.axvline(GRADE_A_MAX, color="green", ls="--", lw=1, label="A|B = 1 µm")
    ax.axvline(GRADE_B_MAX, color="red", ls="--", lw=1, label="B|C = 5 µm (spec)")
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["A", "B", "C"])
    ax.set_xlabel("measured chipping [µm]"); ax.set_ylabel("metrology grade")
    ax.set_title("(b) Lasertec metrology rule", fontsize=10)
    ax.legend(fontsize=7); ax.set_xlim(0, min(20, chips.max()))

    # (c) agreement confusion: CNN-from-image vs metrology grade
    ax = fig.add_subplot(gs[1, 1])
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    im = ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(["A", "B", "C"]); ax.set_yticklabels(["A", "B", "C"])
    ax.set_xlabel("CNN (from image)"); ax.set_ylabel("metrology grade")
    ax.set_title(f"(c) Agreement\nacc={metrics['accuracy']:.3f} "
                 f"reject-recall={metrics['detection_recall']:.3f}", fontsize=10)

    # (d) learning curve
    ax = fig.add_subplot(gs[1, 2])
    if history is not None:
        ax.plot(history["epoch"], history["train_loss"], "o-", color="tab:blue",
                label="train loss")
        ax.set_xlabel("epoch"); ax.set_ylabel("train loss", color="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(history["epoch"], history["val_metric"], "s-",
                 color="tab:green")
        ax2.set_ylabel("val acc", color="tab:green"); ax2.set_ylim(0, 1.02)
        ax.set_title("(d) CNN learning curve", fontsize=10)
    else:
        ax.axis("off")

    # (e) the one-line story
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    txt = (
        "Intelligent inspection — one pipeline:\n"
        "  [1] METROLOGY (Lasertec-style): confocal/Weibull chipping  ->  grade A/B/C   [instrument ground truth]\n"
        "  [2] VISION (this CNN): kerf image  ->  predicts the SAME grade from pixels   "
        f"[agreement acc={metrics['accuracy']:.2f}, reject-recall={metrics['detection_recall']:.2f}]\n"
        "  [3] REAL IMAGES: same backbone on NEU photos, WM-811K warm start -> 0.96   [generalizes to real data]\n"
        "\n"
        "Complementary, not redundant: the CNN grades chipping at speed; metrology still adds\n"
        "subsurface crack depth + PL defect density that an optical image cannot see.")
    ax.text(0.01, 0.95, txt, va="top", ha="left", fontsize=10, family="monospace")

    fig.suptitle("検査の知能化 — From optical metrology to learned visual inspection",
                 fontsize=13)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure saved -> {out_path}")


# ── demo entry point ─────────────────────────────────────────────────────────

def run_demo(n_per_process: int = 120, epochs: int = 15, seed: int = 0,
             pretrained: Optional[str] = None,
             out_path: Optional[str] = None) -> Dict[str, object]:
    from sklearn.model_selection import train_test_split

    t0 = time.time()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = out_path or os.path.join(repo_root, "results",
                                        "intelligent_inspection.png")

    print(f"[1/3] Metrology → grades + calibrated images "
          f"({len(PROCESSES)} processes x {n_per_process}) ...")
    X, y, chips, procs = build_dataset(n_per_process=n_per_process, seed=seed)
    counts = {GRADE_NAMES[g]: int((y == g).sum()) for g in range(3)}
    print(f"      {len(X)} dies  grades={counts}")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed)

    print("[2/3] Vision arm: CNN learns the metrology grade from pixels ...")
    y_pred, metrics, history = train_vision_grader(
        X_tr, y_tr, X_te, y_te, epochs=epochs, seed=seed, pretrained=pretrained)
    print(f"      agreement acc={metrics['accuracy']:.3f} "
          f"macro-F1={metrics['macro_f1']:.3f} "
          f"reject-recall={metrics['detection_recall']:.3f} "
          f"reject-FPR={metrics['false_positive_rate']:.3f}")

    print("[3/3] Plotting unified story ...")
    plot_story(X, y, chips, procs, y_te, y_pred, metrics, history, out_path)

    stats = {"grade_counts": counts, "agreement": metrics,
             "n_dies": len(X), "processes": PROCESSES}
    with open(os.path.join(os.path.dirname(out_path),
                           "intelligent_inspection_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=float)
    print(f"Done in {time.time() - t0:.1f}s")
    return {"agreement": metrics}


def main() -> None:
    p = argparse.ArgumentParser(description="Intelligent inspection pipeline")
    p.add_argument("--n-per-process", type=int, default=120)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pretrained", type=str, default=None,
                   help="WM-811K backbone to warm-start (results/wm811k_backbone.pt)")
    p.add_argument("--quick", action="store_true", help="fast smoke run")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    if args.quick:
        args.n_per_process, args.epochs = 50, 8
    run_demo(n_per_process=args.n_per_process, epochs=args.epochs,
             seed=args.seed, pretrained=args.pretrained, out_path=args.out)


if __name__ == "__main__":
    main()
