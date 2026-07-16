#!/usr/bin/env python3
"""lesion_cad.py — the same inspection pipeline, aimed at medical images:
adaptive local-statistics lesion detection with a FROC evaluation.

FUJIFILM is now a medical-imaging company (CT, endoscopy, the REiLI AI-
diagnosis brand). The detectors that find a particle on a wafer are the same
math that finds a nodule in tissue: model the local background, score every
pixel by how surprising it is, and evaluate with the metric radiologists
actually use. This module carries the wafer-inspection stack over to a
synthetic medical-imaging task, ground truth and all.

  SYNTHETIC IMAGES  a low-frequency "organ" intensity field + correlated
      tissue speckle (the texture a fixed threshold trips on) + Gaussian
      lesions of assorted size and — crucially — subtle low contrast, with
      pixel-precise masks and lesion centroids.

  DETECTORS  three, escalating: a global FIXED threshold (the naive tool);
      the ADAPTIVE multi-scale z-score |I - median_k| / (1.4826 MAD_k) that
      de-weights busy tissue (the wafer die-to-die detector, transferred
      as-is); and a Laplacian-of-Gaussian MATCHED filter tuned to the lesion
      scale (the classic CAD blob detector).

  EVALUATION  pixel AUROC (threshold-free separability) AND a FROC curve —
      lesion sensitivity vs false-positives PER IMAGE, the standard for
      computer-aided detection — with the operating sensitivity at 1 FP/image
      reported. The honest transfer result: the wafer anomaly detector does
      NOT win here — with no clean reference "die", its local z-score fires on
      tissue texture. The scale-matched filter is the right medical detector.
      What transfers is the framework — synthesize-with-ground-truth, compare
      detectors, and evaluate with the metric radiologists actually use — and
      the judgment to pick the scale-matched detector rather than blindly
      reuse the wafer one.

numpy + scipy + matplotlib.  Run:  python3 fujifilm_med/lesion_cad.py
"""
import json
import os

import numpy as np
from scipy import ndimage as ndi

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "lesion_cad_results.json")

H = W = 128
LESION_SCALE = 3.0          # matched-filter lesion sigma [px]


def make_image(seed):
    """Synthetic medical image: organ field + tissue speckle + lesions.
    Returns (img[0,1], mask(bool), centroids[list of (y,x)])."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W] / H
    # low-frequency organ intensity
    a = rng.uniform(-0.3, 0.3, 6)
    organ = 0.5 + a[0] * yy + a[1] * xx + a[2] * yy * xx \
        + a[3] * np.sin(3 * np.pi * yy + a[4]) * 0.15
    # correlated tissue speckle (smoothed noise = texture)
    speckle = ndi.gaussian_filter(rng.normal(0, 1, (H, W)), 2.0) * 0.10
    img = organ + speckle
    mask = np.zeros((H, W), bool)
    cents = []
    n_les = rng.integers(2, 5)
    for _ in range(n_les):
        cy, cx = rng.integers(14, H - 14), rng.integers(14, W - 14)
        r = rng.uniform(3.0, 7.0)
        amp = rng.uniform(0.06, 0.20)          # contrast — some are subtle
        d2 = (yy * H - cy) ** 2 + (xx * W - cx) ** 2
        blob = amp * np.exp(-d2 / (2 * r ** 2))
        img = img + blob
        mask |= d2 <= r ** 2
        cents.append((cy, cx))
    img = img + rng.normal(0, 0.02, (H, W))     # detector read noise
    img = np.clip(img, 0, 1.3)
    return img, mask, cents


# ── detectors (transferred from wafer inspection) ───────────────────────────
def score_fixed(img):
    return np.abs(img - img.mean())


def _adaptive_z(img, k):
    med = ndi.median_filter(img, size=k)
    resid = img - med
    mad = ndi.median_filter(np.abs(resid), size=k) + 1e-4
    return np.abs(resid) / (1.4826 * mad)


def score_adaptive(img):
    z = np.maximum(_adaptive_z(img, 7), _adaptive_z(img, 21))
    return ndi.gaussian_filter(z, 1.0)


def score_log(img, sigma=LESION_SCALE):
    """Laplacian-of-Gaussian matched filter (bright-blob response)."""
    return -ndi.gaussian_laplace(img, sigma) * sigma ** 2


DETECTORS = [("fixed |I-mean|", score_fixed),
             ("adaptive multi-scale z", score_adaptive),
             ("LoG matched filter", score_log)]


# ── evaluation ──────────────────────────────────────────────────────────────
def auroc(scores, labels):
    s = scores.ravel().astype(np.float64)
    y = labels.ravel()
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, np.float64)
    ranks[order] = np.arange(1, s.size + 1)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    return (ranks[y.ravel()].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def froc_counts(score, mask, cents, thr):
    """At threshold thr: (n_true_detected, n_false_positive) via connected
    candidate regions matched to lesion centroids."""
    lab, n = ndi.label(score > thr)
    if n == 0:
        return 0, 0, len(cents)
    hit_labels = set()
    detected = 0
    for (cy, cx) in cents:
        lid = lab[int(cy), int(cx)]
        if lid == 0:
            # allow a small search window (candidate near the lesion)
            y0, y1 = max(int(cy) - 4, 0), min(int(cy) + 5, H)
            x0, x1 = max(int(cx) - 4, 0), min(int(cx) + 5, W)
            around = lab[y0:y1, x0:x1]
            nz = around[around > 0]
            lid = int(np.bincount(nz).argmax()) if nz.size else 0
        if lid > 0:
            detected += 1
            hit_labels.add(lid)
    fp = n - len(hit_labels)
    return detected, fp, len(cents)


def run():
    n_img = 24
    data = [make_image(100 + i) for i in range(n_img)]

    # pixel AUROC per detector
    aurocs = {}
    for name, fn in DETECTORS:
        vals = []
        for img, mask, _ in data:
            a = auroc(fn(img), mask)
            if not np.isnan(a):
                vals.append(a)
        aurocs[name] = round(float(np.mean(vals)), 4)

    # FROC per detector: sweep thresholds, sensitivity vs FP/image
    froc = {}
    sens_at_1fp = {}
    for name, fn in DETECTORS:
        smaps = [fn(img) for img, _, _ in data]
        allv = np.concatenate([s.ravel() for s in smaps])
        thrs = np.quantile(allv, np.linspace(0.80, 0.9995, 26))
        curve = []
        for t in thrs:
            tp = fp = tot = 0
            for s, (img, mask, cents) in zip(smaps, data):
                d, f, c = froc_counts(s, mask, cents, t)
                tp += d; fp += f; tot += c
            curve.append((fp / n_img, tp / max(tot, 1)))
        curve = np.array(curve)
        froc[name] = curve
        # sensitivity at ~1 false positive per image (interp)
        fpi, se = curve[:, 0], curve[:, 1]
        order = np.argsort(fpi)
        sens_at_1fp[name] = round(float(np.interp(1.0, fpi[order],
                                                  se[order])), 3)

    return {
        "n_images": n_img, "image_px": [H, W],
        "pixel_auroc": aurocs,
        "sensitivity_at_1fp_per_image": sens_at_1fp,
        "_froc": {k: v.tolist() for k, v in froc.items()},
    }


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12.6, 7.4))
    gs = fig.add_gridspec(2, 3)

    # top row: example image, GT, matched-filter detection
    img, mask, cents = make_image(100)
    sc = score_log(img)
    for k, (M, title, cmap) in enumerate([
            (img, "synthetic medical image", "gray"),
            (mask, "lesion ground truth", "gray"),
            (sc, "LoG matched-filter detection", "inferno")]):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(M, cmap=cmap)
        for (cy, cx) in cents:
            ax.add_patch(plt.Circle((cx, cy), 9, ec="#39d0ff", fc="none",
                                    lw=1.2))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # bottom-left: pixel AUROC bars
    ax = fig.add_subplot(gs[1, 0])
    names = list(res["pixel_auroc"].keys())
    vals = [res["pixel_auroc"][n] for n in names]
    cols = ["#8a8f98", "#c0392b", "#1a7a4a"]
    ax.bar(range(len(names)), vals, color=cols)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["fixed", "wafer z", "LoG match"], fontsize=8.5)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("pixel AUROC")
    ax.set_title("Pixel separability", fontsize=10)
    for i, v in enumerate(vals):
        ax.text(i, v + .006, f"{v:.2f}", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # bottom-mid+right: FROC curves
    ax = fig.add_subplot(gs[1, 1:])
    for (name, _), c in zip(DETECTORS, cols):
        cur = np.array(res["_froc"][name])
        o = np.argsort(cur[:, 0])
        ax.plot(cur[o, 0], cur[o, 1], "-o", ms=3, color=c, label=name)
    ax.axvline(1.0, color="0.6", ls=":", lw=1)
    ax.set_xlabel("false positives per image")
    ax.set_ylabel("lesion sensitivity")
    ax.set_xlim(0, 4); ax.set_ylim(0, 1.02)
    s = res["sensitivity_at_1fp_per_image"]["LoG matched filter"]
    ax.set_title(f"FROC: the CAD metric — the scale-matched filter reaches "
                 f"{s:.0%} sensitivity at 1 FP/image", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    fig.suptitle("Wafer-inspection detectors, carried over to medical lesion "
                 "detection (CADe)", fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=115)
    plt.close(fig)


def main():
    res = run()
    out = {k: v for k, v in res.items() if k != "_froc"}
    with open(RESULTS_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "fujifilm_lesion_cad.png"), res)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
