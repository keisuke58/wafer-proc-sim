#!/usr/bin/env python3
"""real_anomaly.py — the inspection detectors, validated on REAL photographs.

`defect_inspect.py` proves the pipeline on synthetic dies where ground truth is
exact. This module answers the natural follow-up — *does it work on real
images?* — using three real steel-surface defect sets released by NEU
(Northeastern University) that ship **pixel-precise ground-truth masks**
(the same evaluation structure as MVTec AD, whose download is blocked here):

  NEU_SDI   20 pairs  640x480 RGB photo + binary defect mask
  NEU_Oil   16 pairs  (oil-pollution defects)
  NEU_SPDI  15 pairs  (steel-plate defects)

There is no defect-free reference die on a lone photograph, so the die-to-die
detectors don't transfer directly. What does transfer is the *statistical*
detector: model the image's own background texture and score every pixel in
units of the local robust noise —

  score(x) = |I - median_k(I)| / (1.4826 * MAD_k)        (adaptive z, k-local)

plus a multi-scale version that ORs coarse and fine background models. As a
baseline, a global fixed threshold on |I - mean| (what a naive tool does).

Evaluation is **pixel-level AUROC** against the real masks — threshold-free,
the standard metric for anomaly localization on MVTec-style data.

Run:  python3 real_anomaly.py       (numpy + scipy + PIL + matplotlib)
"""
import glob
import json
import os

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "external")
SETS = ["NEU_SDI", "NEU_Oil", "NEU_SPDI"]


DOWN = 2      # 2x downsample: 640x480 -> 320x240 (keeps defects >> 1 px,
              # cuts the median-filter cost ~4x per scale)


def load_pairs(setname):
    """Return list of (gray float [0,1], mask bool) for one NEU set."""
    pairs = []
    for jp in sorted(glob.glob(os.path.join(DATA, setname, "Imgs", "*.jpg"))):
        mp = jp[:-4] + ".png"
        if not os.path.exists(mp):
            continue
        im = Image.open(jp).convert("L")
        mk = Image.open(mp).convert("L")
        if DOWN > 1:
            im = im.resize((im.width // DOWN, im.height // DOWN), Image.BILINEAR)
            mk = mk.resize((mk.width // DOWN, mk.height // DOWN), Image.NEAREST)
        img = np.asarray(im, float) / 255.0
        msk = np.asarray(mk, float) > 127
        pairs.append((img, msk))
    return pairs


# ─────────────────────────── detectors ──────────────────────────────────────
def score_fixed(img):
    """Naive baseline: |deviation from the global mean|."""
    return np.abs(img - img.mean())


def _adaptive_z(img, k):
    """|I - median_k| in units of the local MAD — one background scale."""
    med = ndi.median_filter(img, size=k)
    resid = img - med
    mad = ndi.median_filter(np.abs(resid), size=k) + 1e-4
    return np.abs(resid) / (1.4826 * mad)


def score_adaptive(img, k=15):
    """Single-scale adaptive statistical detector (as in defect_inspect)."""
    return _adaptive_z(img, k)


def score_multiscale(img):
    """Multi-scale adaptive z: max over coarse+fine background models, smoothed."""
    z = np.maximum(_adaptive_z(img, 9), _adaptive_z(img, 31))
    return ndi.gaussian_filter(z, 1.5)


DETECTORS = [
    ("fixed |I-mean|", score_fixed),
    ("adaptive z (k=31)", score_adaptive),
    ("multi-scale z", score_multiscale),
]


# ─────────────────────────── evaluation ─────────────────────────────────────
def auroc(scores, labels):
    """Pixel AUROC via rank statistic (Mann-Whitney), memory-light."""
    s = scores.ravel()
    y = labels.ravel()
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1)
    # midranks for ties
    s_sorted = s[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    n_pos = int(y.sum()); n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    return (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    figs = os.path.join(HERE, "figures")
    os.makedirs(figs, exist_ok=True)

    results = {}
    per_image = {name: [] for name, _ in DETECTORS}
    gallery = []          # (setname, img, mask, best-score-map)

    for setname in SETS:
        pairs = load_pairs(setname)
        aucs = {name: [] for name, _ in DETECTORS}
        for idx, (img, msk) in enumerate(pairs):
            for name, fn in DETECTORS:
                a = auroc(fn(img), msk)
                if not np.isnan(a):
                    aucs[name].append(a)
                    per_image[name].append(a)
            if idx in (0, 1) and len(gallery) < 6:
                gallery.append((setname, img, msk, score_multiscale(img)))
        results[setname] = {
            "pairs": len(pairs),
            **{name: round(float(np.mean(v)), 3) for name, v in aucs.items()},
        }
        print(setname, results[setname])

    overall = {name: round(float(np.mean(v)), 3) for name, v in per_image.items()}
    print("overall", overall)

    # ── gallery figure: photo | GT mask | detection map ──
    n = len(gallery)
    fig, ax = plt.subplots(n, 3, figsize=(8.2, 2.3 * n))
    for r, (setname, img, msk, sm) in enumerate(gallery):
        ax[r, 0].imshow(img, cmap="gray"); ax[r, 0].set_ylabel(setname, fontsize=8)
        ax[r, 0].set_title("real photo" if r == 0 else "", fontsize=9)
        ax[r, 1].imshow(msk, cmap="gray")
        ax[r, 1].set_title("ground-truth mask" if r == 0 else "", fontsize=9)
        ax[r, 2].imshow(sm, cmap="inferno")
        ax[r, 2].set_title("multi-scale z detection" if r == 0 else "", fontsize=9)
        for c in range(3):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    fig.suptitle("Real steel-surface photographs — unsupervised detection vs pixel ground truth",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "real_gallery.png"), dpi=130)
    plt.close(fig)

    # ── AUROC bar chart ──
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    xs = np.arange(len(SETS) + 1)
    w = 0.26
    colors = ["#8a8f98", "#1f77b4", "#d62728"]
    for d, (name, _) in enumerate(DETECTORS):
        vals = [results[s][name] for s in SETS] + [overall[name]]
        ax.bar(xs + (d - 1) * w, vals, w, label=name, color=colors[d])
        for x, v in zip(xs + (d - 1) * w, vals):
            ax.text(x, v + .008, f"{v:.2f}", ha="center", fontsize=7.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([s.replace("NEU_", "") for s in SETS] + ["ALL"])
    ax.set_ylim(0.5, 1.02)
    ax.axhline(0.5, color="#999", lw=.8, ls=":")
    ax.set_ylabel("pixel AUROC (vs real GT masks)")
    ax.set_title("Unsupervised defect localization on real photographs", fontsize=10.5)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "real_auroc.png"), dpi=130)
    plt.close(fig)

    out = {"sets": results, "overall_pixel_auroc": overall,
           "n_pairs_total": int(sum(results[s]["pairs"] for s in SETS))}
    with open(os.path.join(HERE, "real_anomaly_results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
