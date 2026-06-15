"""
WM-811K dataset exploration → overview figure + stats for the report.

WM-811K is the de-facto benchmark for wafer-map defect-pattern recognition:
811,457 real silicon wafer maps from 46,393 lots, collected from a real fab
(released by NTU MIR Lab). Each wafer map is a 2-D integer grid with values
{0 = no die / outside wafer, 1 = pass die, 2 = fail die}. ~172,950 maps carry
an expert label: one of 8 spatial failure patterns (Center, Donut, Edge-Loc,
Edge-Ring, Loc, Near-full, Random, Scratch) or ``none``.

Why this matters for DISCO: the *spatial pattern* of failing die encodes the
root-cause process signature (edge-ring → etch/CMP uniformity, center → spin
coat, scratch → handling). Auto-classifying it = "intelligent inspection",
the same problem class as kerf-chipping grading and CFRP defect localization.

Usage
-----
    python vision/wm811k_explore.py --trust            # full stats (~1-2 min)
    python vision/wm811k_explore.py --trust --sample 40000   # faster per-wafer stats

Outputs
-------
    results/wm811k_overview.png   — 9-class gallery + distributions
    results/wm811k_stats.json     — machine-readable statistics
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PKL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "external", "WM-811K",
    "MIR-WM811K", "Python", "WM811K.pkl",
)
CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
           "Near-full", "Random", "Scratch", "none"]
DEFECT_CLASSES = [c for c in CLASSES if c != "none"]


def flatten_label(v) -> str:
    """failureType cells are stored as str, [], or nested np arrays."""
    while isinstance(v, (list, tuple, np.ndarray)):
        if len(v) == 0:
            return ""
        v = v[0]
    return str(v).strip()


def defect_fraction(wm: np.ndarray) -> float:
    """Fail die / (pass + fail die). 0 if no die present."""
    wm = np.asarray(wm)
    n_die = int((wm >= 1).sum())
    if n_die == 0:
        return 0.0
    return float((wm == 2).sum()) / n_die


def main() -> None:
    ap = argparse.ArgumentParser(description="Explore WM-811K → figure + stats")
    ap.add_argument("--trust", action="store_true",
                    help="acknowledge that WM811K.pkl will be unpickled")
    ap.add_argument("--pkl", default=PKL_PATH)
    ap.add_argument("--sample", type=int, default=60000,
                    help="rows sampled for per-wafer (size/defect) stats")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not args.trust:
        raise PermissionError(
            "Refusing to unpickle WM811K.pkl without --trust "
            "(pickle executes arbitrary code; source is an external mirror).")

    import pandas as pd

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_png = args.out or os.path.join(repo_root, "results", "wm811k_overview.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    print(f"[1/4] loading {args.pkl} ...")
    df = pd.read_pickle(args.pkl)  # noqa: S301 — gated behind --trust
    n_total = len(df)
    n_lots = df["lotName"].nunique()
    labels = df["failureType"].map(flatten_label)
    labeled_mask = labels.isin(CLASSES)              # excludes "" (unlabeled)
    n_labeled = int(labeled_mask.sum())
    print(f"      {n_total:,} wafer maps in {n_lots:,} lots; "
          f"{n_labeled:,} labelled")

    # ── class distribution (cheap, full data) ────────────────────────────────
    counts = {c: int((labels == c).sum()) for c in CLASSES}
    n_unlabeled = n_total - n_labeled
    defect_total = sum(counts[c] for c in DEFECT_CLASSES)
    print("[2/4] class counts:")
    for c in CLASSES:
        print(f"        {c:>10}: {counts[c]:>7,}")
    print(f"        {'(unlabeled)':>10}: {n_unlabeled:>7,}")

    # ── per-wafer stats on a sample (size + defect fraction) ──────────────────
    print(f"[3/4] per-wafer stats on {min(args.sample, n_total):,} sampled maps ...")
    rng = np.random.default_rng(0)
    samp_idx = rng.choice(n_total, min(args.sample, n_total), replace=False)
    sdf = df.iloc[samp_idx]
    slabels = labels.iloc[samp_idx].to_numpy()

    heights, widths, die_counts, defect_fracs = [], [], [], []
    for wm in sdf["waferMap"].values:
        wm = np.asarray(wm)
        heights.append(wm.shape[0])
        widths.append(wm.shape[1])
        die_counts.append(int((wm >= 1).sum()))
        defect_fracs.append(defect_fraction(wm))
    heights = np.array(heights); widths = np.array(widths)
    die_counts = np.array(die_counts); defect_fracs = np.array(defect_fracs)

    # defect fraction per defect class (sampled)
    frac_by_class = {}
    for c in DEFECT_CLASSES:
        m = slabels == c
        if m.any():
            frac_by_class[c] = defect_fracs[m]

    stats = {
        "n_total": n_total, "n_lots": int(n_lots), "n_labeled": n_labeled,
        "n_unlabeled": int(n_unlabeled), "defect_total": defect_total,
        "class_counts": counts,
        "sample_size": int(len(samp_idx)),
        "wafer_height": {"min": int(heights.min()), "max": int(heights.max()),
                         "median": float(np.median(heights))},
        "wafer_width": {"min": int(widths.min()), "max": int(widths.max()),
                        "median": float(np.median(widths))},
        "die_count": {"min": int(die_counts.min()), "max": int(die_counts.max()),
                      "median": float(np.median(die_counts))},
        "unique_shapes_in_sample": int(len({(int(h), int(w))
                                            for h, w in zip(heights, widths)})),
    }
    stats_path = os.path.join(os.path.dirname(out_png), "wm811k_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"      stats → {stats_path}")

    # ── figure ────────────────────────────────────────────────────────────────
    print("[4/4] rendering figure ...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # wafer colormap: 0=blank(gray) 1=pass(green) 2=fail(red)
    wmap = ListedColormap(["#f0f0f0", "#3a8a3a", "#d33"])

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.15, 1.15, 1.0],
                          hspace=0.45, wspace=0.35)

    # (a) one representative wafer per class — pick a clear, medium-size example
    def pick_example(cls):
        cand = df[(labels == cls)]
        if len(cand) == 0:
            return None
        # prefer a map close to median size for legibility
        for _, row in cand.head(400).iterrows():
            wm = np.asarray(row["waferMap"])
            if 25 <= wm.shape[0] <= 60 and 25 <= wm.shape[1] <= 60:
                return wm
        return np.asarray(cand.iloc[0]["waferMap"])

    for i, cls in enumerate(CLASSES):
        r, cc = divmod(i, 6)
        ax = fig.add_subplot(gs[r, cc])
        wm = pick_example(cls)
        if wm is not None:
            ax.imshow(wm, cmap=wmap, vmin=0, vmax=2, interpolation="nearest")
        ax.set_title(f"{cls}\n(n={counts[cls]:,})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    # (b) class distribution (log) — spans bottom-left
    ax = fig.add_subplot(gs[2, 0:3])
    order = DEFECT_CLASSES + ["none"]
    vals = [counts[c] for c in order]
    colors = ["#d33"] * len(DEFECT_CLASSES) + ["#3a8a3a"]
    ax.bar(range(len(order)), vals, color=colors, log=True)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("wafer maps (log)")
    ax.set_title(f"(b) Labelled class distribution — heavy imbalance "
                 f"({defect_total:,} defect vs {counts['none']:,} none)",
                 fontsize=9)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=6.5)

    # (c) wafer-map size scatter (variable resolution)
    ax = fig.add_subplot(gs[2, 3:5])
    ax.scatter(widths, heights, s=3, alpha=0.15, color="tab:blue",
               edgecolors="none")
    ax.set_xlabel("map width (die)"); ax.set_ylabel("map height (die)")
    ax.set_title(f"(c) Variable wafer-map sizes\n"
                 f"{stats['unique_shapes_in_sample']} unique shapes in sample",
                 fontsize=9)

    # (d) defect-die fraction per defect class
    ax = fig.add_subplot(gs[2, 5])
    bp_data = [frac_by_class[c] for c in DEFECT_CLASSES if c in frac_by_class]
    bp_lab = [c for c in DEFECT_CLASSES if c in frac_by_class]
    ax.boxplot(bp_data, vert=True, showfliers=False)
    ax.set_xticklabels(bp_lab, rotation=90, fontsize=6.5)
    ax.set_ylabel("fail-die fraction")
    ax.set_title("(d) Defect severity", fontsize=9)

    fig.suptitle(
        f"WM-811K — {n_total:,} real wafer maps / {n_lots:,} lots / "
        f"{n_labeled:,} expert-labelled (9 classes). "
        "0=no die (gray) · 1=pass (green) · 2=fail (red)",
        fontsize=12, y=0.98)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"      figure → {out_png}")
    print("Done.")


if __name__ == "__main__":
    main()
