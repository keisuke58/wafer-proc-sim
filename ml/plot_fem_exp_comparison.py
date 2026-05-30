"""
FEM vs Experimental comparison figure.

Panel 1: Experimental chipping (Micro2026) + GP curve vs cut depth
Panel 2: FEM deletion_fraction vs cut depth
Panel 3: FEM cutting force (RF2) vs cut depth

Usage:
    python ml/plot_fem_exp_comparison.py
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from validation.experimental_data import CHIPPING_DATA
from ml.train_from_experimental import (
    ExperimentalGPSurrogate, FEATURE_COLS, MODEL_PATH, REF, _sweep_array,
)

RESULTS  = os.path.join(os.path.dirname(__file__), "..", "results")
FEM_CSV  = os.path.join(RESULTS, "parametric_summary_extended.csv")

# ── Load data ─────────────────────────────────────────────────────────────────
exp_df = pd.DataFrame(CHIPPING_DATA)

# Micro2026 reference sweep: blade_W=23, feed=1.0, spindle=30000
ref_mask = (
    (exp_df["blade_W_um"]  == 23) &
    (exp_df["feed_mm_s"]   == 1.0) &
    (exp_df["spindle_rpm"] == 30000)
)
micro_df = exp_df[ref_mask].sort_values("cut_depth_um")

fem_df = pd.read_csv(FEM_CSV).sort_values("cut_depth_um")

# GP surrogate (load or train)
try:
    model = ExperimentalGPSurrogate.load(MODEL_PATH)
except Exception:
    from ml.train_from_experimental import load_data
    X, y, _ = load_data()
    model = ExperimentalGPSurrogate().fit(X, y)

depths_gp = np.linspace(60, 420, 120)
X_sw = _sweep_array("cut_depth_um", depths_gp,
                    {f: REF[f] for f in FEATURE_COLS if f != "cut_depth_um"})
mu_gp, sig_gp = model.predict(X_sw, return_std=True)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 4.5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

BLUE  = "#2166ac"
RED   = "#d62728"
GREEN = "#2ca02c"

# ── Panel 1: Experimental GP + data ──────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.plot(depths_gp, mu_gp, color=BLUE, lw=2, label="GP surrogate")
ax1.fill_between(depths_gp, mu_gp - 2*sig_gp, mu_gp + 2*sig_gp,
                 alpha=0.15, color=BLUE)
ax1.scatter(micro_df["cut_depth_um"], micro_df["chipping_um"],
            color=BLUE, s=70, zorder=5, edgecolors="k", lw=0.6,
            label="Micro2026 (exp)")
ax1.axhline(15.0, color="gray", ls=":", lw=1.2, label="15 µm threshold")
ax1.set_xlabel("Cut Depth [µm]", fontsize=11)
ax1.set_ylabel("Front Chipping [µm]", fontsize=11)
ax1.set_title("① Experimental GP\n(blade_W=23µm, feed=1mm/s, spindle=30krpm)",
              fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.25)
ax1.set_ylim(bottom=0)

# ── Panel 2: FEM deletion_fraction ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.scatter(fem_df["cut_depth_um"], fem_df["deletion_fraction"] * 100,
            color=RED, s=100, zorder=5, edgecolors="k", lw=0.7,
            marker="^", label="FEM (Drucker-Prager)")
# trend line (polynomial fit)
z = np.polyfit(fem_df["cut_depth_um"], fem_df["deletion_fraction"] * 100, 2)
x_fit = np.linspace(60, 380, 100)
ax2.plot(x_fit, np.polyval(z, x_fit), color=RED, lw=1.5, ls="--", alpha=0.6)
ax2.set_xlabel("Cut Depth [µm]", fontsize=11)
ax2.set_ylabel("Deleted Elements [%]", fontsize=11)
ax2.set_title("② FEM Element Deletion\n(deletion_fraction as chipping proxy)",
              fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.25)
ax2.set_ylim(bottom=0)

# annotate peak
peak_row = fem_df.loc[fem_df["deletion_fraction"].idxmax()]
ax2.annotate(f"peak\nd={peak_row['cut_depth_um']:.0f}µm",
             xy=(peak_row["cut_depth_um"], peak_row["deletion_fraction"]*100),
             xytext=(peak_row["cut_depth_um"]+30, peak_row["deletion_fraction"]*100+0.5),
             fontsize=8, arrowprops=dict(arrowstyle="->", color="k", lw=0.8))

# ── Panel 3: FEM cutting force RF2 ───────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
rf2_kn = fem_df["max_RF2_N"] / 1e3
ax3.scatter(fem_df["cut_depth_um"], rf2_kn,
            color=GREEN, s=100, zorder=5, edgecolors="k", lw=0.7,
            marker="s", label="FEM RF2 (vertical)")
z3 = np.polyfit(fem_df["cut_depth_um"], rf2_kn, 1)
ax3.plot(x_fit, np.polyval(z3, x_fit), color=GREEN, lw=1.5, ls="--", alpha=0.6)
ax3.set_xlabel("Cut Depth [µm]", fontsize=11)
ax3.set_ylabel("Max Vertical Force RF2 [kN]", fontsize=11)
ax3.set_title("③ FEM Cutting Force\n(deeper cut → lower resistance)",
              fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.25)
ax3.set_ylim(bottom=0)

# shared annotation
fig.suptitle(
    "4H-SiC Blade Dicing — Experimental GP + FEM Validation  "
    "(blade_W=23 µm, Drucker-Prager + DuctileDamageInitiation)",
    fontsize=11, y=1.02,
)

out = os.path.join(RESULTS, "fem_exp_comparison.png")
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"[✓] Saved → {out}")
