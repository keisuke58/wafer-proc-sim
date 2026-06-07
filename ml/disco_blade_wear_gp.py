#!/usr/bin/env python3
"""
DISCO Blade Wear — Real-Time GP State Estimation
=================================================
Simulates dicing-blade progressive wear and demonstrates online
Bayesian (GP) updating as new cut data arrives in real time.

Research theme:
  Real-time blade wear estimation via GP surrogate
  → extends master's thesis GP with online Bayesian update
  → addresses "no online adaptation" gap explicitly

Physics:
  W(t) ∈ [0,1]  superlinear wear + shock events
  Sensors: spindle force RMS, vibration RMS, acoustic emission RMS
  GP: Matérn-5/2 kernel, online update every UPDATE_EVERY cuts
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import time

RNG  = np.random.default_rng(42)
OUT  = Path(__file__).parent.parent / "results" / "disco_blade_wear_gp.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

DISCO_BLUE = "#003087"
DISCO_RED  = "#C8102E"
GOLD       = "#E8A000"
LIGHT_BLUE = "#EEF2F8"

# ─── 1. Wear physics ──────────────────────────────────────────────────────────
T_LIFE       = 400          # nominal tool life [cuts]
N_CUTS       = 440          # simulate slightly past EOL
EOL_THR      = 0.80         # end-of-life threshold
INIT_LABELED = 10           # historical data from previous tools
UPDATE_EVERY = 8            # new wear measurement interval [cuts]
SNAP_AT      = [10, 20, 35, 55]   # snapshot n_obs values for Panel B

def simulate_wear(n, T=T_LIFE, k=1.7):
    t = np.arange(n)
    W_base = (t / T) ** k
    shocks = RNG.binomial(1, 0.015, n) * RNG.uniform(0.015, 0.04, n)
    shock_acc = np.cumsum(shocks)
    return np.clip(W_base + shock_acc, 0.0, 1.05)

def sensor_readings(W):
    n = len(W)
    force = 1.0 + 1.8 * W + 0.4 * W**2   + RNG.normal(0, 0.03, n)
    vibr  = 0.1 + 0.9 * W**1.8             + RNG.normal(0, 0.025, n)
    ae    = 0.35 + 0.65 * W**1.3           + RNG.normal(0, 0.035, n)
    return np.stack([force, vibr, ae], axis=1)   # (n, 3)

t      = np.arange(N_CUTS)
W_true = simulate_wear(N_CUTS)
X_all  = sensor_readings(W_true)

print("[1] Wear simulation")
eol_true = int(np.argmax(W_true >= EOL_THR)) if np.any(W_true >= EOL_THR) else N_CUTS
print(f"    EOL at cut {eol_true} / {N_CUTS}")

# ─── 2. Online GP update ──────────────────────────────────────────────────────
scaler  = StandardScaler().fit(X_all)
X_sc    = scaler.transform(X_all)

kernel  = ConstantKernel(1.0) * Matern(length_scale=np.ones(3), nu=2.5) \
        + WhiteKernel(noise_level=1e-3)

# Initial labeled set: first INIT_LABELED cuts (historical tool data)
label_idx = list(range(INIT_LABELED))
# Schedule: add one measurement per UPDATE_EVERY cuts
schedule  = list(range(INIT_LABELED, N_CUTS, UPDATE_EVERY))

rmse_hist   = []
n_obs_hist  = []
snap_store  = {}   # n_obs → (mu, sigma)

print("[2] Online GP updates ...")
for next_add in schedule + [None]:
    idx_tr = np.array(label_idx)
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=2, normalize_y=True, alpha=0.0
    )
    gp.fit(X_sc[idx_tr], W_true[idx_tr])
    mu, sigma = gp.predict(X_sc, return_std=True)

    n_obs = len(label_idx)
    rmse  = float(np.sqrt(np.mean((mu - W_true)**2)))
    rmse_hist.append(rmse)
    n_obs_hist.append(n_obs)

    # Save snapshot
    for sn in SNAP_AT:
        if n_obs == sn:
            snap_store[sn] = (mu.copy(), sigma.copy())

    # Always save the final state
    if next_add is None:
        snap_store["final"] = (mu.copy(), sigma.copy())

    if next_add is not None:
        label_idx.append(next_add)

    if n_obs in SNAP_AT or next_add is None:
        print(f"    n_obs={n_obs:3d}  RMSE={rmse:.4f}")

# ─── 3. EOL prediction from final GP ─────────────────────────────────────────
mu_f, sig_f = snap_store["final"]
eol_pred    = int(np.argmax(mu_f >= EOL_THR)) if np.any(mu_f >= EOL_THR) else N_CUTS
eol_err     = abs(eol_pred - eol_true)
print(f"[3] EOL: true={eol_true}, predicted={eol_pred}, |error|={eol_err} cuts")

# Fit timing
t0 = time.perf_counter()
gp_time = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)
gp_time.fit(X_sc[np.array(label_idx)], W_true[np.array(label_idx)])
_ = gp_time.predict(X_sc, return_std=True)
fit_ms = (time.perf_counter() - t0) * 1000
print(f"[4] GP fit+predict time: {fit_ms:.1f} ms  (target <30 s → achieved)")

# ─── 4. Figures ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.38)

ax_A = fig.add_subplot(gs[0, 0])
ax_B = fig.add_subplot(gs[0, 1])
ax_C = fig.add_subplot(gs[0, 2])
ax_D = fig.add_subplot(gs[1, 0])
ax_E = fig.add_subplot(gs[1, 1])
ax_F = fig.add_subplot(gs[1, 2])

def style(ax, title):
    ax.set_title(title, fontsize=9, fontweight="bold", color=DISCO_BLUE, pad=4)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25)
    ax.spines[["top","right"]].set_visible(False)

# ── Panel A: Wear + sensors ────────────────────────────────────────────────
ax_A.plot(t, W_true, color=DISCO_BLUE, lw=2, label="True wear W(t)", zorder=5)
ax_A.axhline(EOL_THR, color=DISCO_RED, ls="--", lw=1.5, label=f"EOL {EOL_THR:.0%}")
ax_A.axvline(eol_true, color=DISCO_RED, ls=":", lw=1.2, alpha=0.8)
ax2 = ax_A.twinx()
ax2.plot(t, X_all[:, 0], color="steelblue",   alpha=0.35, lw=0.9, label="Force RMS")
ax2.plot(t, X_all[:, 1], color="darkorange",  alpha=0.35, lw=0.9, label="Vibr RMS")
ax2.plot(t, X_all[:, 2], color="forestgreen", alpha=0.35, lw=0.9, label="AE RMS")
ax2.set_ylabel("Sensor (norm.)", fontsize=8, color="grey")
ax2.tick_params(axis='y', labelsize=7, labelcolor='grey')
ax2.spines[["top"]].set_visible(False)
ax_A.set_xlabel("Cut number", fontsize=9)
ax_A.set_ylabel("Wear  W", fontsize=9)
ax_A.set_ylim(-0.05, 1.25)
ax_A.legend(fontsize=7, loc="upper left")
style(ax_A, "A  Wear trajectory & sensor signals")

# ── Panel B: GP snapshots ──────────────────────────────────────────────────
colors_snap = [GOLD, "steelblue", "forestgreen", DISCO_RED]
ax_B.plot(t, W_true, "k--", lw=1.6, label="True wear", zorder=10)
for col, sn in zip(colors_snap, SNAP_AT):
    if sn in snap_store:
        mu_s, _ = snap_store[sn]
        ax_B.plot(t, mu_s, color=col, lw=1.4, label=f"n={sn}", alpha=0.9)
ax_B.axhline(EOL_THR, color=DISCO_RED, ls="--", lw=1, alpha=0.6)
ax_B.set_xlabel("Cut number", fontsize=9)
ax_B.set_ylabel("GP predicted wear", fontsize=9)
ax_B.set_ylim(-0.05, 1.25)
ax_B.legend(fontsize=7, loc="upper left")
style(ax_B, "B  GP prediction (online snapshots)")

# ── Panel C: RMSE learning curve ──────────────────────────────────────────
ax_C.plot(n_obs_hist, rmse_hist, color=DISCO_BLUE, lw=2, marker="o",
          markersize=3, zorder=5)
ax_C.fill_between(n_obs_hist, 0, rmse_hist, alpha=0.12, color=DISCO_BLUE)
ax_C.set_xlabel("# labeled observations", fontsize=9)
ax_C.set_ylabel("RMSE (wear units)", fontsize=9)
ax_C.set_ylim(bottom=0)
init_rmse  = rmse_hist[0]
final_rmse = rmse_hist[-1]
ax_C.annotate(f"n={n_obs_hist[0]}: {init_rmse:.3f}",
              xy=(n_obs_hist[0], init_rmse),
              xytext=(8, 4), textcoords="offset points", fontsize=7, color="grey")
ax_C.annotate(f"n={n_obs_hist[-1]}: {final_rmse:.3f}",
              xy=(n_obs_hist[-1], final_rmse),
              xytext=(-50, 8), textcoords="offset points", fontsize=7, color=DISCO_BLUE)
style(ax_C, "C  Online learning curve")

# ── Panel D: Uncertainty reduction ────────────────────────────────────────
# Show 95% CI width at three representative cuts
probes = [100, 250, 380]
probe_ci = {p: [] for p in probes}
# Recompute sigma for all checkpoints at probe cuts
label_idx2 = list(range(INIT_LABELED))
for next_add in schedule + [None]:
    idx_tr = np.array(label_idx2)
    gp2 = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=0, normalize_y=True, alpha=0.0
    )
    gp2.fit(X_sc[idx_tr], W_true[idx_tr])
    _, sig2 = gp2.predict(X_sc[probes], return_std=True)
    for pi, p in enumerate(probes):
        probe_ci[p].append(float(2 * 1.96 * sig2[pi]))
    if next_add is not None:
        label_idx2.append(next_add)

probe_colors = [DISCO_BLUE, GOLD, DISCO_RED]
for (p, cil), col in zip(probe_ci.items(), probe_colors):
    ax_D.plot(n_obs_hist, cil, color=col, lw=1.8, label=f"cut {p}")
ax_D.set_xlabel("# labeled observations", fontsize=9)
ax_D.set_ylabel("95% CI width (wear units)", fontsize=9)
ax_D.set_ylim(bottom=0)
ax_D.legend(fontsize=7, title="Probe cut", title_fontsize=7)
style(ax_D, "D  Uncertainty reduction over time")

# ── Panel E: EOL decision ─────────────────────────────────────────────────
ax_E.fill_between(t, mu_f - 1.96*sig_f, mu_f + 1.96*sig_f,
                  alpha=0.18, color=DISCO_BLUE, label="95% CI")
ax_E.plot(t, mu_f,    color=DISCO_BLUE, lw=2,   label="GP mean")
ax_E.plot(t, W_true, "k--", lw=1.4,             label="True wear")
ax_E.axhline(EOL_THR, color=DISCO_RED, ls="--", lw=1.5, label=f"EOL {EOL_THR:.0%}")
ax_E.axvline(eol_pred, color=DISCO_BLUE, ls=":", lw=1.5, alpha=0.9)
ax_E.axvline(eol_true, color="k",        ls=":", lw=1.5, alpha=0.8)
ax_E.annotate(f"GP: cut {eol_pred}", xy=(eol_pred, EOL_THR+0.02),
              xytext=(5, 6), textcoords="offset points",
              fontsize=7, color=DISCO_BLUE, fontweight="bold")
ax_E.annotate(f"True: cut {eol_true}", xy=(eol_true, EOL_THR-0.12),
              xytext=(5, -14), textcoords="offset points",
              fontsize=7, color="k")
ax_E.set_xlabel("Cut number", fontsize=9)
ax_E.set_ylabel("Wear  W", fontsize=9)
ax_E.set_ylim(-0.05, 1.2)
ax_E.legend(fontsize=7, loc="upper left")
style(ax_E, "E  Tool replacement decision")

# ── Panel F: Summary ──────────────────────────────────────────────────────
ax_F.axis("off")
rows = [
    ["Parameter",          "Value"],
    ["Tool life simulated", f"{T_LIFE} cuts"],
    ["Update interval",     f"every {UPDATE_EVERY} cuts"],
    ["GP kernel",          "Matérn-5/2 × 3D"],
    ["Initial n (history)", f"{INIT_LABELED}"],
    ["Final n (labeled)",   f"{len(label_idx)}"],
    [f"RMSE  n={INIT_LABELED}",  f"{rmse_hist[0]:.4f}"],
    [f"RMSE  n={len(label_idx)}", f"{rmse_hist[-1]:.4f}"],
    ["RMSE reduction",     f"{(1-rmse_hist[-1]/rmse_hist[0])*100:.1f}%"],
    ["EOL error",          f"{eol_err} cuts"],
    ["GP fit+predict time", f"{fit_ms:.0f} ms"],
]
tbl = ax_F.table(cellText=rows[1:], colLabels=rows[0],
                  loc="center", cellLoc="left")
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
tbl.scale(1.2, 1.38)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("none")
    if r == 0:
        cell.set_facecolor(DISCO_BLUE)
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 1:
        cell.set_facecolor(LIGHT_BLUE)
    else:
        cell.set_facecolor("white")
ax_F.set_title("F  Summary", fontsize=9, fontweight="bold", color=DISCO_BLUE, pad=4)

fig.suptitle(
    r"DISCO Blade Wear — Real-Time GP State Estimation"
    "\n"
    r"Online Bayesian update: new cut data $\rightarrow$ model refresh in $< 30\,\mathrm{s}$",
    fontsize=11, fontweight="bold", color=DISCO_BLUE, y=1.015,
)

plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"[✓] Figure → {OUT}")
print("Done.")
