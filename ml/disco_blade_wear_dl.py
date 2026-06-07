#!/usr/bin/env python3
"""
DISCO Blade Wear — DL Comparison (GP vs LSTM vs Hybrid)
========================================================
Extends disco_blade_wear_gp.py with deep learning:

  GP     : Matérn-5/2, sparse labels, uncertainty-aware
  LSTM   : 3-sensor sequence → wear, temporal dynamics
  Hybrid : LSTM mean + online GP residual correction
           → temporal structure + calibrated uncertainty

Research narrative:
  - GP excels in cold-start (n=10–35 labels)
  - LSTM learns temporal autocorrelation from dense data
  - Hybrid: LSTM handles trend, GP handles residuals → best RMSE + UQ
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

RNG = np.random.default_rng(42)
torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(__file__).parent.parent / "results" / "disco_blade_wear_dl.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

DISCO_BLUE = "#003087"
DISCO_RED  = "#C8102E"
GOLD       = "#E8A000"
GREEN      = "#2E7D32"
LIGHT_BLUE = "#EEF2F8"

print(f"[device] {DEVICE}")

# ─── 1. Simulation (same physics as disco_blade_wear_gp.py) ──────────────────
T_LIFE   = 400
N_CUTS   = 440
EOL_THR  = 0.80
WIN      = 20         # LSTM look-back window [cuts]

def simulate_wear(n, T=T_LIFE, k=1.7):
    t = np.arange(n)
    W_base = (t / T) ** k
    shocks = RNG.binomial(1, 0.015, n) * RNG.uniform(0.015, 0.04, n)
    return np.clip(W_base + np.cumsum(shocks), 0.0, 1.05)

def sensor_readings(W):
    n = len(W)
    force = 1.0 + 1.8*W + 0.4*W**2  + RNG.normal(0, 0.03, n)
    vibr  = 0.1 + 0.9*W**1.8         + RNG.normal(0, 0.025, n)
    ae    = 0.35 + 0.65*W**1.3       + RNG.normal(0, 0.035, n)
    return np.stack([force, vibr, ae], axis=1)

t      = np.arange(N_CUTS)
W_true = simulate_wear(N_CUTS)
X_all  = sensor_readings(W_true)          # (N_CUTS, 3)

eol_true = int(np.argmax(W_true >= EOL_THR)) if np.any(W_true >= EOL_THR) else N_CUTS
print(f"[1] EOL at cut {eol_true}")

# ─── 2. GP baseline (same as before) ─────────────────────────────────────────
INIT_LABELED = 10
UPDATE_EVERY = 8

scaler = StandardScaler().fit(X_all)
X_sc   = scaler.transform(X_all)
kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(3), nu=2.5) + WhiteKernel(1e-3)

label_idx = list(range(INIT_LABELED))
for nx in range(INIT_LABELED, N_CUTS, UPDATE_EVERY):
    label_idx.append(nx)

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True)
gp.fit(X_sc[label_idx], W_true[label_idx])
gp_mu, gp_sig = gp.predict(X_sc, return_std=True)
gp_rmse = float(np.sqrt(np.mean((gp_mu - W_true)**2)))
print(f"[2] GP  RMSE={gp_rmse:.4f}  (n_labels={len(label_idx)})")

# ─── 3. LSTM ──────────────────────────────────────────────────────────────────
class WearLSTM(nn.Module):
    def __init__(self, input_size=3, hidden=48, layers=2, dropout=0.15):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head  = nn.Sequential(
            nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x):                      # x: (B, W, 3)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)   # (B,)

def make_windows(X, Y, win=WIN):
    """Sliding windows: X[i-win:i] → Y[i] for i in [win, N]."""
    Xw = np.stack([X[i-win:i] for i in range(win, len(X))], axis=0)
    Yw = Y[win:]
    return Xw.astype(np.float32), Yw.astype(np.float32)

Xw, Yw = make_windows(X_sc, W_true, WIN)           # (N-WIN, WIN, 3), (N-WIN,)
n_train = int(0.8 * len(Xw))
Xw_tr, Yw_tr = Xw[:n_train], Yw[:n_train]
Xw_te, Yw_te = Xw[n_train:], Yw[n_train:]

loader = DataLoader(TensorDataset(torch.from_numpy(Xw_tr), torch.from_numpy(Yw_tr)),
                    batch_size=32, shuffle=True)

lstm  = WearLSTM().to(DEVICE)
optim = torch.optim.Adam(lstm.parameters(), lr=3e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=150)
loss_fn = nn.MSELoss()

print("[3] Training LSTM ...")
train_losses = []
for epoch in range(200):
    lstm.train()
    ep_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optim.zero_grad()
        pred = lstm(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(lstm.parameters(), 1.0)
        optim.step()
        ep_loss += loss.item() * len(xb)
    sched.step()
    train_losses.append(ep_loss / len(Xw_tr))
    if (epoch + 1) % 50 == 0:
        print(f"    epoch {epoch+1:3d}  loss={train_losses[-1]:.5f}")

lstm.eval()
with torch.no_grad():
    lstm_pred_win = lstm(torch.from_numpy(Xw).to(DEVICE)).cpu().numpy()
    lstm_pred_te  = lstm(torch.from_numpy(Xw_te).to(DEVICE)).cpu().numpy()

# Align LSTM output with full t array (padding first WIN steps with NaN)
lstm_pred_full = np.full(N_CUTS, np.nan)
lstm_pred_full[WIN:] = lstm_pred_win

lstm_rmse = float(np.sqrt(np.mean((lstm_pred_te - Yw_te)**2)))
print(f"    LSTM test RMSE={lstm_rmse:.4f}")

# ─── 4. Hybrid = LSTM trend + GP residual correction ─────────────────────────
# GP trained on LSTM residuals (labeled points only)
resid_all = W_true[WIN:] - lstm_pred_win                   # LSTM residuals
# Use labeled cut indices (offset by WIN)
lab_in_win = [i - WIN for i in label_idx if i >= WIN]
lab_in_win = [i for i in lab_in_win if i < len(resid_all)]

resid_train = resid_all[lab_in_win]
X_resid_tr  = X_sc[WIN:][lab_in_win]

gp_res = GaussianProcessRegressor(
    kernel=ConstantKernel(0.1) * Matern(length_scale=np.ones(3), nu=2.5) + WhiteKernel(1e-3),
    n_restarts_optimizer=2, normalize_y=False,
)
gp_res.fit(X_resid_tr, resid_train)
resid_mu, resid_sig = gp_res.predict(X_sc[WIN:], return_std=True)

hybrid_mu   = lstm_pred_win + resid_mu
hybrid_sig  = resid_sig
hybrid_full = np.full(N_CUTS, np.nan)
hybrid_full[WIN:] = hybrid_mu
hybrid_sig_full = np.full(N_CUTS, np.nan)
hybrid_sig_full[WIN:] = hybrid_sig

hybrid_rmse = float(np.sqrt(np.mean((hybrid_mu - W_true[WIN:])**2)))
print(f"[4] Hybrid RMSE={hybrid_rmse:.4f}")

# ─── 5. Data efficiency sweep ─────────────────────────────────────────────────
print("[5] Data efficiency sweep ...")
n_label_sweep = [5, 10, 15, 20, 30, 40, 55]
gp_rmse_sw, hyb_rmse_sw = [], []
for nl in n_label_sweep:
    idx_s = np.linspace(0, N_CUTS-1, nl, dtype=int)
    # GP
    gp_s = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, normalize_y=True)
    gp_s.fit(X_sc[idx_s], W_true[idx_s])
    mu_s, _ = gp_s.predict(X_sc, return_std=True)
    gp_rmse_sw.append(float(np.sqrt(np.mean((mu_s - W_true)**2))))
    # Hybrid: same LSTM, update GP residual on nl points ∩ win
    idx_win = [i - WIN for i in idx_s if i >= WIN and i - WIN < len(resid_all)]
    if len(idx_win) >= 2:
        gp_rs = GaussianProcessRegressor(
            kernel=ConstantKernel(0.1) * Matern(length_scale=np.ones(3), nu=2.5) + WhiteKernel(1e-3),
            n_restarts_optimizer=1
        )
        gp_rs.fit(X_sc[WIN:][idx_win], resid_all[idx_win])
        rm_s, _ = gp_rs.predict(X_sc[WIN:], return_std=True)
        hyb_mu_s = lstm_pred_win + rm_s
        hyb_rmse_sw.append(float(np.sqrt(np.mean((hyb_mu_s - W_true[WIN:])**2))))
    else:
        hyb_rmse_sw.append(np.nan)
print(f"    GP  sweep:  {[f'{r:.3f}' for r in gp_rmse_sw]}")
print(f"    Hyb sweep:  {[f'{r:.3f}' if not np.isnan(r) else 'nan' for r in hyb_rmse_sw]}")

# ─── 6. EOL predictions ───────────────────────────────────────────────────────
eol_gp  = int(np.argmax(gp_mu >= EOL_THR))   if np.any(gp_mu >= EOL_THR)          else N_CUTS
eol_lst = int(np.argmax(lstm_pred_full >= EOL_THR)) if np.any(np.nan_to_num(lstm_pred_full) >= EOL_THR) else N_CUTS
eol_hyb = int(np.argmax(np.nan_to_num(hybrid_full) >= EOL_THR)) if np.any(np.nan_to_num(hybrid_full) >= EOL_THR) else N_CUTS

# ─── 7. Plot ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.38)
ax = {k: fig.add_subplot(gs[r, c]) for k, (r, c) in {
    "A":(0,0),"B":(0,1),"C":(0,2),"D":(1,0),"E":(1,1),"F":(1,2)}.items()}

def style(a, title):
    a.set_title(title, fontsize=9, fontweight="bold", color=DISCO_BLUE, pad=4)
    a.tick_params(labelsize=8)
    a.grid(alpha=0.25)
    a.spines[["top","right"]].set_visible(False)

# Panel A — Training curve
ax["A"].plot(train_losses, color=DISCO_BLUE, lw=1.8)
ax["A"].set_xlabel("Epoch", fontsize=9); ax["A"].set_ylabel("MSE loss", fontsize=9)
ax["A"].set_yscale("log")
style(ax["A"], "A  LSTM training curve")

# Panel B — GP vs LSTM vs Hybrid prediction
ax["B"].plot(t, W_true, "k--", lw=1.5, label="True wear", zorder=10)
ax["B"].plot(t, gp_mu,         color=DISCO_BLUE, lw=1.6, label=f"GP  ({gp_rmse:.3f})")
ax["B"].plot(t, lstm_pred_full, color=GREEN,      lw=1.6, label=f"LSTM ({lstm_rmse:.3f})")
ax["B"].plot(t, hybrid_full,    color=GOLD,       lw=1.8, label=f"Hybrid ({hybrid_rmse:.3f})", ls="-")
ax["B"].axhline(EOL_THR, color=DISCO_RED, ls="--", lw=1.2)
ax["B"].set_xlabel("Cut #", fontsize=9); ax["B"].set_ylabel("Wear W", fontsize=9)
ax["B"].set_ylim(-0.05, 1.25); ax["B"].legend(fontsize=7, loc="upper left")
style(ax["B"], "B  GP / LSTM / Hybrid comparison")

# Panel C — Hybrid with uncertainty band
ax["C"].fill_between(t[WIN:], (hybrid_full - 1.96*hybrid_sig_full)[WIN:],
                              (hybrid_full + 1.96*hybrid_sig_full)[WIN:],
                     alpha=0.2, color=GOLD, label="95% CI")
ax["C"].plot(t, W_true,      "k--", lw=1.5, label="True wear")
ax["C"].plot(t, hybrid_full,  color=GOLD, lw=2,   label="Hybrid mean")
ax["C"].axhline(EOL_THR, color=DISCO_RED, ls="--", lw=1.2)
ax["C"].axvline(eol_hyb, color=GOLD, ls=":", lw=1.5)
ax["C"].axvline(eol_true, color="k", ls=":", lw=1.2)
ax["C"].set_xlabel("Cut #", fontsize=9); ax["C"].set_ylabel("Wear W", fontsize=9)
ax["C"].set_ylim(-0.05, 1.25); ax["C"].legend(fontsize=7)
style(ax["C"], "C  Hybrid: LSTM + GP residual (UQ)")

# Panel D — Data efficiency
ax["D"].plot(n_label_sweep, gp_rmse_sw,  color=DISCO_BLUE, lw=2, marker="o", ms=5, label="GP only")
hyb_valid = [(n, r) for n, r in zip(n_label_sweep, hyb_rmse_sw) if not np.isnan(r)]
if hyb_valid:
    nv, rv = zip(*hyb_valid)
    ax["D"].plot(nv, rv, color=GOLD, lw=2, marker="s", ms=5, label="Hybrid")
ax["D"].axhline(lstm_rmse, color=GREEN, ls="--", lw=1.5, label=f"LSTM (full train) {lstm_rmse:.3f}")
ax["D"].set_xlabel("# labeled observations", fontsize=9)
ax["D"].set_ylabel("RMSE (wear units)", fontsize=9)
ax["D"].set_ylim(bottom=0); ax["D"].legend(fontsize=7)
style(ax["D"], "D  Data efficiency: labeled obs vs RMSE")

# Panel E — EOL comparison bar chart
models_eol = ["True EOL", "GP", "LSTM", "Hybrid"]
eol_vals   = [eol_true,  eol_gp, eol_lst, eol_hyb]
colors_eol = ["k", DISCO_BLUE, GREEN, GOLD]
bars = ax["E"].bar(models_eol, eol_vals, color=colors_eol, width=0.55, alpha=0.85)
ax["E"].axhline(eol_true, color="k", ls="--", lw=1.2)
for bar, val in zip(bars, eol_vals):
    ax["E"].text(bar.get_x() + bar.get_width()/2, val + 3,
                 f"{val}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax["E"].set_ylabel("EOL cut #", fontsize=9); ax["E"].set_ylim(0, N_CUTS + 40)
style(ax["E"], "E  End-of-life prediction accuracy")

# Panel F — Summary table
ax["F"].axis("off")
rows = [
    ["Model",    "RMSE",              "EOL error",                "UQ"],
    ["GP",       f"{gp_rmse:.4f}",   f"|{abs(eol_gp-eol_true)}| cuts", "Yes (GP)"],
    ["LSTM",     f"{lstm_rmse:.4f}", f"|{abs(eol_lst-eol_true)}| cuts", "No"],
    ["Hybrid",   f"{hybrid_rmse:.4f}", f"|{abs(eol_hyb-eol_true)}| cuts", "Yes (GP)"],
]
tbl = ax["F"].table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.25, 1.7)
row_colors = [DISCO_BLUE, LIGHT_BLUE, "#E8F5E9", "#FFF8E1"]
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("none")
    if r == 0:
        cell.set_facecolor(DISCO_BLUE); cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor(row_colors[r])
ax["F"].set_title("F  Summary", fontsize=9, fontweight="bold", color=DISCO_BLUE, pad=4)

fig.suptitle(
    "DISCO Blade Wear — Deep Learning Comparison\n"
    r"GP (UQ) $\;\cdot\;$ LSTM (temporal) $\;\cdot\;$ Hybrid LSTM + GP residual (best of both)",
    fontsize=11, fontweight="bold", color=DISCO_BLUE, y=1.015,
)

plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"[✓] Figure → {OUT}")
print("Done.")
