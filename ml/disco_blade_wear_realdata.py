#!/usr/bin/env python3
"""
DISCO Blade Wear GP/Hybrid — Real Data Validation
===================================================
Dataset : UCI Milling (Mill.csv) — 167 cuts, real flank wear VB [mm]
          https://archive.ics.uci.edu/dataset/197/milling
Sensors : Fx, Fy, Fz (force), vx, vy, vz (vibration), AE_spindle, AE_table
Target  : VB → normalized wear W = VB / VB_max ∈ [0,1]

Models  : GP (Matérn-5/2) | Hybrid (LSTM trend + GP residual)
Output  : results/disco_blade_wear_realdata.png
"""
import sys, os, io, zipfile, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

RNG = np.random.default_rng(42)
torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(__file__).parent.parent / "results" / "disco_blade_wear_realdata.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DISCO_BLUE = "#003087"
DISCO_RED  = "#C8102E"
GOLD       = "#E8A000"
GREEN      = "#2E7D32"
LIGHT_BLUE = "#EEF2F8"

# ─── 1. Download & load ───────────────────────────────────────────────────────
CSV_PATH = DATA_DIR / "mill.csv"
URL = "https://archive.ics.uci.edu/static/public/197/milling.zip"

if not CSV_PATH.exists():
    print(f"[download] {URL}")
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            # find the csv inside
            names = zf.namelist()
            csv_name = next(n for n in names if n.lower().endswith(".csv"))
            with zf.open(csv_name) as f:
                CSV_PATH.write_bytes(f.read())
        print(f"[download] saved → {CSV_PATH}")
    except Exception as e:
        print(f"[download] FAILED: {e}")
        sys.exit(1)
else:
    print(f"[load] using cached {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"[1] Loaded {len(df)} rows, columns: {list(df.columns)}")
print(df.head(3))

# ─── 2. Preprocess ───────────────────────────────────────────────────────────
# Columns: case, run, feedrate, material, VB (flank wear mm),
#          time-avg sensors: Fx Fy Fz vx vy vz AE_spindle AE_table smcAC smcDC
# Drop rows where VB is NaN or 0 (no measurement)
df = df.dropna(subset=["VB"])
df = df[df["VB"] >= 0].copy()

# Group by (case, run) = one tool life trajectory
df["tool_id"] = df["case"].astype(str) + "_" + df["run"].astype(str)
groups = df["tool_id"].unique()
print(f"[2] {len(groups)} tool life trajectories")

# Use 3 sensor aggregates analogous to our simulation
#   force_rms  ← sqrt(Fx²+Fy²+Fz²)
#   vibr_rms   ← sqrt(vx²+vy²+vz²)
#   ae_rms     ← AE_spindle
sensor_cols = ["Fx", "Fy", "Fz", "vx", "vy", "vz", "AE_spindle"]
for c in sensor_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=sensor_cols)

df["force_rms"] = np.sqrt(df["Fx"]**2 + df["Fy"]**2 + df["Fz"]**2)
df["vibr_rms"]  = np.sqrt(df["vx"]**2 + df["vy"]**2 + df["vz"]**2)
df["ae_rms"]    = df["AE_spindle"].abs()

FEAT_COLS = ["force_rms", "vibr_rms", "ae_rms"]
VB_MAX = df["VB"].max()
EOL_THR = 0.30  # ISO: VB > 0.3mm = tool worn out

df["W"] = (df["VB"] / VB_MAX).clip(0, 1)
df["eol"] = df["VB"] >= EOL_THR
print(f"    VB range: {df['VB'].min():.3f}–{VB_MAX:.3f} mm  (EOL threshold {EOL_THR} mm)")
print(f"    W range:  {df['W'].min():.3f}–{df['W'].max():.3f}")

# Re-group after cleaning
groups = df["tool_id"].unique()
print(f"    Clean trajectories: {len(groups)}")

# ─── 3. Build arrays: stack all cuts, record group ────────────────────────────
X_raw = df[FEAT_COLS].values.astype(np.float32)
y_all = df["W"].values.astype(np.float32)
grp   = df["tool_id"].values

scaler = StandardScaler().fit(X_raw)
X_sc   = scaler.transform(X_raw)

# ─── 4. GP cross-validation (LOGO: leave-one-tool-out) ───────────────────────
print("[3] GP leave-one-tool-out CV ...")
kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(3), nu=2.5) + WhiteKernel(1e-3)

logo = LeaveOneGroupOut()
gp_preds = np.zeros_like(y_all)
gp_stds  = np.zeros_like(y_all)
fold_rmse_gp = []

for i, (tr, te) in enumerate(logo.split(X_sc, y_all, grp)):
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)
    gp.fit(X_sc[tr], y_all[tr])
    mu, sig = gp.predict(X_sc[te], return_std=True)
    gp_preds[te] = mu
    gp_stds[te]  = sig
    rmse = float(np.sqrt(np.mean((mu - y_all[te])**2)))
    fold_rmse_gp.append(rmse)
    if i % 5 == 0:
        print(f"    fold {i:2d}  RMSE={rmse:.4f}")

gp_cv_rmse = float(np.mean(fold_rmse_gp))
print(f"    GP LOO-CV RMSE = {gp_cv_rmse:.4f}")

# ─── 5. LSTM cross-validation ─────────────────────────────────────────────────
WIN = 10   # window size (cuts per window) — smaller than simulation (fewer cuts/tool)

class WearLSTM(nn.Module):
    def __init__(self, inp=3, hid=32, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(inp, hid, layers, batch_first=True,
                            dropout=0.1 if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hid, 16), nn.ReLU(),
                                  nn.Linear(16, 1), nn.Sigmoid())
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)

def make_windows_grp(X, y, g, groups, win=WIN):
    """Make sliding windows per group (tool trajectory)."""
    Xw_list, yw_list, gw_list = [], [], []
    for grp_id in groups:
        mask = g == grp_id
        Xg, yg = X[mask], y[mask]
        if len(Xg) <= win:
            continue
        for i in range(win, len(Xg)):
            Xw_list.append(Xg[i-win:i])
            yw_list.append(yg[i])
            gw_list.append(grp_id)
    if not Xw_list:
        return np.zeros((0, win, X.shape[1]), dtype=np.float32), \
               np.zeros(0, dtype=np.float32), np.array([])
    return (np.stack(Xw_list).astype(np.float32),
            np.array(yw_list, dtype=np.float32),
            np.array(gw_list))

print("[4] LSTM + Hybrid LOO-CV ...")
Xw, yw, gw = make_windows_grp(X_sc, y_all, grp, groups, WIN)
print(f"    Window samples: {len(Xw)}")

# Map group IDs back to original indices for alignment
df_idx = df.reset_index(drop=True)
df_idx["_orig"] = np.arange(len(df_idx))

lstm_preds_w = np.zeros(len(Xw))
hyb_preds_w  = np.zeros(len(Xw))
fold_rmse_lstm, fold_rmse_hyb = [], []
grp_unique_w  = np.unique(gw)

for i, held_grp in enumerate(grp_unique_w):
    tr_mask = gw != held_grp
    te_mask = gw == held_grp
    if te_mask.sum() == 0:
        continue

    Xw_tr, yw_tr = Xw[tr_mask], yw[tr_mask]
    Xw_te, yw_te = Xw[te_mask], yw[te_mask]

    # Train LSTM
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xw_tr), torch.from_numpy(yw_tr)),
        batch_size=16, shuffle=True
    )
    lstm = WearLSTM().to(DEVICE)
    opt  = torch.optim.Adam(lstm.parameters(), lr=5e-3, weight_decay=1e-4)
    for _ in range(80):
        lstm.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = nn.MSELoss()(lstm(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(lstm.parameters(), 1.0)
            opt.step()

    lstm.eval()
    with torch.no_grad():
        lstm_tr_pred = lstm(torch.from_numpy(Xw_tr).to(DEVICE)).cpu().numpy()
        lstm_te_pred = lstm(torch.from_numpy(Xw_te).to(DEVICE)).cpu().numpy()

    lstm_preds_w[te_mask] = lstm_te_pred
    rmse_lstm = float(np.sqrt(np.mean((lstm_te_pred - yw_te)**2)))
    fold_rmse_lstm.append(rmse_lstm)

    # GP residual correction (trained on training residuals)
    resid_tr = yw_tr - lstm_tr_pred
    # Use last (non-windowed) feature of each training window as GP input
    X_gp_tr = Xw_tr[:, -1, :]   # (n_tr, 3)
    X_gp_te = Xw_te[:, -1, :]

    gp_res = GaussianProcessRegressor(
        kernel=ConstantKernel(0.1) * Matern(length_scale=np.ones(3), nu=2.5) + WhiteKernel(1e-3),
        n_restarts_optimizer=1, normalize_y=False
    )
    gp_res.fit(X_gp_tr, resid_tr)
    resid_te_mu, _ = gp_res.predict(X_gp_te, return_std=True)
    hyb_te = lstm_te_pred + resid_te_mu
    hyb_preds_w[te_mask] = hyb_te
    rmse_hyb = float(np.sqrt(np.mean((hyb_te - yw_te)**2)))
    fold_rmse_hyb.append(rmse_hyb)

    if i % 5 == 0:
        print(f"    fold {i:2d}  LSTM={rmse_lstm:.4f}  Hyb={rmse_hyb:.4f}")

lstm_cv_rmse = float(np.mean(fold_rmse_lstm))
hyb_cv_rmse  = float(np.mean(fold_rmse_hyb))
print(f"    LSTM LOO-CV RMSE = {lstm_cv_rmse:.4f}")
print(f"    Hybrid LOO-CV RMSE = {hyb_cv_rmse:.4f}")

# ─── 6. Pick one tool for visualization ───────────────────────────────────────
# Use the trajectory with most cuts
demo_grp = max(groups, key=lambda g: (grp == g).sum())
mask_d  = grp == demo_grp
X_d, y_d = X_sc[mask_d], y_all[mask_d]
n_d = len(y_d)
print(f"[5] Demo trajectory '{demo_grp}': {n_d} cuts")

# Full GP on remaining tools → predict demo
tr_d = ~mask_d
gp_demo = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)
gp_demo.fit(X_sc[tr_d], y_all[tr_d])
gp_mu_d, gp_sig_d = gp_demo.predict(X_d, return_std=True)

# EOL detection
eol_true_d = int(np.argmax(y_d >= (EOL_THR / VB_MAX))) if np.any(y_d >= (EOL_THR / VB_MAX)) else n_d
eol_gp_d   = int(np.argmax(gp_mu_d >= (EOL_THR / VB_MAX))) if np.any(gp_mu_d >= (EOL_THR / VB_MAX)) else n_d

# ─── 7. Plot ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.38)
axes = {k: fig.add_subplot(gs[r, c]) for k, (r, c) in {
    "A":(0,0),"B":(0,1),"C":(0,2),"D":(1,0),"E":(1,1),"F":(1,2)}.items()}

def style(a, title):
    a.set_title(title, fontsize=9, fontweight="bold", color=DISCO_BLUE, pad=4)
    a.tick_params(labelsize=8)
    a.grid(alpha=0.25)
    a.spines[["top","right"]].set_visible(False)

# Panel A — Wear distribution across all tools
axes["A"].hist(y_all, bins=25, color=DISCO_BLUE, alpha=0.75, edgecolor="white")
axes["A"].axvline(EOL_THR / VB_MAX, color=DISCO_RED, ls="--", lw=1.8,
                  label=f"EOL (W={EOL_THR/VB_MAX:.2f})")
axes["A"].set_xlabel("Normalized wear W", fontsize=9)
axes["A"].set_ylabel("# cuts", fontsize=9)
axes["A"].legend(fontsize=7)
style(axes["A"], "A  Wear distribution (167 real cuts)")

# Panel B — Demo trajectory: truth vs GP
cuts_d = np.arange(n_d)
axes["B"].fill_between(cuts_d, gp_mu_d - 1.96*gp_sig_d, gp_mu_d + 1.96*gp_sig_d,
                       alpha=0.18, color=DISCO_BLUE)
axes["B"].plot(cuts_d, y_d,      "k--", lw=1.5, label="True wear")
axes["B"].plot(cuts_d, gp_mu_d,  color=DISCO_BLUE, lw=2, label="GP (LOO)")
axes["B"].axhline(EOL_THR / VB_MAX, color=DISCO_RED, ls="--", lw=1.2)
axes["B"].axvline(eol_true_d, color="k",        ls=":", lw=1.2)
axes["B"].axvline(eol_gp_d,   color=DISCO_BLUE, ls=":", lw=1.4)
axes["B"].set_xlabel("Cut # (within tool)", fontsize=9)
axes["B"].set_ylabel("Wear W", fontsize=9)
axes["B"].legend(fontsize=7)
style(axes["B"], "B  Demo trajectory: GP vs truth")

# Panel C — Scatter: GP predicted vs true (all tools LOO)
# Align GP predictions on non-windowed data
axes["C"].scatter(y_all, gp_preds, s=8, alpha=0.5, color=DISCO_BLUE, label="GP")
lim = [0, 1.05]
axes["C"].plot(lim, lim, "k--", lw=1)
axes["C"].set_xlabel("True wear W", fontsize=9)
axes["C"].set_ylabel("Predicted wear W", fontsize=9)
axes["C"].set_xlim(lim); axes["C"].set_ylim(lim)
axes["C"].legend(fontsize=7)
r2_gp = float(1 - np.sum((gp_preds - y_all)**2) / np.sum((y_all - y_all.mean())**2))
axes["C"].text(0.05, 0.88, f"RMSE={gp_cv_rmse:.3f}\nR²={r2_gp:.3f}",
               transform=axes["C"].transAxes, fontsize=8, color=DISCO_BLUE)
style(axes["C"], "C  GP LOO-CV: predicted vs true")

# Panel D — Scatter: Hybrid predicted vs true (windowed)
axes["D"].scatter(yw, hyb_preds_w, s=8, alpha=0.5, color=GOLD, label="Hybrid")
axes["D"].plot(lim, lim, "k--", lw=1)
axes["D"].set_xlabel("True wear W", fontsize=9)
axes["D"].set_ylabel("Predicted wear W", fontsize=9)
axes["D"].set_xlim(lim); axes["D"].set_ylim(lim)
r2_hyb = float(1 - np.sum((hyb_preds_w - yw)**2) / np.sum((yw - yw.mean())**2))
axes["D"].text(0.05, 0.88, f"RMSE={hyb_cv_rmse:.3f}\nR²={r2_hyb:.3f}",
               transform=axes["D"].transAxes, fontsize=8, color=GOLD)
axes["D"].legend(fontsize=7)
style(axes["D"], "D  Hybrid LOO-CV: predicted vs true")

# Panel E — Per-fold RMSE comparison
n_folds_common = min(len(fold_rmse_gp), len(fold_rmse_lstm), len(fold_rmse_hyb))
x_f = np.arange(n_folds_common)
axes["E"].bar(x_f - 0.25, fold_rmse_gp[:n_folds_common],  0.25, color=DISCO_BLUE,
              alpha=0.8, label=f"GP   ({gp_cv_rmse:.3f})")
axes["E"].bar(x_f,        fold_rmse_lstm[:n_folds_common], 0.25, color=GREEN,
              alpha=0.8, label=f"LSTM ({lstm_cv_rmse:.3f})")
axes["E"].bar(x_f + 0.25, fold_rmse_hyb[:n_folds_common], 0.25, color=GOLD,
              alpha=0.8, label=f"Hyb  ({hyb_cv_rmse:.3f})")
axes["E"].set_xlabel("Fold (tool ID)", fontsize=9)
axes["E"].set_ylabel("RMSE", fontsize=9)
axes["E"].set_ylim(bottom=0)
axes["E"].legend(fontsize=7)
style(axes["E"], "E  Per-fold RMSE (real data LOO-CV)")

# Panel F — Summary table
axes["F"].axis("off")
rows = [
    ["Metric",           "GP",                  "LSTM",              "Hybrid"],
    ["Dataset",          "UCI Milling",          "UCI Milling",       "UCI Milling"],
    ["# cuts",           "167",                  "167",               "167"],
    ["CV method",        "LOO-tool",             "LOO-tool",          "LOO-tool"],
    ["RMSE",             f"{gp_cv_rmse:.4f}",   f"{lstm_cv_rmse:.4f}", f"{hyb_cv_rmse:.4f}"],
    ["R²",               f"{r2_gp:.3f}",         "—",                 f"{r2_hyb:.3f}"],
    ["UQ",               "Yes",                  "No",                "Yes"],
    ["EOL err (demo)",   f"|{abs(eol_gp_d-eol_true_d)}| cuts", "—",  "—"],
]
tbl = axes["F"].table(cellText=rows[1:], colLabels=rows[0],
                       loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1.15, 1.45)
col_clrs = [DISCO_BLUE, DISCO_BLUE, GREEN, GOLD]
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("none")
    if r == 0:
        cell.set_facecolor(col_clrs[c] if c < len(col_clrs) else DISCO_BLUE)
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 1:
        cell.set_facecolor(LIGHT_BLUE)
    else:
        cell.set_facecolor("white")
style(axes["F"], "F  Summary — Real data validation")

fig.suptitle(
    "DISCO Blade Wear GP/Hybrid — Real Data Validation\n"
    r"UCI Milling dataset (167 cuts, VB flank wear)  $\cdot$  Leave-one-tool-out CV",
    fontsize=11, fontweight="bold", color=DISCO_BLUE, y=1.015,
)

plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"[✓] Figure → {OUT}")
print("Done.")
