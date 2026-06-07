#!/usr/bin/env python3
"""
DISCO Blade Wear GP/Hybrid — Real Data Validation
===================================================
Dataset : IEEE PHM 2012 PRONOSTIA (FEMTO-ST) Bearing Dataset
          6 run-to-failure bearing experiments (Learning Set)
          Sensors: horizontal + vertical acceleration [g], 2560 samples/file
          https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset

Wear proxy: W = file_index / n_files  (normalized time-to-failure)
            Vibration RMS ↑ monotonically as bearing degrades → maps to blade wear

Features  : h_rms, v_rms, kurtosis_h  (analogous to force/vibr/AE)
Models    : GP (Matérn-5/2) | LSTM | Hybrid (LSTM trend + GP residual)
CV        : Leave-One-Bearing-Out
Output    : results/disco_blade_wear_realdata.png
"""
import sys, os, io, zipfile
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from collections import defaultdict

RNG = np.random.default_rng(42)
torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT    = Path(__file__).parent.parent / "results" / "disco_blade_wear_realdata.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
ZIP_PATH = Path(__file__).parent.parent / "data" / "pronostia.zip"

DISCO_BLUE = "#003087"
DISCO_RED  = "#C8102E"
GOLD       = "#E8A000"
GREEN      = "#2E7D32"
LIGHT_BLUE = "#EEF2F8"

print(f"[device] {DEVICE}")

# ─── 1. Load PRONOSTIA bearing data from zip ──────────────────────────────────
if not ZIP_PATH.exists():
    print(f"[ERROR] {ZIP_PATH} not found. Download the PRONOSTIA dataset first:")
    print("  curl -L -o data/pronostia.zip https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset/archive/refs/heads/master.zip")
    sys.exit(1)

print(f"[1] Loading PRONOSTIA from {ZIP_PATH.name} ...")

BEARING_SETS = {
    "Cond1": ["Bearing1_1", "Bearing1_2"],
    "Cond2": ["Bearing2_1", "Bearing2_2"],
    "Cond3": ["Bearing3_1", "Bearing3_2"],
}
ALL_BEARINGS = [b for bs in BEARING_SETS.values() for b in bs]
STEP = 5    # subsample every STEP files (speed up)

def extract_features(raw_bytes):
    """From one acc_*.csv file, extract h_rms, v_rms, kurtosis_h."""
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes), header=None,
                         names=["hr","min","sec","usec","h","v"])
        h = df["h"].values.astype(float)
        v = df["v"].values.astype(float)
        h_rms = float(np.sqrt(np.mean(h**2)))
        v_rms = float(np.sqrt(np.mean(v**2)))
        mu_h  = h.mean()
        std_h = h.std() + 1e-9
        kurt  = float(np.mean(((h - mu_h)/std_h)**4))
        return h_rms, v_rms, kurt
    except Exception:
        return None

bearing_data = defaultdict(list)   # bearing_name → [(h_rms, v_rms, kurt, W), ...]

with zipfile.ZipFile(ZIP_PATH) as zf:
    all_names = zf.namelist()
    for bearing in ALL_BEARINGS:
        prefix = f"ieee-phm-2012-data-challenge-dataset-master/Learning_set/{bearing}/"
        files = sorted([n for n in all_names if n.startswith(prefix) and "acc_" in n])
        n_files = len(files)
        print(f"    {bearing}: {n_files} files → subsample every {STEP} → {len(files[::STEP])} points")
        for i, fname in enumerate(files[::STEP]):
            with zf.open(fname) as f:
                feat = extract_features(f.read())
            if feat is not None:
                W = i / max(len(files[::STEP]) - 1, 1)   # normalized 0→1
                bearing_data[bearing].append((feat[0], feat[1], feat[2], W))

# Build arrays
all_X, all_y, all_grp = [], [], []
for bearing, samples in bearing_data.items():
    arr = np.array(samples)      # (n, 4): h_rms, v_rms, kurt, W
    if len(arr) < 5:
        continue
    all_X.append(arr[:, :3])
    all_y.append(arr[:, 3])
    all_grp.extend([bearing] * len(arr))

X_raw = np.vstack(all_X).astype(np.float32)
y_all  = np.concatenate(all_y).astype(np.float32)
grp    = np.array(all_grp)

scaler = StandardScaler().fit(X_raw)
X_sc   = scaler.transform(X_raw)
groups = list(bearing_data.keys())

print(f"[2] Total samples: {len(y_all)}, bearings: {len(groups)}")
print(f"    W range: {y_all.min():.3f}–{y_all.max():.3f}")

# ─── 2. GP LOO-CV ─────────────────────────────────────────────────────────────
print("[3] GP leave-one-bearing-out CV ...")
kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(3), nu=2.5) + WhiteKernel(1e-3)

gp_preds = np.zeros_like(y_all)
gp_stds  = np.zeros_like(y_all)
fold_rmse_gp = []

for held in groups:
    tr = grp != held
    te = grp == held
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)
    gp.fit(X_sc[tr], y_all[tr])
    mu, sig = gp.predict(X_sc[te], return_std=True)
    gp_preds[te] = mu
    gp_stds[te]  = sig
    rmse = float(np.sqrt(np.mean((mu - y_all[te])**2)))
    fold_rmse_gp.append(rmse)
    print(f"    {held}: RMSE={rmse:.4f}")

gp_cv_rmse = float(np.mean(fold_rmse_gp))
r2_gp = float(1 - np.sum((gp_preds - y_all)**2) / np.sum((y_all - y_all.mean())**2))
print(f"    GP LOO-CV RMSE={gp_cv_rmse:.4f}  R²={r2_gp:.3f}")

# ─── 3. LSTM + Hybrid LOO-CV ──────────────────────────────────────────────────
WIN = 15

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

def make_windows(X, y, g, groups, win):
    Xw, yw, gw = [], [], []
    for grp_id in groups:
        m = g == grp_id
        Xg, yg = X[m], y[m]
        if len(Xg) <= win:
            continue
        for i in range(win, len(Xg)):
            Xw.append(Xg[i-win:i])
            yw.append(yg[i])
            gw.append(grp_id)
    if not Xw:
        return np.zeros((0, win, X.shape[1]), np.float32), np.zeros(0, np.float32), np.array([])
    return np.stack(Xw).astype(np.float32), np.array(yw, np.float32), np.array(gw)

Xw, yw, gw = make_windows(X_sc, y_all, grp, groups, WIN)
print(f"[4] LSTM+Hybrid LOO-CV  ({len(Xw)} windows, WIN={WIN}) ...")

lstm_preds_w = np.zeros(len(Xw))
hyb_preds_w  = np.zeros(len(Xw))
fold_rmse_lstm, fold_rmse_hyb = [], []

for held in groups:
    tr_m = gw != held
    te_m = gw == held
    if te_m.sum() == 0:
        continue
    Xw_tr, yw_tr = Xw[tr_m], yw[tr_m]
    Xw_te, yw_te = Xw[te_m], yw[te_m]

    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xw_tr), torch.from_numpy(yw_tr)),
        batch_size=32, shuffle=True
    )
    lstm = WearLSTM().to(DEVICE)
    opt  = torch.optim.Adam(lstm.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    for _ in range(120):
        lstm.train()
        for xb, yb in loader:
            opt.zero_grad()
            nn.MSELoss()(lstm(xb.to(DEVICE)), yb.to(DEVICE)).backward()
            nn.utils.clip_grad_norm_(lstm.parameters(), 1.0)
            opt.step()
        sched.step()

    lstm.eval()
    with torch.no_grad():
        lstm_tr_pred = lstm(torch.from_numpy(Xw_tr).to(DEVICE)).cpu().numpy()
        lstm_te_pred = lstm(torch.from_numpy(Xw_te).to(DEVICE)).cpu().numpy()

    lstm_preds_w[te_m] = lstm_te_pred
    rmse_lstm = float(np.sqrt(np.mean((lstm_te_pred - yw_te)**2)))
    fold_rmse_lstm.append(rmse_lstm)

    resid_tr = yw_tr - lstm_tr_pred
    gp_res = GaussianProcessRegressor(
        kernel=ConstantKernel(0.1) * Matern(length_scale=np.ones(3), nu=2.5) + WhiteKernel(1e-3),
        n_restarts_optimizer=1, normalize_y=False
    )
    gp_res.fit(Xw_tr[:, -1, :], resid_tr)
    resid_mu, _ = gp_res.predict(Xw_te[:, -1, :], return_std=True)
    hyb_te = lstm_te_pred + resid_mu
    hyb_preds_w[te_m] = hyb_te
    rmse_hyb = float(np.sqrt(np.mean((hyb_te - yw_te)**2)))
    fold_rmse_hyb.append(rmse_hyb)
    print(f"    {held}: LSTM={rmse_lstm:.4f}  Hyb={rmse_hyb:.4f}")

lstm_cv_rmse = float(np.mean(fold_rmse_lstm))
hyb_cv_rmse  = float(np.mean(fold_rmse_hyb))
r2_hyb = float(1 - np.sum((hyb_preds_w - yw)**2) / np.sum((yw - yw.mean())**2))
print(f"    LSTM LOO-CV RMSE={lstm_cv_rmse:.4f}")
print(f"    Hybrid LOO-CV RMSE={hyb_cv_rmse:.4f}  R²={r2_hyb:.3f}")

# ─── 4. Demo bearing trajectory ───────────────────────────────────────────────
demo = max(groups, key=lambda g: (grp == g).sum())
mask_d  = grp == demo
X_d, y_d = X_sc[mask_d], y_all[mask_d]
gp_demo = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)
gp_demo.fit(X_sc[~mask_d], y_all[~mask_d])
gp_mu_d, gp_sig_d = gp_demo.predict(X_d, return_std=True)
print(f"[5] Demo bearing: {demo}  ({len(y_d)} points)")

# ─── 5. Plot ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.38)
ax  = {k: fig.add_subplot(gs[r, c]) for k, (r, c) in {
    "A":(0,0),"B":(0,1),"C":(0,2),"D":(1,0),"E":(1,1),"F":(1,2)}.items()}

def style(a, title):
    a.set_title(title, fontsize=9, fontweight="bold", color=DISCO_BLUE, pad=4)
    a.tick_params(labelsize=8); a.grid(alpha=0.25)
    a.spines[["top","right"]].set_visible(False)

# A — Sensor signals for demo bearing
t_d = np.arange(len(y_d))
X_d_raw = scaler.inverse_transform(X_d)
ax["A"].plot(t_d, X_d_raw[:, 0], color=DISCO_BLUE, lw=1.2, alpha=0.8, label="h-RMS")
ax["A"].plot(t_d, X_d_raw[:, 1], color=GREEN,      lw=1.2, alpha=0.8, label="v-RMS")
ax2 = ax["A"].twinx()
ax2.plot(t_d, y_d, color=DISCO_RED, lw=1.8, label="Wear W", ls="--")
ax2.set_ylabel("Wear W", fontsize=8, color=DISCO_RED)
ax2.tick_params(axis="y", labelsize=7, labelcolor=DISCO_RED)
ax2.spines[["top"]].set_visible(False)
ax["A"].set_xlabel("Time step", fontsize=9); ax["A"].set_ylabel("Vibr. RMS [g]", fontsize=9)
ax["A"].legend(fontsize=7, loc="upper left")
style(ax["A"], f"A  Real bearing signals ({demo})")

# B — GP prediction on demo trajectory
cuts_d = np.arange(len(y_d))
ax["B"].fill_between(cuts_d, gp_mu_d - 1.96*gp_sig_d, gp_mu_d + 1.96*gp_sig_d,
                     alpha=0.18, color=DISCO_BLUE)
ax["B"].plot(cuts_d, y_d,     "k--", lw=1.5, label="True wear W")
ax["B"].plot(cuts_d, gp_mu_d, color=DISCO_BLUE, lw=2, label="GP (LOO-CV)")
ax["B"].axhline(0.9, color=DISCO_RED, ls="--", lw=1.2, label="EOL (W=0.9)")
ax["B"].set_xlabel("Time step", fontsize=9); ax["B"].set_ylabel("W", fontsize=9)
ax["B"].set_ylim(-0.05, 1.15); ax["B"].legend(fontsize=7)
style(ax["B"], "B  GP LOO-CV on real bearing")

# C — Scatter GP
ax["C"].scatter(y_all, gp_preds, s=8, alpha=0.5, color=DISCO_BLUE)
ax["C"].plot([0,1],[0,1],"k--",lw=1)
ax["C"].set_xlabel("True W", fontsize=9); ax["C"].set_ylabel("GP predicted W", fontsize=9)
ax["C"].text(0.05, 0.88, f"RMSE={gp_cv_rmse:.3f}\nR²={r2_gp:.3f}",
             transform=ax["C"].transAxes, fontsize=8, color=DISCO_BLUE)
ax["C"].set_xlim(0,1); ax["C"].set_ylim(0,1)
style(ax["C"], "C  GP: predicted vs true")

# D — Scatter Hybrid
ax["D"].scatter(yw, hyb_preds_w, s=8, alpha=0.5, color=GOLD)
ax["D"].plot([0,1],[0,1],"k--",lw=1)
ax["D"].set_xlabel("True W", fontsize=9); ax["D"].set_ylabel("Hybrid predicted W", fontsize=9)
ax["D"].text(0.05, 0.88, f"RMSE={hyb_cv_rmse:.3f}\nR²={r2_hyb:.3f}",
             transform=ax["D"].transAxes, fontsize=8, color=GOLD)
ax["D"].set_xlim(0,1); ax["D"].set_ylim(0,1)
style(ax["D"], "D  Hybrid: predicted vs true")

# E — Per-bearing RMSE bars
n_b = len(fold_rmse_gp)
x_b = np.arange(n_b)
ax["E"].bar(x_b-0.25, fold_rmse_gp,   0.25, color=DISCO_BLUE, alpha=0.85, label=f"GP   ({gp_cv_rmse:.3f})")
ax["E"].bar(x_b,      fold_rmse_lstm,  0.25, color=GREEN,      alpha=0.85, label=f"LSTM ({lstm_cv_rmse:.3f})")
ax["E"].bar(x_b+0.25, fold_rmse_hyb,  0.25, color=GOLD,       alpha=0.85, label=f"Hyb  ({hyb_cv_rmse:.3f})")
ax["E"].set_xticks(x_b); ax["E"].set_xticklabels(groups, rotation=30, fontsize=7)
ax["E"].set_ylabel("RMSE", fontsize=9); ax["E"].set_ylim(bottom=0)
ax["E"].legend(fontsize=7)
style(ax["E"], "E  Per-bearing RMSE (LOO-CV)")

# F — Summary table
ax["F"].axis("off")
rows = [
    ["Metric",       "GP",                "LSTM",              "Hybrid"],
    ["Dataset",      "PRONOSTIA",         "PRONOSTIA",         "PRONOSTIA"],
    ["# bearings",   "6",                 "6",                 "6"],
    ["CV method",    "LOO-bearing",       "LOO-bearing",       "LOO-bearing"],
    ["RMSE",         f"{gp_cv_rmse:.4f}", f"{lstm_cv_rmse:.4f}", f"{hyb_cv_rmse:.4f}"],
    ["R²",           f"{r2_gp:.3f}",      "—",                 f"{r2_hyb:.3f}"],
    ["UQ",           "Yes",               "No",                "Yes"],
    ["Sensors",      "h-RMS,v-RMS,kurt",  "same",              "same"],
]
tbl = ax["F"].table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1.15, 1.45)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("none")
    if r == 0:
        clr = [DISCO_BLUE, DISCO_BLUE, GREEN, GOLD][min(c, 3)]
        cell.set_facecolor(clr); cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 1:
        cell.set_facecolor(LIGHT_BLUE)
    else:
        cell.set_facecolor("white")
style(ax["F"], "F  Summary — Real data validation")

fig.suptitle(
    "DISCO Blade Wear GP/Hybrid — Real Data Validation\n"
    r"IEEE PHM 2012 PRONOSTIA Bearing Dataset  $\cdot$  Leave-one-bearing-out CV",
    fontsize=11, fontweight="bold", color=DISCO_BLUE, y=1.015,
)

plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"[✓] Figure → {OUT}")
print("Done.")
