#!/usr/bin/env python3
"""
DISCO — Chip Volume & End-Market Breakdown
==========================================
Pie charts + proportion bars for semiconductor end-market:
  数百億枚 overall → 1000万枚 AI GPU → 1万施設 data centers
"""
import numpy as np
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

OUT = Path(__file__).parent.parent / "results" / "disco_chip_volume.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

DISCO_BLUE = "#003087"
DISCO_RED  = "#C8102E"
GOLD       = "#E8A000"
GREEN      = "#2E7D32"
LIGHT_BLUE = "#EEF2F8"
PURPLE     = "#6A1B9A"
TEAL       = "#00695C"

# ─── Data ─────────────────────────────────────────────────────────────────────

# 半導体エンドマーケット：売上高シェア (2024)
MARKET_REVENUE = {
    "データセンター\n/サーバー":  30,
    "スマートフォン":             24,
    "PC/ノートPC":               18,
    "自動車":                     12,
    "産業機器":                    9,
    "家電・その他":                7,
}

# 同：枚数シェア（売上と逆転：安いチップが多い）
MARKET_VOLUME = {
    "IoT/MCU\n/センサー":        40,
    "スマートフォン":             22,
    "PC/ノートPC":               15,
    "自動車":                      8,
    "産業機器":                    7,
    "データセンター\n/サーバー":    5,
    "その他":                      3,
}

# DISCOの売上寄与推定（用途別）
DISCO_REVENUE_SHARE = {
    "研削\n(バックグラインド)": 35,
    "ブレード\nダイシング":     30,
    "レーザー\nダイシング":     18,
    "計測・\nその他":           17,
}

# 用途別単価指数（スマホ=1）
UNIT_PRICE = {
    "IoT/MCU": 0.3,
    "スマホ":   1.0,
    "PC/CPU":  3.0,
    "自動車":  5.0,
    "産業":    8.0,
    "AIチップ\n(GPU等)": 80,
    "H100-class": 800,
}

# AI GPU出荷予測（百万枚）
YEARS     = [2022, 2023, 2024, 2025, 2026, 2028, 2030]
AI_ALL    = [  2,    4,   10,   16,   22,   35,   55]
AI_HIGH   = [0.3,  0.8,    3,    5,    7,   12,   20]

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.52, wspace=0.42)
axes = {k: fig.add_subplot(gs[r, c]) for k, (r, c) in {
    "A":(0,0), "B":(0,1), "C":(0,2), "G":(0,3),
    "D":(1,0), "E":(1,1), "F":(1,2), "H":(1,3)}.items()}

def style(a, title):
    a.set_title(title, fontsize=9, fontweight="bold", color=DISCO_BLUE, pad=5)
    a.tick_params(labelsize=8)

PIE_COLORS_REV = [DISCO_RED, "#FF6B6B", "#4A90D9", GOLD, GREEN, GRAY := "#90A4AE"]
PIE_COLORS_VOL = ["#B0BEC5", "#FF6B6B", "#4A90D9", GOLD, GREEN, DISCO_RED, "#CFD8DC"]
PIE_COLORS_DISCO = [DISCO_BLUE, "#1565C0", "#42A5F5", LIGHT_BLUE]

# ── Panel A: 売上高シェア（円グラフ）────────────────────────────────────────
labels_r = list(MARKET_REVENUE.keys())
vals_r   = list(MARKET_REVENUE.values())
explode_r = [0.06 if "データ" in l else 0 for l in labels_r]
wedges, texts, auts = axes["A"].pie(
    vals_r, labels=None, autopct="%1.0f%%",
    colors=PIE_COLORS_REV, explode=explode_r,
    startangle=120, pctdistance=0.78,
    wedgeprops=dict(linewidth=0.8, edgecolor="white")
)
for t in auts:
    t.set_fontsize(8)
axes["A"].legend(wedges, labels_r, fontsize=6.5, loc="lower center",
                 bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
style(axes["A"], "A  エンドマーケット別 売上高シェア (2024)")

# ── Panel B: 枚数シェア（円グラフ）──────────────────────────────────────────
labels_v = list(MARKET_VOLUME.keys())
vals_v   = list(MARKET_VOLUME.values())
explode_v = [0.06 if "IoT" in l else 0 for l in labels_v]
wedges2, _, auts2 = axes["B"].pie(
    vals_v, labels=None, autopct="%1.0f%%",
    colors=PIE_COLORS_VOL, explode=explode_v,
    startangle=90, pctdistance=0.78,
    wedgeprops=dict(linewidth=0.8, edgecolor="white")
)
for t in auts2:
    t.set_fontsize(8)
axes["B"].legend(wedges2, labels_v, fontsize=6.5, loc="lower center",
                 bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
style(axes["B"], "B  エンドマーケット別 枚数シェア (2024)")

# ── Panel C: DISCO売上内訳（円グラフ）────────────────────────────────────────
labels_d = list(DISCO_REVENUE_SHARE.keys())
vals_d   = list(DISCO_REVENUE_SHARE.values())
wedges3, _, auts3 = axes["C"].pie(
    vals_d, labels=None, autopct="%1.0f%%",
    colors=PIE_COLORS_DISCO, startangle=45, pctdistance=0.75,
    wedgeprops=dict(linewidth=0.8, edgecolor="white")
)
for t in auts3:
    t.set_fontsize(8)
axes["C"].legend(wedges3, labels_d, fontsize=7, loc="lower center",
                 bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
style(axes["C"], "C  DISCO 売上内訳（装置種別推定）")

# ── Panel D: 枚数 vs 売上 — ミスマッチを可視化 ───────────────────────────────
categories_d = ["データセンター\n/AI", "自動車", "スマホ", "PC", "IoT/MCU"]
vol_pct      = [5,   8,  22, 15,  40]   # volume %
rev_pct      = [30,  12, 24, 18,   7]   # revenue %
x_d = np.arange(len(categories_d))
w = 0.36
b1 = axes["D"].bar(x_d - w/2, vol_pct, w, color=GRAY, alpha=0.85, label="枚数シェア(%)")
b2 = axes["D"].bar(x_d + w/2, rev_pct, w, color=DISCO_BLUE, alpha=0.85, label="売上シェア(%)")
axes["D"].set_xticks(x_d)
axes["D"].set_xticklabels(categories_d, fontsize=7)
axes["D"].set_ylabel("%", fontsize=9)
axes["D"].legend(fontsize=7)
axes["D"].grid(axis="y", alpha=0.25)
axes["D"].spines[["top","right"]].set_visible(False)
# Annotate reversal
for xi, (vp, rp) in enumerate(zip(vol_pct, rev_pct)):
    ratio = rp / vp
    color = DISCO_RED if ratio > 2 else ("grey" if ratio < 0.5 else GOLD)
    axes["D"].text(xi, max(vp, rp) + 1.2, f"{ratio:.1f}×",
                   ha="center", fontsize=7.5, fontweight="bold", color=color)
axes["D"].text(0, 31, "売上/枚数 倍率", fontsize=6.5, color="grey", style="italic")
style(axes["D"], "D  枚数 vs 売上 ミスマッチ（AI圧倒的）")

# ── Panel E: 用途別単価指数（横棒、対数）────────────────────────────────────
up_labels = list(UNIT_PRICE.keys())
up_vals   = [UNIT_PRICE[k] for k in up_labels]
colors_up = [GRAY if v < 5 else (GOLD if v < 50 else DISCO_RED) for v in up_vals]
axes["E"].barh(up_labels, up_vals, color=colors_up, alpha=0.85)
axes["E"].set_xscale("log")
axes["E"].set_xlabel("単価指数（スマホ=1）", fontsize=9)
axes["E"].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}×"))
for i, (label, val) in enumerate(zip(up_labels, up_vals)):
    axes["E"].text(val * 1.15, i, f"{val}×",
                   va="center", fontsize=8,
                   color=DISCO_RED if val >= 50 else DISCO_BLUE,
                   fontweight="bold")
axes["E"].grid(axis="x", alpha=0.2)
axes["E"].spines[["top","right"]].set_visible(False)
style(axes["E"], "E  用途別チップ単価（対数軸）")

# ── Panel F: AIチップ出荷予測 ─────────────────────────────────────────────────
axes["F"].fill_between(YEARS, AI_ALL,  alpha=0.15, color=DISCO_BLUE)
axes["F"].fill_between(YEARS, AI_HIGH, alpha=0.15, color=DISCO_RED)
axes["F"].plot(YEARS, AI_ALL,  "o-", color=DISCO_BLUE, lw=2.2, ms=6, label="AIチップ全般")
axes["F"].plot(YEARS, AI_HIGH, "s-", color=DISCO_RED,  lw=2.2, ms=6, label="H100クラス以上")
axes["F"].axvline(2025, color="grey", ls=":", lw=1.2)
axes["F"].text(2025.1, max(AI_ALL) * 0.88, "現在", fontsize=7.5, color="grey")
axes["F"].set_xlabel("年", fontsize=9)
axes["F"].set_ylabel("出荷量（百万枚/年）", fontsize=9)
axes["F"].set_xticks(YEARS)
axes["F"].legend(fontsize=7)
axes["F"].grid(alpha=0.25)
axes["F"].spines[["top","right"]].set_visible(False)
# Annotate CAGR
cagr_all  = (AI_ALL[-1]  / AI_ALL[0])  ** (1/(len(YEARS)-1)) - 1
cagr_high = (AI_HIGH[-1] / AI_HIGH[0]) ** (1/(len(YEARS)-1)) - 1
axes["F"].text(2028, AI_ALL[-2] * 0.7,
               f"CAGR\n全般: +{cagr_all*100:.0f}%\n高性能: +{cagr_high*100:.0f}%",
               fontsize=7.5, color=DISCO_BLUE,
               bbox=dict(boxstyle="round,pad=0.3", facecolor=LIGHT_BLUE, alpha=0.8))
style(axes["F"], "F  AIチップ出荷量予測 2022〜2030（百万枚）")

# ── Panel G: Donut — 枚数(外輪) vs 売上(内輪) ──────────────────────────────
# Outer ring = volume share, inner ring = revenue share (same segments)
seg_labels = ["DC/AI", "スマホ", "PC", "自動車", "産業", "IoT/その他"]
seg_vol = [5,  22, 15,  8,  7,  43]
seg_rev = [30, 24, 18, 12,  9,   7]
seg_colors = [DISCO_RED, "#FF6B6B", "#4A90D9", GOLD, GREEN, "#B0BEC5"]

# Outer ring (volume)
w_o, t_o = axes["G"].pie(
    seg_vol, radius=1.0, labels=None, autopct=None,
    colors=seg_colors, startangle=90,
    wedgeprops=dict(width=0.38, edgecolor="white", linewidth=1.2))
# Inner ring (revenue)
w_i, t_i = axes["G"].pie(
    seg_rev, radius=0.60, labels=None, autopct=None,
    colors=seg_colors, startangle=90,
    wedgeprops=dict(width=0.38, edgecolor="white", linewidth=1.2))
axes["G"].text(0, 0, "外: 枚数\n内: 売上", ha="center", va="center",
               fontsize=7.5, fontweight="bold", color=DISCO_BLUE)
legend_patches = [mpatches.Patch(color=c, label=l)
                  for c, l in zip(seg_colors, seg_labels)]
axes["G"].legend(handles=legend_patches, fontsize=6.5, loc="lower center",
                 bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
style(axes["G"], "G  ドーナツ: 枚数(外)vs売上(内) 比較")

# ── Panel H: 積み上げ面グラフ — 半導体市場成長 2024→2030 ────────────────────
years_h = [2024, 2025, 2026, 2027, 2028, 2029, 2030]
# Market size by segment ($B), growing at different rates
dc_ai  = np.array([150, 185, 225, 270, 330, 400, 480])
mobile = np.array([120, 125, 130, 135, 140, 145, 150])
pc     = np.array([ 90,  93,  96,  98, 100, 102, 104])
auto   = np.array([ 60,  68,  78,  90, 104, 118, 134])
indus  = np.array([ 45,  48,  52,  57,  63,  69,  76])
iot    = np.array([ 35,  36,  37,  38,  39,  40,  41])

stack_data = [iot, indus, auto, pc, mobile, dc_ai]
stack_colors = ["#B0BEC5", GREEN, GOLD, "#4A90D9", "#FF6B6B", DISCO_RED]
stack_labels = ["IoT/その他", "産業", "自動車", "PC", "スマホ", "DC/AI"]

bottom = np.zeros(len(years_h))
for dat, col, lab in zip(stack_data, stack_colors, stack_labels):
    axes["H"].fill_between(years_h, bottom, bottom + dat, alpha=0.82,
                           color=col, label=lab)
    bottom += dat

total_2024 = sum(d[0] for d in stack_data)
total_2030 = sum(d[-1] for d in stack_data)
axes["H"].text(2024.1, total_2024 + 10, f"${total_2024}B", fontsize=7.5, color="grey")
axes["H"].text(2029.5, total_2030 + 10, f"${total_2030}B", fontsize=7.5,
               color=DISCO_BLUE, fontweight="bold")
axes["H"].set_xlabel("年", fontsize=9)
axes["H"].set_ylabel("市場規模 ($B)", fontsize=9)
axes["H"].legend(fontsize=6.5, loc="upper left", ncol=2)
axes["H"].set_xlim(2024, 2030)
axes["H"].grid(axis="y", alpha=0.2)
axes["H"].spines[["top","right"]].set_visible(False)
style(axes["H"], "H  半導体エンドマーケット成長予測 2024→2030")

fig.suptitle(
    "DISCO Corporation — 半導体エンドマーケット構造と DISCO のポジション\n"
    r"枚数シェアは小さくても売上インパクトが大きい AI チップ層に重心移動中",
    fontsize=11, fontweight="bold", color=DISCO_BLUE, y=1.01,
)

plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"[✓] {OUT}")
