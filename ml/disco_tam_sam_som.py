#!/usr/bin/env python3
"""
DISCO — TAM / SAM / SOM Market Sizing
======================================
三重構造で DISCO の市場ポジションを可視化。

TAM: Total Addressable Market   = 全半導体製造装置市場
SAM: Serviceable Addressable Market = DISCO が狙える中工程装置
SOM: Serviceable Obtainable Market  = DISCO の実際の売上
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).parent.parent / "results" / "disco_tam_sam_som.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

DISCO_BLUE  = "#003087"
DISCO_RED   = "#C8102E"
GOLD        = "#E8A000"
GREEN       = "#2E7D32"
LIGHT_BLUE  = "#EEF2F8"
GRAY        = "#90A4AE"

# ─── Data (2024, $B) ──────────────────────────────────────────────────────────
TAM_TOTAL = 108   # 全半導体製造装置市場 (WFE $90B + ATME $18B)
SAM_DISCO = 10    # DISCO狙える中工程装置 (研削/ダイシング/レーザー)
SOM_DISCO = 2.5   # DISCO実売上 (FY2024推定)

# AI特化 TAM/SAM/SOM
TAM_AI = 18       # AI向け製造装置全体 ($B)
SAM_AI = 3.5      # AI向けDISCO狙える装置
SOM_AI = 1.8      # AI向けDISCO売上推定

# 2030年予測
TAM_2030 = 180
SAM_2030 = 22
SOM_2030 = 6.5

TAM_AI_2030 = 45
SAM_AI_2030 = 12
SOM_AI_2030 = 7.0

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 7))
fig.patch.set_facecolor("white")

def style(a, title):
    a.set_title(title, fontsize=10, fontweight="bold", color=DISCO_BLUE, pad=8)
    a.spines[["top","right","left","bottom"]].set_visible(False)

# ── Panel A: 同心円 TAM/SAM/SOM (現在) ───────────────────────────────────────
ax = axes[0]
ax.set_aspect("equal")
ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
ax.axis("off")

radii  = [1.0, 0.62, 0.30]
colors = [LIGHT_BLUE, "#BBDEFB", DISCO_BLUE]
labels = [
    f"TAM\n全装置市場\n${TAM_TOTAL}B",
    f"SAM\nDISCO狙える\n${SAM_DISCO}B",
    f"SOM\nDISCO売上\n${SOM_DISCO}B",
]
text_colors = [DISCO_BLUE, DISCO_BLUE, "white"]

for r, c, lbl, tc in zip(radii, colors, labels, text_colors):
    circle = plt.Circle((0, 0), r, color=c, zorder=3 - radii.index(r))
    ax.add_patch(circle)

# Re-draw in correct z-order (largest first)
ax.cla(); ax.set_aspect("equal")
ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.axis("off")
for r, c in zip(reversed(radii), reversed(colors)):
    ax.add_patch(plt.Circle((0, 0), r, color=c, ec="white", lw=2))

# Labels
positions = [(0, 0.80), (0, 0.44), (0, 0)]
for pos, lbl, tc in zip(positions, labels, text_colors):
    ax.text(pos[0], pos[1], lbl, ha="center", va="center",
            fontsize=8.5, color=tc, fontweight="bold", linespacing=1.5)

# Percentage labels on right side
for r, val, total, nm in zip(radii[1:], [SAM_DISCO, SOM_DISCO],
                              [TAM_TOTAL, SAM_DISCO], ["SAM/TAM", "SOM/SAM"]):
    pct = val / total * 100
    ax.text(r + 0.08, 0.05, f"{pct:.0f}%", fontsize=8, color=DISCO_BLUE,
            fontweight="bold", ha="left")
    ax.annotate("", xy=(r, 0), xytext=(r - 0.01, 0),
                arrowprops=dict(arrowstyle="-", color=DISCO_BLUE, lw=1))

style(ax, "A  DISCO TAM / SAM / SOM (2024)")
ax.text(0, -1.08, f"シェア余地: SOM/TAM = {SOM_DISCO/TAM_TOTAL*100:.1f}%",
        ha="center", fontsize=8, color=DISCO_RED, fontweight="bold")

# ── Panel B: AI特化 同心円 ────────────────────────────────────────────────────
ax = axes[1]
ax.set_aspect("equal")
ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.axis("off")

radii_ai  = [1.0,
             np.sqrt(SAM_AI / TAM_AI),
             np.sqrt(SOM_AI / TAM_AI)]
colors_ai = [LIGHT_BLUE, "#FFCDD2", DISCO_RED]
labels_ai = [
    f"TAM\nAI向け装置\n${TAM_AI}B",
    f"SAM\nDISCO狙える\n${SAM_AI}B",
    f"SOM\nDISCO AI売上\n${SOM_AI}B",
]
tc_ai = [DISCO_BLUE, DISCO_BLUE, "white"]

for r, c in zip(reversed(radii_ai), reversed(colors_ai)):
    ax.add_patch(plt.Circle((0, 0), r, color=c, ec="white", lw=2))

pos_ai = [(0, 0.80), (0, 0.44), (0, 0)]
for pos, lbl, tc in zip(pos_ai, labels_ai, tc_ai):
    ax.text(pos[0], pos[1], lbl, ha="center", va="center",
            fontsize=8.5, color=tc, fontweight="bold", linespacing=1.5)

style(ax, "B  AI チップ特化 TAM / SAM / SOM (2024)")
ax.text(0, -1.08,
        f"AI SOM/TAM = {SOM_AI/TAM_AI*100:.0f}%  vs  全体 {SOM_DISCO/TAM_TOTAL*100:.1f}%",
        ha="center", fontsize=8, color=DISCO_RED, fontweight="bold")

# ── Panel C: 2024→2030 成長バー ───────────────────────────────────────────────
ax = axes[2]
ax.set_facecolor("white"); ax.spines[["top","right"]].set_visible(False)

cats = ["TAM\n全体", "SAM\n全体", "SOM\n全体", "TAM\nAI", "SAM\nAI", "SOM\nAI"]
vals_24 = [TAM_TOTAL, SAM_DISCO, SOM_DISCO, TAM_AI, SAM_AI, SOM_AI]
vals_30 = [TAM_2030,  SAM_2030,  SOM_2030,  TAM_AI_2030, SAM_AI_2030, SOM_AI_2030]
x = np.arange(len(cats))
w = 0.38

bar_colors = [GRAY, GRAY, DISCO_BLUE, "#FFCDD2", "#EF9A9A", DISCO_RED]
b1 = ax.bar(x - w/2, vals_24, w, color=bar_colors, alpha=0.55, label="2024")
b2 = ax.bar(x + w/2, vals_30, w, color=bar_colors, alpha=0.92, label="2030予測")

# Growth annotation
for xi, (v24, v30) in enumerate(zip(vals_24, vals_30)):
    cagr = (v30 / v24) ** (1/6) - 1
    ax.text(xi, v30 + 2, f"+{cagr*100:.0f}%\nCAGR",
            ha="center", fontsize=7, color=bar_colors[xi],
            fontweight="bold", linespacing=1.3)

ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8)
ax.set_ylabel("市場規模 ($B)", fontsize=9)
ax.tick_params(labelsize=8)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.25)
ax.set_ylim(0, max(vals_30) * 1.25)

# Divider between overall and AI
ax.axvline(2.5, color=GRAY, ls="--", lw=1, alpha=0.6)
ax.text(1.0, max(vals_30)*1.18, "全体", ha="center", fontsize=8,
        color=DISCO_BLUE, style="italic")
ax.text(4.0, max(vals_30)*1.18, "AI特化", ha="center", fontsize=8,
        color=DISCO_RED, style="italic")

style(ax, "C  TAM / SAM / SOM 成長予測 2024→2030")

# ─── Overall title ────────────────────────────────────────────────────────────
fig.suptitle(
    "DISCO Corporation — TAM / SAM / SOM 市場ポジション分析\n"
    "SOM/TAM = 2.3%（全体）→ AI特化では 10%。成長余地が大きい。",
    fontsize=11, fontweight="bold", color=DISCO_BLUE, y=1.02,
)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"[✓] {OUT}")
