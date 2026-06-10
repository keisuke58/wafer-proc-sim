"""
日本の「上位」は世界では？ — グローバル資産分布可視化
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
BG, PANEL, GRID, WHITE, DIM = "#0d0d0d", "#141414", "#2a2a2a", "#f0f0f0", "#666666"

def sa(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE, labelsize=8)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
    ax.grid(color=GRID, lw=0.5, alpha=0.5)

# ════════════════════════════════════════════════════════════════════════════
# グローバル資産分布 (Credit Suisse Global Wealth Report 2023)
# 単位: 万円 (1USD = 150円換算)
# ════════════════════════════════════════════════════════════════════════════

# 世界の成人人口: 約58億人
GLOBAL_POPULATION_B = 5.8

GLOBAL_WEALTH_DIST = {
    # パーセンタイル: 資産額 (万円)
    1:    0,       # 負債超過
    10:   2,       # ¥2万
    20:   5,       # ¥5万
    30:   15,      # ¥15万
    40:   30,      # ¥30万
    50:   130,     # 中央値 ¥130万 ($8,654)
    60:   300,     # ¥300万
    70:   700,     # ¥700万
    80:   1500,    # ¥1500万
    90:   5000,    # ¥5000万 ($333k) ← 世界上位10%の壁
    95:   10000,   # ¥1億
    99:   15000,   # ¥1.5億 ($1M) ← 世界上位1%の壁
    99.5: 50000,   # ¥5億
    99.9: 75000,   # ¥7.5億 ($5M)
    99.99:750000,  # ¥75億 ($50M)
}

# 国別 資産中央値 (万円, 2023)
COUNTRY_MEDIAN = {
    "Australia":   "オーストラリア",
    "Belgium":     "ベルギー",
    "Japan":       "日本",
    "USA":         "米国",
    "UK":          "英国",
    "Germany":     "ドイツ",
    "France":      "フランス",
    "China":       "中国",
    "Brazil":      "ブラジル",
    "India":       "インド",
    "Nigeria":     "ナイジェリア",
    "DRCongo":     "コンゴ",
}
COUNTRY_MEDIAN_JPY = {
    "Australia": 28000,
    "Belgium":   25000,
    "Japan":     21000,   # ¥2,100万 ($140k)
    "USA":       11000,   # ¥1,100万 ($73k) ← 不平等が大きいので中央値は低め
    "UK":        14000,
    "Germany":   17000,
    "France":    13000,
    "China":      2500,
    "Brazil":      400,
    "India":       120,
    "Nigeria":      30,
    "DRCongo":       5,
}

# Disco キャリア 2080年時点の資産 (万円)
DISCO_SCENARIOS = {
    "disco_10pct":  ("Disco一本 投資10%",   600,    "#90CAF9"),
    "disco_25pct":  ("Disco一本 投資25%",   2500,   "#42A5F5"),
    "disco_35pct":  ("Disco一本 投資35%",   5000,   "#1565C0"),
    "disco_nvidia": ("Disco→NVIDIA",        10000,  "#76FF03"),
}

def wealth_to_global_pct(wealth_man_yen):
    """資産額(万円)→世界パーセンタイル"""
    dist = sorted(GLOBAL_WEALTH_DIST.items())
    for i in range(len(dist)-1):
        lo_p, lo_w = dist[i]
        hi_p, hi_w = dist[i+1]
        if lo_w <= wealth_man_yen < hi_w:
            t = (wealth_man_yen - lo_w) / (hi_w - lo_w + 1e-9)
            return lo_p + t * (hi_p - lo_p)
    if wealth_man_yen < dist[0][1]: return 0.5
    return 99.99

# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def p_global_dist(ax):
    sa(ax)
    ax.set_title("世界の資産分布 — 中央値¥130万の衝撃", fontsize=10, pad=5)

    pcts  = sorted(GLOBAL_WEALTH_DIST.keys())
    vals  = [GLOBAL_WEALTH_DIST[p] for p in pcts]
    color_arr = plt.cm.RdYlGn([p/100 for p in pcts])

    ax.barh(pcts, vals, height=3.5, color=color_arr, alpha=0.85, edgecolor="#00000033")

    # 注目ラインとラベル
    marks = [
        (130,   "世界中央値 ¥130万",  "#FFF176", 5),
        (5000,  "世界上位10% ¥5000万","#FF9800", 91),
        (15000, "世界上位1% ¥1.5億",  "#FF5722", 99.5),
    ]
    for val, label, color, pct_y in marks:
        ax.axvline(val, color=color, lw=1.5, ls="--", alpha=0.8)
        ax.text(val*1.1, pct_y, label, color=color, fontsize=7.5, va="center")

    ax.set_xscale("log")
    ax.set_xlabel("資産額 (万円, 対数スケール)")
    ax.set_ylabel("世界パーセンタイル (%)")
    ax.set_xlim(0.5, 2000000)
    ax.set_ylim(0, 101)

    # 日本の中央値
    ax.axvline(COUNTRY_MEDIAN_JPY["Japan"], color="#4FC3F7", lw=2, ls="-", alpha=0.9)
    ax.text(COUNTRY_MEDIAN_JPY["Japan"]*1.1, 30,
            "日本中央値\n¥2,100万\n→ 世界上位14%",
            color="#4FC3F7", fontsize=8, fontweight="bold")


def p_country_compare(ax):
    sa(ax)
    ax.set_title("国別 資産中央値 (万円) + 世界パーセンタイル", fontsize=10, pad=5)

    countries  = list(COUNTRY_MEDIAN_JPY.keys())
    labels_jp  = [COUNTRY_MEDIAN[c] for c in countries]
    medians    = [COUNTRY_MEDIAN_JPY[c] for c in countries]
    global_pct = [wealth_to_global_pct(m) for m in medians]
    colors     = plt.cm.RdYlGn([p/100 for p in global_pct])

    y = np.arange(len(countries))
    bars = ax.barh(y, medians, color=colors, alpha=0.88, height=0.65, edgecolor="#ffffff11")
    for bar, m, gp in zip(bars, medians, global_pct):
        ax.text(m * 1.05, bar.get_y() + bar.get_height()/2,
                f"¥{m:,}万  (世界{gp:.0f}%ile)",
                color=WHITE, va="center", fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels_jp, color=WHITE, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("資産中央値 (万円)")
    ax.set_xlim(1, 500000)


def p_disco_global(ax):
    sa(ax)
    ax.set_title("Disco キャリア — 日本内 vs 世界 パーセンタイル比較", fontsize=10, pad=5)

    japan_pct = {
        "disco_10pct":  30,
        "disco_25pct":  12,
        "disco_35pct":  7,
        "disco_nvidia": 3,
    }

    keys   = list(DISCO_SCENARIOS.keys())
    labels = [DISCO_SCENARIOS[k][0] for k in keys]
    assets = [DISCO_SCENARIOS[k][1] for k in keys]
    colors = [DISCO_SCENARIOS[k][2] for k in keys]
    j_pcts = [japan_pct[k] for k in keys]
    g_pcts = [wealth_to_global_pct(a) for a in assets]

    x  = np.arange(len(keys))
    w  = 0.32
    b1 = ax.bar(x - w/2, [100 - j for j in j_pcts], w,
                color=colors, alpha=0.65, label="日本国内 上位%", edgecolor="#ffffff22")
    b2 = ax.bar(x + w/2, [100 - g for g in g_pcts], w,
                color=colors, alpha=1.0,  label="世界 上位%",    edgecolor=WHITE, lw=0.8)

    for bar, jp, gp, asset in zip(b1, j_pcts, g_pcts, assets):
        ax.text(bar.get_x()+bar.get_width()/2, 100-jp+0.5,
                f"日本\n上位{jp}%", ha="center", va="bottom", color=WHITE, fontsize=7.5)
    for bar, jp, gp, asset in zip(b2, j_pcts, g_pcts, assets):
        ax.text(bar.get_x()+bar.get_width()/2, 100-gp+0.5,
                f"世界\n上位{100-gp:.1f}%", ha="center", va="bottom",
                color=WHITE, fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=WHITE, fontsize=8)
    ax.set_ylabel("上位 % (高いほど上位)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, facecolor="#222", edgecolor="#444", labelcolor=WHITE)

    note = "薄色=日本基準 / 濃色=世界基準\n→ 日本の「上位」は世界では更に上"
    ax.text(0.01, 0.01, note, transform=ax.transAxes, color=DIM, fontsize=8)


def p_population_pyramid(ax):
    sa(ax)
    ax.set_title("世界58億人のうち — あなたの資産は何人目？", fontsize=10, pad=5)

    scenarios = [
        ("日本平均 ¥1000万",   1000,  "#546e7a"),
        ("Disco入社初年度 ¥0", 0,     "#90CAF9"),
        ("Disco 10年 ¥500万",  500,   "#42A5F5"),
        ("Disco投資25% 最終\n¥2,500万", 2500, "#1565C0"),
        ("Disco投資35% 最終\n¥5,000万", 5000, "#0D47A1"),
        ("→NVIDIA 最終\n¥1億",  10000, "#76FF03"),
    ]

    for i, (label, asset_man, color) in enumerate(scenarios):
        gp  = wealth_to_global_pct(asset_man)
        n_above = GLOBAL_POPULATION_B * (100 - gp) / 100
        n_below = GLOBAL_POPULATION_B * gp / 100

        y = i
        ax.barh(y, n_below, color=color,   alpha=0.85, height=0.6)
        ax.barh(y, n_above, left=n_below, color="#333333", alpha=0.5, height=0.6)
        ax.text(n_below + 0.05, y,
                f"{n_above:.2f}億人があなたより上 ({100-gp:.1f}%)",
                color=WHITE, va="center", fontsize=7.5)
        ax.text(-0.05, y, label, ha="right", va="center", color=color, fontsize=7.5)

    ax.axvline(GLOBAL_POPULATION_B / 2, color=DIM, lw=1, ls="--", alpha=0.6)
    ax.text(GLOBAL_POPULATION_B/2+0.05, -0.5, "中央値", color=DIM, fontsize=7.5)
    ax.set_xlabel("世界人口 (億人)")
    ax.set_xlim(-3.5, GLOBAL_POPULATION_B + 2.5)
    ax.set_yticks([])
    ax.set_ylim(-0.8, len(scenarios) - 0.2)


def p_verdict(ax):
    sa(ax)
    ax.set_facecolor("#060606")
    ax.axis("off")
    ax.set_title("結論: 世界基準で見ると", fontsize=10, pad=5)

    lines = [
        ("日本にいる時点で", "#FFF176",
         "日本の中央値 ¥2,100万 = 世界上位14%\n"
         "日本で「普通」= 世界では上位14人に入る富裕層\n"
         "→ 日本生まれは既に宝くじに当たっている"),

        ("Disco一本 投資10%", "#90CAF9",
         "¥600万 (2080年) → 世界上位 約20%\n"
         "世界58億人中 上位11億人\n"
         "「豊か」だが世界の超富裕層とは別次元"),

        ("Disco一本 投資25%", "#42A5F5",
         "¥2,500万 (2080年) → 世界上位 約8%\n"
         "世界58億人中 上位4.5億人\n"
         "先進国の「普通の上」= 世界では明確な富裕層"),

        ("Disco→NVIDIA 積極投資", "#76FF03",
         "¥1億超 (2080年) → 世界上位 約1%\n"
         "世界58億人中 上位5,800万人\n"
         "ここまで来て初めて「資本が自走」し始める"),

        ("残酷な真実", "#FF5722",
         "どのシナリオでも 世界のTop 0.1% には届かない\n"
         "それは資産 ¥7.5億超 = 労働収入では不可能な領域\n"
         "→ r > g の「資本側」にいる人間だけが到達できる"),
    ]

    y_pos = [0.85, 0.68, 0.50, 0.32, 0.10]
    for (title, color, body), y in zip(lines, y_pos):
        ax.text(0.02, y + 0.08, f"◆ {title}", transform=ax.transAxes,
                color=color, fontsize=9, fontweight="bold")
        ax.text(0.02, y, body, transform=ax.transAxes,
                color=WHITE, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#111",
                          edgecolor=color, alpha=0.7))


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    fig = plt.figure(figsize=(22, 26), facecolor=BG)
    fig.suptitle(
        "日本の「上位」は世界では？ — グローバル資産分布の現実",
        fontsize=17, color=WHITE, fontweight="bold", y=0.987
    )

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        hspace=0.50, wspace=0.35,
        left=0.12, right=0.97,
        top=0.960, bottom=0.04,
    )

    p_global_dist(       fig.add_subplot(gs[0, 0]))
    p_country_compare(   fig.add_subplot(gs[0, 1]))
    p_disco_global(      fig.add_subplot(gs[1, 0]))
    p_population_pyramid(fig.add_subplot(gs[1, 1]))
    p_verdict(           fig.add_subplot(gs[2, 0]))

    # 右下: キーナンバー一覧
    ax_num = fig.add_subplot(gs[2, 1])
    sa(ax_num)
    ax_num.axis("off")
    ax_num.set_title("世界パーセンタイル 早見表 (2024)", fontsize=10, pad=5)

    entries = [
        ("世界中央値",       "¥130万",    50.0,  "#888888"),
        ("日本人の中央値",   "¥2,100万",  86.0,  "#4FC3F7"),
        ("世界上位10%",     "¥5,000万",  90.0,  "#FF9800"),
        ("世界上位1%",      "¥1.5億",    99.0,  "#FF5722"),
        ("世界上位0.1%",    "¥7.5億",    99.9,  "#FF1744"),
        ("イーロン・マスク", "¥60兆",    99.9999,"#FFD700"),
    ]
    for i, (label, asset, pct, color) in enumerate(entries):
        y = 0.88 - i * 0.14
        ax_num.text(0.02, y, label,  transform=ax_num.transAxes,
                    color=color, fontsize=9, fontweight="bold")
        ax_num.text(0.45, y, asset,  transform=ax_num.transAxes,
                    color=WHITE, fontsize=9)
        ax_num.text(0.70, y, f"上位 {100-pct:.2g}%",
                    transform=ax_num.transAxes, color=color, fontsize=9, fontweight="bold")

        # バー
        ax_num.barh(y - 0.04, pct/100, height=0.05, left=0.02,
                    color=color, alpha=0.4, transform=ax_num.transAxes)

    out = os.path.join(OUT_DIR, "global_wealth_position.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
