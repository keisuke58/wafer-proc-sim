"""
富の集中 × イーロン・マスクの「夢想」可視化
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
import matplotlib.patheffects as pe

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
BG, PANEL, GRID, WHITE, DIM = "#0d0d0d", "#141414", "#2a2a2a", "#f0f0f0", "#777777"

def sa(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE, labelsize=8)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
    ax.grid(color=GRID, lw=0.5, alpha=0.6)

# ── データ ───────────────────────────────────────────────────────────────────

TOP10_VS_BOTTOM = {
    "groups": ["Top 1人\n(Musk)", "Top 10人", "Top 100人", "Top 1000人",
               "下位\n10億人", "下位\n20億人", "下位\n40億人"],
    "wealth_trillion_usd": [0.40, 5.8, 14.0, 28.0, 0.5, 1.2, 2.5],
    "colors": ["#ff1744","#ff5722","#ff9800","#ffc107","#546e7a","#455a64","#37474f"],
}

BILLIONAIRE_GROWTH = {
    "years": [2000,2005,2008,2010,2015,2019,2020,2021,2022,2023,2024],
    "total": [0.9,  2.1,  2.3,  4.0,  6.5,  8.3,  8.0, 13.1, 11.0, 12.2, 14.2],
    "top10": [0.2,  0.5,  0.5,  1.0,  1.8,  2.4,  2.5,  4.5,  3.8,  4.9,  5.8],
    "musk":  [0.0,  0.0,  0.0,  0.0,  0.01, 0.02, 0.03, 0.30, 0.14, 0.22, 0.40],
}

SLEEPING_INCOME = {
    "labels": ["日本人\n平均年収", "医師\n(年収)", "プロ野球\n(年収最高)", "ベゾス\n(1日)", "マスク\n(1日)", "マスク\n(1秒)"],
    "jpy_M":  [4.6,              12,              8,                    340,              1200,             0.014],
    "colors": ["#4CAF50","#8BC34A","#CDDC39","#FF9800","#FF5722","#FF1744"],
}

MUSK_PROJECTS = {
    # name, category, announced, status(0-10), reality_score, dream_scale(T$), color
    "SpaceX_Falcon": ("Falcon 9 再利用", "実績",   2011, 10, 10, 0.05, "#4CAF50"),
    "Tesla_EV":      ("Tesla EV 量産",   "実績",   2012, 10, 9,  0.8,  "#4CAF50"),
    "Starlink":      ("Starlink 衛星",   "実績",   2019, 9,  8,  0.3,  "#8BC34A"),
    "Tesla_FSD":     ("Tesla 完全自動運転","進行中", 2016, 5,  5,  0.5,  "#FFC107"),
    "Neuralink":     ("Neuralink 脳チップ","進行中", 2019, 4,  5,  1.0,  "#FFC107"),
    "Optimus":       ("Optimus ロボット","進行中",  2022, 4,  4,  3.0,  "#FF9800"),
    "xAI_AGI":       ("xAI → AGI 達成", "夢想",   2023, 2,  2,  10.0, "#FF5722"),
    "Mars_City":     ("火星都市 100万人","夢想",   2016, 2,  2,  50.0, "#FF1744"),
    "DOGE_2T":       ("DOGE $2T 削減",  "失速",   2024, 1,  1,  0.1,  "#9E9E9E"),
    "Hyperloop":     ("Hyperloop",       "失速",   2013, 1,  1,  0.2,  "#9E9E9E"),
    "X_everything":  ("X = 全部入りApp","進行中",  2022, 3,  3,  0.4,  "#FF9800"),
}

R_G_DATA = {
    "years": list(range(1950, 2026, 5)),
    "r_capital": [5.0,5.2,5.5,5.8,6.2,7.0,7.5,8.0,8.5,9.0,9.2,9.5,10.0,10.2,9.8,9.5],
    "g_gdp":     [4.5,4.8,5.2,5.0,4.5,3.5,3.0,3.2,2.8,2.5,2.2,1.8, 2.0, 2.2, 2.5, 2.3],
}

WEALTH_WHERE = {
    "labels": ["金融資産\n(株・債券)", "不動産", "プライベート\nエクイティ", "現金・預金", "その他"],
    "pct":    [52, 23, 15, 7, 3],
    "colors": ["#e53935","#fb8c00","#fdd835","#43a047","#1e88e5"],
}

# ── プロット ─────────────────────────────────────────────────────────────────

def p_top10_vs_bottom(ax):
    sa(ax)
    ax.set_title("富の集中 — Top10人 vs 下位40億人 (2024)", fontsize=10, pad=5)
    g = TOP10_VS_BOTTOM["groups"]
    w = TOP10_VS_BOTTOM["wealth_trillion_usd"]
    c = TOP10_VS_BOTTOM["colors"]
    x = np.arange(len(g))
    bars = ax.bar(x, w, color=c, alpha=0.88, width=0.65, edgecolor="#ffffff22")
    for bar, val in zip(bars, w):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.3,
                f"${val:.1f}T", ha="center", va="bottom", color=WHITE, fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(g, color=WHITE, fontsize=8)
    ax.set_ylabel("資産総額 (兆ドル)")
    ax.set_ylim(0, 32)
    # 比較矢印
    ax.annotate("", xy=(1, 5.8), xytext=(6, 2.5),
                arrowprops=dict(arrowstyle="<->", color="#ff4444", lw=1.5))
    ax.text(3.5, 4.2, "Top10 > 下位40億人", color="#ff4444", fontsize=8.5, fontweight="bold", ha="center")
    ax.text(3.5, 3.5, "$5.8T   vs   $2.5T", color="#ff9999", fontsize=8, ha="center")


def p_growth(ax):
    sa(ax)
    ax.set_title("億万長者の資産増加 — コロナ禍で爆発", fontsize=10, pad=5)
    y = BILLIONAIRE_GROWTH["years"]
    ax.fill_between(y, BILLIONAIRE_GROWTH["total"], alpha=0.15, color="#ff5722")
    ax.plot(y, BILLIONAIRE_GROWTH["total"], "o-", color="#ff5722", lw=2, ms=4, label="全億万長者")
    ax.plot(y, BILLIONAIRE_GROWTH["top10"], "s-", color="#ff1744", lw=2, ms=4, label="Top10")
    ax.plot(y, [m*100 for m in BILLIONAIRE_GROWTH["musk"]], "^-",
            color="#FF9800", lw=2, ms=4, label="マスク×100倍スケール")
    ax.axvspan(2020, 2021, alpha=0.08, color="#ff0000")
    ax.text(2020.1, 13.5, "コロナ禍\nで2倍", color="#ff4444", fontsize=7.5)
    ax.set_ylabel("資産 (兆ドル)")
    ax.legend(fontsize=7.5, facecolor="#222", edgecolor="#444", labelcolor=WHITE)
    ax.set_ylim(0, 16)


def p_sleeping_income(ax):
    sa(ax)
    ax.set_title("「寝ながら稼ぐ」格差 — 1日・1秒の収入", fontsize=10, pad=5)
    lb = SLEEPING_INCOME["labels"]
    v  = SLEEPING_INCOME["jpy_M"]
    c  = SLEEPING_INCOME["colors"]
    x  = np.arange(len(lb))
    bars = ax.bar(x, v, color=c, alpha=0.88, width=0.6, edgecolor="#ffffff22")
    for bar, val in zip(bars, v):
        label = f"¥{val:.3f}M" if val < 1 else f"¥{val:.0f}M"
        ax.text(bar.get_x()+bar.get_width()/2, val*1.05,
                label, ha="center", va="bottom", color=WHITE, fontsize=7.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(lb, color=WHITE, fontsize=8)
    ax.set_ylabel("金額 (百万円/単位時間)")
    ax.set_yscale("log")
    ax.set_ylim(1, 3000)
    ax.text(0.02, 0.95, "対数スケール", transform=ax.transAxes, color=DIM, fontsize=7)
    note = "マスクの1秒 = ¥14,000\n日本人の平均時給 ≈ ¥1,100"
    ax.text(0.65, 0.55, note, transform=ax.transAxes, color="#ff9800", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="#1a1a1a", edgecolor="#ff9800", alpha=0.8))


def p_musk_projects(ax):
    sa(ax)
    ax.set_title("マスクの「夢想」マップ — 現実度 vs 市場規模", fontsize=10, pad=5)
    for key, (name, cat, year, status, reality, scale, color) in MUSK_PROJECTS.items():
        ax.scatter(reality, status, s=scale*12+40, c=color, alpha=0.80,
                   edgecolors=WHITE, lw=0.8, zorder=3)
        offset = (0.3, 0.3)
        if key == "Mars_City":    offset = (0.3, -0.8)
        if key == "DOGE_2T":      offset = (-2.5, 0.3)
        if key == "Hyperloop":    offset = (-2.8, -0.8)
        if key == "SpaceX_Falcon":offset = (-3.5, 0.3)
        ax.annotate(f"{name}\n({year})",
                    (reality, status),
                    xytext=(reality+offset[0], status+offset[1]),
                    color=WHITE, fontsize=6.5,
                    arrowprops=dict(arrowstyle="-", color="#555", lw=0.8))
    ax.axvline(5, color=GRID, lw=1, ls="--")
    ax.axhline(5, color=GRID, lw=1, ls="--")
    ax.text(7.5, 9.5, "実現済み", color="#4CAF50", fontsize=9, fontweight="bold")
    ax.text(0.5, 1.0, "失速・撤退", color="#9E9E9E", fontsize=9, fontweight="bold")
    ax.text(0.5, 9.5, "夢想継続中", color="#FF5722", fontsize=9, fontweight="bold")
    ax.text(7.5, 1.0, "?", color="#FFC107", fontsize=20, fontweight="bold")
    ax.set_xlabel("現実度スコア (0=夢想, 10=完全実現)")
    ax.set_ylabel("進捗スコア (0=未着手, 10=完成)")
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 11.5)
    note = "円のサイズ = 想定市場規模 ($T)"
    ax.text(0.01, 0.01, note, transform=ax.transAxes, color=DIM, fontsize=7)

    # 凡例
    for cat, color in [("実績", "#4CAF50"), ("進行中", "#FFC107"), ("夢想", "#FF5722"), ("失速", "#9E9E9E")]:
        ax.scatter([], [], c=color, s=60, label=cat, edgecolors=WHITE, lw=0.6)
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE,
              loc="lower right")


def p_r_vs_g(ax):
    sa(ax)
    ax.set_title("r > g — ピケティの「富の集中」メカニズム", fontsize=10, pad=5)
    y = R_G_DATA["years"]
    r = R_G_DATA["r_capital"]
    g = R_G_DATA["g_gdp"]
    ax.plot(y, r, "o-", color="#e53935", lw=2.2, ms=4, label="r: 資本収益率 (株・不動産)")
    ax.plot(y, g, "s-", color="#1e88e5", lw=2.2, ms=4, label="g: GDP成長率")
    ax.fill_between(y, r, g, where=[ri > gi for ri,gi in zip(r,g)],
                    alpha=0.18, color="#e53935", label="格差拡大ゾーン")
    ax.text(1955, 4.6, "g ≈ r\n(戦後奇跡)", color="#1e88e5", fontsize=7.5)
    ax.text(2010, 5.0, "r >> g\n(21世紀の常態)", color="#e53935", fontsize=8, fontweight="bold")
    ax.axvline(1980, color=DIM, lw=1, ls=":", alpha=0.7)
    ax.text(1981, 9.8, "1980: 新自由主義\n規制緩和・税率低下", color=DIM, fontsize=7)
    ax.set_ylabel("年率 (%)")
    ax.set_ylim(1, 11)
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE, loc="upper left")
    note = "r=資本収益率 が g=経済成長率 を上回り続ける限り\n富は自動的に集中する (Piketty 2014)"
    ax.text(0.01, 0.04, note, transform=ax.transAxes, color=DIM, fontsize=7.5)


def p_where_wealth(ax):
    sa(ax)
    ax.set_title("富裕層の資産の内訳 — なぜ株が最強か", fontsize=10, pad=5)
    wedges, texts, autotexts = ax.pie(
        WEALTH_WHERE["pct"],
        labels=WEALTH_WHERE["labels"],
        colors=WEALTH_WHERE["colors"],
        autopct="%1.0f%%",
        startangle=140,
        pctdistance=0.72,
        wedgeprops=dict(edgecolor="#0d0d0d", lw=1.5),
    )
    for t in texts:
        t.set_color(WHITE)
        t.set_fontsize(8)
    for at in autotexts:
        at.set_color(WHITE)
        at.set_fontsize(8)
        at.set_fontweight("bold")
    note = (
        "株・債券が52%\n"
        "→ 株価上昇だけで寝ながら資産増\n"
        "→ 労働収入と無関係に雪だるま式"
    )
    ax.text(1.0, -0.8, note, color="#ff9800", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="#1a1a1a", edgecolor="#ff9800", alpha=0.8))


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    fig = plt.figure(figsize=(22, 26), facecolor=BG)
    fig.suptitle(
        "富の集中 × マスクの「夢想」全可視化\nTop10が下位40億人を超える世界",
        fontsize=18, color=WHITE, fontweight="bold", y=0.985
    )

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.50, wspace=0.38,
        left=0.06, right=0.97,
        top=0.955, bottom=0.04,
    )

    p_top10_vs_bottom(  fig.add_subplot(gs[0, :2]))
    p_where_wealth(     fig.add_subplot(gs[0,  2]))
    p_growth(           fig.add_subplot(gs[1, :2]))
    p_sleeping_income(  fig.add_subplot(gs[1,  2]))
    p_musk_projects(    fig.add_subplot(gs[2, :2]))
    p_r_vs_g(           fig.add_subplot(gs[2,  2]))

    out = os.path.join(OUT_DIR, "wealth_concentration_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
