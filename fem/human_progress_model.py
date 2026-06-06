"""
人類の「底辺の底上げ」— 200年の進歩を全グラフ化
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.ticker as mticker

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ── データ ──────────────────────────────────────────────────────────────────

POVERTY = {
    "years": [1820, 1870, 1910, 1950, 1970, 1990, 2000, 2010, 2015, 2020, 2023],
    "rate":  [90,   80,   72,   60,   50,   36,   28,   19,   10,   9.2,  9.0],
}

INFANT_MORTALITY = {
    "years": [1800, 1850, 1900, 1920, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2023],
    "rate":  [430,  400,  360,  310,  216,  185,  145,  105,  93,   76,   53,   37],
}

LIFE_EXPECTANCY = {
    "years": [1800, 1850, 1900, 1930, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2023],
    "world": [29,   31,   33,   38,   48,   52,   57,   62,   65,   67,   70,   73],
    "japan": [None, None, 44,   46,   61,   68,   72,   76,   79,   81,   83,   84],
}

LITERACY = {
    "years": [1820, 1870, 1900, 1930, 1950, 1970, 1990, 2000, 2010, 2023],
    "rate":  [12,   18,   21,   38,   36,   57,   71,   79,   83,   87],
}

CALORIES = {
    "years": [1961, 1970, 1980, 1990, 2000, 2010, 2020],
    "world": [2196, 2411, 2530, 2688, 2788, 2870, 2960],
    "dev":   [1960, 2100, 2270, 2440, 2600, 2750, 2900],
    "rich":  [2980, 3090, 3150, 3230, 3380, 3440, 3500],
}

CHINA_INDIA_POVERTY = {
    "years": [1981, 1990, 2000, 2010, 2015, 2020, 2023],
    "china": [88,   67,   36,   11,   4,    1,    0.4],
    "india": [60,   46,   42,   33,   22,   14,   12],
    "sub_sahara": [55, 57, 58, 48, 41, 38, 36],
}

GINI = {
    "years": [1980, 1990, 2000, 2010, 2020, 2023],
    "usa":   [0.40, 0.43, 0.46, 0.47, 0.49, 0.49],
    "china": [0.28, 0.33, 0.43, 0.48, 0.46, 0.47],
    "japan": [0.31, 0.33, 0.34, 0.34, 0.33, 0.33],
    "world": [0.69, 0.68, 0.66, 0.63, 0.61, 0.60],
}

BILLIONAIRE_WEALTH = {
    "years": [2000, 2005, 2010, 2015, 2020, 2024],
    "total_trillion_usd": [0.9, 2.1, 4.0, 6.5, 8.0, 14.2],
    "top10_trillion_usd": [0.2, 0.5, 1.0, 1.8, 2.5,  5.8],
}

TECH_ADOPTION = {
    "tech": ["電話\n(固定)", "テレビ", "PC", "携帯", "スマホ", "SNS", "AI\nツール"],
    "rich_country_years":  [1920, 1955, 1985, 1995, 2010, 2012, 2023],
    "global_50pct_years":  [1980, 1990, 2005, 2008, 2018, 2020, 2026],
    "adoption_gap_years":  [60,   35,   20,   13,    8,    8,    3],
}

FUTURE_RISKS = {
    "risks": ["気候変動", "AI失業", "米中対立", "高齢化\n財政", "パンデミック", "核戦争"],
    "probability_pct": [95, 70, 60, 90, 40, 10],
    "gdp_impact_pct":  [10, 20, 15, 12,  8, 80],
    "reversibility":   [2,   5,  6,  4,   7,  1],   # 1=不可逆, 10=回復可能
    "colors": ["#ff7f0e", "#9467bd", "#d62728", "#8c564b", "#e377c2", "#1f0a0a"],
}

# ── スタイル ─────────────────────────────────────────────────────────────────

BG = "#0d0d0d"
PANEL = "#141414"
GRID = "#2a2a2a"
WHITE = "#f0f0f0"
DIM = "#888888"

def sa(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE, labelsize=8)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
    ax.grid(color=GRID, lw=0.5, alpha=0.6)


# ── プロット関数 ──────────────────────────────────────────────────────────────

def p_poverty(ax):
    sa(ax)
    ax.set_title("極貧率の崩壊 (1820→2023)", fontsize=10, pad=5)
    y = POVERTY["years"]
    r = POVERTY["rate"]
    ax.fill_between(y, r, alpha=0.25, color="#d62728")
    ax.plot(y, r, "o-", color="#ff4444", lw=2.2, ms=5)
    ax.axhline(9, color="#ff4444", lw=0.8, ls="--", alpha=0.5)
    ax.text(1825, 92, "90%  ← 農奴・小作が当たり前", color=DIM, fontsize=7.5)
    ax.text(2010, 11, "9%", color="#ff4444", fontsize=9, fontweight="bold")
    ax.set_ylabel("極貧率 (%) [$1.90/日以下]")
    ax.set_ylim(0, 100)
    ax.set_xlim(1815, 2030)


def p_infant(ax):
    sa(ax)
    ax.set_title("乳幼児死亡率 (5歳未満, /1000人)", fontsize=10, pad=5)
    y = INFANT_MORTALITY["years"]
    r = INFANT_MORTALITY["rate"]
    ax.fill_between(y, r, alpha=0.2, color="#ff7f0e")
    ax.plot(y, r, "s-", color="#ff9933", lw=2.2, ms=5)
    ax.text(1805, 440, "430人 (半数近くが5歳前に死亡)", color=DIM, fontsize=7)
    ax.annotate("37人 (2023)", xy=(2023, 37), xytext=(2000, 80),
                color="#ff9933", fontsize=8, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#ff9933", lw=1))
    ax.set_ylabel("死亡数 (/1000出生)")
    ax.set_ylim(0, 480)
    ax.set_xlim(1795, 2030)


def p_lifeexp(ax):
    sa(ax)
    ax.set_title("平均寿命", fontsize=10, pad=5)
    yw = LIFE_EXPECTANCY["years"]
    ww = LIFE_EXPECTANCY["world"]
    yj = [y for y, v in zip(yw, LIFE_EXPECTANCY["japan"]) if v is not None]
    jj = [v for v in LIFE_EXPECTANCY["japan"] if v is not None]
    ax.plot(yw, ww, "o-", color="#2196F3", lw=2.2, ms=5, label="世界平均")
    ax.plot(yj, jj, "^-", color="#4CAF50", lw=2.2, ms=5, label="日本")
    ax.fill_between(yw, ww, 20, alpha=0.12, color="#2196F3")
    ax.axhline(73, color="#2196F3", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(84, color="#4CAF50", lw=0.8, ls="--", alpha=0.5)
    ax.text(2024, 74, "73歳", color="#2196F3", fontsize=8)
    ax.text(2024, 85, "84歳", color="#4CAF50", fontsize=8)
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE, loc="upper left")
    ax.set_ylabel("平均寿命 (歳)")
    ax.set_ylim(20, 95)
    ax.set_xlim(1795, 2035)


def p_literacy(ax):
    sa(ax)
    ax.set_title("世界識字率", fontsize=10, pad=5)
    y = LITERACY["years"]
    r = LITERACY["rate"]
    ax.fill_between(y, r, alpha=0.2, color="#9C27B0")
    ax.plot(y, r, "D-", color="#CE93D8", lw=2.2, ms=5)
    ax.text(1825, 14, "12%  ← 読み書きは特権階級のみ", color=DIM, fontsize=7.5)
    ax.text(2010, 88.5, "87%", color="#CE93D8", fontsize=9, fontweight="bold")
    ax.set_ylabel("識字率 (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(1815, 2030)


def p_calories(ax):
    sa(ax)
    ax.set_title("1日摂取カロリー (平均)", fontsize=10, pad=5)
    y = CALORIES["years"]
    ax.plot(y, CALORIES["rich"],  "o-", color="#4CAF50", lw=2, ms=4, label="先進国")
    ax.plot(y, CALORIES["world"], "s-", color="#FFC107", lw=2, ms=4, label="世界平均")
    ax.plot(y, CALORIES["dev"],   "^-", color="#FF5722", lw=2, ms=4, label="途上国")
    ax.axhline(2000, color=DIM, lw=1, ls=":", alpha=0.7)
    ax.text(1962, 2010, "2000kcal (最低必要量)", color=DIM, fontsize=7)
    ax.fill_between(y, CALORIES["dev"], 2000,
                    where=[d < 2000 for d in CALORIES["dev"]],
                    alpha=0.3, color="#FF5722", label="栄養不足ゾーン")
    ax.legend(fontsize=7.5, facecolor="#222", edgecolor="#444", labelcolor=WHITE)
    ax.set_ylabel("kcal/日")
    ax.set_ylim(1800, 3600)


def p_china_india(ax):
    sa(ax)
    ax.set_title("極貧率 — 国別 (中国・インド・サブサハラ)", fontsize=10, pad=5)
    y = CHINA_INDIA_POVERTY["years"]
    ax.plot(y, CHINA_INDIA_POVERTY["china"],     "o-", color="#e74c3c", lw=2.2, ms=5, label="中国")
    ax.plot(y, CHINA_INDIA_POVERTY["india"],     "s-", color="#f39c12", lw=2.2, ms=5, label="インド")
    ax.plot(y, CHINA_INDIA_POVERTY["sub_sahara"],"^-", color="#8e44ad", lw=2.2, ms=5, label="サブサハラ")
    ax.annotate("0.4% (2023)\n→ 歴史的偉業", xy=(2023, 0.4), xytext=(2010, 15),
                color="#e74c3c", fontsize=7.5,
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1))
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE)
    ax.set_ylabel("極貧率 (%)")
    ax.set_ylim(-2, 95)


def p_gini(ax):
    sa(ax)
    ax.set_title("ジニ係数 — 国内格差の拡大", fontsize=10, pad=5)
    y = GINI["years"]
    ax.plot(y, GINI["usa"],   "o-", color="#3498db", lw=2.2, ms=5, label="米国")
    ax.plot(y, GINI["china"], "s-", color="#e74c3c", lw=2.2, ms=5, label="中国")
    ax.plot(y, GINI["japan"], "^-", color="#2ecc71", lw=2.2, ms=5, label="日本")
    ax.plot(y, GINI["world"], "D-", color="#f39c12", lw=2.2, ms=5, label="世界全体")
    ax.axhline(0.4, color=DIM, lw=1, ls=":", alpha=0.6)
    ax.text(1981, 0.405, "警戒ライン 0.4", color=DIM, fontsize=7)
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE, loc="center right")
    ax.set_ylabel("ジニ係数 (0=完全平等, 1=完全不平等)")
    ax.set_ylim(0.25, 0.75)
    note = "↑ 国内格差は拡大 (米・中)\n↓ 世界全体の格差は縮小 (中国・インドの台頭)"
    ax.text(0.02, 0.05, note, transform=ax.transAxes, color=DIM, fontsize=7)


def p_billionaire(ax):
    sa(ax)
    ax.set_title("億万長者の資産総額 vs Top10", fontsize=10, pad=5)
    y = BILLIONAIRE_WEALTH["years"]
    t = BILLIONAIRE_WEALTH["total_trillion_usd"]
    top = BILLIONAIRE_WEALTH["top10_trillion_usd"]
    ax.fill_between(y, t, alpha=0.2, color="#e74c3c")
    ax.plot(y, t,   "o-", color="#e74c3c", lw=2.2, ms=5, label="全億万長者 合計")
    ax.plot(y, top, "s-", color="#ff4444", lw=2.2, ms=5, label="Top10のみ")
    ax.text(2001, 15.0, "$14.2T (2024)", color="#e74c3c", fontsize=8, fontweight="bold")
    ax.text(2001, 6.5,  "$5.8T — Top10だけ", color="#ff4444", fontsize=7.5)
    ax.set_ylabel("資産総額 (兆ドル)")
    ax.set_ylim(0, 17)
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE)
    note = "比較: 世界下位40億人の合計資産 ≈ $2.5T\nイーロン・マスク1人 ≈ $400B"
    ax.text(0.02, 0.72, note, transform=ax.transAxes, color=DIM, fontsize=7)


def p_tech_gap(ax):
    sa(ax)
    ax.set_title("テクノロジー普及ギャップ — 縮小している", fontsize=10, pad=5)
    techs = TECH_ADOPTION["tech"]
    gaps  = TECH_ADOPTION["adoption_gap_years"]
    x = np.arange(len(techs))
    colors = ["#546e7a","#607d8b","#78909c","#e57373","#ef9a9a","#ffccbc","#a5d6a7"]
    bars = ax.bar(x, gaps, color=colors, alpha=0.88, width=0.6)
    for bar, g in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, g + 0.5, f"{g}年",
                ha="center", va="bottom", color=WHITE, fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(techs, color=WHITE, fontsize=8)
    ax.set_ylabel("先進国→世界普及率50%までの年数差")
    ax.set_ylim(0, 70)
    ax.annotate("", xy=(6, 3), xytext=(0, 60),
                arrowprops=dict(arrowstyle="-|>", color="#4CAF50", lw=2))
    ax.text(3, 55, "格差は急速に縮小中", color="#4CAF50", fontsize=9, fontweight="bold")


def p_risks(ax):
    sa(ax)
    ax.set_title("未来リスク — 確率 × GDPインパクト × 回復可能性", fontsize=10, pad=5)
    risks  = FUTURE_RISKS["risks"]
    prob   = FUTURE_RISKS["probability_pct"]
    impact = FUTURE_RISKS["gdp_impact_pct"]
    rev    = FUTURE_RISKS["reversibility"]
    colors = FUTURE_RISKS["colors"]

    sc = ax.scatter(prob, impact, s=[r * 80 for r in rev],
                    c=colors, alpha=0.85, edgecolors=WHITE, lw=0.8)
    for i, (r, p, im) in enumerate(zip(risks, prob, impact)):
        ax.annotate(r, (p, im),
                    textcoords="offset points", xytext=(6, 4),
                    color=WHITE, fontsize=8)
    ax.set_xlabel("発生確率 (%)")
    ax.set_ylabel("GDP損失インパクト (%)")
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 95)
    note = "円のサイズ = 回復可能性 (大=回復しやすい)"
    ax.text(0.01, 0.94, note, transform=ax.transAxes, color=DIM, fontsize=7)
    ax.axvline(50, color=GRID, lw=1, ls="--")
    ax.axhline(20, color=GRID, lw=1, ls="--")
    ax.text(52, 88, "高確率×高ダメージ", color="#ff4444", fontsize=7.5, alpha=0.8)


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    fig = plt.figure(figsize=(22, 28), facecolor=BG)
    fig.suptitle(
        "人類200年の進歩 — 全データ可視化\n「底辺の底上げ」は本当に起きているか",
        fontsize=18, color=WHITE, fontweight="bold", y=0.985
    )

    gs = gridspec.GridSpec(
        4, 3,
        figure=fig,
        hspace=0.50, wspace=0.36,
        left=0.06, right=0.97,
        top=0.96, bottom=0.04,
    )

    p_poverty(    fig.add_subplot(gs[0, 0]))
    p_infant(     fig.add_subplot(gs[0, 1]))
    p_lifeexp(    fig.add_subplot(gs[0, 2]))
    p_literacy(   fig.add_subplot(gs[1, 0]))
    p_calories(   fig.add_subplot(gs[1, 1]))
    p_china_india(fig.add_subplot(gs[1, 2]))
    p_gini(       fig.add_subplot(gs[2, 0]))
    p_billionaire(fig.add_subplot(gs[2, 1]))
    p_tech_gap(   fig.add_subplot(gs[2, 2]))
    p_risks(      fig.add_subplot(gs[3, :]))

    out = os.path.join(OUT_DIR, "human_progress_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
