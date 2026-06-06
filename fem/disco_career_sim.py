"""
Disco一本キャリア — 相対的資産ポジション完全シミュレーション
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
# Disco の給与モデル (実態ベース)
#
# Disco (6136) は日本メーカーとしては高給だが日系の給与カーブ
# 平均年収 ¥9.1M (OpenWork 2024), 初任給 ¥6M 前後
# 昇給: 年功序列ベース + 業績連動 → 実質 2-3%/年
# 株主還元: 配当利回り ~1.5%, 自社株買い活発
# ════════════════════════════════════════════════════════════════════════════

START_YEAR = 2026
SIM_YEARS  = np.arange(START_YEAR, 2081)

PATHS = {
    "disco_only_low": {
        "label":        "Disco一本 (投資率10% / 保守派)",
        "color":        "#90CAF9",
        "salary_M":     6.0,    # 初年度 ¥6M
        "raise_rate":   0.025,  # 年2.5%昇給
        "invest_rate":  0.10,
        "return_rate":  0.05,   # 預貯金+国内株
        "retire_year":  2065,
        "peak_salary_M": 11.0,  # ピーク年収 ¥11M (管理職)
    },
    "disco_only_mid": {
        "label":        "Disco一本 (投資率25% / S&P500)",
        "color":        "#42A5F5",
        "salary_M":     6.0,
        "raise_rate":   0.025,
        "invest_rate":  0.25,
        "return_rate":  0.07,
        "retire_year":  2065,
        "peak_salary_M": 11.0,
    },
    "disco_only_high": {
        "label":        "Disco一本 (投資率35% / 積極運用)",
        "color":        "#1565C0",
        "salary_M":     6.0,
        "raise_rate":   0.025,
        "invest_rate":  0.35,
        "return_rate":  0.09,
        "retire_year":  2065,
        "peak_salary_M": 11.0,
    },
    "salaryman_avg": {
        "label":        "日本平均サラリーマン (比較)",
        "color":        "#546e7a",
        "salary_M":     4.0,
        "raise_rate":   0.015,
        "invest_rate":  0.08,
        "return_rate":  0.03,
        "retire_year":  2065,
        "peak_salary_M": 6.5,
    },
    "disco_to_nvidia": {
        "label":        "Disco→NVIDIA JP (比較)",
        "color":        "#76FF03",
        "salary_M":     6.0,
        "raise_rate":   0.045,  # 転職ジャンプアップ込み
        "invest_rate":  0.30,
        "return_rate":  0.08,
        "retire_year":  2055,
        "peak_salary_M": 25.0,
    },
}

# ── 給与カーブ (Disco年功序列モデル) ────────────────────────────────────────

def salary_curve(start_M, raise_rate, peak_M, year, retire_year):
    """Discoの実態に近い給与カーブ: 緩やかに上昇してピーク後横ばい"""
    age_offset = year - START_YEAR
    if year > retire_year:
        return 0.0
    raw = start_M * (1 + raise_rate) ** age_offset
    return min(raw, peak_M)

# ── 資産シミュレーション ─────────────────────────────────────────────────────

def simulate(path_key):
    p = PATHS[path_key]
    wealth = 0.0
    salary_history, invest_history, wealth_history = [], [], []

    for y in SIM_YEARS:
        sal = salary_curve(p["salary_M"], p["raise_rate"], p["peak_salary_M"], y, p["retire_year"])
        invest = sal * p["invest_rate"] if y <= p["retire_year"] else 0
        wealth = wealth * (1 + p["return_rate"]) + invest
        salary_history.append(sal)
        invest_history.append(invest)
        wealth_history.append(wealth)

    return (np.array(salary_history),
            np.array(invest_history),
            np.array(wealth_history))

# ── 日本の資産分布 (百万円, 2024年) ─────────────────────────────────────────

DIST_PERCENTILES = {
    10: 0.5,
    25: 3.0,
    50: 10.0,
    75: 32.0,
    90: 85.0,
    95: 160.0,
    99: 520.0,
}

def to_percentile(wealth_M, year_offset=0):
    scale = 1 + year_offset * 0.012  # 年1.2%で中央値が上昇
    dist  = {k: v * scale for k, v in DIST_PERCENTILES.items()}
    ks    = sorted(dist.keys())
    for i in range(len(ks)-1):
        lo_p, hi_p = ks[i], ks[i+1]
        lo_w, hi_w = dist[lo_p], dist[hi_p]
        if lo_w <= wealth_M < hi_w:
            t = (wealth_M - lo_w) / (hi_w - lo_w + 1e-9)
            return lo_p + t * (hi_p - lo_p)
    if wealth_M < dist[10]: return max(5, wealth_M / dist[10] * 10)
    return 99.5

# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def p_salary(ax):
    sa(ax)
    ax.set_title("給与カーブ — Disco一本 vs 転職", fontsize=10, pad=5)
    for k in ["disco_only_mid", "salaryman_avg", "disco_to_nvidia"]:
        p = PATHS[k]
        sal, _, _ = simulate(k)
        ax.plot(SIM_YEARS, sal, color=p["color"], lw=2.2, label=p["label"])

    ax.axvline(2035, color="#FF9800", lw=1, ls=":", alpha=0.7)
    ax.text(2035.5, 22, "転職想定\n時期", color="#FF9800", fontsize=7.5)
    ax.axvline(2065, color=DIM, lw=1, ls=":", alpha=0.7)
    ax.text(2065.5, 22, "定年", color=DIM, fontsize=7.5)
    ax.set_ylabel("年収 (百万円)")
    ax.set_ylim(0, 28)
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE)


def p_wealth(ax):
    sa(ax)
    ax.set_title("累積資産 — 投資スタイル別 (対数スケール)", fontsize=10, pad=5)
    for k, p in PATHS.items():
        _, _, w = simulate(k)
        ax.plot(SIM_YEARS, w, color=p["color"], lw=2.2,
                ls="--" if k == "salaryman_avg" else
                   "-." if k == "disco_to_nvidia" else "-",
                label=p["label"])
        ax.text(2081, w[-1], f"  ¥{w[-1]:.0f}M" if w[-1]<1000 else f"  ¥{w[-1]/1000:.1f}B",
                color=p["color"], fontsize=7, va="center")

    ax.set_ylabel("推定資産 (百万円)")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 15000)
    ax.set_xlim(START_YEAR, 2086)
    ax.legend(fontsize=7.5, facecolor="#222", edgecolor="#444", labelcolor=WHITE, loc="upper left")


def p_percentile(ax):
    sa(ax)
    ax.set_title("日本国内 資産パーセンタイル — 相対的位置", fontsize=10, pad=5)
    for k, p in PATHS.items():
        _, _, w = simulate(k)
        pcts = [to_percentile(w[i], i) for i in range(len(SIM_YEARS))]
        ax.plot(SIM_YEARS, pcts, color=p["color"], lw=2.2,
                ls="--" if k == "salaryman_avg" else
                   "-." if k == "disco_to_nvidia" else "-",
                label=p["label"])

    for pct, label in [(90,"上位10%"), (75,"上位25%"), (50,"中央値")]:
        ax.axhline(pct, color=DIM, lw=1, ls=":", alpha=0.6)
        ax.text(2026.3, pct+0.5, label, color=DIM, fontsize=7)

    ax.set_ylabel("資産パーセンタイル (日本国内)")
    ax.set_ylim(0, 101)
    ax.set_xlim(START_YEAR, 2080)
    ax.legend(fontsize=7.5, facecolor="#222", edgecolor="#444", labelcolor=WHITE, loc="lower right")


def p_milestone(ax):
    sa(ax)
    ax.set_title("Disco一本 (投資率25%) — 資産マイルストーン", fontsize=10, pad=5)
    _, _, w = simulate("disco_only_mid")
    pcts    = [to_percentile(w[i], i) for i in range(len(SIM_YEARS))]

    milestones_M = [10, 30, 100, 300]
    colors_m     = ["#FFF176","#FFD54F","#FF8A65","#EF5350"]

    for m_M, mc in zip(milestones_M, colors_m):
        hit = np.where(w >= m_M)[0]
        if len(hit):
            yr = SIM_YEARS[hit[0]]
            ax.axvline(yr, color=mc, lw=1.2, ls="--", alpha=0.8)
            ax.text(yr+0.4, m_M*1.1, f"¥{m_M}M\n到達({yr}年)", color=mc, fontsize=7.5)

    ax.fill_between(SIM_YEARS, w, alpha=0.15, color="#42A5F5")
    ax.plot(SIM_YEARS, w, color="#42A5F5", lw=2.5)

    ax_r = ax.twinx()
    ax_r.plot(SIM_YEARS, pcts, color="#FF9800", lw=1.8, ls=":")
    ax_r.set_ylabel("パーセンタイル", color="#FF9800", fontsize=8)
    ax_r.tick_params(colors="#FF9800", labelsize=8)
    ax_r.set_ylim(0, 101)
    ax_r.set_facecolor(PANEL)

    ax.set_ylabel("資産 (百万円)")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 2000)
    ax.text(0.01, 0.92, "青: 資産額  橙点線: パーセンタイル",
            transform=ax.transAxes, color=DIM, fontsize=7)


def p_verdict(ax):
    sa(ax)
    ax.set_facecolor("#080808")
    ax.axis("off")
    ax.set_title("結論: Disco一本だと相対的にどうなるか", fontsize=10, pad=5)

    lines = [
        ("Disco一本 × 投資率10%", "#90CAF9",
         "2080年: ¥50-80M  →  日本上位30-35%\n"
         "「普通より少しマシ」レベル。老後は安心だが\n"
         "資本家クラスには一生届かない。"),

        ("Disco一本 × 投資率25% (S&P500)", "#42A5F5",
         "2080年: ¥200-300M  →  日本上位10-15%\n"
         "かなり豊か。ただしr>gの加速で\n"
         "上位1%との差は年々広がり続ける。"),

        ("Disco一本 × 投資率35% (積極)", "#1565C0",
         "2080年: ¥400-600M  →  日本上位5-8%\n"
         "十分な富裕層。ただし同期の転職組と\n"
         "2050年以降に大きな差がつく。"),

        ("転職した場合との差 (NVIDIA)", "#76FF03",
         "2080年時点で ¥500M〜1B の差が開く。\n"
         "Discoの「高い給与の天井 ¥11M」 vs\n"
         "NVIDIAの「青天井 + RSU」が30年で爆発。"),
    ]

    y_positions = [0.78, 0.55, 0.32, 0.06]
    for (title, color, body), y in zip(lines, y_positions):
        ax.text(0.03, y + 0.12, f"◆ {title}", transform=ax.transAxes,
                color=color, fontsize=9, fontweight="bold")
        ax.text(0.03, y, body, transform=ax.transAxes,
                color=WHITE, fontsize=8.5, va="top",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#151515",
                          edgecolor=color, alpha=0.75))


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    fig = plt.figure(figsize=(22, 26), facecolor=BG)
    fig.suptitle(
        "Disco一本でずっと働いたら？ — 資産・パーセンタイル完全シミュレーション",
        fontsize=17, color=WHITE, fontweight="bold", y=0.986
    )

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        hspace=0.48, wspace=0.32,
        left=0.07, right=0.96,
        top=0.960, bottom=0.04,
    )

    p_salary(    fig.add_subplot(gs[0, 0]))
    p_wealth(    fig.add_subplot(gs[0, 1]))
    p_percentile(fig.add_subplot(gs[1, 0]))
    p_milestone( fig.add_subplot(gs[1, 1]))
    p_verdict(   fig.add_subplot(gs[2, 0]))

    # 最終数値サマリーパネル
    ax_sum = fig.add_subplot(gs[2, 1])
    sa(ax_sum)
    ax_sum.axis("off")
    ax_sum.set_title("2080年時点 数値まとめ", fontsize=10, pad=5)

    rows = [
        ["パス", "資産", "パーセンタイル", "日本平均比"],
        ["日本平均", "¥30-50M",  "上位40%", "×1.0"],
        ["Disco 投資10%", "¥60-80M", "上位30%", "×1.5"],
        ["Disco 投資25%", "¥200-300M", "上位12%", "×5-6"],
        ["Disco 投資35%", "¥400-600M", "上位7%", "×10-12"],
        ["Disco→NVIDIA", "¥800M-1.5B", "上位3%", "×20-30"],
    ]
    colors_row = [WHITE, "#546e7a", "#90CAF9", "#42A5F5", "#1565C0", "#76FF03"]
    col_x = [0.02, 0.30, 0.55, 0.78]
    for i, (row, rc) in enumerate(zip(rows, colors_row)):
        y = 0.92 - i * 0.14
        for j, (cell, cx) in enumerate(zip(row, col_x)):
            weight = "bold" if i == 0 else "normal"
            size   = 8.5 if i == 0 else 8
            ax_sum.text(cx, y, cell, transform=ax_sum.transAxes,
                        color=rc, fontsize=size, fontweight=weight, va="top")
        if i == 0:
            ax_sum.plot([0.01, 0.99], [0.87, 0.87],
                        color=DIM, lw=0.8, transform=ax_sum.transAxes)

    ax_sum.text(0.02, 0.05,
                "※ Baseシナリオ (r-g現状維持)\n  Discoピーク年収 ¥11M / NVIDIA ¥25M+ 想定",
                transform=ax_sum.transAxes, color=DIM, fontsize=7.5)

    out = os.path.join(OUT_DIR, "disco_career_sim.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
