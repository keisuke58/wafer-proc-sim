"""
r > g 2129年予測 × 半導体エンジニアの個人資産シミュレーション
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
# 1. r と g の 2129年予測モデル
#
# AI が生産性を上げる → g も上がる
# ただし AI の利益は資本家に集中 → r はさらに上がる
# 結果: r-g の差は拡大し続ける (Base/Accel シナリオ)
# 例外: UBI・累進課税・AI民主化 → g が追いつく (Utopia シナリオ)
# ════════════════════════════════════════════════════════════════════════════

YEARS = np.arange(1950, 2130, 1)

def make_r(scenario):
    r = np.zeros(len(YEARS))
    for i, y in enumerate(YEARS):
        if y < 1980:
            r[i] = 5.0 + (y - 1950) * 0.05
        elif y < 2000:
            r[i] = 6.5 + (y - 1980) * 0.12
        elif y < 2025:
            r[i] = 9.0 + (y - 2000) * 0.05
        else:
            base = 10.2
            t = y - 2025
            if scenario == "utopia":
                # AI民主化・税制改革でrが抑制
                r[i] = base + t * 0.02
            elif scenario == "base":
                # AI普及で資本収益率がじわり上昇
                r[i] = base + t * 0.06 + 0.001 * t**1.3
            elif scenario == "accel":
                # AGI到達後、資本が指数的に増殖
                r[i] = base + t * 0.15 + 0.003 * t**1.6
    return np.clip(r, 0, 60)

def make_g(scenario):
    g = np.zeros(len(YEARS))
    for i, y in enumerate(YEARS):
        if y < 1980:
            g[i] = 4.5 + (y - 1950) * 0.02
        elif y < 2000:
            g[i] = 3.5 - (y - 1980) * 0.05
        elif y < 2025:
            g[i] = 2.5 + (y - 2000) * 0.02
        else:
            base = 3.0
            t = y - 2025
            if scenario == "utopia":
                # AI生産性革命がg押し上げ + 富の再分配
                g[i] = base + t * 0.12 + 0.002 * t**1.4
            elif scenario == "base":
                # AIでgはやや改善するが限定的
                g[i] = base + t * 0.04
            elif scenario == "accel":
                # 自動化で雇用消滅 → g は低迷
                g[i] = base - t * 0.01 + 0.0005 * t**1.1
    return np.clip(g, -2, 40)

SCENARIOS = {
    "accel":  ("AGI加速\n(資本独占)", "#FF1744"),
    "base":   ("現状維持\n(緩やか拡大)", "#FF9800"),
    "utopia": ("UBI・税制改革\n(格差縮小)", "#4CAF50"),
}

# ════════════════════════════════════════════════════════════════════════════
# 2. 個人資産シミュレーション
#    半導体エンジニア (西岡さん, 2026年 現在)
#    前提: 現在26歳想定、Disco→Lam→NVIDIA キャリア
# ════════════════════════════════════════════════════════════════════════════

SIM_YEARS = np.arange(2026, 2080, 1)

CAREER_PATHS = {
    "salaryman": {
        "label": "日本平均サラリーマン",
        "color": "#546e7a",
        "salary_start": 4.0,    # 百万円/年
        "salary_growth": 0.015, # 年1.5%昇給
        "invest_rate":  0.10,   # 収入の10%投資
        "return_rate":  0.04,   # 年4%リターン
        "retire_age":   65,
        "retire_year":  2065,
    },
    "disco_nvidia": {
        "label": "Disco→Lam→NVIDIA JP (西岡さん想定)",
        "color": "#4FC3F7",
        "salary_start": 7.0,
        "salary_growth": 0.045,  # 外資ジャンプアップで年4.5%
        "invest_rate":  0.25,
        "return_rate":  0.07,    # 株式メイン
        "retire_age":   55,
        "retire_year":  2055,
    },
    "disco_nvidia_aggressive": {
        "label": "NVIDIA JP + 積極投資 (S&P500 + AI株)",
        "color": "#00E5FF",
        "salary_start": 7.0,
        "salary_growth": 0.045,
        "invest_rate":  0.40,
        "return_rate":  0.10,
        "retire_age":   50,
        "retire_year":  2050,
    },
    "capital_class": {
        "label": "資本家クラス (初期資産¥50M + 投資)",
        "color": "#FF5722",
        "salary_start": 0,
        "salary_growth": 0,
        "invest_rate":  1.0,
        "return_rate":  0.12,
        "retire_age":   0,
        "retire_year":  2026,
        "_initial_asset_M": 50,
    },
}

def simulate_wealth(path_key, scenario="base"):
    p = CAREER_PATHS[path_key]
    r_arr = make_r(scenario)
    base_return = p["return_rate"]

    wealth = p.get("_initial_asset_M", 0.0)
    salary = p["salary_start"]
    wealth_history = []

    for i, y in enumerate(SIM_YEARS):
        # 資本収益率をシナリオに応じてスケール
        r_now = r_arr[YEARS == y][0] / 100
        # 個人投資家の実効リターン = base × (r_now/r_2026)
        r_2026 = r_arr[YEARS == 2026][0] / 100
        eff_return = base_return * (r_now / r_2026)
        eff_return = min(eff_return, 0.25)  # 上限25%

        if y <= p["retire_year"] and salary > 0:
            invest_amount = salary * p["invest_rate"]
            salary *= (1 + p["salary_growth"])
        else:
            invest_amount = 0

        wealth = wealth * (1 + eff_return) + invest_amount
        wealth_history.append(wealth)

    return np.array(wealth_history)

# ════════════════════════════════════════════════════════════════════════════
# 3. 相対的位置 (パーセンタイル) シミュレーション
# ════════════════════════════════════════════════════════════════════════════

# 日本の資産分布 (2024年ベース, 百万円)
# 参考: 金融広報中央委員会「家計の金融行動に関する世論調査」
PERCENTILES_2024 = {
    10: 0,      # 貯蓄なし
    25: 2.5,
    50: 10.0,   # 中央値
    75: 30.0,
    90: 80.0,
    95: 150.0,
    99: 500.0,
    99.9: 5000.0,
}

def wealth_to_percentile(wealth_M, year_offset):
    """資産額(百万円)からパーセンタイルを推定 (格差拡大を考慮)"""
    # 格差拡大係数: 2024から年々上位に資産集中
    inflation = 1 + year_offset * 0.015
    adjusted = {k: v * inflation for k, v in PERCENTILES_2024.items()}
    keys = sorted(adjusted.keys())
    for i in range(len(keys)-1):
        lo_p, hi_p = keys[i], keys[i+1]
        lo_w, hi_w = adjusted[lo_p], adjusted[hi_p]
        if lo_w <= wealth_M <= hi_w:
            t = (wealth_M - lo_w) / (hi_w - lo_w + 1e-9)
            return lo_p + t * (hi_p - lo_p)
    if wealth_M < adjusted[10]: return 5
    if wealth_M > adjusted[99.9]: return 99.9
    return 99.5

# ════════════════════════════════════════════════════════════════════════════
# 4. 可視化
# ════════════════════════════════════════════════════════════════════════════

def p_rg_forecast(ax):
    sa(ax)
    ax.set_title("r (資本収益率) vs g (GDP成長率) — 2129年まで予測", fontsize=10, pad=5)

    for sc_key, (sc_label, sc_color) in SCENARIOS.items():
        r = make_r(sc_key)
        g = make_g(sc_key)
        diff = r - g
        mask = YEARS >= 2025
        ax.plot(YEARS[mask], r[mask], color=sc_color, lw=1.8, ls="-",
                label=f"r [{sc_label.split(chr(10))[0]}]", alpha=0.9)
        ax.plot(YEARS[mask], g[mask], color=sc_color, lw=1.8, ls="--", alpha=0.6)

    # 過去データ (実績)
    hist_y = [1950,1960,1970,1980,1990,2000,2010,2020,2025]
    hist_r = [5.0, 5.5, 6.0, 6.5, 7.5, 9.0, 9.5,10.0,10.2]
    hist_g = [4.5, 5.2, 5.0, 3.5, 3.0, 2.5, 2.2, 2.5, 3.0]
    ax.plot(hist_y, hist_r, "o-", color=WHITE, lw=2.5, ms=5, label="r 実績", zorder=5)
    ax.plot(hist_y, hist_g, "s-", color="#90CAF9", lw=2.5, ms=5, label="g 実績", zorder=5)

    ax.axvline(2025, color=DIM, lw=1.2, ls=":")
    ax.text(2026, 28, "← 実績  予測 →", color=DIM, fontsize=8)
    ax.axvline(2045, color="#FF9800", lw=0.8, ls=":", alpha=0.6)
    ax.text(2046, 25, "AGI?", color="#FF9800", fontsize=7.5)
    ax.set_ylabel("年率 (%)")
    ax.set_xlim(1945, 2132)
    ax.set_ylim(-1, 62)

    from matplotlib.lines import Line2D
    legend_elem = [
        Line2D([0],[0], color=WHITE,    lw=2, label="r 実績"),
        Line2D([0],[0], color="#90CAF9",lw=2, label="g 実績"),
        Line2D([0],[0], color="#FF1744",lw=2, label="r/g [AGI加速]"),
        Line2D([0],[0], color="#FF9800",lw=2, label="r/g [現状維持]"),
        Line2D([0],[0], color="#4CAF50",lw=2, label="r/g [UBI改革]"),
        Line2D([0],[0], color=WHITE, lw=1.5, ls="--", label="g (各シナリオ)"),
    ]
    ax.legend(handles=legend_elem, fontsize=7.5, facecolor="#222",
              edgecolor="#444", labelcolor=WHITE, ncol=2, loc="upper left")


def p_rg_gap(ax):
    sa(ax)
    ax.set_title("r - g (格差拡大速度) — シナリオ別", fontsize=10, pad=5)
    for sc_key, (sc_label, sc_color) in SCENARIOS.items():
        r = make_r(sc_key)
        g = make_g(sc_key)
        diff = r - g
        ax.plot(YEARS, diff, color=sc_color, lw=2, label=sc_label.replace("\n"," "))
        if sc_key == "accel":
            ax.fill_between(YEARS, diff, 0, where=diff>0, alpha=0.08, color=sc_color)

    ax.axhline(0, color=WHITE, lw=1, ls="-", alpha=0.4)
    ax.axvline(2025, color=DIM, lw=1, ls=":")
    ax.set_ylabel("r - g (格差拡大速度, %/年)")
    ax.set_xlim(1945, 2132)
    ax.set_ylim(-2, 50)
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE)
    ax.text(2060, 38, "AGI加速シナリオ:\nr-g = 30%超", color="#FF1744", fontsize=8.5)
    ax.text(2090, 6, "現状維持", color="#FF9800", fontsize=8)
    ax.text(2090, -1.2, "UBI改革", color="#4CAF50", fontsize=8)


def p_personal_wealth(ax):
    sa(ax)
    ax.set_title("個人資産シミュレーション (2026→2079) — 西岡さんはどこ？", fontsize=10, pad=5)

    for path_key, path in CAREER_PATHS.items():
        w = simulate_wealth(path_key, "base")
        ax.plot(SIM_YEARS, w, color=path["color"], lw=2.2,
                label=path["label"])
        # 最終値アノテーション
        ax.text(2079.5, w[-1], f"  ¥{w[-1]:.0f}M",
                color=path["color"], fontsize=7.5, va="center")

    # 中央値ライン
    median_growth = np.array([PERCENTILES_2024[50] * (1.015 ** (y-2024)) for y in SIM_YEARS])
    ax.plot(SIM_YEARS, median_growth, color=DIM, lw=1.5, ls=":",
            label="日本資産中央値 (成長込み)")

    ax.axvspan(2026, 2035, alpha=0.05, color="#4FC3F7")
    ax.text(2026.5, 1800, "Disco期", color="#4FC3F7", fontsize=7)
    ax.axvspan(2035, 2042, alpha=0.05, color="#FF9800")
    ax.text(2035.5, 1800, "Lam/AMAT", color="#FF9800", fontsize=7)
    ax.axvspan(2042, 2060, alpha=0.05, color="#00E5FF")
    ax.text(2042.5, 1800, "NVIDIA JP", color="#00E5FF", fontsize=7)

    ax.set_ylabel("資産 (百万円)")
    ax.set_yscale("log")
    ax.set_ylim(1, 60000)
    ax.legend(fontsize=7.5, facecolor="#222", edgecolor="#444", labelcolor=WHITE,
              loc="upper left")
    ax.text(0.01, 0.01, "対数スケール / Baseシナリオ (現状維持)",
            transform=ax.transAxes, color=DIM, fontsize=7)


def p_scenario_comparison(ax):
    sa(ax)
    ax.set_title("西岡さんシナリオ × r-g シナリオ — 2079年時点資産", fontsize=10, pad=5)

    paths = ["salaryman", "disco_nvidia", "disco_nvidia_aggressive", "capital_class"]
    sc_keys = ["utopia", "base", "accel"]
    sc_labels = ["UBI改革", "現状維持", "AGI加速"]
    sc_colors = ["#4CAF50", "#FF9800", "#FF1744"]

    x = np.arange(len(paths))
    w_bars = 0.22
    for j, (sc_key, sc_label, sc_color) in enumerate(zip(sc_keys, sc_labels, sc_colors)):
        vals = []
        for p_key in paths:
            w = simulate_wealth(p_key, sc_key)
            vals.append(w[-1])
        bars = ax.bar(x + j * w_bars, vals, w_bars, color=sc_color, alpha=0.82,
                      label=sc_label, edgecolor="#ffffff11")
        for bar, v in zip(bars, vals):
            if v > 100:
                ax.text(bar.get_x()+bar.get_width()/2, v*1.05,
                        f"¥{v:.0f}M" if v < 1000 else f"¥{v/1000:.1f}B",
                        ha="center", va="bottom", color=WHITE, fontsize=6.5, fontweight="bold")

    labels_short = ["平均\nサラリーマン", "Disco→\nNVIDIA", "NVIDIA+\n積極投資", "資本家\nクラス"]
    ax.set_xticks(x + w_bars)
    ax.set_xticklabels(labels_short, color=WHITE, fontsize=8)
    ax.set_ylabel("2079年時点 推定資産 (百万円)")
    ax.set_yscale("log")
    ax.set_ylim(1, 500000)
    ax.legend(fontsize=8.5, facecolor="#222", edgecolor="#444", labelcolor=WHITE)


def p_percentile(ax):
    sa(ax)
    ax.set_title("日本国内 資産パーセンタイル推移 — 西岡さんのポジション", fontsize=10, pad=5)

    for path_key in ["salaryman", "disco_nvidia", "disco_nvidia_aggressive"]:
        p = CAREER_PATHS[path_key]
        w_base = simulate_wealth(path_key, "base")
        pcts = [wealth_to_percentile(w, i) for i, w in enumerate(w_base)]
        ax.plot(SIM_YEARS, pcts, color=p["color"], lw=2.2, label=p["label"])

    ax.axhline(90, color=DIM, lw=1, ls="--", alpha=0.7)
    ax.axhline(75, color=DIM, lw=1, ls="--", alpha=0.5)
    ax.axhline(50, color=DIM, lw=1, ls="--", alpha=0.4)
    ax.text(2026.5, 91, "上位10%", color=DIM, fontsize=7.5)
    ax.text(2026.5, 76, "上位25%", color=DIM, fontsize=7.5)
    ax.text(2026.5, 51, "中央値", color=DIM, fontsize=7.5)

    ax.set_ylabel("日本国内 資産パーセンタイル")
    ax.set_ylim(40, 100.5)
    ax.set_xlim(2026, 2079)
    ax.legend(fontsize=7.5, facecolor="#222", edgecolor="#444", labelcolor=WHITE,
              loc="lower right")
    note = "格差拡大を考慮: 同じ資産でも相対的位置は下がる可能性あり"
    ax.text(0.01, 0.02, note, transform=ax.transAxes, color=DIM, fontsize=7)


def p_key_decisions(ax):
    sa(ax)
    ax.set_facecolor("#0a0a0a")
    ax.axis("off")
    ax.set_title("結論: 西岡さんへの処方箋", fontsize=10, pad=5)

    items = [
        ("早期投資開始", "#4CAF50",
         "Disco入社初年度から月収の25-30%を\nS&P500/全世界株に。\n複利の力は開始年齢で9割決まる。"),
        ("外資ジャンプアップを急ぐ", "#4FC3F7",
         "Disco3-5年でLam/AMAT→NVIDIA。\n年収差¥3-5Mの差が\n40年で¥100-200Mの差になる。"),
        ("r > g を味方に", "#FF9800",
         "給与所得より資本所得を育てる。\nRSU(株式報酬)の多い外資を選ぶ\nのも戦略のうち。"),
        ("AGI前に資産基盤を", "#FF5722",
         "2035-2040年頃のAGI以降は\n雇用構造が激変する可能性。\nその前に投資元本を積み上げる。"),
    ]

    y_pos = [0.82, 0.58, 0.35, 0.10]
    for (title, color, body), y in zip(items, y_pos):
        ax.text(0.03, y+0.10, f"◆ {title}", transform=ax.transAxes,
                color=color, fontsize=9.5, fontweight="bold")
        ax.text(0.03, y, body, transform=ax.transAxes,
                color=WHITE, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#1c1c1c",
                          edgecolor=color, alpha=0.7))


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    fig = plt.figure(figsize=(22, 28), facecolor=BG)
    fig.suptitle(
        "r > g  2129年予測 × 半導体エンジニアの個人資産シミュレーション",
        fontsize=17, color=WHITE, fontweight="bold", y=0.986
    )

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.50, wspace=0.38,
        left=0.06, right=0.97,
        top=0.960, bottom=0.04,
    )

    p_rg_forecast(      fig.add_subplot(gs[0, :2]))
    p_rg_gap(           fig.add_subplot(gs[0,  2]))
    p_personal_wealth(  fig.add_subplot(gs[1, :2]))
    p_scenario_comparison(fig.add_subplot(gs[1, 2]))
    p_percentile(       fig.add_subplot(gs[2, :2]))
    p_key_decisions(    fig.add_subplot(gs[2,  2]))

    out = os.path.join(OUT_DIR, "rg_forecast_2129.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
