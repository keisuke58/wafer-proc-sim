"""
M&A EXIT ¥5億 確実達成モデル
「確実性重視でM&Aで最低¥5億」を逆算で設計する
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
BG, PANEL, GRID, WHITE, DIM = "#0d0d0d", "#141414", "#2a2a2a", "#f0f0f0", "#666666"
GOLD = "#FFD700"

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
# 逆算: ¥5億を手元に残すには何が必要か
#
# 創業者持分: Seed後 85% → SeriesA後 65% → exit時 ~55-65%
# 清算優先権考慮後の実効持分: ~50%と保守的に見る
# ¥5億 ÷ 50% = バリュエーション ¥10億 が最低ライン
# ARRマルチプル 10倍(保守) → ARR ¥1億 が目標
# ════════════════════════════════════════════════════════════════════════════

TARGET_TAKE_M   = 500    # ¥5億
FOUNDER_PCT_AT_EXIT = 0.55  # 保守的に55%
TARGET_VALUATION_M  = TARGET_TAKE_M / FOUNDER_PCT_AT_EXIT   # ≈ ¥909M = ¥90億
MULTIPLE_CONSERVATIVE = 10
MULTIPLE_BASE         = 15
TARGET_ARR_M_LOW  = TARGET_VALUATION_M / MULTIPLE_BASE         # ≈ ¥6億
TARGET_ARR_M_HIGH = TARGET_VALUATION_M / MULTIPLE_CONSERVATIVE # ≈ ¥9億

# ════════════════════════════════════════════════════════════════════════════
# ARR成長シミュレーション
# ════════════════════════════════════════════════════════════════════════════

SIM_YEARS = np.arange(2031, 2042)  # 起業2031年想定

def arr_growth(arr0_M, growth_rate, years):
    arr = [arr0_M]
    for _ in years[1:]:
        arr.append(arr[-1] * (1 + growth_rate))
    return np.array(arr)

ARR_SCENARIOS = {
    "slow":   ("成長率50%/年\n(保守)", arr_growth(20, 0.50, SIM_YEARS), "#546e7a"),
    "base":   ("成長率80%/年\n(標準)", arr_growth(20, 0.80, SIM_YEARS), "#FF9800"),
    "fast":   ("成長率120%/年\n(積極)", arr_growth(20, 1.20, SIM_YEARS), "#4CAF50"),
}
# 初期ARR ¥2,000万 = 顧客2社×¥1,000万契約

# ════════════════════════════════════════════════════════════════════════════
# 顧客獲得モデル: 誰が最初の顧客になるか
# ════════════════════════════════════════════════════════════════════════════

CUSTOMER_SEGMENTS = {
    "大手装置メーカー\n(国内)": {
        "examples":     "Disco, TEL, 荏原製作所, アドバンテスト",
        "contract_M":   1500,   # ¥1,500万/年
        "sales_cycle_months": 12,
        "win_prob":     0.30,
        "n_potential":  20,
        "color":        "#4CAF50",
        "note":         "Disco在籍時の人脈で\n直接アクセス可能",
    },
    "大手半導体\nメーカー": {
        "examples":     "Kioxia, ソニーセミコン, ルネサス",
        "contract_M":   2000,
        "sales_cycle_months": 18,
        "win_prob":     0.15,
        "n_potential":  15,
        "color":        "#FF9800",
        "note":         "プロセスエンジニアと\nの繋がりが鍵",
    },
    "外資装置メーカー\n(日本法人)": {
        "examples":     "Lam Japan, AMAT Japan, KLA Japan",
        "contract_M":   3000,
        "sales_cycle_months": 24,
        "win_prob":     0.10,
        "n_potential":  10,
        "color":        "#FF5722",
        "note":         "転職後の人脈が活きる\n外資は高単価",
    },
    "大学・研究機関": {
        "examples":     "東大, 東工大, AIST, IMEC",
        "contract_M":   200,
        "sales_cycle_months": 3,
        "win_prob":     0.60,
        "n_potential":  50,
        "color":        "#42A5F5",
        "note":         "単価低いが実績作りに\n最速で契約できる",
    },
    "海外スタートアップ\n(SaaS型)": {
        "examples":     "Rapidus関連, 台湾新興Fab",
        "contract_M":   500,
        "sales_cycle_months": 6,
        "win_prob":     0.25,
        "n_potential":  30,
        "color":        "#9C27B0",
        "note":         "グローバル展開の足がかり\n英語論文の実績が効く",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# ¥5億達成までのマイルストーン
# ════════════════════════════════════════════════════════════════════════════

MILESTONES = [
    (2026, "Disco入社\n副業開始",      "#42A5F5",
     "wafer-proc-simを大学・研究機関に\n無料提供 → フィードバック収集\n論文1本: ECTC/VLSI投稿"),
    (2028, "最初の有料契約",           "#4CAF50",
     "大学 or 中小装置メーカーに\n年¥100-300万で有償提供\n「払ってもらう」経験が全て"),
    (2031, "独立・会社設立",           "#FF9800",
     "ARR ¥2,000万 (顧客2-3社)\nNEDO補助金申請 → ¥3,000万\n個人貯蓄¥1,500万でランウェイ確保"),
    (2033, "ARR ¥1億突破",            "#FF5722",
     "顧客8-10社\nSeed/エンジェル ¥3,000万調達\nエンジニア2名採用"),
    (2035, "M&A打診が来る",           "#9C27B0",
     "ARR ¥3-6億 で買収打診が来始める\nSynopsys/TEL/Lam が接触してくる\nこの時点で売るかどうか判断"),
    (2037, "EXIT ¥5億以上",           GOLD,
     "ARR ¥6-8億 × 10-15倍\n= バリュエーション ¥60-120億\n創業者55% → 手元¥33-66億\n★ 最低でも¥5億は確実"),
]

# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def p_backsolve(ax):
    sa(ax)
    ax.set_title("逆算: ¥5億手元に残すために必要なARR", fontsize=10, pad=5)

    multiples = np.arange(8, 22, 0.5)
    for founder_pct, label, color in [
        (0.70, "創業者持分70% (外部調達なし)", "#4CAF50"),
        (0.55, "創業者持分55% (Seed+A調達後)", "#FF9800"),
        (0.40, "創業者持分40% (フル希薄化)", "#FF5722"),
    ]:
        needed_val = TARGET_TAKE_M / founder_pct
        needed_arr = needed_val / multiples
        ax.plot(multiples, needed_arr, lw=2.2, color=color, label=label)

    ax.axhline(100, color=GOLD, lw=2, ls="--")
    ax.text(8.2, 102, "ARR ¥1億 (現実的な達成ライン)", color=GOLD, fontsize=8.5, fontweight="bold")
    ax.fill_between(multiples,
                    TARGET_TAKE_M / 0.55 / multiples,
                    TARGET_TAKE_M / 0.70 / multiples,
                    alpha=0.12, color="#4CAF50")
    ax.text(16, 55, "ここに入れば\n¥5億達成ゾーン", color="#4CAF50", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="#1a1a1a", edgecolor="#4CAF50", alpha=0.8))

    ax.set_xlabel("ARRマルチプル (買収時)")
    ax.set_ylabel("必要ARR (百万円)")
    ax.set_ylim(0, 300)
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE)
    note = f"目標: ¥{TARGET_TAKE_M}M手元\n保守的試算: ARR¥{TARGET_ARR_M_HIGH:.0f}M × 10倍 × 持分55%"
    ax.text(0.01, 0.97, note, transform=ax.transAxes, color=GOLD,
            fontsize=8.5, va="top", fontweight="bold")


def p_arr_path(ax):
    sa(ax)
    ax.set_title("ARR成長経路 — ¥5億EXITラインまでの軌跡", fontsize=10, pad=5)

    for key, (label, arr, color) in ARR_SCENARIOS.items():
        ax.plot(SIM_YEARS, arr, lw=2.5, color=color, label=label)
        # EXIT判断年 (ARR≥TARGET_ARR_M_LOW を超えた最初の年)
        hit = np.where(arr >= TARGET_ARR_M_LOW)[0]
        if len(hit):
            yr = SIM_YEARS[hit[0]]
            ax.scatter([yr], [arr[hit[0]]], s=120, color=color,
                       edgecolors=WHITE, lw=1.5, zorder=5)
            ax.annotate(f"EXIT可能\n{yr}年",
                        (yr, arr[hit[0]]),
                        xytext=(yr+0.3, arr[hit[0]]+30),
                        color=color, fontsize=7.5,
                        arrowprops=dict(arrowstyle="->", color=color, lw=1))

    ax.axhline(TARGET_ARR_M_LOW,  color=GOLD,     lw=2,   ls="--")
    ax.axhline(TARGET_ARR_M_HIGH, color="#FF5722", lw=1.5, ls=":")
    ax.text(2031.1, TARGET_ARR_M_LOW+5,
            f"¥5億EXITライン(15x): ARR≥¥{TARGET_ARR_M_LOW:.0f}M",
            color=GOLD, fontsize=8, fontweight="bold")
    ax.text(2031.1, TARGET_ARR_M_HIGH+5,
            f"¥5億EXITライン(10x): ARR≥¥{TARGET_ARR_M_HIGH:.0f}M",
            color="#FF5722", fontsize=7.5)

    ax.set_ylabel("ARR (百万円)")
    ax.set_ylim(0, 800)
    ax.set_xlim(2030.5, 2041.5)
    ax.legend(fontsize=8.5, facecolor="#222", edgecolor="#444", labelcolor=WHITE)
    note = "初期ARR ¥2,000万 = 顧客2社 × ¥1,000万/年から出発"
    ax.text(0.01, 0.02, note, transform=ax.transAxes, color=DIM, fontsize=8)


def p_customers(ax):
    sa(ax)
    ax.set_title("顧客セグメント — 単価 × 獲得難易度", fontsize=10, pad=5)

    segs   = list(CUSTOMER_SEGMENTS.keys())
    prices = [CUSTOMER_SEGMENTS[s]["contract_M"] for s in segs]
    probs  = [CUSTOMER_SEGMENTS[s]["win_prob"]*100 for s in segs]
    cycles = [CUSTOMER_SEGMENTS[s]["sales_cycle_months"] for s in segs]
    colors = [CUSTOMER_SEGMENTS[s]["color"] for s in segs]
    notes  = [CUSTOMER_SEGMENTS[s]["note"] for s in segs]

    sc = ax.scatter(probs, prices, s=[c*15 for c in cycles],
                    c=colors, alpha=0.85, edgecolors=WHITE, lw=1, zorder=3)

    for i, (seg, p, pr, note) in enumerate(zip(segs, prices, probs, notes)):
        ax.annotate(seg.replace("\n"," "),
                    (pr, p),
                    xytext=(pr+2, p+80),
                    color=WHITE, fontsize=7.5,
                    arrowprops=dict(arrowstyle="-", color="#555", lw=0.8))

    ax.set_xlabel("受注確率 (%)")
    ax.set_ylabel("年間契約単価 (百万円)")
    ax.set_ylim(-200, 3500)
    ax.set_xlim(0, 80)
    ax.axhline(1000, color=GOLD, lw=1, ls="--", alpha=0.6)
    ax.text(1, 1050, "¥1,000万/年以上が理想ライン", color=GOLD, fontsize=7.5)
    note2 = "円のサイズ = 営業サイクル (大=時間がかかる)\n→ まず大学で実績、次に国内装置メーカーが最短経路"
    ax.text(0.01, 0.01, note2, transform=ax.transAxes, color=DIM, fontsize=7.5)


def p_milestone_chart(ax):
    sa(ax)
    ax.set_title("¥5億EXIT までのマイルストーンロードマップ", fontsize=10, pad=5)
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, len(MILESTONES) - 0.2)

    for i, (year, title, color, body) in enumerate(MILESTONES):
        y = len(MILESTONES) - 1 - i
        # 年ラベル
        ax.text(0.2, y, str(year), color=color, fontsize=11,
                fontweight="bold", va="center")
        # タイトル
        ax.text(1.2, y+0.15, title.replace("\n", " "),
                color=color, fontsize=9, fontweight="bold", va="center")
        # 内容
        ax.text(1.2, y-0.22, body,
                color=WHITE, fontsize=7.5, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#111",
                          edgecolor=color, alpha=0.7))
        # ライン
        if i < len(MILESTONES)-1:
            ax.plot([0.2, 0.2], [y-0.35, y-0.85],
                    color=color, lw=2, alpha=0.5)
            ax.annotate("", xy=(0.2, y-0.80), xytext=(0.2, y-0.40),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))


def p_numbers(ax):
    sa(ax)
    ax.set_facecolor("#060606")
    ax.axis("off")
    ax.set_title("¥5億EXITに必要な数字 — 全部逆算", fontsize=11, pad=5)

    rows = [
        ("目標: 手元¥5億",                GOLD,    ""),
        ("必要バリュエーション",           WHITE,   f"¥{TARGET_TAKE_M/FOUNDER_PCT_AT_EXIT:.0f}M ≈ ¥90億"),
        ("必要ARR (15倍マルチプル想定)",   WHITE,   f"¥{TARGET_ARR_M_LOW:.0f}M ≈ ¥6億"),
        ("必要ARR (10倍で保守的に)",       WHITE,   f"¥{TARGET_ARR_M_HIGH:.0f}M ≈ ¥9億"),
        ("─────────────────────",         DIM,     ""),
        ("顧客パターン①",                 "#4CAF50","装置メーカー5社 × ¥1,500万 = ARR¥7,500万\n+ 半導体メーカー3社 × ¥2,000万 = ARR¥6,000万\n→ 合計ARR ¥1.35億 (もう少し必要)"),
        ("顧客パターン②",                 "#FF9800","外資装置メーカー4社 × ¥3,000万 = ARR¥1.2億\n国内メーカー5社 × ¥1,500万 = ARR¥7,500万\n→ 合計ARR ¥1.95億 → EXIT射程圏"),
        ("顧客パターン③ (¥6億ARR)",       GOLD,    "大手10社 × ¥3,000万\n+ 中堅20社 × ¥1,500万\n= ARR ¥6億 → バリュエーション¥60-90億\n→ 創業者取り分 ¥33-50億 ★達成"),
        ("─────────────────────",         DIM,     ""),
        ("最重要: 最初の1社",             "#FF5722","「Discoの元同僚に使ってもらう」\nこれだけ考えればいい。\nその1社から全てが始まる。"),
    ]

    y = 0.97
    for label, color, value in rows:
        if label.startswith("─"):
            y -= 0.03
            ax.plot([0.01, 0.99], [y, y], color="#333",
                    lw=0.8, transform=ax.transAxes)
            y -= 0.02
            continue
        ax.text(0.02, y, label, transform=ax.transAxes,
                color=color, fontsize=8.5, fontweight="bold", va="top")
        if value:
            ax.text(0.42, y, value, transform=ax.transAxes,
                    color=WHITE, fontsize=8, va="top")
        y -= 0.09 if "\n" in value else 0.07


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    fig = plt.figure(figsize=(22, 26), facecolor=BG)
    fig.suptitle(
        f"M&A EXIT 最低¥5億 — 逆算で設計する起業戦略",
        fontsize=17, color=GOLD, fontweight="bold", y=0.987
    )

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        hspace=0.46, wspace=0.34,
        left=0.06, right=0.97,
        top=0.962, bottom=0.04,
    )

    p_backsolve(       fig.add_subplot(gs[0, 0]))
    p_arr_path(        fig.add_subplot(gs[0, 1]))
    p_customers(       fig.add_subplot(gs[1, 0]))
    p_milestone_chart( fig.add_subplot(gs[1, 1]))
    p_numbers(         fig.add_subplot(gs[2, 0]))

    ax_final = fig.add_subplot(gs[2, 1])
    sa(ax_final)
    ax_final.set_facecolor("#060606")
    ax_final.axis("off")
    ax_final.set_title("確実性を上げる5つの原則", fontsize=11, pad=5)

    principles = [
        ("#1  顧客を先に見つける",  "#4CAF50",
         "コードを書く前に\n「年¥1,000万払う人がいるか」を確認\nDiscoの元同僚に聞くだけでいい"),
        ("#2  売上でなくARRで考える","#FF9800",
         "単発売上より年間契約\n1社からの¥1,000万/年契約は\n売却時に¥1億-1.5億の価値になる"),
        ("#3  M&A打診を自分から作る","#FF5722",
         "ARR¥1億になったら\nSynopsys/TELに自分から接触する\n「買いたい会社」として認識させる"),
        ("#4  希薄化を最小化する",   "#9C27B0",
         "補助金を最大活用してVC依存を減らす\nSeed調達は¥3,000万以内に抑える\n→ EXIT時の持分を60%以上守る"),
        ("#5  失敗コストを下げる",   "#42A5F5",
         "Disco在籍中にMVPを完成させる\n独立するのは有料顧客が1社ついてから\n→ ゼロからのリスクを消す"),
    ]

    for i, (title, color, body) in enumerate(principles):
        y = 0.86 - i * 0.175
        ax_final.text(0.02, y, title, transform=ax_final.transAxes,
                      color=color, fontsize=9, fontweight="bold")
        ax_final.text(0.02, y-0.04, body, transform=ax_final.transAxes,
                      color=WHITE, fontsize=8, va="top",
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="#111",
                                edgecolor=color, alpha=0.75))

    out = os.path.join(OUT_DIR, "ma_target_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
