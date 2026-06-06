"""
Exit vs IPO — 現実的シナリオ比較 + VC株式の仕組み
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
# 1. 株式希薄化の仕組み — VC/エンジェルは何を「取る」か
# ════════════════════════════════════════════════════════════════════════════

# シナリオ: 資本金¥100万で設立、株1000株発行
DILUTION_STORY = [
    # (ラウンド, 新規発行株数, 投資額M, バリュエーションM, 投資家種別, color)
    ("創業",      0,    0,      10,   "創業者",    "#42A5F5"),
    ("エンジェル", 200, 200,   1200,  "エンジェル","#FF9800"),
    ("Seed VC",   300, 3000,  12000,  "Seed VC",   "#FF5722"),
    ("Series A",  500,15000,  60000,  "Series A",  "#9C27B0"),
]

def calc_cap_table():
    total_shares  = 1000
    founder_shares= 1000
    rows = []
    for rnd, new_shares, invest_M, val_M, inv_type, color in DILUTION_STORY:
        founder_pct = founder_shares / total_shares * 100
        rows.append({
            "round": rnd, "total": total_shares,
            "founder": founder_shares, "founder_pct": founder_pct,
            "invest_M": invest_M, "val_M": val_M,
            "color": color, "inv_type": inv_type,
        })
        total_shares  += new_shares
        # founder株は変わらない
    return rows

# ════════════════════════════════════════════════════════════════════════════
# 2. 優先株 vs 普通株 — VCが要求する権利
# ════════════════════════════════════════════════════════════════════════════

VC_RIGHTS = [
    ("清算優先権\n(Liquidation Preference)", "#FF1744",
     "EXIT時にVCが先に回収する権利\n1x: 投資額を先に回収\n2x: 投資額の2倍を先に回収\n→ 安すぎるEXITで創業者がゼロになる"),
    ("反希薄化条項\n(Anti-dilution)", "#FF5722",
     "次ラウンドが低バリュエーションの場合\nVCの持分を自動調整する条項\n→ 創業者の持分が更に減る"),
    ("拒否権\n(Veto Right)", "#FF9800",
     "M&A・IPO・新規調達に対して\nVCが拒否できる権利\n→ EXIT戦略をVCに縛られる"),
    ("取締役指名権\n(Board Seat)", "#FFC107",
     "VC が取締役を1-2名指名できる\n→ 経営判断にVCが直接関与"),
    ("情報開示権\n(Information Rights)", "#4CAF50",
     "月次・四半期の財務情報開示義務\n→ 競合他社に近いVCには注意"),
]

# ════════════════════════════════════════════════════════════════════════════
# 3. EXIT(M&A) vs IPO 比較
# ════════════════════════════════════════════════════════════════════════════

EXIT_COMPARE = {
    "M&A\n(戦略的買収)": {
        "realistic_arr_M":   100,    # ARR ¥1億から現実的
        "typical_multiple":  15,
        "timeline_years":    4,
        "probability":       0.15,   # 15%
        "founder_take_M":    1500,   # ¥15億
        "color":             "#4CAF50",
        "pros": ["早い(4-6年)", "確実性高い", "ARR¥1億から可能",
                 "事業継続の場合も多い"],
        "cons": ["ブランドが消える", "VCの同意が必要",
                 "値下げ交渉される", "PMI(統合)が大変"],
        "buyers": ["Synopsys", "Cadence", "ASML", "TEL", "KLA",
                   "Disco自身", "NTTデータ", "富士通"],
    },
    "IPO\n(東証グロース)": {
        "realistic_arr_M":   500,    # ARR ¥5億から現実的
        "typical_multiple":  20,
        "timeline_years":    8,
        "probability":       0.03,
        "founder_take_M":    10000,
        "color":             "#FF9800",
        "pros": ["青天井", "ブランド価値", "採用力UP",
                 "部分売却で現金化"],
        "cons": ["8-10年かかる", "監査コスト年¥5000万+",
                 "株主へのIR義務", "短期業績プレッシャー"],
        "buyers": ["市場(個人・機関投資家)"],
    },
    "セカンダリー\n(持分売却)": {
        "realistic_arr_M":   200,
        "typical_multiple":  12,
        "timeline_years":    5,
        "probability":       0.10,
        "founder_take_M":    2400,
        "color":             "#42A5F5",
        "pros": ["IPO前に現金化", "会社は存続",
                 "リスク分散", "Series Bで一部売却可能"],
        "cons": ["バリュエーション低め",
                 "既存VCの同意が必要"],
        "buyers": ["成長VC", "PE", "他の戦略投資家"],
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 4. 半導体ソフトウェアの M&A 実例
# ════════════════════════════════════════════════════════════════════════════

REAL_MA = [
    # (買収側, 被買収, 年, 金額B$, ARR_M$, multiple, color)
    ("Synopsys",  "Ansys",           2024, 35.0,  2400, 14.6, "#FF5722"),
    ("Cadence",   "OpenEye Sci.",    2021,  0.5,    40, 12.5, "#FF9800"),
    ("ASML",      "Cymer",           2013,  2.5,   150, 16.7, "#FFC107"),
    ("KLA",       "Orbotech",        2019,  3.4,   500,  6.8, "#4CAF50"),
    ("Lam Res.",  "Coventor",        2017,  0.25,   20, 12.5, "#42A5F5"),
    ("TEL",       "Oerlikon (一部)", 2012,  0.5,    60,  8.3, "#9C27B0"),
]

# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def p_dilution_story(ax):
    sa(ax)
    ax.set_title("VCに株を「取られる」仕組み — キャップテーブル推移", fontsize=10, pad=5)

    rows = calc_cap_table()
    rounds      = [r["round"] for r in rows]
    founder_pct = [r["founder_pct"] for r in rows]
    val_M       = [r["val_M"] for r in rows]
    colors      = [r["color"] for r in rows]

    x = np.arange(len(rows))
    bars = ax.bar(x, founder_pct, color=colors, alpha=0.85, width=0.5)

    for bar, p, r in zip(bars, founder_pct, rows):
        ax.text(bar.get_x()+bar.get_width()/2, p+0.8,
                f"創業者\n{p:.0f}%", ha="center", color=WHITE,
                fontsize=8.5, fontweight="bold")
        ax.text(bar.get_x()+bar.get_width()/2, p/2,
                f"バリュ\n¥{r['val_M']/100:.0f}億",
                ha="center", color=WHITE, fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(rounds, color=WHITE, fontsize=9)
    ax.set_ylabel("創業者持分 (%)")
    ax.set_ylim(0, 120)
    ax.axhline(50, color="#FF5722", lw=1.2, ls="--", alpha=0.6)
    ax.text(0.1, 51, "50%: 経営権の実質的な境界線", color="#FF5722", fontsize=7.5)

    note = ("重要: 株は「取られる」のではなく「売る」\n"
            "その代わりに現金が入る。バリュエーションが高いほど少ない株で多くの資金を得られる")
    ax.text(0.01, 0.02, note, transform=ax.transAxes, color=DIM, fontsize=8)


def p_vc_rights(ax):
    sa(ax)
    ax.set_facecolor("#060606")
    ax.axis("off")
    ax.set_title("VCが要求する「権利」— 株以外に何を渡すか", fontsize=10, pad=5)

    for i, (title, color, body) in enumerate(VC_RIGHTS):
        y = 0.88 - i * 0.175
        ax.text(0.02, y, title.replace("\n", " "), transform=ax.transAxes,
                color=color, fontsize=8.5, fontweight="bold")
        ax.text(0.02, y-0.04, body, transform=ax.transAxes,
                color=WHITE, fontsize=7.5, va="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#111",
                          edgecolor=color, alpha=0.7))

    ax.text(0.5, 0.01,
            "対策: タームシートは弁護士と一緒に読む。清算優先権は1xまで、拒否権は最小限に交渉する",
            transform=ax.transAxes, ha="center",
            color="#FFD700", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="#111", edgecolor="#FFD700", alpha=0.8))


def p_exit_compare(ax):
    sa(ax)
    ax.set_title("M&A vs IPO vs セカンダリー — 現実的な比較", fontsize=10, pad=5)

    methods = list(EXIT_COMPARE.keys())
    x = np.arange(len(methods))
    w = 0.22

    metrics = {
        "必要ARR\n(÷100=億円)": [EXIT_COMPARE[m]["realistic_arr_M"]/100 for m in methods],
        "タイムライン\n(年)":    [EXIT_COMPARE[m]["timeline_years"] for m in methods],
        "成功確率\n(×10%)":     [EXIT_COMPARE[m]["probability"]*100 for m in methods],
    }
    bar_colors = ["#42A5F5","#FF9800","#4CAF50"]

    for i, (label, vals) in enumerate(metrics.items()):
        bars = ax.bar(x + (i-1)*w, vals,
                      w, alpha=0.85, color=bar_colors[i], label=label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.1,
                    f"{v:.0f}", ha="center", color=WHITE, fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, color=WHITE, fontsize=9)
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE)
    ax.set_ylim(0, 18)

    # 創業者取り分
    for i, m in enumerate(methods):
        take = EXIT_COMPARE[m]["founder_take_M"]
        c    = EXIT_COMPARE[m]["color"]
        ax.text(i, 16.5, f"創業者取り分\n¥{take/100:.0f}億",
                ha="center", color=c, fontsize=8.5, fontweight="bold")


def p_ma_examples(ax):
    sa(ax)
    ax.set_title("半導体ソフト M&A 実例 — マルチプル相場", fontsize=10, pad=5)

    buyers   = [f"{r[0]}→{r[1]}" for r in REAL_MA]
    mults    = [r[5] for r in REAL_MA]
    amounts  = [r[3] for r in REAL_MA]
    colors   = [r[6] for r in REAL_MA]
    years    = [r[2] for r in REAL_MA]

    y = np.arange(len(REAL_MA))
    bars = ax.barh(y, mults, color=colors, alpha=0.85, height=0.55)
    for bar, m, a, yr in zip(bars, mults, amounts, years):
        ax.text(m+0.1, bar.get_y()+bar.get_height()/2,
                f"{m:.1f}x ARR  (${a}B, {yr}年)",
                color=WHITE, va="center", fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels(buyers, color=WHITE, fontsize=8)
    ax.set_xlabel("ARR マルチプル")
    ax.set_xlim(0, 22)
    ax.axvline(12, color="#FFD700", lw=1.5, ls="--", alpha=0.7)
    ax.text(12.2, 5.5, "相場: 10-15x", color="#FFD700", fontsize=8.5)

    note = "半導体ソフトは「スイッチングコストが高い」ため\n一般SaaSより高マルチプルで買収される"
    ax.text(0.01, 0.02, note, transform=ax.transAxes, color=DIM, fontsize=8)


def p_realistic_path(ax):
    sa(ax)
    ax.set_facecolor("#060606")
    ax.axis("off")
    ax.set_title("wafer-proc-sim の現実的EXIT経路", fontsize=11, pad=5)

    scenarios = [
        ("最短・最現実的\n(4-6年)", "#4CAF50",
         "M&A by 大手半導体装置メーカー\n"
         "Synopsys / Cadence / ASML / TEL / Lam\n"
         "ARR ¥1-3億 × 12-20倍 = ¥12-60億\n"
         "創業者取り分: ¥5-20億\n"
         "→ 「資本側」に確実に到達"),
        ("中期・現実的\n(6-8年)", "#FF9800",
         "セカンダリー + 事業継続\n"
         "Series Bで創業者株の20-30%を売却\n"
         "¥10-30億を手元に確保しながら\n"
         "会社は成長を続ける"),
        ("長期・理想\n(8-12年)", "#FF5722",
         "東証グロース IPO\n"
         "ARR ¥5-10億達成後に上場\n"
         "時価総額 ¥100-300億\n"
         "創業者取り分: ¥25-75億\n"
         "→ ただし成功確率3%"),
        ("最重要な真実", "#FFD700",
         "wafer-proc-simのような\n「ニッチ×高参入障壁×B2B」ツールは\n"
         "M&Aが一番現実的で一番コスパが高い\n"
         "Synopsysがプロセスシミュレーション部門を\n¥30-50億で買いたいと言ったら\n迷わず売っていい"),
    ]

    positions = [(0.01, 0.72), (0.51, 0.72), (0.01, 0.28), (0.51, 0.28)]
    for (title, color, body), (x, y) in zip(scenarios, positions):
        ax.text(x+0.23, y+0.14, title, transform=ax.transAxes,
                ha="center", color=color, fontsize=9, fontweight="bold")
        ax.text(x+0.01, y, body, transform=ax.transAxes,
                color=WHITE, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#111",
                          edgecolor=color, alpha=0.8, linewidth=1.5),
                wrap=True)


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    fig = plt.figure(figsize=(22, 26), facecolor=BG)
    fig.suptitle(
        "VC/エンジェル × Exit vs IPO — 半導体SaaS 創業者のためのお金の真実",
        fontsize=16, color=WHITE, fontweight="bold", y=0.987
    )

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        hspace=0.46, wspace=0.32,
        left=0.06, right=0.97,
        top=0.962, bottom=0.04,
    )

    p_dilution_story(fig.add_subplot(gs[0, 0]))
    p_vc_rights(     fig.add_subplot(gs[0, 1]))
    p_exit_compare(  fig.add_subplot(gs[1, 0]))
    p_ma_examples(   fig.add_subplot(gs[1, 1]))
    p_realistic_path(fig.add_subplot(gs[2, :]))

    out = os.path.join(OUT_DIR, "exit_strategy_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
