"""
起業資金調達 × 個人財務ランウェイ 完全モデル
半導体シミュレーションSaaS (wafer-proc-sim系) を想定
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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
# 1. 起業フェーズ × 必要資金 × 調達源
# ════════════════════════════════════════════════════════════════════════════

PHASES = [
    {
        "name": "0. Disco在籍\n副業MVP",
        "year": "2026-2031",
        "duration_months": 60,
        "fund_needed_M": 0,
        "monthly_burn_M": 0,
        "personal_salary_M": 7.0,
        "savings_end_M": 1500,   # 5年で¥1500万貯蓄
        "color": "#42A5F5",
        "sources": ["給与100%"],
        "milestone": "wafer-proc-simを\n1社に無料提供\n反応を見る",
    },
    {
        "name": "1. 補助金\nPre-seed",
        "year": "2031-2032",
        "duration_months": 12,
        "fund_needed_M": 500,
        "monthly_burn_M": 30,
        "personal_salary_M": 1.2,   # 自分の給与を最低限に
        "savings_end_M": 1200,
        "color": "#4CAF50",
        "sources": ["自己資金¥200M", "NEDO/JST¥300M"],
        "milestone": "MVP完成\n有料顧客1社獲得\n(年¥300-500万契約)",
    },
    {
        "name": "2. エンジェル\nSeed",
        "year": "2032-2033",
        "duration_months": 12,
        "fund_needed_M": 3000,
        "monthly_burn_M": 150,
        "personal_salary_M": 2.4,
        "savings_end_M": 900,
        "color": "#FF9800",
        "sources": ["エンジェル¥1500M", "VC Seed¥1500M"],
        "milestone": "顧客5社\nARR ¥3,000万\nエンジニア採用2名",
    },
    {
        "name": "3. Series A\nVC",
        "year": "2033-2035",
        "duration_months": 24,
        "fund_needed_M": 15000,
        "monthly_burn_M": 500,
        "personal_salary_M": 6.0,
        "savings_end_M": 900,
        "color": "#FF5722",
        "sources": ["Series A VC\n¥15,000M"],
        "milestone": "顧客30社\nARR ¥2億\nチーム15名\n海外展開開始",
    },
    {
        "name": "4. Series B\nグローバル",
        "year": "2035-2038",
        "duration_months": 36,
        "fund_needed_M": 50000,
        "monthly_burn_M": 1500,
        "personal_salary_M": 12.0,
        "savings_end_M": 900,
        "color": "#9C27B0",
        "sources": ["Series B\n¥50,000M\n(国内外VC)"],
        "milestone": "顧客200社\nARR ¥20億\nチーム80名\nIPO準備",
    },
]

# ════════════════════════════════════════════════════════════════════════════
# 2. 個人ランウェイ計算 (起業後に生活できる期間)
# ════════════════════════════════════════════════════════════════════════════

MONTHLY_LIVING_M = 0.25   # 月¥25万 (独身・東京)

def runway_months(savings_M, monthly_salary_M=0):
    net_monthly_burn = MONTHLY_LIVING_M - monthly_salary_M / 12
    if net_monthly_burn <= 0:
        return 999
    return savings_M / net_monthly_burn

# ════════════════════════════════════════════════════════════════════════════
# 3. VCの仕組み — 希薄化と創業者持分
# ════════════════════════════════════════════════════════════════════════════

DILUTION_ROUNDS = [
    ("創業時",    100.0, 0,      "#42A5F5"),
    ("Seed",       72.0, 3000,   "#4CAF50"),
    ("Series A",   50.0, 15000,  "#FF9800"),
    ("Series B",   35.0, 50000,  "#FF5722"),
    ("IPO",        25.0, 200000, "#9C27B0"),
]

# ════════════════════════════════════════════════════════════════════════════
# 4. 調達方法の比較
# ════════════════════════════════════════════════════════════════════════════

FUNDING_TYPES = {
    "補助金\n(NEDO/JST)": {
        "amount_M": 300,
        "cost": 0,        # 返済・持分なし
        "speed_months": 6,
        "conditions": "研究開発目的\n採択率20-30%",
        "best_for": "MVP開発\n技術検証",
        "color": "#4CAF50",
        "score_cost": 10,
        "score_speed": 4,
        "score_amount": 3,
    },
    "エンジェル\n投資家": {
        "amount_M": 1000,
        "cost": 10,       # 株式10%程度
        "speed_months": 3,
        "conditions": "人脈・紹介が必要\n条件交渉可",
        "best_for": "初期顧客獲得\nメンタリング",
        "color": "#FF9800",
        "score_cost": 7,
        "score_speed": 8,
        "score_amount": 5,
    },
    "Seed VC": {
        "amount_M": 3000,
        "cost": 15,
        "speed_months": 4,
        "conditions": "トラクション必要\n取締役参加",
        "best_for": "チーム組成\nPMF探索",
        "color": "#FF5722",
        "score_cost": 5,
        "score_speed": 6,
        "score_amount": 7,
    },
    "DEEPCORE\n(NVIDIA系)": {
        "amount_M": 5000,
        "cost": 15,
        "speed_months": 5,
        "conditions": "AI/半導体特化\nNVIDIAネットワーク",
        "best_for": "半導体SaaS\nに最適",
        "color": "#76FF03",
        "score_cost": 5,
        "score_speed": 5,
        "score_amount": 8,
    },
    "銀行借入\n(日本政策金融)": {
        "amount_M": 2000,
        "cost": 2,        # 金利2%
        "speed_months": 2,
        "conditions": "返済義務あり\n担保・保証人",
        "best_for": "運転資金\n設備投資",
        "color": "#546e7a",
        "score_cost": 8,
        "score_speed": 9,
        "score_amount": 6,
    },
    "事業会社\n出資(CVC)": {
        "amount_M": 10000,
        "cost": 20,
        "speed_months": 8,
        "conditions": "戦略的アライアンス\n独占条項リスク",
        "best_for": "大企業顧客\n確保と同時",
        "color": "#9C27B0",
        "score_cost": 4,
        "score_speed": 3,
        "score_amount": 9,
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 5. EXITシミュレーション (IPO/M&A)
# ════════════════════════════════════════════════════════════════════════════

ARR_MULTIPLES = {
    "SaaS B2B (標準)":        8,
    "SaaS B2B (高成長)":      15,
    "半導体ソフト (ニッチ独占)": 20,
    "EDA類似 (Synopsys級)":   30,
}

def exit_value(arr_M, multiple, founder_pct):
    valuation = arr_M * multiple
    founder_take = valuation * founder_pct / 100
    return valuation, founder_take

EXIT_SCENARIOS = [
    ("ARR ¥2億 × 8倍",   200,  8,  35, "#546e7a"),
    ("ARR ¥5億 × 15倍",  500,  15, 30, "#FF9800"),
    ("ARR ¥20億 × 20倍", 2000, 20, 25, "#FF5722"),
    ("ARR ¥100億 × 30倍",10000,30, 20, "#FFD700"),
]

# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def p_phases(ax):
    sa(ax)
    ax.set_title("起業フェーズ × 資金需要 × 個人ランウェイ", fontsize=10, pad=5)

    x = np.arange(len(PHASES))
    funds = [p["fund_needed_M"]/100 for p in PHASES]  # 百万円→億円
    savings= [p["savings_end_M"]/100 for p in PHASES]
    colors = [p["color"] for p in PHASES]

    bars = ax.bar(x, funds, color=colors, alpha=0.85, width=0.4, label="調達額 (億円)")
    ax.plot(x, savings, "o--", color="#FFD700", lw=2, ms=7, label="個人貯蓄残高 (億円)", zorder=5)

    for bar, f in zip(bars, funds):
        if f > 0:
            ax.text(bar.get_x()+bar.get_width()/2, f+0.5,
                    f"¥{f:.0f}億", ha="center", color=WHITE, fontsize=8, fontweight="bold")

    for i, (s, p) in enumerate(zip(savings, PHASES)):
        rw = runway_months(p["savings_end_M"], p["personal_salary_M"])
        rw_label = f"ランウェイ\n∞" if rw > 200 else f"ランウェイ\n{rw:.0f}ヶ月"
        ax.text(i, s + 1.5, rw_label, ha="center", color="#FFD700", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([p["name"] for p in PHASES], color=WHITE, fontsize=7.5)
    ax.set_ylabel("金額 (億円)")
    ax.set_ylim(0, 60)
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE)


def p_dilution(ax):
    sa(ax)
    ax.set_title("資金調達 = 株の希薄化 — 創業者持分の変化", fontsize=10, pad=5)

    rounds  = [r[0] for r in DILUTION_ROUNDS]
    pcts    = [r[1] for r in DILUTION_ROUNDS]
    raises  = [r[2] for r in DILUTION_ROUNDS]
    colors  = [r[3] for r in DILUTION_ROUNDS]

    x = np.arange(len(rounds))
    bars = ax.bar(x, pcts, color=colors, alpha=0.85, width=0.55)
    for bar, p, r in zip(bars, pcts, raises):
        ax.text(bar.get_x()+bar.get_width()/2, p+0.8,
                f"{p:.0f}%", ha="center", color=WHITE, fontsize=9, fontweight="bold")
        if r > 0:
            ax.text(bar.get_x()+bar.get_width()/2, p/2,
                    f"+¥{r/100:.0f}億調達", ha="center", color=WHITE, fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(rounds, color=WHITE, fontsize=9)
    ax.set_ylabel("創業者持分 (%)")
    ax.set_ylim(0, 115)
    ax.axhline(20, color="#FF5722", lw=1.2, ls="--", alpha=0.7)
    ax.text(0.1, 21, "20%を下回ると創業者のコントロールが弱まる", color="#FF5722", fontsize=7.5)
    note = "IPO時に25%持っていれば\nバリュエーション×0.25が創業者の取り分"
    ax.text(3.5, 80, note, ha="center", color=DIM, fontsize=8,
            bbox=dict(boxstyle="round", facecolor="#1a1a1a", edgecolor="#555"))


def p_funding_radar(ax):
    sa(ax)
    ax.set_title("調達方法 比較 — コスト/スピード/金額", fontsize=10, pad=5)

    methods = list(FUNDING_TYPES.keys())
    x = np.arange(len(methods))
    w = 0.22

    cost_scores  = [FUNDING_TYPES[m]["score_cost"]   for m in methods]
    speed_scores = [FUNDING_TYPES[m]["score_speed"]  for m in methods]
    amt_scores   = [FUNDING_TYPES[m]["score_amount"] for m in methods]
    colors       = [FUNDING_TYPES[m]["color"]        for m in methods]

    ax.bar(x - w, cost_scores,  w, color=colors, alpha=0.6,  label="コストの低さ")
    ax.bar(x,     speed_scores, w, color=colors, alpha=0.8,  label="スピード")
    ax.bar(x + w, amt_scores,   w, color=colors, alpha=1.0,  label="調達額")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, color=WHITE, fontsize=8)
    ax.set_ylabel("スコア (10=最良)")
    ax.set_ylim(0, 12)
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE)

    for i, (m, c) in enumerate(zip(methods, colors)):
        amt = FUNDING_TYPES[m]["amount_M"]
        ax.text(i, 11.2, f"最大¥{amt/100:.0f}億" if amt >= 100 else f"最大¥{amt}M",
                ha="center", color=c, fontsize=7, fontweight="bold")


def p_exit(ax):
    sa(ax)
    ax.set_title("EXIT シミュレーション — 創業者の取り分", fontsize=10, pad=5)

    labels  = [s[0] for s in EXIT_SCENARIOS]
    vals_M  = [exit_value(s[1], s[2], s[3])[0] for s in EXIT_SCENARIOS]
    takes_M = [exit_value(s[1], s[2], s[3])[1] for s in EXIT_SCENARIOS]
    colors  = [s[4] for s in EXIT_SCENARIOS]

    x = np.arange(len(EXIT_SCENARIOS))
    w = 0.32
    ax.bar(x - w/2, [v/100 for v in vals_M],  w, color=colors, alpha=0.5, label="会社バリュエーション")
    ax.bar(x + w/2, [t/100 for t in takes_M], w, color=colors, alpha=1.0, label="創業者取り分")

    for i, (v, t, sc) in enumerate(zip(vals_M, takes_M, EXIT_SCENARIOS)):
        founder_pct = sc[3]
        ax.text(i+w/2, t/100+5, f"¥{t/100:.0f}億\n(持分{founder_pct}%)",
                ha="center", color=WHITE, fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=WHITE, fontsize=8)
    ax.set_ylabel("金額 (億円)")
    ax.set_yscale("log")
    ax.set_ylim(1, 50000)
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE)

    ax.axhline(100, color="#FFD700", lw=1.5, ls="--", alpha=0.7)
    ax.text(0.01, 101, "¥100億 = 労働では絶対到達不可能な水準",
            color="#FFD700", fontsize=7.5)


def p_cashflow(ax):
    sa(ax)
    ax.set_title("個人キャッシュフロー全体像 (2026→2040)", fontsize=10, pad=5)

    years = np.arange(2026, 2041)

    # Disco在籍期 (2026-2031)
    salary = np.where(years <= 2031,
                      np.linspace(7, 9, len(years)),
                      np.where(years <= 2033, 2.4, 6.0))
    living = np.full(len(years), 3.0)
    net    = salary - living

    # 累積貯蓄
    savings = np.zeros(len(years))
    s = 0
    for i, n in enumerate(net):
        s = s * 1.07 + n    # 7%運用
        savings[i] = s

    ax.bar(years, salary, color="#42A5F5", alpha=0.7, label="年収 (給与+役員報酬)")
    ax.bar(years, -living, color="#FF5722", alpha=0.7, label="生活費")
    ax.plot(years, savings, "o-", color="#FFD700", lw=2.5, ms=5,
            label="累積資産 (運用込み)", zorder=5)

    ax.axvline(2031, color="#4CAF50", lw=1.5, ls="--")
    ax.axvline(2033, color="#FF9800", lw=1.5, ls="--")
    ax.text(2031.1, 35, "起業\n独立", color="#4CAF50", fontsize=8)
    ax.text(2033.1, 35, "Series A\n調達", color="#FF9800", fontsize=8)
    ax.axhline(0, color=WHITE, lw=0.8, alpha=0.4)

    ax.set_ylabel("百万円/年  /  累積資産(百万円)")
    ax.set_ylim(-10, 55)
    ax.legend(fontsize=7.5, facecolor="#222", edgecolor="#444", labelcolor=WHITE,
              loc="upper left")
    note = "灰色期間: 給与低下 → 貯蓄で生活\n黄色線: 株式運用で資産は増え続ける"
    ax.text(0.01, 0.02, note, transform=ax.transAxes, color=DIM, fontsize=7.5)


def p_key_numbers(ax):
    sa(ax)
    ax.set_facecolor("#060606")
    ax.axis("off")
    ax.set_title("お金で絶対に知っておくべき数字", fontsize=11, pad=5)

    entries = [
        ("ランウェイの鉄則",  "#FF5722",
         "最低18ヶ月分の個人生活費を確保してから起業\n月¥25万 × 18ヶ月 = ¥450万 (最低ライン)\n→ Disco 5年で¥1,500万貯めれば5年分"),
        ("VCに渡す株は慎重に", "#FF9800",
         "Seed で20%以上渡すと後で後悔しやすい\nSeed 15% + Series A 20% + B 15% = 残り50%\n→ IPO時の創業者持分 25-30% を死守する"),
        ("ARRが全て",         "#4CAF50",
         "SaaSの価値 = ARR × マルチプル\nARR ¥1億 × 20倍 = バリュエーション ¥20億\n→ 顧客10社 × 年¥1,000万契約 = ARR ¥1億"),
        ("補助金を使い倒す",   "#42A5F5",
         "NEDO: 最大¥1億 (返済不要、株式も渡さない)\nJST START: ¥4,000万 (大学発技術)\nものづくり補助金: ¥1,250万\n→ 合計¥1.5億を希薄化ゼロで調達できる可能性"),
        ("失敗コストの現実",   "#9C27B0",
         "ソフトウェア起業の失敗コスト = 時間だけ\n資本金¥100万+生活費¥500万 = 最大¥600万のロス\n30代前半なら十分取り返せる"),
    ]

    y_positions = [0.84, 0.66, 0.48, 0.28, 0.08]
    for (title, color, body), y in zip(entries, y_positions):
        ax.text(0.02, y+0.09, f"◆ {title}", transform=ax.transAxes,
                color=color, fontsize=9, fontweight="bold")
        ax.text(0.02, y, body, transform=ax.transAxes,
                color=WHITE, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#111",
                          edgecolor=color, alpha=0.75))


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    fig = plt.figure(figsize=(22, 30), facecolor=BG)
    fig.suptitle(
        "起業資金調達 完全ガイド — 半導体SaaS (wafer-proc-sim系) を想定",
        fontsize=17, color=WHITE, fontweight="bold", y=0.987
    )

    gs = gridspec.GridSpec(
        4, 2,
        figure=fig,
        hspace=0.48, wspace=0.32,
        left=0.06, right=0.97,
        top=0.962, bottom=0.03,
    )

    p_cashflow(      fig.add_subplot(gs[0, :]))
    p_phases(        fig.add_subplot(gs[1, 0]))
    p_dilution(      fig.add_subplot(gs[1, 1]))
    p_funding_radar( fig.add_subplot(gs[2, 0]))
    p_exit(          fig.add_subplot(gs[2, 1]))
    p_key_numbers(   fig.add_subplot(gs[3, 0]))

    ax_timeline = fig.add_subplot(gs[3, 1])
    sa(ax_timeline)
    ax_timeline.set_facecolor("#060606")
    ax_timeline.axis("off")
    ax_timeline.set_title("西岡さんの推奨タイムライン", fontsize=11, pad=5)

    tl = [
        ("2026-2031", "#42A5F5", "Disco在籍\n・毎月¥15-20万投資\n・wafer-proc-simを1社に無料提供\n・顧客の「痛み」を徹底リサーチ\n・補助金の調査・準備"),
        ("2031",      "#4CAF50", "独立\n・貯蓄¥1,500万 + NEDO補助金申請\n・最初の有料顧客1社を獲得\n・年¥300-500万契約が目標"),
        ("2032",      "#FF9800", "Seed調達\n・有料顧客3-5社 + ARR ¥1,000万\n・エンジェル or Seed VC から¥3,000万\n・エンジニア1名採用"),
        ("2033-2035", "#FF5722", "Series A\n・ARR ¥2億達成\n・¥15,000万調達\n・チーム15名\n・DEEPCORE/NVIDIA連携"),
        ("2037-2040", "#FFD700", "EXIT or IPO\n・ARR ¥20億\n・創業者取り分 ¥50-200億\n・「資本側」へ完全移行"),
    ]

    for i, (period, color, content) in enumerate(tl):
        y = 0.90 - i * 0.18
        ax_timeline.text(0.02, y, period, transform=ax_timeline.transAxes,
                         color=color, fontsize=8.5, fontweight="bold")
        ax_timeline.text(0.22, y, content, transform=ax_timeline.transAxes,
                         color=WHITE, fontsize=7.5, va="top",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="#151515",
                                   edgecolor=color, alpha=0.75))
        if i < len(tl)-1:
            ax_timeline.annotate("", xy=(0.10, y-0.10), xytext=(0.10, y-0.01),
                                 xycoords="axes fraction", textcoords="axes fraction",
                                 arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))

    out = os.path.join(OUT_DIR, "startup_finance_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
