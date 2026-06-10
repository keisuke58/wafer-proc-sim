"""
Disco 魔改造アイデア — 全10案の市場規模・実現性・インパクト可視化
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
BG, PANEL, GRID, WHITE, DIM = "#0d0d0d", "#0a0a0a", "#2a2a2a", "#f0f0f0", "#666666"

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
# 全10アイデア データ
# ════════════════════════════════════════════════════════════════════════════

IDEAS = {
    "diamond_cut": {
        "label":       "人工ダイヤモンド\n精密カット",
        "category":    "物理転用",
        "market_B$":   25,
        "feasibility": 9,     # 0-10 (技術的実現性)
        "time_years":  2,     # 実現までの年数
        "disruption":  8,     # 既存業界への破壊度
        "revenue_type":"装置販売",
        "color":       "#00E5FF",
        "summary":     "ブレードはダイヤ砥粒\n既に技術ある\n宝飾業界の精度は\nDiscoの1/100",
        "first_step":  "ジュエリーメーカーへ\nサンプル提供",
    },
    "joint_grinding": {
        "label":       "人工関節\n表面研削",
        "category":    "物理転用",
        "market_B$":   20,
        "feasibility": 8,
        "time_years":  4,
        "disruption":  6,
        "revenue_type":"装置+消耗品",
        "color":       "#FF6E40",
        "summary":     "股関節摺動面 =\nDiscoの研削砥石\nと同じ技術\nStrykerへOEM",
        "first_step":  "医療グレード認証\n(QMS/ISO13485)\n取得から",
    },
    "process_db": {
        "label":       "半導体プロセス\nデータベース",
        "category":    "データ",
        "market_B$":   50,
        "feasibility": 7,
        "time_years":  3,
        "disruption":  10,
        "revenue_type":"SaaS/データ",
        "color":       "#E040FB",
        "summary":     "世界30,000台の\nセンサーデータ\n今は全部捨てている\n世界最大のDB",
        "first_step":  "装置にIoTセンサー\n追加 → データ収集\nインフラ整備",
    },
    "consumable_saas": {
        "label":       "消耗品\nサブスク化",
        "category":    "ビジネスモデル",
        "market_B$":   15,
        "feasibility": 8,
        "time_years":  2,
        "disruption":  7,
        "revenue_type":"MRR",
        "color":       "#69F0AE",
        "summary":     "1000カット¥X\nの従量課金へ\nRolls-Royceの\n'Power by Hour'",
        "first_step":  "既存顧客5社で\nパイロット契約",
    },
    "rapidus_equity": {
        "label":       "Rapidus\n株主化",
        "category":    "事業構造",
        "market_B$":   100,
        "feasibility": 6,
        "time_years":  2,
        "disruption":  8,
        "revenue_type":"株式+装置",
        "color":       "#FF5252",
        "summary":     "装置を売るだけ\nでなく出資する\nRapidus成功なら\nDiscoも10倍",
        "first_step":  "Rapidus CFOと\n戦略的提携協議",
    },
    "sic_vertical": {
        "label":       "SiCウェーハ\n自社生産",
        "category":    "事業構造",
        "market_B$":   30,
        "feasibility": 5,
        "time_years":  7,
        "disruption":  9,
        "revenue_type":"素材+装置",
        "color":       "#FF9800",
        "summary":     "消耗品の原材料\nを川上統合\nWolfspeed的\nポジションへ",
        "first_step":  "SiCメーカーへの\nM&A/出資検討",
    },
    "laser_spinoff": {
        "label":       "レーザー部門\n分社化IPO",
        "category":    "事業構造",
        "market_B$":   40,
        "feasibility": 7,
        "time_years":  5,
        "disruption":  6,
        "revenue_type":"ストック",
        "color":       "#40C4FF",
        "summary":     "レーザーは装置+\nSaaS的収益構造\n分社化でSaaS\nバリュエーション獲得",
        "first_step":  "レーザー事業の\nP&L分離から",
    },
    "dna_cut": {
        "label":       "DNA\n物理的切断",
        "category":    "ぶっ飛び",
        "market_B$":   200,
        "feasibility": 3,
        "time_years":  15,
        "disruption":  10,
        "revenue_type":"ライセンス",
        "color":       "#B2FF59",
        "summary":     "CRISPRは化学的\nDiscoは物理的に\nnmスケール切断\nバイオ×半導体",
        "first_step":  "東大/慶應の\nバイオ研究室と\n共同研究",
    },
    "space_mining": {
        "label":       "宇宙資源\n採掘装置",
        "category":    "ぶっ飛び",
        "market_B$":   500,
        "feasibility": 3,
        "time_years":  20,
        "disruption":  10,
        "revenue_type":"装置販売",
        "color":       "#EA80FC",
        "summary":     "月面レゴリスの\n切断/採掘\nDiscoの真空対応\n装置がそのまま使える",
        "first_step":  "JAXAとの\n宇宙向け装置\n共同開発",
    },
}

CATEGORIES = {
    "物理転用":       "#00BCD4",
    "データ":         "#E040FB",
    "ビジネスモデル": "#69F0AE",
    "事業構造":       "#FF9800",
    "ぶっ飛び":       "#FF5252",
}

# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def p_matrix(ax):
    sa(ax)
    ax.set_title("魔改造マトリクス — 実現性 × 市場規模 × 破壊力", fontsize=11, pad=5)

    for key, idea in IDEAS.items():
        size = idea["market_B$"] * 3
        ax.scatter(idea["feasibility"], idea["disruption"],
                   s=size, c=idea["color"], alpha=0.82,
                   edgecolors=WHITE, lw=1.2, zorder=3)
        offset_x = 0.15
        offset_y = 0.25
        if key == "space_mining":   offset_y = -0.6
        if key == "dna_cut":        offset_x = -2.2
        if key == "rapidus_equity": offset_y = -0.6
        if key == "laser_spinoff":  offset_x = -2.5
        ax.annotate(idea["label"].replace("\n"," "),
                    (idea["feasibility"], idea["disruption"]),
                    xytext=(idea["feasibility"]+offset_x,
                            idea["disruption"]+offset_y),
                    color=idea["color"], fontsize=7.5, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="#444", lw=0.8))

    ax.axvline(5, color=DIM, lw=1, ls="--", alpha=0.5)
    ax.axhline(7, color=DIM, lw=1, ls="--", alpha=0.5)
    ax.text(0.5, 9.6, "高破壊力", color=DIM, fontsize=8)
    ax.text(8.5, 0.3, "高実現性", color=DIM, fontsize=8)
    ax.text(7.5, 9.5, "ゴールゾーン", color="#FFD700", fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="#1a1a1a", edgecolor="#FFD700"))
    ax.set_xlabel("実現性 (0=SF / 10=今すぐできる)")
    ax.set_ylabel("既存業界への破壊度 (0=ニッチ / 10=革命)")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 11)
    note = "円のサイズ = 市場規模 ($B)"
    ax.text(0.01, 0.01, note, transform=ax.transAxes, color=DIM, fontsize=8)


def p_timeline(ax):
    sa(ax)
    ax.set_title("実現タイムライン × 収益ポテンシャル", fontsize=10, pad=5)

    for key, idea in IDEAS.items():
        revenue = idea["market_B$"] * idea["feasibility"] / 10
        ax.scatter(idea["time_years"], revenue,
                   s=120, c=idea["color"], alpha=0.88,
                   edgecolors=WHITE, lw=1, zorder=3)
        ax.annotate(idea["label"].split("\n")[0],
                    (idea["time_years"], revenue),
                    xytext=(idea["time_years"]+0.2, revenue+1),
                    color=idea["color"], fontsize=7.5)

    ax.axvline(3, color="#4CAF50", lw=1.2, ls=":", alpha=0.7)
    ax.axvline(7, color="#FF9800", lw=1.2, ls=":", alpha=0.7)
    ax.text(0.3, 42, "短期\n(〜3年)", color="#4CAF50", fontsize=8)
    ax.text(4.2, 42, "中期\n(3〜7年)", color="#FF9800", fontsize=8)
    ax.text(8.5, 42, "長期\n(7年〜)", color="#FF5722", fontsize=8)
    ax.set_xlabel("実現までの年数")
    ax.set_ylabel("収益ポテンシャル (市場規模 × 実現性, $B換算)")
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 45)


def p_revenue_model(ax):
    sa(ax)
    ax.set_title("収益モデル別 — Disco の現状からの変化量", fontsize=10, pad=5)

    disco_current = {
        "装置販売":   393 * 0.68,
        "消耗品":     393 * 0.32,
        "SaaS/データ": 0,
        "MRR":        0,
        "株式":       0,
    }

    disco_madmod = {
        "装置販売":   393 * 0.68 + 80,   # ダイヤ+医療+食品+宇宙
        "消耗品":     393 * 0.32 + 30,
        "SaaS/データ": 120,               # プロセスDB
        "MRR":        90,                 # サブスク
        "株式":       200,                # Rapidus出資など
    }

    labels = list(disco_current.keys())
    x = np.arange(len(labels))
    w = 0.32
    c1 = ax.bar(x - w/2, list(disco_current.values()), w,
                color="#546e7a", alpha=0.85, label="現状 FY2024")
    c2 = ax.bar(x + w/2, list(disco_madmod.values()), w,
                color="#FF9800", alpha=0.85, label="魔改造後 (5〜10年)")

    for bar in c2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x()+bar.get_width()/2, h+3,
                    f"¥{h:.0f}B", ha="center", color=WHITE, fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=WHITE, fontsize=9)
    ax.set_ylabel("売上 (億円/年)")
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE)
    total_cur = sum(disco_current.values())
    total_mad = sum(disco_madmod.values())
    ax.text(0.98, 0.95, f"現状合計: ¥{total_cur:.0f}B\n魔改造後: ¥{total_mad:.0f}B (+{(total_mad/total_cur-1)*100:.0f}%)",
            transform=ax.transAxes, ha="right", va="top",
            color="#FFD700", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="#111", edgecolor="#FFD700"))


def p_cards(ax):
    sa(ax)
    ax.set_facecolor("#060606")
    ax.axis("off")
    ax.set_title("全10案 カード一覧", fontsize=11, pad=5)

    keys = list(IDEAS.keys())
    cols, rows = 5, 2
    for idx, key in enumerate(keys):
        idea = IDEAS[key]
        col  = idx % cols
        row  = idx // cols
        x    = 0.01 + col * 0.196
        y    = 0.88 - row * 0.47

        # カテゴリ色バー
        cat_color = CATEGORIES[idea["category"]]
        ax.add_patch(FancyBboxPatch(
            (x, y-0.38), 0.183, 0.40,
            boxstyle="round,pad=0.005",
            facecolor="#111111", edgecolor=idea["color"],
            linewidth=1.5, transform=ax.transAxes, zorder=2
        ))
        ax.text(x+0.091, y,      idea["label"],
                transform=ax.transAxes, ha="center", va="top",
                color=idea["color"], fontsize=8, fontweight="bold")
        ax.text(x+0.091, y-0.07, idea["summary"],
                transform=ax.transAxes, ha="center", va="top",
                color=WHITE, fontsize=6.5)

        metrics = (f"市場 ${idea['market_B$']}B  |  "
                   f"実現{idea['feasibility']}/10  |  "
                   f"{idea['time_years']}年")
        ax.text(x+0.091, y-0.24, metrics,
                transform=ax.transAxes, ha="center",
                color=DIM, fontsize=6.5)
        ax.text(x+0.091, y-0.30, f"► {idea['first_step']}",
                transform=ax.transAxes, ha="center", va="top",
                color=cat_color, fontsize=6.5)
        ax.text(x+0.091, y-0.375, idea["category"],
                transform=ax.transAxes, ha="center",
                color=cat_color, fontsize=6.5, fontstyle="italic")


def p_priority(ax):
    sa(ax)
    ax.set_title("優先順位 — 「今すぐ着手すべき」ランキング", fontsize=10, pad=5)

    # スコア = 実現性 × 市場規模 / (年数^0.8)
    scored = sorted(IDEAS.items(),
                    key=lambda x: x[1]["feasibility"] * x[1]["market_B$"] / (x[1]["time_years"]**0.8),
                    reverse=True)
    labels = [v["label"].replace("\n"," ") for _, v in scored]
    scores = [v["feasibility"] * v["market_B$"] / (v["time_years"]**0.8) for _, v in scored]
    colors = [v["color"] for _, v in scored]
    y = np.arange(len(scored))

    bars = ax.barh(y, scores, color=colors, alpha=0.88, height=0.65)
    for bar, (_, idea), sc in zip(bars, scored, scores):
        ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                f"{sc:.0f}pt  ({idea['time_years']}年・${idea['market_B$']}B)",
                color=WHITE, va="center", fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=WHITE, fontsize=8)
    ax.set_xlabel("優先度スコア (実現性 × 市場規模 / 年数)")
    ax.set_xlim(0, max(scores)*1.5)
    ax.invert_yaxis()

    ax.axhline(2.5, color="#FFD700", lw=1.5, ls="--", alpha=0.7)
    ax.text(max(scores)*0.7, 2.7, "この上は今すぐ着手", color="#FFD700", fontsize=8)


# ── メイン ────────────────────────────────────────────────────────────────────

def main():
    fig = plt.figure(figsize=(24, 30), facecolor=BG)
    fig.suptitle(
        "Disco 魔改造アイデア全10案\n「精密切断」という核を世界に解き放つ",
        fontsize=18, color=WHITE, fontweight="bold", y=0.987
    )
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        hspace=0.44, wspace=0.32,
        left=0.05, right=0.98, top=0.960, bottom=0.04,
    )

    p_matrix(      fig.add_subplot(gs[0, 0]))
    p_timeline(    fig.add_subplot(gs[0, 1]))
    p_cards(       fig.add_subplot(gs[1, :]))
    p_revenue_model(fig.add_subplot(gs[2, 0]))
    p_priority(    fig.add_subplot(gs[2, 1]))

    out = os.path.join(OUT_DIR, "disco_madmod.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
