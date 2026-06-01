"""
半導体装置 × SiC 市場分析
===========================
4 セグメントを定量的に分析:
  1. SiC パワーデバイス市場  — EV/産業/エネルギー需要 × 主要プレイヤー
  2. 後工程装置市場 (OSAT)   — K&S/Besi/Shinkawa + ASE/Amkor
  3. TEL 洗浄装置市場        — CELLESTA vs Screen/SEMES
  4. 装置バリューチェーン     — ASML/TEL/Disco/Advantest/K&S の TAM × SiC 依存度

データ出典: Yole Group / SEMI / 各社 IR / アナリストコンセンサス (公開情報)
数値は概算。実際の投資判断には最新 IR を参照のこと。

実行:
    python scripts/market_analysis.py
    python scripts/market_analysis.py --segment sic
    python scripts/market_analysis.py --segment osat
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

YEARS = np.arange(2022, 2031)

# ════════════════════════════════════════════════════════════════════════════
# 1. SiC パワーデバイス市場データ
# ════════════════════════════════════════════════════════════════════════════

# 市場規模 [億ドル] — Yole Développement 2024 レポート準拠
# 2024 は EV 需要鈍化で一部修正、2025 以降は ADAS/産業用途が牽引
SIC_MARKET = {
    "total_B_USD": np.array([2.0, 2.5, 3.0, 4.2, 5.8, 7.5, 9.2, 11.0, 13.0]),
    "ev":          np.array([1.1, 1.4, 1.7, 2.4, 3.3, 4.3, 5.3, 6.3,  7.5]),   # EV/HEV
    "industrial":  np.array([0.4, 0.5, 0.6, 0.8, 1.1, 1.5, 1.8, 2.2,  2.6]),   # 産業・太陽光
    "telecom":     np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 1.0, 1.2,  1.4]),   # 5G/電源
    "other":       np.array([0.3, 0.3, 0.3, 0.4, 0.7, 0.9, 1.1, 1.3,  1.5]),   # 航空・医療
}

# プレイヤー別市場シェア (2023 → 2030 推移)
SIC_PLAYERS = {
    "ST Micro":    {"share_2023": 30, "share_2030": 25, "color": "#2166ac"},
    "Wolfspeed":   {"share_2023": 18, "share_2030": 17, "color": "#d62728"},
    "ローム":      {"share_2023": 14, "share_2030": 12, "color": "#ff7f0e"},
    "ON Semi":     {"share_2023": 10, "share_2030": 12, "color": "#2ca02c"},
    "Infineon":    {"share_2023":  8, "share_2030": 10, "color": "#9467bd"},
    "三菱電機":    {"share_2023":  6, "share_2030":  7, "color": "#8c564b"},
    "BYD/中国勢":  {"share_2023":  4, "share_2030": 10, "color": "#e377c2"},
    "Others":      {"share_2023": 10, "share_2030":  7, "color": "#cccccc"},
}

# ════════════════════════════════════════════════════════════════════════════
# 2. 後工程装置・OSAT 市場
# ════════════════════════════════════════════════════════════════════════════

# ワイヤーボンディング装置市場 [億ドル]
WIRE_BOND_MARKET = {
    "total":     np.array([1.1, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.5, 2.8]),
    "players": {
        "K&S":      {"share": 54, "rev_2023_B": 0.76, "color": "#2166ac",
                     "ticker": "KLIC", "note": "Cu wire 転換需要で堅調"},
        "Besi":     {"share": 24, "rev_2023_B": 0.34, "color": "#d62728",
                     "ticker": "BESI.AS", "note": "Hybrid bonding で成長加速"},
        "Shinkawa": {"share": 12, "rev_2023_B": 0.17, "color": "#ff7f0e",
                     "ticker": "6274.T", "note": "SiC/GaN 対応強み"},
        "Others":   {"share": 10, "rev_2023_B": 0.14, "color": "#cccccc",
                     "ticker": "—",      "note": ""},
    }
}

# OSAT（外部委託封止・試験）市場 [億ドル]
OSAT_MARKET = {
    "total_2023_B": 35.0,
    "cagr_pct": 6.0,
    "players": {
        "ASE Group":    {"share": 25, "color": "#2166ac"},
        "Amkor":        {"share": 18, "color": "#d62728"},
        "JCET":         {"share": 12, "color": "#ff7f0e"},
        "Powertech":    {"share":  8, "color": "#2ca02c"},
        "UTAC":         {"share":  5, "color": "#9467bd"},
        "Others":       {"share": 32, "color": "#cccccc"},
    }
}

# ════════════════════════════════════════════════════════════════════════════
# 3. 洗浄装置市場
# ════════════════════════════════════════════════════════════════════════════

CLEANING_MARKET = {
    "total":  np.array([2.8, 3.2, 3.5, 3.9, 4.4, 5.0, 5.6, 6.2, 6.9]),  # 億ドル
    "players": {
        "TEL (CELLESTA)":     {"share_2023": 35, "share_2030": 37, "color": "#d62728"},
        "Screen Holdings":    {"share_2023": 25, "share_2030": 24, "color": "#2166ac"},
        "SEMES (Samsung)":    {"share_2023": 15, "share_2030": 15, "color": "#ff7f0e"},
        "Lam Research":       {"share_2023": 10, "share_2030":  9, "color": "#2ca02c"},
        "ACM Research":       {"share_2023":  5, "share_2030":  7, "color": "#9467bd"},
        "Others":             {"share_2023": 10, "share_2030":  8, "color": "#cccccc"},
    },
    # SiC 特需: Si 比 2.5× の装置単価 (化学的安定性 → 薬液コスト高)
    "sic_premium_x": 2.5,
}

# ════════════════════════════════════════════════════════════════════════════
# 4. 装置バリューチェーン サマリー
# ════════════════════════════════════════════════════════════════════════════

EQUIPMENT_CHAIN = {
    "ASML":       {"tam_B": 22, "rev_2023_B": 28.0, "sic_exposure_pct":  3,
                   "cagr_5y": 15, "color": "#2166ac",
                   "segment": "Lithography", "ticker": "ASML"},
    "TEL":        {"tam_B": 18, "rev_2023_B": 17.5, "sic_exposure_pct": 18,
                   "cagr_5y": 12, "color": "#d62728",
                   "segment": "Deposition/Cleaning", "ticker": "8035.T"},
    "Disco":      {"tam_B":  3, "rev_2023_B":  2.9, "sic_exposure_pct": 45,
                   "cagr_5y": 18, "color": "#ff7f0e",
                   "segment": "Dicing/Grinding", "ticker": "6146.T"},
    "Advantest":  {"tam_B":  6, "rev_2023_B":  4.2, "sic_exposure_pct": 12,
                   "cagr_5y": 14, "color": "#2ca02c",
                   "segment": "ATE", "ticker": "6857.T"},
    "K&S":        {"tam_B":  2, "rev_2023_B":  1.4, "sic_exposure_pct": 22,
                   "cagr_5y": 10, "color": "#9467bd",
                   "segment": "Wire Bonding", "ticker": "KLIC"},
    "Screen":     {"tam_B":  4, "rev_2023_B":  3.1, "sic_exposure_pct": 14,
                   "cagr_5y": 11, "color": "#8c564b",
                   "segment": "Cleaning/Coating", "ticker": "7735.T"},
}


# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))

    # ── Panel 1: SiC 市場規模 × 用途別積み上げ ───────────────────────────
    ax = axes[0, 0]
    colors = ["#2196f3", "#4caf50", "#ff9800", "#9c27b0"]
    labels = ["EV/HEV", "Industrial/Solar", "5G/Telecom", "Aerospace/Medical"]
    bottom = np.zeros(len(YEARS))
    for key, col, lbl in zip(["ev", "industrial", "telecom", "other"], colors, labels):
        vals = SIC_MARKET[key]
        ax.bar(YEARS, vals, bottom=bottom, color=col, label=lbl, alpha=0.88, width=0.7)
        bottom += vals
    ax.plot(YEARS, SIC_MARKET["total_B_USD"], "ko-", lw=2, ms=5, label="Total")
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Market size [B USD]", fontsize=10)
    ax.set_title("SiC Power Device Market\n(Yole 2024 consensus)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.2, axis="y")
    cagr = (SIC_MARKET["total_B_USD"][-1] / SIC_MARKET["total_B_USD"][0]) ** (1/8) - 1
    ax.text(2025.5, 11.5, f"CAGR\n{cagr*100:.0f}%", fontsize=11,
            fontweight="bold", color="#d62728", ha="center")

    # ── Panel 2: SiC プレイヤー シェア変化 (2023 vs 2030) ─────────────────
    ax2 = axes[0, 1]
    players = list(SIC_PLAYERS.keys())
    shares_23 = [SIC_PLAYERS[p]["share_2023"] for p in players]
    shares_30 = [SIC_PLAYERS[p]["share_2030"] for p in players]
    colors_p  = [SIC_PLAYERS[p]["color"] for p in players]
    x = np.arange(len(players))
    w = 0.38
    bars23 = ax2.bar(x - w/2, shares_23, w, color=colors_p, alpha=0.50, edgecolor="k", lw=0.5, label="2023")
    bars30 = ax2.bar(x + w/2, shares_30, w, color=colors_p, alpha=0.95, edgecolor="k", lw=0.5, label="2030E")
    # 変化矢印
    for i, (s23, s30) in enumerate(zip(shares_23, shares_30)):
        dy = s30 - s23
        if abs(dy) >= 1:
            col = "#2ca02c" if dy > 0 else "#d62728"
            ax2.annotate(f"{dy:+d}", xy=(x[i], max(s23, s30) + 0.3),
                         ha="center", fontsize=8, color=col, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(players, fontsize=8, rotation=25)
    ax2.set_ylabel("Market share [%]", fontsize=10)
    ax2.set_title("SiC Player Share: 2023 vs 2030\n(薄=2023, 濃=2030)", fontsize=10)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.2, axis="y")

    # ── Panel 3: 後工程装置 + OSAT ────────────────────────────────────────
    ax3 = axes[0, 2]
    # ワイヤーボンド装置 シェア (ドーナツ)
    wp = WIRE_BOND_MARKET["players"]
    wedge_labels = [f"{k}\n{v['share']}%" for k, v in wp.items()]
    wedge_sizes  = [v["share"] for v in wp.values()]
    wedge_colors = [v["color"] for v in wp.values()]
    wedges, _ = ax3.pie(wedge_sizes, labels=None, colors=wedge_colors,
                        startangle=90, wedgeprops={"width": 0.55, "edgecolor": "white"})
    ax3.legend(wedges, wedge_labels, loc="center left",
               bbox_to_anchor=(0.85, 0.5), fontsize=8)
    # 中央テキスト
    ax3.text(0, 0, f"Wire Bond\nEquip.\n$1.5B\n(2023)", ha="center", va="center",
             fontsize=9, fontweight="bold")
    ax3.set_title("Wire Bond Equipment Market\nK&S 54% dominance", fontsize=10)

    # ── Panel 4: 洗浄装置 市場規模 + TEL シェア ───────────────────────────
    ax4 = axes[1, 0]
    cm = CLEANING_MARKET
    tel_rev = [cm["total"][i] * cm["players"]["TEL (CELLESTA)"]["share_2023"] / 100
               for i in range(len(YEARS))]
    screen_rev = [cm["total"][i] * cm["players"]["Screen Holdings"]["share_2023"] / 100
                  for i in range(len(YEARS))]
    sic_add = [cm["total"][i] * 0.05  # SiC 洗浄の追加需要 (全体の 5% → 12%)
               * (1 + 0.15) ** (YEARS[i] - 2022) for i in range(len(YEARS))]

    ax4.stackplot(YEARS,
                  [np.array(tel_rev), np.array(screen_rev),
                   cm["total"] * 0.15, cm["total"] * 0.10,
                   cm["total"] * 0.08],
                  labels=["TEL (CELLESTA)", "Screen", "SEMES", "Lam", "Others"],
                  colors=["#d62728","#2166ac","#ff7f0e","#2ca02c","#cccccc"],
                  alpha=0.85)
    ax4.plot(YEARS, cm["total"], "ko--", lw=1.5, ms=4, label="Total market")
    ax4.fill_between(YEARS, 0, sic_add, alpha=0.3, color="purple",
                     label="SiC 追加需要 (est)")
    ax4.set_xlabel("Year", fontsize=10)
    ax4.set_ylabel("Market size [B USD]", fontsize=10)
    ax4.set_title("Wafer Cleaning Equipment Market\nTEL CELLESTA vs competitors", fontsize=10)
    ax4.legend(fontsize=7, loc="upper left"); ax4.grid(alpha=0.2, axis="y")
    ax4.text(2027, 1.8, f"SiC 洗浄\n単価 {cm['sic_premium_x']}× premium", fontsize=8,
             color="purple", ha="center",
             bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "purple"})

    # ── Panel 5: 装置バリューチェーン — SiC 依存度 vs 売上成長 ───────────
    ax5 = axes[1, 1]
    for name, d in EQUIPMENT_CHAIN.items():
        size = d["rev_2023_B"] * 15  # バブルサイズ ∝ 売上
        sc = ax5.scatter(d["sic_exposure_pct"], d["cagr_5y"],
                         s=size, color=d["color"], alpha=0.80,
                         edgecolors="k", linewidths=0.8, zorder=5)
        offset = {"ASML": (-4, 0.5), "TEL": (1.5, 0.5), "Disco": (1.5, -1.2),
                  "Advantest": (1.5, 0.5), "K&S": (1.5, -1.2), "Screen": (-6, -1.5)}
        dx, dy = offset.get(name, (1.5, 0.5))
        ax5.annotate(f"{name}\n${d['rev_2023_B']:.1f}B",
                     xy=(d["sic_exposure_pct"], d["cagr_5y"]),
                     xytext=(d["sic_exposure_pct"] + dx, d["cagr_5y"] + dy),
                     fontsize=8, ha="center",
                     arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "gray"})
    ax5.axvline(20, color="purple", ls="--", lw=1, alpha=0.5, label="SiC 依存度 20%")
    ax5.set_xlabel("SiC exposure [% of revenue]", fontsize=10)
    ax5.set_ylabel("Revenue CAGR 5Y [%]", fontsize=10)
    ax5.set_title("Equipment Value Chain\nSiC dependency vs growth (bubble=2023 revenue)", fontsize=10)
    ax5.grid(alpha=0.2)
    # 象限ラベル
    ax5.text(35, 8.5, "High SiC\nHigh Growth", fontsize=8, color="#d62728",
             alpha=0.6, ha="center")
    ax5.text(5, 8.5, "Low SiC\nHigh Growth", fontsize=8, color="#2166ac",
             alpha=0.6, ha="center")

    # ── Panel 6: 投資サマリー表 ───────────────────────────────────────────
    ax6 = axes[1, 2]
    ax6.axis("off")
    rows = []
    for name, d in EQUIPMENT_CHAIN.items():
        sic_rev = d["rev_2023_B"] * d["sic_exposure_pct"] / 100
        rows.append([
            name,
            d["ticker"],
            d["segment"],
            f"${d['rev_2023_B']:.1f}B",
            f"{d['sic_exposure_pct']}%",
            f"${sic_rev:.2f}B",
            f"{d['cagr_5y']}%",
        ])
    col_labels = ["Company", "Ticker", "Segment", "Rev'23", "SiC%", "SiC Rev", "CAGR"]
    tbl = ax6.table(cellText=rows, colLabels=col_labels,
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    tbl.scale(1.35, 2.1)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#222"); cell.set_text_props(color="w")
        elif r % 2 == 0:
            cell.set_facecolor("#f5f5f5")
        # SiC 依存度 > 20% を強調
        if c == 4 and r > 0:
            pct = int(rows[r-1][4].replace("%", ""))
            if pct >= 20:
                cell.set_facecolor("#ffe0b2")
    ax6.set_title("Semiconductor Equipment Value Chain\nSiC Revenue Exposure (2023)", fontsize=10)

    fig.suptitle("半導体装置 × SiC 市場分析 2023–2030\n"
                 "SiC デバイス市場 CAGR ~25% | Disco が最高 SiC 依存度 45%",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(out_dir, "market_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    _print_summary()
    print(f"\n[OK] market analysis -> {out}")


def _print_summary():
    print("\n" + "="*64)
    print("  半導体装置 × SiC 市場分析 サマリー")
    print("="*64)

    # SiC 市場
    mkt_23 = SIC_MARKET["total_B_USD"][1]
    mkt_30 = SIC_MARKET["total_B_USD"][-1]
    cagr   = (mkt_30 / mkt_23) ** (1/7) - 1
    print(f"\n【SiC パワーデバイス市場】")
    print(f"  2023: ${mkt_23:.1f}B → 2030E: ${mkt_30:.1f}B  (CAGR {cagr*100:.0f}%)")
    print(f"  EV/HEV 主導 ({SIC_MARKET['ev'][-1]/SIC_MARKET['total_B_USD'][-1]*100:.0f}% of 2030)")
    print(f"  BYD/中国勢が 4% → 10% に台頭 (最大リスク)")

    # 装置 SiC 依存度ランキング
    print(f"\n【装置メーカー SiC 依存度ランキング】")
    ranked = sorted(EQUIPMENT_CHAIN.items(),
                    key=lambda x: x[1]["sic_exposure_pct"], reverse=True)
    for name, d in ranked:
        sic_rev = d["rev_2023_B"] * d["sic_exposure_pct"] / 100
        print(f"  {name:<12} SiC {d['sic_exposure_pct']:>2}%  "
              f"(${sic_rev:.2f}B)  CAGR {d['cagr_5y']}%")

    # 洗浄装置
    cm = CLEANING_MARKET
    print(f"\n【洗浄装置市場 (TEL CELLESTA)】")
    print(f"  全体: $3.5B (2023) → $6.9B (2030E)")
    print(f"  TEL シェア: {cm['players']['TEL (CELLESTA)']['share_2023']}% → "
          f"{cm['players']['TEL (CELLESTA)']['share_2030']}% (拡大傾向)")
    print(f"  SiC 単価プレミアム: {cm['sic_premium_x']}× (化学的安定性 → 薬液コスト高)")

    # OSAT
    print(f"\n【後工程 (OSAT) 市場】")
    print(f"  総市場: ${OSAT_MARKET['total_2023_B']:.0f}B  CAGR {OSAT_MARKET['cagr_pct']}%")
    for name, d in OSAT_MARKET["players"].items():
        print(f"  {name:<14} {d['share']:>2}%")

    print("\n【注目トレンド】")
    print("  - Au→Cu ワイヤー移行: K&S の Cu 対応装置需要 +15%/y")
    print("  - Hybrid bonding 普及: Besi の成長加速 (AI チップ積層)")
    print("  - SiC 洗浄の単価高さ: TEL/Screen の収益性向上ドライバー")
    print("  - 中国 SiC 内製化: Wolfspeed/ST のシェア侵食リスク 2026–")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", default="all",
                    choices=["all", "sic", "osat", "cleaning", "chain"])
    args = ap.parse_args()
    plot_all()


if __name__ == "__main__":
    main()
