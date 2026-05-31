"""
Disco Corporation (6146) — DCF Valuation Model
================================================
投資研究用。数値は公開情報 + アナリスト推定に基づく概算。
実際の投資判断には最新の決算資料・IR を参照のこと。

前提:
  - 会計年度: 4月始まり（FY2024 = 2023/4–2024/3）
  - 通貨: 億円
  - 基準年: FY2024（実績ベース）

実行:
  python scripts/disco_dcf.py
  python scripts/disco_dcf.py --bear   # 悲観シナリオ
  python scripts/disco_dcf.py --bull   # 楽観シナリオ
"""

import argparse
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ── 実績データ（億円）────────────────────────────────────────────────────────
# 出所: Disco IR、Bloomberg コンセンサス推定（一部）
ACTUALS = {
    # FY: [Revenue, EBIT, CAPEX, D&A]
    "FY2021": [1_830, 460, 120, 90],
    "FY2022": [2_270, 730, 170, 100],
    "FY2023": [2_200, 650, 190, 110],   # 在庫調整サイクルの谷
    "FY2024": [2_900, 980, 210, 120],   # HBM 第1波
}

# ── シナリオ共通パラメータ ────────────────────────────────────────────────
WACC        = 0.090   # 9.0%（Beta≈1.2, Rf=0.8%, ERP=7%@JPX, D/E低い）
TAX_RATE    = 0.28
SHARES_MM   = 37.5    # 百万株（自己株除き）
NET_CASH    = 1_500   # 億円（FY2024末 現預金 - 有利子負債）

# ── セグメント成長ドライバー ──────────────────────────────────────────────
# 売上構成（FY2024推定）
#   ダイシングソー装置   35%  → HBM/SiC 牽引
#   研削機装置           25%  → HBM 薄ダイ研削ボトルネック
#   消耗品（ブレード等） 30%  → リカーリング、景気連動低
#   その他               10%
SEGMENTS_FY2024 = {
    "dicing_saw":  2_900 * 0.35,
    "grinder":     2_900 * 0.25,
    "consumables": 2_900 * 0.30,
    "other":       2_900 * 0.10,
}

# ── シナリオ定義 ─────────────────────────────────────────────────────────
SCENARIOS = {
    "base": {
        "label": "Base（中心シナリオ）",
        "color": "#2166ac",
        # セグメント別売上成長率 FY2025–2030
        "seg_growth": {
            #              25    26    27    28    29    30
            "dicing_saw":  [.22, .18, .05, .12, .10, .08],
            "grinder":     [.30, .25, .02, .15, .12, .09],
            "consumables": [.12, .10, .08, .10, .09, .08],
            "other":       [.08, .08, .06, .06, .06, .05],
        },
        "op_margin":  [.355, .370, .330, .355, .360, .365],  # EBIT/Revenue
        "capex_pct":  [.075, .080, .075, .070, .068, .065],  # CAPEX/Revenue
        "da_pct":     [.043, .044, .044, .043, .042, .042],  # D&A/Revenue
        "nwc_pct":    [.020, .020, .018, .018, .017, .017],  # ΔNWC/Rev change
        "terminal_g": 0.025,
    },
    "bull": {
        "label": "Bull（楽観：HBM4 + SiC 超加速）",
        "color": "#2ca02c",
        "seg_growth": {
            "dicing_saw":  [.28, .25, .12, .15, .13, .11],
            "grinder":     [.38, .32, .10, .18, .15, .12],
            "consumables": [.15, .13, .11, .12, .11, .10],
            "other":       [.10, .10, .08, .08, .07, .06],
        },
        "op_margin":  [.370, .385, .360, .375, .380, .385],
        "capex_pct":  [.080, .085, .078, .073, .070, .068],
        "da_pct":     [.045, .046, .046, .045, .044, .043],
        "nwc_pct":    [.022, .022, .020, .020, .018, .018],
        "terminal_g": 0.030,
    },
    "bear": {
        "label": "Bear（悲観：AI バブル崩壊 + 中国制裁）",
        "color": "#d62728",
        "seg_growth": {
            "dicing_saw":  [.10, -.05, -.15, .08, .10, .08],
            "grinder":     [.15, -.08, -.20, .10, .12, .09],
            "consumables": [.08, .03, -.02, .06, .07, .07],
            "other":       [.05, .00, -.05, .03, .04, .04],
        },
        "op_margin":  [.320, .280, .240, .300, .320, .335],
        "capex_pct":  [.070, .065, .058, .062, .064, .064],
        "da_pct":     [.042, .042, .040, .040, .040, .041],
        "nwc_pct":    [.018, .015, .010, .015, .016, .016],
        "terminal_g": 0.015,
    },
}

YEARS = ["FY2025", "FY2026", "FY2027", "FY2028", "FY2029", "FY2030"]


# ── FCF 計算 ────────────────────────────────────────────────────────────────

def project_fcf(sc: dict) -> dict:
    """シナリオパラメータ → 売上・EBIT・FCF の時系列を返す。"""
    revenues    = []
    ebits       = []
    nopats      = []
    capexes     = []
    das         = []
    delta_nwcs  = []
    fcfs        = []

    seg = {k: v for k, v in SEGMENTS_FY2024.items()}
    rev_prev = sum(seg.values())

    for i, yr in enumerate(YEARS):
        # セグメント別売上
        for k in seg:
            seg[k] *= (1 + sc["seg_growth"][k][i])
        rev = sum(seg.values())

        ebit     = rev * sc["op_margin"][i]
        nopat    = ebit * (1 - TAX_RATE)
        capex    = rev * sc["capex_pct"][i]
        da       = rev * sc["da_pct"][i]
        d_nwc    = (rev - rev_prev) * sc["nwc_pct"][i]
        fcf      = nopat + da - capex - d_nwc

        revenues.append(rev)
        ebits.append(ebit)
        nopats.append(nopat)
        capexes.append(capex)
        das.append(da)
        delta_nwcs.append(d_nwc)
        fcfs.append(fcf)
        rev_prev = rev

    return {
        "revenues": revenues,
        "ebits":    ebits,
        "nopats":   nopats,
        "capexes":  capexes,
        "das":      das,
        "delta_nwcs": delta_nwcs,
        "fcfs":     fcfs,
    }


def dcf_valuation(fcfs: list, terminal_g: float, wacc: float = WACC) -> dict:
    """FCF リスト → EV / 株価算出。"""
    # PV of explicit period
    pv_fcfs = [fcf / (1 + wacc) ** (i + 1) for i, fcf in enumerate(fcfs)]
    sum_pv  = sum(pv_fcfs)

    # Terminal value (Gordon Growth)
    tv      = fcfs[-1] * (1 + terminal_g) / (wacc - terminal_g)
    pv_tv   = tv / (1 + wacc) ** len(fcfs)

    ev          = sum_pv + pv_tv + NET_CASH
    price_per_share = ev / SHARES_MM * 100  # 億円 → 円（1億 = 10^8 / 10^6 = 100円/株）

    return {
        "pv_fcfs":   pv_fcfs,
        "sum_pv":    sum_pv,
        "tv":        tv,
        "pv_tv":     pv_tv,
        "ev":        ev,
        "price":     price_per_share,
        "tv_pct":    pv_tv / ev * 100,
    }


# ── 感度分析 ────────────────────────────────────────────────────────────────

def sensitivity(fcfs: list, terminal_g_center: float) -> np.ndarray:
    """WACC × Terminal g のマトリクス → 株価テーブル。"""
    waccs    = np.arange(0.07, 0.122, 0.010)
    tgs      = np.arange(0.010, 0.041, 0.010)
    mat      = np.zeros((len(waccs), len(tgs)))
    for i, w in enumerate(waccs):
        for j, g in enumerate(tgs):
            if w <= g:
                mat[i, j] = np.nan
            else:
                mat[i, j] = dcf_valuation(fcfs, g, w)["price"]
    return mat, waccs, tgs


# ── プロット ─────────────────────────────────────────────────────────────────

def plot_all(results: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Disco Corporation (6146) — DCF Valuation Model\n"
                 "※ 投資研究用概算。実際の投資判断には IR・最新決算を参照",
                 fontsize=13, y=0.98)

    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.38)

    # ── Panel 1: 売上予測 ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    act_yrs  = list(ACTUALS.keys())
    act_revs = [v[0] for v in ACTUALS.values()]
    ax1.bar(act_yrs, act_revs, color="#aec7e8", label="実績", zorder=3)
    for sc_name, res in results.items():
        sc = SCENARIOS[sc_name]
        ax1.plot(YEARS, res["proj"]["revenues"],
                 color=sc["color"], lw=2.5, marker="o", ms=5, label=sc["label"])
    ax1.axvline(3.5, color="gray", ls=":", lw=1.0)
    ax1.text(3.6, max(act_revs)*0.95, "予測→", fontsize=9, color="gray")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"¥{x/100:.0f}B"))
    ax1.set_title("売上高予測（億円）", fontsize=11)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.25, zorder=0)

    # ── Panel 2: EBIT マージン ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    for sc_name, res in results.items():
        sc = SCENARIOS[sc_name]
        margins = [e / r * 100 for e, r in
                   zip(res["proj"]["ebits"], res["proj"]["revenues"])]
        ax2.plot(YEARS, margins, color=sc["color"], lw=2, marker="o", ms=4)
    act_margins = [v[1] / v[0] * 100 for v in ACTUALS.values()]
    ax2.plot(act_yrs, act_margins, "k--o", ms=4, lw=1.5, label="実績")
    ax2.set_title("EBIT マージン [%]", fontsize=11)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)

    # ── Panel 3: FCF 棒グラフ ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    x    = np.arange(len(YEARS))
    w    = 0.28
    offs = [-w, 0, w]
    for idx, (sc_name, res) in enumerate(results.items()):
        sc  = SCENARIOS[sc_name]
        ax3.bar(x + offs[idx], res["proj"]["fcfs"], width=w,
                color=sc["color"], alpha=0.85, label=sc["label"], zorder=3)
    ax3.set_xticks(x); ax3.set_xticklabels(YEARS, fontsize=9)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"¥{x:.0f}B"))
    ax3.set_title("フリーキャッシュフロー予測（億円）", fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.25, zorder=0)

    # ── Panel 4: DCF 株価サマリー ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    prices = {sc_name: res["val"]["price"] for sc_name, res in results.items()}
    rows = [["シナリオ", "EV (兆円)", "株価 (円)", "TV比率"]]
    for sc_name, res in results.items():
        v = res["val"]
        rows.append([
            SCENARIOS[sc_name]["label"].split("（")[0],
            f"¥{v['ev']/10000:.1f}T",
            f"¥{v['price']:,.0f}",
            f"{v['tv_pct']:.0f}%",
        ])
    tbl = ax4.table(cellText=rows[1:], colLabels=rows[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2166ac"); cell.set_text_props(color="white")
        elif r == 1:
            cell.set_facecolor("#d4e8f7")
        elif r == 2:
            cell.set_facecolor("#d4f7d4")
        elif r == 3:
            cell.set_facecolor("#f7d4d4")
    ax4.set_title("DCF 株価試算\n(Base WACC=9.0%)", fontsize=10)

    # ── Panel 5: 感度分析ヒートマップ（Base） ────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    base_fcfs = results["base"]["proj"]["fcfs"]
    mat, waccs, tgs = sensitivity(base_fcfs, SCENARIOS["base"]["terminal_g"])
    im = ax5.imshow(mat, aspect="auto", cmap="RdYlGn",
                    vmin=mat[~np.isnan(mat)].min(),
                    vmax=mat[~np.isnan(mat)].max())
    plt.colorbar(im, ax=ax5, label="株価試算 [円]")
    ax5.set_xticks(range(len(tgs)))
    ax5.set_xticklabels([f"{g*100:.1f}%" for g in tgs], fontsize=9)
    ax5.set_yticks(range(len(waccs)))
    ax5.set_yticklabels([f"{w*100:.1f}%" for w in waccs], fontsize=9)
    ax5.set_xlabel("Terminal Growth Rate", fontsize=10)
    ax5.set_ylabel("WACC", fontsize=10)
    ax5.set_title("感度分析: WACC × Terminal Growth → 株価（Baseシナリオ FCF）", fontsize=10)
    for i in range(len(waccs)):
        for j in range(len(tgs)):
            if not np.isnan(mat[i, j]):
                ax5.text(j, i, f"{mat[i,j]:,.0f}", ha="center", va="center",
                         fontsize=8, color="black")

    # ── Panel 6: セグメント別売上構成（Base） ───────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    seg_labels  = ["Dicing Saw", "Grinder", "Consumables", "Other"]
    seg_colors  = ["#2166ac", "#d62728", "#2ca02c", "#ff7f0e"]
    base_rev    = results["base"]["proj"]["revenues"]
    seg_vals_30 = []
    seg = {k: v for k, v in SEGMENTS_FY2024.items()}
    sc  = SCENARIOS["base"]
    for i in range(len(YEARS)):
        for k in seg:
            seg[k] *= (1 + sc["seg_growth"][k][i])
    seg_vals_30 = list(seg.values())
    wedges, texts, autotexts = ax6.pie(
        seg_vals_30, labels=seg_labels, colors=seg_colors,
        autopct="%1.0f%%", startangle=90, pctdistance=0.75)
    for t in autotexts:
        t.set_fontsize(9)
    ax6.set_title("セグメント構成 FY2030（Base）", fontsize=10)

    plt.savefig(os.path.join(out_dir, "disco_dcf.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] DCF チャート → {out_dir}/disco_dcf.png")


# ── テキストサマリー ──────────────────────────────────────────────────────

def print_summary(results: dict):
    print("\n" + "=" * 65)
    print("  Disco Corporation (6146) — DCF 試算サマリー")
    print("  WACC=9.0%  |  予測期間 FY2025–2030  |  単位: 億円")
    print("=" * 65)

    for sc_name, res in results.items():
        sc  = SCENARIOS[sc_name]
        p   = res["proj"]
        v   = res["val"]
        print(f"\n【{sc['label']}】")
        print(f"  {'年度':>8} {'売上':>8} {'EBIT率':>7} {'FCF':>8}")
        for i, yr in enumerate(YEARS):
            margin = p["ebits"][i] / p["revenues"][i] * 100
            print(f"  {yr:>8} {p['revenues'][i]:>7.0f}億 {margin:>6.1f}% {p['fcfs'][i]:>7.0f}億")
        print(f"  ─────────────────────────────────────────")
        print(f"  FCF PV 合計:    ¥{v['sum_pv']:>7.0f}億")
        print(f"  Terminal Value: ¥{v['pv_tv']:>7.0f}億  ({v['tv_pct']:.0f}%)")
        print(f"  Enterprise Value: ¥{v['ev']/10000:.2f}兆円")
        print(f"  → 株価試算:     ¥{v['price']:>10,.0f}")

    print("\n" + "─" * 65)
    print("  参考: 現在株価レンジ（2024-2025）¥25,000–¥50,000 程度")
    print("  ※ 実際の株価・財務数値は IR・Bloomberg で確認のこと")
    print("=" * 65 + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bear", action="store_true")
    ap.add_argument("--bull", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    if args.bear:
        sc_names = ["bear"]
    elif args.bull:
        sc_names = ["bull"]
    else:
        sc_names = ["base", "bull", "bear"]

    results = {}
    for sc_name in sc_names:
        sc   = SCENARIOS[sc_name]
        proj = project_fcf(sc)
        val  = dcf_valuation(proj["fcfs"], sc["terminal_g"])
        results[sc_name] = {"proj": proj, "val": val}

    print_summary(results)

    if not args.no_plot and len(results) == 3:
        plot_all(results, OUT_DIR)


if __name__ == "__main__":
    main()
