"""
SMIC — 中国ファウンドリ / 地政学的制裁モデル
==========================================================
中国最大の半導体ファウンドリ (中芯国際集成電路製造)。
DUV オンリーで 7nm 相当 (N+2) を実現; EUV 禁輸下での技術進化を物理モデル化。

Key physics & geopolitics:
  - N+1/N+2 DUV マルチパターニング: SADP/SAQP → 実効 CD
  - DUV vs EUV コスト比較: マスク枚数 × コスト → プロセス経済性
  - 制裁影響モデル: 機器調達難 → 歩留まり劣化 × 生産コスト増
  - 市場シェア推移: 対米制裁前後
  - Huawei Kirin 9010: N+2 で製造 → 性能 vs TSMC N7 比較

実行:
    python fem/smic_model.py
    python fem/smic_model.py --patterning --node N+2
    python fem/smic_model.py --geopolitics

参考文献:
    SMIC Annual Report (2023)
    Levinson (2011) Principles of Lithography — SADP/SAQP
    BIS Export Control Rules (2022, 2023) — 制裁規定
    TechInsights Kirin 9010 analysis (2023)
"""

import argparse
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SMIC_NODES = {
    "28nm_HKMG": {
        "CD_nm": 28, "masks": 35, "EUV_required": False,
        "D0": 0.06, "wafer_cost_usd": 3500,
        "density_MT_mm2": 18, "color": "#2ca02c",
        "ref": "SMIC 28nm HKMG (2016)"},
    "14nm_FinFET": {
        "CD_nm": 14, "masks": 48, "EUV_required": False,
        "D0": 0.08, "wafer_cost_usd": 6000,
        "density_MT_mm2": 35, "color": "#ff7f0e",
        "ref": "SMIC N1 14nm (2019)"},
    "N+1_7nm_eq": {
        "CD_nm": 7, "masks": 72, "EUV_required": False,
        "D0": 0.12, "wafer_cost_usd": 10000,
        "density_MT_mm2": 70, "color": "#d62728",
        "ref": "SMIC N+1 (~7nm DUV, 2022)"},
    "N+2_7nm_adv": {
        "CD_nm": 5, "masks": 90, "EUV_required": False,
        "D0": 0.14, "wafer_cost_usd": 13000,
        "density_MT_mm2": 85, "color": "#2166ac",
        "ref": "SMIC N+2 (~5-7nm DUV, 2023)"},
}

TSMC_COMPARE = {
    "N7":  {"CD_nm": 7,  "masks": 55, "D0": 0.07, "wafer_cost_usd": 9346,
            "EUV_masks": 0,  "density_MT_mm2": 91},
    "N5":  {"CD_nm": 5,  "masks": 65, "D0": 0.08, "wafer_cost_usd": 16988,
            "EUV_masks": 14, "density_MT_mm2": 134},
    "N3E": {"CD_nm": 3,  "masks": 75, "D0": 0.085,"wafer_cost_usd": 20000,
            "EUV_masks": 25, "density_MT_mm2": 167},
}

def duv_multipatterning(target_CD_nm: float = 7.0,
                         tool_NA: float = 1.35,
                         wavelength_nm: float = 193.0) -> dict:
    """
    DUV マルチパターニング: SADP / SAQP で EUV なしに微細化。

    k1 = CD × NA / λ
    SADP: effective pitch = pitch/2
    SAQP: effective pitch = pitch/4
    必要パターニング回数 = ceil(log2(λ/(NA × target_CD) / 0.25))

    参考: Levinson (2011) — マルチパターニング
    """
    k1_min = 0.25   # 分解限界
    single_CD_nm = k1_min * wavelength_nm / tool_NA
    n_patterning = max(1, math.ceil(math.log2(single_CD_nm / target_CD_nm)))
    method = {1: "Single", 2: "SADP", 3: "SAQP", 4: "SAQP+"}
    extra_masks = (2 ** n_patterning - 1) * 4   # 追加マスク (spacer + etch)
    cost_penalty = 1.0 + (n_patterning - 1) * 0.35
    overlay_nm = 2.5 * n_patterning  # マルチパターニングオーバーレイ誤差増大

    return {
        "target_CD_nm": target_CD_nm,
        "single_pass_CD_nm": round(single_CD_nm, 1),
        "n_patterning": n_patterning,
        "method": method.get(n_patterning, "SAQP+"),
        "extra_masks": extra_masks,
        "cost_penalty_factor": round(cost_penalty, 2),
        "overlay_3sigma_nm": round(overlay_nm, 1),
        "DUV_feasible": n_patterning <= 4,
    }


def smic_yield_vs_sanctions(node: str = "N+2_7nm_adv", sanction_level: int = 2) -> dict:
    """
    制裁レベル別の歩留まり劣化モデル。

    sanction_level:
      0 = 制裁なし (2019以前)
      1 = 2020制裁 (機器輸出制限)
      2 = 2022制裁 (先進装置禁輸)
      3 = 2023制裁 (AIチップ + EDA ツール)

    歩留まり劣化: 制裁 → 代替機器 → 精度低下 → D0 増加
    """
    p = SMIC_NODES.get(node, SMIC_NODES["N+2_7nm_adv"])
    D0_base = p["D0"]
    sanction_D0_multiplier = {0: 1.0, 1: 1.15, 2: 1.40, 3: 1.70}
    D0_eff = D0_base * sanction_D0_multiplier.get(sanction_level, 1.0)
    # Die cost for 100mm² die
    A_cm2 = 100.0 / 100.0
    val = (1.0 - math.exp(-D0_eff * A_cm2)) / max(D0_eff * A_cm2, 1e-12)
    Y = val ** 2
    cost_premium = p["wafer_cost_usd"] * sanction_D0_multiplier.get(sanction_level, 1.0)
    return {
        "node": node,
        "sanction_level": sanction_level,
        "D0_effective": round(D0_eff, 4),
        "yield_pct": round(Y * 100, 1),
        "wafer_cost_est_usd": round(cost_premium, 0),
        "CPGD_100mm2_usd": round(cost_premium / max(200 * Y, 1), 0),
    }


def smic_vs_tsmc(smic_node: str = "N+2_7nm_adv",
                  tsmc_node: str = "N7") -> dict:
    """
    SMIC vs TSMC: 密度・コスト・オーバーレイ比較。
    """
    smic = SMIC_NODES.get(smic_node, SMIC_NODES["N+2_7nm_adv"])
    tsmc = TSMC_COMPARE.get(tsmc_node, TSMC_COMPARE["N7"])
    dup  = duv_multipatterning(smic["CD_nm"])
    density_gap_pct = (tsmc["density_MT_mm2"] - smic["density_MT_mm2"]) / max(tsmc["density_MT_mm2"], 1) * 100
    cost_ratio = smic["wafer_cost_usd"] / max(tsmc["wafer_cost_usd"], 1)
    return {
        "smic_node": smic_node, "tsmc_node": tsmc_node,
        "smic_density_MT_mm2": smic["density_MT_mm2"],
        "tsmc_density_MT_mm2": tsmc["density_MT_mm2"],
        "density_gap_pct": round(density_gap_pct, 1),
        "smic_wafer_cost_usd": smic["wafer_cost_usd"],
        "tsmc_wafer_cost_usd": tsmc["wafer_cost_usd"],
        "cost_ratio_smic_tsmc": round(cost_ratio, 2),
        "smic_extra_masks": dup["extra_masks"],
        "smic_method": dup["method"],
    }


def market_revenue_trend() -> dict:
    years  = [2019, 2020, 2021, 2022, 2023, 2024]
    rev_B  = [3.1,  3.9,  5.4,  7.3,  6.3,  7.0]   # SMIC revenue estimate [$B]
    share_pct = [4.5, 5.2, 5.8, 6.2, 5.4, 5.6]      # foundry market share
    return {"years": years, "revenue_B": rev_B, "share_pct": share_pct}


def plot_all(out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(17, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    ax = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    # Panel 0: DUV マルチパターニング: CD vs k1
    CD_arr = np.linspace(3, 30, 100)
    for NA, col in [(1.35, "#d62728"), (1.20, "#2166ac"), (0.93, "#2ca02c")]:
        k1_arr = [NA * CD / 193.0 for CD in CD_arr]
        ax[0].plot(CD_arr, k1_arr, color=col, lw=2.2, label=f"NA={NA}")
    ax[0].axhline(0.25, color="k", ls="--", lw=1.5, label="k1=0.25 分解限界")
    ax[0].axhspan(0.0, 0.25, alpha=0.1, color="red", label="DUV 不可領域")
    for node, p in SMIC_NODES.items():
        ax[0].scatter([p["CD_nm"]], [1.35 * p["CD_nm"] / 193.0], s=80,
                      color=p["color"], zorder=5)
    ax[0].set_xlabel("ターゲット CD [nm]"); ax[0].set_ylabel("k1 ファクタ")
    ax[0].set_title("DUV マルチパターニング空間\n(SADP/SAQP 適用領域)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)

    # Panel 1: 制裁レベル → 歩留まり劣化
    sanc_levels = [0, 1, 2, 3]
    for node, col in [(n, SMIC_NODES[n]["color"]) for n in SMIC_NODES]:
        y_arr = [smic_yield_vs_sanctions(node, s)["yield_pct"] for s in sanc_levels]
        ax[1].plot(sanc_levels, y_arr, "o-", color=col, lw=2.2, ms=8,
                   label=node.replace("_", " "))
    ax[1].set_xlabel("制裁レベル (0=なし, 3=最大)"); ax[1].set_ylabel("歩留まり [%]")
    ax[1].set_xticks(sanc_levels)
    ax[1].set_xticklabels(["0\n(制裁前)", "1\n(2020)", "2\n(2022)", "3\n(2023)"], fontsize=8)
    ax[1].set_title("SMIC 歩留まり vs 米国制裁レベル\n(D0 劣化モデル)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)

    # Panel 2: SMIC vs TSMC 密度比較
    nodes_smic = list(SMIC_NODES.keys())
    nodes_tsmc = list(TSMC_COMPARE.keys())
    x1 = np.arange(len(nodes_smic))
    x2 = np.arange(len(nodes_tsmc)) + len(nodes_smic) + 0.5
    bars1 = ax[2].bar(x1, [SMIC_NODES[n]["density_MT_mm2"] for n in nodes_smic],
                       color=[SMIC_NODES[n]["color"] for n in nodes_smic], width=0.7, alpha=0.85, label="SMIC")
    bars2 = ax[2].bar(x2, [TSMC_COMPARE[n]["density_MT_mm2"] for n in nodes_tsmc],
                       color=["#2166ac", "#d62728", "#2ca02c"], width=0.7, alpha=0.85, hatch="///", label="TSMC")
    ax[2].set_xticks(list(x1) + list(x2))
    ax[2].set_xticklabels([n.replace("_", "\n") for n in nodes_smic] + list(nodes_tsmc),
                           fontsize=7)
    ax[2].set_ylabel("密度 [MT/mm²]")
    ax[2].set_title("SMIC vs TSMC トランジスタ密度\n(DUV vs EUV)")
    ax[2].legend(fontsize=9); ax[2].grid(axis="y", alpha=0.25)

    # Panel 3: SMIC 売上 & 市場シェア推移
    trend = market_revenue_trend()
    ax3b = ax[3].twinx()
    ax[3].bar(trend["years"], trend["revenue_B"], color="#d62728", alpha=0.6, label="売上 [$B]", width=0.6)
    ax3b.plot(trend["years"], trend["share_pct"], "D--", color="#2166ac", lw=2.0, ms=7)
    ax[3].set_xlabel("年"); ax[3].set_ylabel("売上 [$B]", color="#d62728")
    ax3b.set_ylabel("市場シェア [%]", color="#2166ac")
    ax[3].set_title("SMIC 売上 & ファウンドリ市場シェア\n(2019-2024)")
    ax[3].legend(loc="upper left", fontsize=8.5)
    ax[3].grid(axis="y", alpha=0.25)

    # Panel 4: ウェーハコスト vs TSMC 比較
    all_nodes = list(SMIC_NODES.keys()) + list(TSMC_COMPARE.keys())
    all_costs  = ([SMIC_NODES[n]["wafer_cost_usd"] for n in SMIC_NODES] +
                  [TSMC_COMPARE[n]["wafer_cost_usd"] for n in TSMC_COMPARE])
    all_cols   = ([SMIC_NODES[n]["color"] for n in SMIC_NODES] +
                  ["#2166ac", "#d62728", "#2ca02c"])
    all_labels = ([n.replace("_", "\n") for n in SMIC_NODES] +
                  [f"TSMC {n}" for n in TSMC_COMPARE])
    ax[4].bar(range(len(all_nodes)), all_costs, color=all_cols, width=0.6, alpha=0.85)
    ax[4].set_xticks(range(len(all_nodes)))
    ax[4].set_xticklabels(all_labels, fontsize=7)
    ax[4].set_ylabel("ウェーハコスト [USD]")
    ax[4].set_title("ウェーハコスト比較\nSMIC (DUV) vs TSMC (EUV)")
    ax[4].grid(axis="y", alpha=0.25)

    # Panel 5: SMIC ノード比較テーブル
    ax[5].axis("off")
    col_labels = ["ノード", "CD\n[nm]", "マスク\n枚数", "D0\n[/cm²]",
                  "密度\n[MT/mm²]", "制裁L2\n歩留まり"]
    tdata = []
    for nn, np_ in SMIC_NODES.items():
        ys = smic_yield_vs_sanctions(nn, 2)
        tdata.append([nn.replace("_", " "),
                      f"{np_['CD_nm']}", f"{np_['masks']}",
                      f"{np_['D0']:.2f}",
                      f"{np_['density_MT_mm2']}", f"{ys['yield_pct']:.1f}%"])
    tbl = ax[5].table(cellText=tdata, colLabels=col_labels,
                      cellLoc="center", loc="center", bbox=[0.0, 0.20, 1.0, 0.75])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#8b0000"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#fff5f5")
    ax[5].set_title("SMIC プロセスノード 比較\n28nm · 14nm · N+1 · N+2")

    fig.suptitle("SMIC — 中国ファウンドリ / 地政学制裁モデル\n"
                 "DUV マルチパターニング · 制裁歩留まり劣化 · TSMC 競合比較",
                 fontsize=12, fontweight="bold")
    out = os.path.join(out_dir, "smic_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] SMIC model → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="SMIC foundry + geopolitics model")
    ap.add_argument("--patterning",  action="store_true")
    ap.add_argument("--geopolitics", action="store_true")
    ap.add_argument("--node",  default="N+2_7nm_adv", choices=list(SMIC_NODES.keys()))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    if args.patterning:
        r = duv_multipatterning(SMIC_NODES[args.node]["CD_nm"])
        print(f"\n{args.node}: {r['method']}  extra_masks={r['extra_masks']}  "
              f"cost_penalty={r['cost_penalty_factor']}×")
    if args.geopolitics:
        print(f"\n制裁レベル別 歩留まり ({args.node})")
        for lv in range(4):
            r = smic_yield_vs_sanctions(args.node, lv)
            print(f"  Lv{lv}  D0={r['D0_effective']:.4f}  "
                  f"yield={r['yield_pct']:.1f}%  CPGD=${r['CPGD_100mm2_usd']}")
    if not args.no_plot:
        plot_all()

if __name__ == "__main__":
    main()
