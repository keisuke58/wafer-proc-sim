"""
ASE / Amkor — OSAT 後工程組立・テスト コストモデル
==========================================================
世界最大の半導体後工程委託サービス (OSAT)。
CoWoS 組立、Flip Chip、Fan-Out WLP (InFO)、3D-IC SoIC の
組立コスト・歩留まり・熱管理モデル。

Key physics & economics:
  - Flip Chip バンプ歩留まり: バンプピッチ × 欠陥密度 → 接続歩留まり
  - Fan-Out WLP (FOWLP): RDL 層数 → 配線コスト + 歩留まり
  - CoWoS 組立コスト: インターポーザー → 実装 → 充填 → テスト
  - 3D-IC ハイブリッドボンディング 接続密度 vs 歩留まり
  - OSAT コスト構造: 材料費 × 労務費 × 設備費
  - ASE vs Amkor 市場シェア

実行:
    python fem/osat_model.py
    python fem/osat_model.py --pkg CoWoS
    python fem/osat_model.py --cost --device GB200

参考文献:
    Lau (2019) IEEE TCPMT — 3D IC packaging cost model
    Stow et al. (2017) DAC — heterogeneous integration cost
    ASE / Amkor Annual Reports (2024)
    TSMC CoWoS cost model (ECTC 2023)
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

# ════════════════════════════════════════════════════════════════════════════
# 1. パッケージング パラメータ テーブル
# ════════════════════════════════════════════════════════════════════════════

PKG_TYPES = {
    "Wire_Bond": {
        "bump_pitch_um": 0.0,
        "rdl_layers": 0,
        "cost_per_unit_usd": 0.20,
        "yield_base_pct": 99.8,
        "throughput_uph": 10000,
        "BW_GBs": 0.1,
        "color": "#ff7f0e",
        "ref": "ASE wire bond brief (2023)",
    },
    "Flip_Chip_BGA": {
        "bump_pitch_um": 150.0,
        "rdl_layers": 2,
        "cost_per_unit_usd": 2.50,
        "yield_base_pct": 99.5,
        "throughput_uph": 2000,
        "BW_GBs": 50.0,
        "color": "#2166ac",
        "ref": "ASE Flip Chip FCBGA brief (2023)",
    },
    "FOWLP_InFO": {
        "bump_pitch_um": 50.0,
        "rdl_layers": 4,
        "cost_per_unit_usd": 8.0,
        "yield_base_pct": 98.5,
        "throughput_uph": 500,
        "BW_GBs": 200.0,
        "color": "#d62728",
        "ref": "TSMC InFO-PoP / ASE Fan-Out (2024)",
    },
    "CoWoS_S": {
        "bump_pitch_um": 130.0,
        "rdl_layers": 3,
        "cost_per_unit_usd": 150.0,
        "yield_base_pct": 96.0,
        "throughput_uph": 20,
        "BW_GBs": 4000.0,
        "color": "#2ca02c",
        "ref": "TSMC CoWoS-S + ASE assembly (2024)",
    },
    "CoWoS_L": {
        "bump_pitch_um": 40.0,
        "rdl_layers": 5,
        "cost_per_unit_usd": 350.0,
        "yield_base_pct": 93.0,
        "throughput_uph": 8,
        "BW_GBs": 10000.0,
        "color": "#9467bd",
        "ref": "TSMC CoWoS-L + Amkor assembly (2024)",
    },
    "SoIC_3D": {
        "bump_pitch_um": 9.0,
        "rdl_layers": 8,
        "cost_per_unit_usd": 600.0,
        "yield_base_pct": 90.0,
        "throughput_uph": 4,
        "BW_GBs": 50000.0,
        "color": "#8c564b",
        "ref": "TSMC SoIC-X + ASE assembly (2024)",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 2. Core Physics / Economics Functions
# ════════════════════════════════════════════════════════════════════════════

def bump_yield(pkg: str = "CoWoS_S", die_area_mm2: float = 800.0,
               D0_bump: float = 0.005) -> dict:
    """
    バンプ接続歩留まり:
    Y_bump = (1 - D0_bump × A_bump) ^ n_bumps
    n_bumps = die_area / (pitch²) × fill_factor

    D0_bump: バンプ欠陥密度 [/bump] (典型 0.1-1%)
    """
    p = PKG_TYPES.get(pkg, PKG_TYPES["CoWoS_S"])
    pitch = p["bump_pitch_um"]
    if pitch < 1.0:
        return {"pkg": pkg, "yield_pct": 100.0, "n_bumps": 0,
                "note": "wire bond: no bumps"}
    fill_factor = 0.70   # 70% area coverage
    n_bumps = int(die_area_mm2 * 1e6 / pitch**2 * fill_factor)
    Y_bump  = (1.0 - D0_bump) ** n_bumps
    Y_total = p["yield_base_pct"] / 100.0 * Y_bump
    return {
        "pkg": pkg, "die_area_mm2": die_area_mm2,
        "n_bumps": n_bumps,
        "D0_bump": D0_bump,
        "bump_yield_pct": round(Y_bump * 100, 3),
        "total_yield_pct": round(Y_total * 100, 2),
    }


def assembly_cost(pkg: str = "CoWoS_S", n_dies: int = 1,
                  die_area_mm2: float = 800.0) -> dict:
    """
    組立総コスト = 材料費 + 加工費 + テスト費
    CoWoS 面積スケーリング: cost ∝ sqrt(total_area)
    """
    p = PKG_TYPES.get(pkg, PKG_TYPES["CoWoS_S"])
    base_cost = p["cost_per_unit_usd"]
    # Area scaling: larger packages cost more
    area_factor = math.sqrt(die_area_mm2 * n_dies / 100.0)
    pkg_cost = base_cost * area_factor
    # Test cost: burn-in + final test
    test_cost = pkg_cost * 0.12
    total = pkg_cost + test_cost
    # Yield-adjusted cost
    yld = bump_yield(pkg, die_area_mm2)["total_yield_pct"] / 100.0
    cost_adjusted = total / max(yld, 0.1)
    return {
        "pkg": pkg, "n_dies": n_dies, "die_area_mm2": die_area_mm2,
        "pkg_cost_usd": round(pkg_cost, 2),
        "test_cost_usd": round(test_cost, 2),
        "total_cost_usd": round(total, 2),
        "yield_adj_cost_usd": round(cost_adjusted, 2),
        "yield_pct": round(yld * 100, 2),
        "BW_GBs": p["BW_GBs"],
    }


def cowos_gb200_cost() -> dict:
    """
    NVIDIA GB200 NVL72: CoWoS-L + SoIC 組立コスト内訳。
    2× B200 GPU + 4× NVLink Switch per NVL72 node。
    """
    gpu_die_cost = 4000.0   # per B200 die
    hbm_cost = 600.0 * 8   # 8× HBM3E stack
    cowos_assy = assembly_cost("CoWoS_L", n_dies=3, die_area_mm2=3000)
    substrate_cost = 800.0
    test_final  = 2500.0
    total_per_gpu = (gpu_die_cost + hbm_cost + cowos_assy["total_cost_usd"] +
                     substrate_cost + test_final)
    return {
        "device": "GB200 (per GPU module)",
        "gpu_die_cost_usd": gpu_die_cost,
        "hbm_cost_usd": hbm_cost,
        "cowos_assembly_usd": round(cowos_assy["total_cost_usd"], 0),
        "substrate_usd": substrate_cost,
        "final_test_usd": test_final,
        "total_per_gpu_usd": round(total_per_gpu, 0),
        "osat_pct_of_total": round(cowos_assy["total_cost_usd"] / total_per_gpu * 100, 1),
    }


def market_share() -> dict:
    return {
        "ASE":   {"share_pct": 30.0, "revenue_B": 17.2, "color": "#d62728",
                  "specialty": "CoWoS, FOWLP, SiP"},
        "Amkor": {"share_pct": 19.0, "revenue_B": 6.1,  "color": "#2166ac",
                  "specialty": "FOWLP, Flip Chip, 3D"},
        "JCET":  {"share_pct": 13.0, "revenue_B": 3.8,  "color": "#2ca02c",
                  "specialty": "Wire Bond, Flip Chip"},
        "Powertech": {"share_pct": 8.0, "revenue_B": 2.3, "color": "#9467bd",
                      "specialty": "Memory packaging"},
        "Others": {"share_pct": 30.0, "revenue_B": 8.5, "color": "#ff7f0e",
                   "specialty": "Various"},
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. Visualization
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(17, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    ax = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    # ── Panel 0: バンプ歩留まり vs 欠陥率 ────────────────────────────────
    D0_arr = np.linspace(0.0001, 0.01, 100)
    for pkg, col in [("Flip_Chip_BGA", "#2166ac"), ("FOWLP_InFO", "#d62728"),
                      ("CoWoS_S", "#2ca02c"), ("CoWoS_L", "#9467bd")]:
        yld = [bump_yield(pkg, 800.0, D0)["bump_yield_pct"] for D0 in D0_arr]
        ax[0].plot(D0_arr * 100, yld, color=col, lw=2.2,
                   label=pkg.replace("_", " "))
    ax[0].set_xlabel("バンプ欠陥率 [%]"); ax[0].set_ylabel("バンプ歩留まり [%]")
    ax[0].set_title("バンプ接続歩留まり vs 欠陥率\n(ダイ面積 800mm²)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25); ax[0].set_ylim(60, 100)

    # ── Panel 1: 組立コスト vs ダイ面積 ─────────────────────────────────
    area_arr = np.linspace(50, 1500, 100)
    for pkg, col in [("Flip_Chip_BGA", "#2166ac"), ("FOWLP_InFO", "#d62728"),
                      ("CoWoS_S", "#2ca02c"), ("CoWoS_L", "#9467bd"), ("SoIC_3D", "#8c564b")]:
        cost = [assembly_cost(pkg, 1, A)["yield_adj_cost_usd"] for A in area_arr]
        ax[1].semilogy(area_arr, cost, color=col, lw=2.2, label=pkg.replace("_", " "))
    ax[1].set_xlabel("ダイ面積 [mm²]"); ax[1].set_ylabel("歩留まり調整後コスト [USD]")
    ax[1].set_title("OSAT 組立コスト vs ダイ面積\n(ASE / Amkor, 歩留まり補正済み)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)

    # ── Panel 2: BW vs コスト トレードオフ ──────────────────────────────
    pkgs = [p for p in PKG_TYPES if p != "Wire_Bond"]
    bws   = [PKG_TYPES[p]["BW_GBs"] / 1000 for p in pkgs]
    costs = [PKG_TYPES[p]["cost_per_unit_usd"] for p in pkgs]
    cols  = [PKG_TYPES[p]["color"] for p in pkgs]
    ax[2].scatter(costs, bws, c=cols, s=150, zorder=5)
    for pkg, cost, bw in zip(pkgs, costs, bws):
        ax[2].annotate(pkg.replace("_", "\n"), (cost, bw), fontsize=7.5,
                       xytext=(5, -12), textcoords="offset points")
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xlabel("基本コスト [USD]"); ax[2].set_ylabel("帯域幅 [TB/s]")
    ax[2].set_title("パッケージング BW vs コスト\n(対数スケール)")
    ax[2].grid(alpha=0.25, which="both")

    # ── Panel 3: GB200 組立コスト内訳 ────────────────────────────────────
    gb = cowos_gb200_cost()
    labels_g = ["GPU ダイ", "HBM3E\n(×8スタック)", "CoWoS-L\n組立", "基板", "最終テスト"]
    values_g = [gb["gpu_die_cost_usd"], gb["hbm_cost_usd"],
                gb["cowos_assembly_usd"], gb["substrate_usd"], gb["final_test_usd"]]
    colors_g = ["#2166ac", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
    ax[3].bar(range(len(labels_g)), values_g, color=colors_g, width=0.6, alpha=0.85)
    ax[3].set_xticks(range(len(labels_g))); ax[3].set_xticklabels(labels_g, fontsize=8)
    ax[3].set_ylabel("コスト [USD]")
    ax[3].set_title(f"NVIDIA GB200 組立コスト内訳\n合計 ${gb['total_per_gpu_usd']:,.0f}/GPU")
    for i, v in enumerate(values_g):
        ax[3].text(i, v + 50, f"${v:,.0f}", ha="center", fontsize=8)
    ax[3].grid(axis="y", alpha=0.25)

    # ── Panel 4: OSAT 市場シェア ─────────────────────────────────────────
    ms = market_share()
    labels_m = list(ms.keys())
    shares_m  = [ms[k]["share_pct"] for k in labels_m]
    colors_m  = [ms[k]["color"] for k in labels_m]
    wedges, texts, autotexts = ax[4].pie(shares_m, labels=labels_m, colors=colors_m,
                                          autopct="%1.1f%%", startangle=90, pctdistance=0.75)
    for at in autotexts: at.set_fontsize(9)
    total_osat_B = sum(ms[k]["revenue_B"] for k in ms)
    ax[4].set_title(f"OSAT 市場シェア (2024推定)\nTAM ~${total_osat_B:.0f}B")

    # ── Panel 5: パッケージング比較テーブル ─────────────────────────────
    ax[5].axis("off")
    col_labels = ["パッケージ", "バンプ\nピッチ", "コスト\n[USD]", "歩留まり\n[%]",
                  "BW\n[TB/s]", "スループット\n[UPH]"]
    tdata = [[pk.replace("_", " "),
              f"{PKG_TYPES[pk]['bump_pitch_um']:.0f}µm" if PKG_TYPES[pk]["bump_pitch_um"] > 0 else "—",
              f"${PKG_TYPES[pk]['cost_per_unit_usd']:.0f}",
              f"{PKG_TYPES[pk]['yield_base_pct']:.1f}",
              f"{PKG_TYPES[pk]['BW_GBs']/1000:.1f}",
              f"{PKG_TYPES[pk]['throughput_uph']}"]
             for pk in PKG_TYPES]
    tbl = ax[5].table(cellText=tdata, colLabels=col_labels,
                      cellLoc="center", loc="center", bbox=[0.0, 0.08, 1.0, 0.88])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1a4a1a"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0fff0")
    ax[5].set_title("OSAT パッケージング比較\nASE / Amkor 取扱い対応")

    fig.suptitle("ASE / Amkor — OSAT 後工程組立 コストモデル\n"
                 "Wire Bond · Flip Chip · FOWLP · CoWoS · SoIC 3D",
                 fontsize=12, fontweight="bold")
    out = os.path.join(out_dir, "osat_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] OSAT model → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="ASE/Amkor OSAT cost model")
    ap.add_argument("--pkg",  default="CoWoS_S", choices=list(PKG_TYPES.keys()))
    ap.add_argument("--cost", action="store_true")
    ap.add_argument("--device", default="GB200")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    if args.cost:
        r = assembly_cost(args.pkg, 1, 800.0)
        print(f"\n{args.pkg}: total=${r['total_cost_usd']:.2f}  "
              f"yield={r['yield_pct']:.2f}%  BW={r['BW_GBs']:.0f} GB/s")
    if not args.no_plot:
        plot_all()

if __name__ == "__main__":
    main()
