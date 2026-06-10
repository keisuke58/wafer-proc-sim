"""
Teradyne — ATE (Automated Test Equipment) モデル
==========================================================
Advantest と並ぶ世界最大の ATE メーカー (US拠点)。
UltraFLEX Plus (SoC/RF)、J750 (汎用)、Eagle (メモリ) をカバー。

Key physics & economics:
  - テスト時間モデル: ベクタ数 × サイクル時間 × 並列度
  - テストカバレッジ: 故障検出率 (DPM = Defective Parts per Million)
  - コスト比較: Teradyne vs Advantest (テスタ価格 × スループット → CPT)
  - メモリテスト: HBM / LPDDR5X パターン生成コスト
  - サプライチェーン: NVIDIA H100 → テスト時間 × コスト
  - Teradyne vs Advantest 市場シェア モデル

実行:
    python fem/teradyne_model.py
    python fem/teradyne_model.py --soc --device A100
    python fem/teradyne_model.py --memory --type HBM3E

参考文献:
    SEMI E10 テスタ稼働率標準 (2023)
    Teradyne UltraFLEX Plus product brief (2024)
    Teradyne Annual Report (2024)
    Daasch et al. (2010) ITC — SoC test economics
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
# 1. テスタ パラメータ テーブル
# ════════════════════════════════════════════════════════════════════════════

TESTERS = {
    # Teradyne
    "UltraFLEX_Plus": {
        "vendor": "Teradyne",
        "target": "SoC/RF/GPU",
        "freq_GHz": 6.0,
        "n_sites_max": 8,
        "price_MUSD": 8.5,
        "pins": 1024,
        "power_W": 15000,
        "footprint_m2": 6.5,
        "color": "#d62728",
        "ref": "Teradyne UltraFLEX Plus brief (2024)",
    },
    "J750": {
        "vendor": "Teradyne",
        "target": "Mixed-signal/MCU",
        "freq_GHz": 0.2,
        "n_sites_max": 32,
        "price_MUSD": 1.2,
        "pins": 256,
        "power_W": 3000,
        "footprint_m2": 3.0,
        "color": "#ff7f0e",
        "ref": "Teradyne J750 brief (2023)",
    },
    "Eagle": {
        "vendor": "Teradyne",
        "target": "DRAM/NAND/HBM",
        "freq_GHz": 3.2,
        "n_sites_max": 128,
        "price_MUSD": 5.0,
        "pins": 4096,
        "power_W": 8000,
        "footprint_m2": 5.0,
        "color": "#2ca02c",
        "ref": "Teradyne Eagle memory tester brief (2024)",
    },
    # Advantest (比較用)
    "T5503": {
        "vendor": "Advantest",
        "target": "SoC/GPU",
        "freq_GHz": 4.0,
        "n_sites_max": 8,
        "price_MUSD": 9.0,
        "pins": 1024,
        "power_W": 18000,
        "footprint_m2": 7.0,
        "color": "#2166ac",
        "ref": "Advantest T5503 brief (2024)",
    },
    "T5577": {
        "vendor": "Advantest",
        "target": "HBM/LPDDR",
        "freq_GHz": 3.6,
        "n_sites_max": 64,
        "price_MUSD": 6.5,
        "pins": 8192,
        "power_W": 12000,
        "footprint_m2": 6.0,
        "color": "#9467bd",
        "ref": "Advantest T5577 brief (2024)",
    },
}

DEVICES = {
    "H100_SXM": {
        "gate_count_B": 80.0,
        "test_vectors_M": 5000.0,
        "freq_test_GHz": 1.5,
        "n_io": 5000,
        "burn_in_hr": 0.5,
        "die_value_usd": 3000.0,
        "tester": "UltraFLEX_Plus",
        "color": "#2166ac",
    },
    "B200": {
        "gate_count_B": 208.0,
        "test_vectors_M": 12000.0,
        "freq_test_GHz": 1.5,
        "n_io": 8000,
        "burn_in_hr": 0.5,
        "die_value_usd": 5000.0,
        "tester": "UltraFLEX_Plus",
        "color": "#d62728",
    },
    "Snapdragon_8_Elite": {
        "gate_count_B": 20.0,
        "test_vectors_M": 800.0,
        "freq_test_GHz": 2.0,
        "n_io": 1200,
        "burn_in_hr": 0.1,
        "die_value_usd": 120.0,
        "tester": "UltraFLEX_Plus",
        "color": "#2ca02c",
    },
    "HBM3E_12Hi": {
        "gate_count_B": 0.0,
        "test_vectors_M": 0.0,
        "freq_test_GHz": 4.8,
        "n_io": 1024,
        "burn_in_hr": 2.0,
        "die_value_usd": 600.0,
        "tester": "Eagle",
        "color": "#9467bd",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 2. Core Physics / Economics Functions
# ════════════════════════════════════════════════════════════════════════════

def test_time(device: str = "H100_SXM", n_sites: int = 1) -> dict:
    """
    テスト時間 = ベクタ数 × (1/freq) × (1/並列度) + バーンイン
    DPM: 故障検出率から残留欠陥率を推定

    参考: SEMI E10 — OEE テスタ稼働率
    """
    d = DEVICES.get(device, DEVICES["H100_SXM"])
    t_test_s  = d["test_vectors_M"] * 1e6 / (d["freq_test_GHz"] * 1e9) / max(n_sites, 1)
    t_burnin_s = d["burn_in_hr"] * 3600
    t_total_s  = t_test_s + t_burnin_s
    # Coverage: empirical 97% (3% stuck-at fault escape)
    coverage_pct = 97.0
    # DPM = (1 - coverage/100) * defect_rate_per_M
    defect_rate_ppm = 500.0   # typical GPU wafer ~500 ppm gross defects
    DPM_escape = defect_rate_ppm * (1.0 - coverage_pct / 100.0)

    return {
        "device": device,
        "n_sites": n_sites,
        "test_time_s": round(t_test_s, 2),
        "burnin_s": round(t_burnin_s, 0),
        "total_time_s": round(t_total_s, 2),
        "coverage_pct": coverage_pct,
        "DPM_escape": round(DPM_escape, 2),
    }


def cost_per_test(device: str = "H100_SXM", tester: str = "UltraFLEX_Plus",
                  n_sites: int = 1, utilization: float = 0.85) -> dict:
    """
    テスト コスト (Cost Per Test):
    CPT = (tester_depreciaton + opex) × test_time / (n_sites × utilization)

    参考: Daasch et al. (2010) ITC
    """
    t = TESTERS.get(tester, TESTERS["UltraFLEX_Plus"])
    d = DEVICES.get(device, DEVICES["H100_SXM"])
    # Depreciation: 5年、15% OpEx
    annual_cost = t["price_MUSD"] * 1e6 / 5.0 + t["price_MUSD"] * 1e6 * 0.15
    cost_per_second = annual_cost / (3600 * 8000 * utilization)
    tt = test_time(device, n_sites)
    CPT = cost_per_second * tt["total_time_s"] / max(n_sites, 1)
    test_cost_pct = CPT / max(d["die_value_usd"], 1.0) * 100.0
    return {
        "device": device, "tester": tester, "n_sites": n_sites,
        "CPT_usd": round(CPT, 2),
        "die_value_usd": d["die_value_usd"],
        "test_cost_pct": round(test_cost_pct, 2),
        "total_test_time_s": tt["total_time_s"],
    }


def memory_test_pattern(mem_type: str = "HBM3E", n_io: int = 1024,
                         freq_GHz: float = 4.8) -> dict:
    """
    メモリテスト パターン生成コスト:
    パターン数 = アドレス空間 × テストアルゴリズム係数 (March-C+: 12n)
    サイクル時間 = 1/freq; テスト時間 ∝ capacity × algo_factor

    参考: van de Goor (1998) Testing Semiconductor Memories
    """
    mem_capacity_Gb = {"HBM3E": 288.0, "HBM3": 192.0, "LPDDR5X": 16.0,
                       "DDR5": 32.0}.get(mem_type, 32.0)
    algo_factor = {"March_C+": 12, "March_B": 17, "Mats+": 4}.get("March_C+", 12)
    n_cells = mem_capacity_Gb * 1e9
    cycle_ns = 1.0 / freq_GHz
    test_time_ms = n_cells * algo_factor * cycle_ns * 1e-6 / max(n_io, 1)
    return {
        "mem_type": mem_type,
        "n_io": n_io,
        "freq_GHz": freq_GHz,
        "capacity_Gb": mem_capacity_Gb,
        "algo_factor": algo_factor,
        "test_time_ms": round(test_time_ms, 3),
        "test_time_s": round(test_time_ms / 1000, 4),
    }


def market_share_model() -> dict:
    """ATE 市場シェア (2024推定) と主要顧客"""
    market = {
        "Teradyne": {
            "share_pct": 50.0,
            "revenue_B": 2.7,
            "key_customers": ["NVIDIA", "Apple", "Qualcomm", "Broadcom"],
            "color": "#d62728",
        },
        "Advantest": {
            "share_pct": 44.0,
            "revenue_B": 5.5,   # FY2024 (含む半導体テスタ以外)
            "key_customers": ["SK Hynix", "Samsung", "TSMC", "Kioxia"],
            "color": "#2166ac",
        },
        "Others": {
            "share_pct": 6.0,
            "revenue_B": 0.3,
            "key_customers": ["Cohu", "FormFactor"],
            "color": "#9467bd",
        },
    }
    HHI = sum((s["share_pct"] / 100) ** 2 for s in market.values())
    return {"market": market, "total_TAM_B": 8.0, "HHI": round(HHI, 4)}


# ════════════════════════════════════════════════════════════════════════════
# 3. Visualization
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(17, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    ax = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    # ── Panel 0: テスト時間 vs サイト数 ─────────────────────────────────
    sites_arr = [1, 2, 4, 8, 16, 32]
    for dev, col in [("H100_SXM", "#2166ac"), ("B200", "#d62728"),
                      ("Snapdragon_8_Elite", "#2ca02c"), ("HBM3E_12Hi", "#9467bd")]:
        tt_arr = [test_time(dev, s)["total_time_s"] for s in sites_arr]
        ax[0].loglog(sites_arr, tt_arr, "o-", color=col, lw=2.2, ms=7,
                     label=dev.replace("_", " "))
    ax[0].set_xlabel("並列サイト数"); ax[0].set_ylabel("テスト時間 [s]")
    ax[0].set_title("テスト時間 vs 並列サイト数\n(UltraFLEX Plus)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25, which="both")

    # ── Panel 1: CPT vs テスタ種別 ──────────────────────────────────────
    devices_cpt = ["H100_SXM", "B200", "Snapdragon_8_Elite"]
    testers_cmp = ["UltraFLEX_Plus", "T5503"]
    x = np.arange(len(devices_cpt)); w = 0.35
    for i, (tname, col) in enumerate([("UltraFLEX_Plus", "#d62728"), ("T5503", "#2166ac")]):
        cpts = [cost_per_test(d, tname)["CPT_usd"] for d in devices_cpt]
        ax[1].bar(x + i*w - w/2, cpts, w, color=col, alpha=0.85,
                  label=TESTERS[tname]["vendor"])
    ax[1].set_xticks(x); ax[1].set_xticklabels([d.replace("_", "\n") for d in devices_cpt], fontsize=8)
    ax[1].set_ylabel("CPT [USD]"); ax[1].set_title("テストコスト比較\nTeradyne vs Advantest")
    ax[1].legend(fontsize=9); ax[1].grid(axis="y", alpha=0.25)

    # ── Panel 2: メモリテスト時間 vs I/O 本数 ───────────────────────────
    io_arr = np.array([256, 512, 1024, 2048, 4096])
    for mtype, col in [("HBM3E", "#d62728"), ("HBM3", "#2166ac"),
                        ("LPDDR5X", "#2ca02c"), ("DDR5", "#ff7f0e")]:
        tt_arr = [memory_test_pattern(mtype, n_io=n, freq_GHz=4.8)["test_time_ms"] for n in io_arr]
        ax[2].loglog(io_arr, tt_arr, "s-", color=col, lw=2.2, ms=7, label=mtype)
    ax[2].set_xlabel("I/O 本数"); ax[2].set_ylabel("テスト時間 [ms]")
    ax[2].set_title("メモリテスト時間 vs I/O 本数\n(March-C+ アルゴリズム, 4.8GHz)")
    ax[2].legend(fontsize=9); ax[2].grid(alpha=0.25, which="both")

    # ── Panel 3: テストコスト vs ダイ価値 ────────────────────────────────
    die_values = np.logspace(1, 5, 100)
    for dev, col in [("H100_SXM", "#2166ac"), ("B200", "#d62728"),
                      ("Snapdragon_8_Elite", "#2ca02c")]:
        tt_s = test_time(dev, 1)["total_time_s"]
        # CPT固定; test_cost_pct = CPT/die_value
        CPT_fixed = cost_per_test(dev, DEVICES[dev]["tester"])["CPT_usd"]
        pct_arr = [CPT_fixed / v * 100 for v in die_values]
        ax[3].semilogx(die_values, pct_arr, color=col, lw=2.2,
                       label=dev.replace("_", " "))
        ax[3].scatter([DEVICES[dev]["die_value_usd"]], [CPT_fixed / DEVICES[dev]["die_value_usd"] * 100],
                      color=col, s=80, zorder=5)
    ax[3].axhline(5.0, color="k", ls="--", lw=1.5, label="5% target")
    ax[3].set_xlabel("ダイ価値 [USD]"); ax[3].set_ylabel("テストコスト比 [%]")
    ax[3].set_title("テストコスト / ダイ価値\n(●= 実際の価値・コスト点)")
    ax[3].legend(fontsize=8); ax[3].grid(alpha=0.25)
    ax[3].set_ylim(0, 30)

    # ── Panel 4: ATE 市場シェア ────────────────────────────────────────
    ms = market_share_model()
    labels_m = list(ms["market"].keys())
    shares    = [ms["market"][k]["share_pct"] for k in labels_m]
    colors_m  = [ms["market"][k]["color"] for k in labels_m]
    wedges, texts, autotexts = ax[4].pie(
        shares, labels=labels_m, colors=colors_m, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75)
    for at in autotexts: at.set_fontsize(10)
    ax[4].set_title(f"ATE 市場シェア (2024推定)\nTAM ~${ms['total_TAM_B']:.1f}B  HHI={ms['HHI']:.3f}")

    # ── Panel 5: テスタ スペック比較テーブル ────────────────────────────
    ax[5].axis("off")
    col_labels = ["テスタ", "メーカー", "対象", "価格\n[$M]", "最大\nサイト", "周波数\n[GHz]"]
    tdata = [[tname.replace("_", " "), tp["vendor"], tp["target"][:15],
              f"${tp['price_MUSD']:.1f}M", str(tp["n_sites_max"]), f"{tp['freq_GHz']:.1f}"]
             for tname, tp in TESTERS.items()]
    tbl = ax[5].table(cellText=tdata, colLabels=col_labels,
                      cellLoc="center", loc="center", bbox=[0.0, 0.10, 1.0, 0.85])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#8b0000"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#fff0f0")
        # Teradyne rows: light red, Advantest rows: light blue
        elif r > 0 and r <= len(tdata):
            vendor = tdata[r-1][1] if r-1 < len(tdata) else ""
            if vendor == "Advantest":
                cell.set_facecolor("#f0f0ff")
    ax[5].set_title("テスタ スペック比較\nTeradyne vs Advantest")

    fig.suptitle("Teradyne — ATE テスト時間 · CPT · メモリテスト · 市場シェア\n"
                 "UltraFLEX Plus · Eagle · J750 vs Advantest T5503/T5577",
                 fontsize=12, fontweight="bold")
    out = os.path.join(out_dir, "teradyne_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Teradyne model → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Teradyne ATE model")
    ap.add_argument("--soc",    action="store_true")
    ap.add_argument("--memory", action="store_true")
    ap.add_argument("--device", default="H100_SXM", choices=list(DEVICES.keys()))
    ap.add_argument("--type",   default="HBM3E")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    if args.soc:
        r = cost_per_test(args.device, DEVICES[args.device]["tester"])
        print(f"\n{args.device}: CPT=${r['CPT_usd']:.2f}  "
              f"テストコスト比={r['test_cost_pct']:.2f}%")
    if args.memory:
        r = memory_test_pattern(args.type)
        print(f"\n{args.type}: テスト時間={r['test_time_ms']:.3f}ms  "
              f"容量={r['capacity_Gb']:.0f}Gb")
    if not args.no_plot:
        plot_all()

if __name__ == "__main__":
    main()
