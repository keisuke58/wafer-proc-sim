"""
Qualcomm — Snapdragon モバイル SoC / AI モデル
==========================================================
世界最大のモバイル SoC メーカー。
Snapdragon 8 Elite (TSMC N3E)、Oryon CPU コア (ARM カスタム)、
Adreno GPU、Hexagon NPU の物理・コスト・市場モデル。

Key physics & economics:
  - Oryon CPU コア: パイプライン深さ × IPC × 周波数 → 性能
  - Hexagon NPU: TOPS スケーリング vs ダイ面積
  - モバイル電力バジェット: 8W TDP → 冷却制約 → スロットリング
  - 大量生産 COO: TSMC N3E × 億枚規模 → CPGD $100前後
  - mmWave 5G パスロス: Friis 伝搬 + 建物減衰
  - Qualcomm vs Apple Silicon: 価格性能比較

実行:
    python fem/qualcomm_model.py
    python fem/qualcomm_model.py --soc --chip SD8Elite
    python fem/qualcomm_model.py --5g --freq 28

参考文献:
    Qualcomm Snapdragon 8 Elite Architecture Guide (2024)
    Qualcomm Annual Report (2024)
    Friis (1946) Proc. IRE — 電波伝搬方程式
    Hennessy & Patterson (2019) — IPC パイプラインモデル
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
# 1. Snapdragon SoC スペック テーブル
# ════════════════════════════════════════════════════════════════════════════

SOC_SPECS = {
    "SD8_Elite": {
        "node": "TSMC N3E",
        "die_area_mm2": 107.0,
        "transistors_B": 20.0,
        "cpu_cluster": {"Prime": 2, "Perf": 6},
        "cpu_freq_GHz": {"Prime": 4.32, "Perf": 3.53},
        "gpu_tops_fp16": 45.0,
        "npu_tops_int8": 45.0,
        "modem": "5G X80",
        "TDP_W": 8.5,
        "price_oem_usd": 130.0,
        "color": "#d62728",
        "ref": "Qualcomm Snapdragon 8 Elite (2024)",
    },
    "SD8_Gen3": {
        "node": "TSMC N4P",
        "die_area_mm2": 115.0,
        "transistors_B": 16.0,
        "cpu_cluster": {"Prime": 1, "Perf": 5, "Eff": 2},
        "cpu_freq_GHz": {"Prime": 3.30, "Perf": 3.15, "Eff": 2.27},
        "gpu_tops_fp16": 34.0,
        "npu_tops_int8": 38.0,
        "modem": "5G X75",
        "TDP_W": 8.0,
        "price_oem_usd": 100.0,
        "color": "#2166ac",
        "ref": "Qualcomm Snapdragon 8 Gen 3 (2023)",
    },
    "SD_X_Elite": {
        "node": "TSMC N4",
        "die_area_mm2": 210.0,
        "transistors_B": 42.0,
        "cpu_cluster": {"Oryon": 12},
        "cpu_freq_GHz": {"Oryon": 4.0},
        "gpu_tops_fp16": 4600.0,
        "npu_tops_int8": 45.0,
        "modem": "Integrated 5G",
        "TDP_W": 23.0,
        "price_oem_usd": 200.0,
        "color": "#2ca02c",
        "ref": "Qualcomm Snapdragon X Elite (2024)",
    },
}

# Apple 競合比較
APPLE_COMPARE = {
    "A18_Pro": {
        "die_area_mm2": 91.0, "transistors_B": 19.0,
        "node": "TSMC N3E", "TDP_W": 8.0,
        "cpu_ipc_factor": 1.05,  # vs Oryon Zen-4 base
        "gpu_tops_fp16": 38.0, "npu_tops_int8": 38.0,
        "color": "#9467bd"},
    "M4": {
        "die_area_mm2": 308.0, "transistors_B": 28.0,
        "node": "TSMC N3E", "TDP_W": 28.0,
        "cpu_ipc_factor": 1.10,
        "gpu_tops_fp16": 140.0, "npu_tops_int8": 38.0,
        "color": "#555555"},
}

# ════════════════════════════════════════════════════════════════════════════
# 2. Core Physics / Performance Functions
# ════════════════════════════════════════════════════════════════════════════

def cpu_performance(soc: str = "SD8_Elite", workload: str = "single") -> dict:
    """
    CPU パフォーマンス: IPC × 周波数 × コア数。
    シングルスレッド: Prime コア × IPC × freq
    マルチスレッド: Σ (n_cores × IPC × freq) × efficiency_derating
    """
    p = SOC_SPECS.get(soc, SOC_SPECS["SD8_Elite"])
    IPC_base = 5.0   # Oryon / Cortex-X5 baseline IPC

    if workload == "single":
        cluster = max(p["cpu_cluster"].keys(), key=lambda k: p["cpu_freq_GHz"].get(k, 0))
        freq = p["cpu_freq_GHz"][cluster]
        perf = IPC_base * freq   # GIPS (Giga Instructions/s)
    else:
        perf = sum(n * IPC_base * p["cpu_freq_GHz"].get(cl, 2.0) * (0.85 if cl != "Prime" and cl != "Oryon" else 1.0)
                   for cl, n in p["cpu_cluster"].items())
    return {
        "soc": soc, "workload": workload,
        "performance_GIPS": round(perf, 1),
        "TDP_W": p["TDP_W"],
        "perf_per_watt": round(perf / max(p["TDP_W"], 0.1), 2),
    }


def thermal_throttling(TDP_W: float = 8.5, ambient_T_C: float = 35.0,
                        skin_T_limit_C: float = 43.0) -> dict:
    """
    モバイル スロットリング モデル:
    熱抵抗 R_th ≈ (T_skin - T_amb) / P_dissipate
    スロットリング開始: T_skin >= 43°C
    サステイン性能 = TDP_W × (43 - T_amb) / (T_junction_max - T_amb)
    """
    R_th_C_W = (skin_T_limit_C - ambient_T_C) / max(TDP_W, 0.1)
    T_junction = ambient_T_C + TDP_W * (R_th_C_W + 5.0)  # +5°C junction-skin
    sustained_W = (skin_T_limit_C - ambient_T_C) / (R_th_C_W + 5.0 / TDP_W)
    sustained_pct = sustained_W / max(TDP_W, 0.1) * 100
    return {
        "TDP_W": TDP_W, "ambient_T_C": ambient_T_C,
        "skin_T_limit_C": skin_T_limit_C,
        "R_th_C_W": round(R_th_C_W, 3),
        "T_junction_C": round(T_junction, 1),
        "sustained_W": round(sustained_W, 1),
        "sustained_pct": round(sustained_pct, 1),
        "throttling": T_junction > skin_T_limit_C + 20.0,
    }


def friis_5g_pathloss(freq_GHz: float = 28.0, distance_m: float = 100.0,
                       environment: str = "urban") -> dict:
    """
    Friis + 都市減衰: mmWave 5G パスロス。
    PL = 20log(4πdf/c) + alpha_env [dB]
    FSPL (Free Space Path Loss); alpha_env = 環境減衰係数 [dB/m]

    参考: Friis (1946); 3GPP TR 38.901
    """
    c = 3e8
    FSPL_dB = 20 * math.log10(4 * math.pi * distance_m * freq_GHz * 1e9 / c)
    env_attn = {"urban": 0.4, "suburban": 0.15, "indoor": 2.0, "LOS": 0.0}
    alpha = env_attn.get(environment, 0.4)
    total_PL_dB = FSPL_dB + alpha * distance_m
    rx_sensitivity_dBm = -90.0  # typical 5G receiver
    tx_power_dBm = 23.0          # typical phone TX
    link_budget_dB = tx_power_dBm - total_PL_dB - (-90.0)
    max_range_m = (10 ** ((tx_power_dBm - rx_sensitivity_dBm - 20 * math.log10(4 * math.pi * freq_GHz * 1e9 / c)) /
                          (20 + alpha * 10))) * 10 ** (20 / (20 + alpha * 10))

    return {
        "freq_GHz": freq_GHz, "distance_m": distance_m,
        "environment": environment,
        "FSPL_dB": round(FSPL_dB, 1),
        "total_PL_dB": round(total_PL_dB, 1),
        "link_budget_dB": round(link_budget_dB, 1),
        "connected": link_budget_dB > 0,
    }


def soc_mass_production_cost(soc: str = "SD8_Elite",
                              annual_volume_M: float = 200.0) -> dict:
    """
    大量生産 COO: ウェーハコスト + パッケージング + テスト
    ボリュームディスカウント: TSMC と長期契約 → wafer cost -15% at 200M/yr
    """
    p = SOC_SPECS.get(soc, SOC_SPECS["SD8_Elite"])
    D0 = 0.085; die_area = p["die_area_mm2"]
    wafer_cost_base = 20000.0  # N3E
    vol_discount = min(0.15, annual_volume_M / 1000.0 * 0.05)
    wafer_cost = wafer_cost_base * (1.0 - vol_discount)
    A_cm2 = die_area / 100.0
    val = (1.0 - math.exp(-D0 * A_cm2)) / max(D0 * A_cm2, 1e-12)
    Y = val ** 2
    R = 150.0
    dies_per_wafer = int((math.pi * R**2 - math.pi * R * math.sqrt(2 * die_area)) / die_area)
    good_dies = max(int(dies_per_wafer * Y), 1)
    die_cost = wafer_cost / good_dies
    pkg_cost  = 3.5   # FOWLP-PoP
    test_cost = 2.0
    total_CPGD = die_cost + pkg_cost + test_cost
    return {
        "soc": soc, "annual_volume_M": annual_volume_M,
        "vol_discount_pct": round(vol_discount * 100, 1),
        "die_cost_usd": round(die_cost, 2),
        "total_CPGD_usd": round(total_CPGD, 2),
        "OEM_price_usd": p["price_oem_usd"],
        "gross_margin_pct": round((p["price_oem_usd"] - total_CPGD) / p["price_oem_usd"] * 100, 1),
        "yield_pct": round(Y * 100, 1),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. Visualization
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(17, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    ax = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    # Panel 0: NPU TOPS vs ダイ面積
    socs = list(SOC_SPECS.keys()) + list(APPLE_COMPARE.keys())
    tops = [SOC_SPECS[s]["npu_tops_int8"] if s in SOC_SPECS
            else APPLE_COMPARE[s]["npu_tops_int8"] for s in socs]
    areas = [SOC_SPECS[s]["die_area_mm2"] if s in SOC_SPECS
             else APPLE_COMPARE[s]["die_area_mm2"] for s in socs]
    cols_ = [SOC_SPECS[s]["color"] if s in SOC_SPECS
             else APPLE_COMPARE[s]["color"] for s in socs]
    ax[0].scatter(areas, tops, c=cols_, s=120, zorder=5)
    for s, A, T in zip(socs, areas, tops):
        ax[0].annotate(s.replace("_", "\n"), (A, T), fontsize=7.5,
                       xytext=(5, 5), textcoords="offset points")
    ax[0].set_xlabel("ダイ面積 [mm²]"); ax[0].set_ylabel("NPU TOPS (INT8)")
    ax[0].set_title("NPU 性能 vs ダイ面積\n(Qualcomm Hexagon vs Apple Neural Engine)")
    ax[0].grid(alpha=0.25)

    # Panel 1: スロットリング — サステイン電力 vs 環境温度
    T_amb_arr = np.linspace(15, 45, 100)
    for TDP, col, lbl in [(8.5, "#d62728", "SD8Elite 8.5W"),
                           (8.0, "#2166ac", "SD8Gen3 8.0W"),
                           (23.0, "#2ca02c", "SDX Elite 23W")]:
        sust = [thermal_throttling(TDP, T)["sustained_pct"] for T in T_amb_arr]
        ax[1].plot(T_amb_arr, sust, color=col, lw=2.2, label=lbl)
    ax[1].axhline(70.0, color="k", ls="--", lw=1.5, label="70% サステイン目標")
    ax[1].set_xlabel("環境温度 [°C]"); ax[1].set_ylabel("サステイン性能 [%]")
    ax[1].set_title("モバイル スロットリング\n(スキン温度 43°C 制約)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)

    # Panel 2: 5G mmWave パスロス vs 距離
    d_arr = np.linspace(10, 500, 200)
    for freq, col, lbl in [(3.5, "#2166ac", "Sub-6GHz 3.5G"),
                            (28.0, "#d62728", "mmWave 28GHz"),
                            (60.0, "#ff7f0e", "mmWave 60GHz")]:
        pl = [friis_5g_pathloss(freq, d, "urban")["total_PL_dB"] for d in d_arr]
        ax[2].plot(d_arr, pl, color=col, lw=2.2, label=lbl)
    ax[2].axhline(113.0, color="k", ls="--", lw=1.5, label="113dB リンクバジェット")
    ax[2].set_xlabel("距離 [m]"); ax[2].set_ylabel("パスロス [dB]")
    ax[2].set_title("5G パスロス vs 距離\n(Friis + 都市減衰モデル)")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.25)

    # Panel 3: 大量生産 COO vs 生産量
    vol_arr = np.linspace(10, 500, 100)
    for soc, col in [(s, SOC_SPECS[s]["color"]) for s in SOC_SPECS]:
        cost = [soc_mass_production_cost(soc, v)["total_CPGD_usd"] for v in vol_arr]
        asp  = SOC_SPECS[soc]["price_oem_usd"]
        ax[3].plot(vol_arr, cost, color=col, lw=2.2,
                   label=f"{soc.replace('_',' ')} (ASP=${asp})")
    ax[3].set_xlabel("年間生産量 [M 個]"); ax[3].set_ylabel("CPGD [USD]")
    ax[3].set_title("Snapdragon 大量生産 COO\n(TSMC N3E, ボリュームディスカウント)")
    ax[3].legend(fontsize=8); ax[3].grid(alpha=0.25)
    ax[3].set_ylim(0, 150)

    # Panel 4: Qualcomm vs Apple 性能比較
    metrics = ["GPU TOPS\n(FP16÷10)", "NPU TOPS\n(INT8)", "TDP [W]", "ダイ面積\n[mm²÷5]"]
    qc_vals = [SOC_SPECS["SD8_Elite"]["gpu_tops_fp16"] / 10,
               SOC_SPECS["SD8_Elite"]["npu_tops_int8"],
               SOC_SPECS["SD8_Elite"]["TDP_W"],
               SOC_SPECS["SD8_Elite"]["die_area_mm2"] / 5]
    ap_vals = [APPLE_COMPARE["A18_Pro"]["gpu_tops_fp16"] / 10,
               APPLE_COMPARE["A18_Pro"]["npu_tops_int8"],
               APPLE_COMPARE["A18_Pro"]["TDP_W"],
               APPLE_COMPARE["A18_Pro"]["die_area_mm2"] / 5]
    x = np.arange(len(metrics)); w = 0.35
    ax[4].bar(x - w/2, qc_vals, w, color="#d62728", alpha=0.85, label="Qualcomm SD8 Elite")
    ax[4].bar(x + w/2, ap_vals, w, color="#555555",  alpha=0.85, label="Apple A18 Pro")
    ax[4].set_xticks(x); ax[4].set_xticklabels(metrics, fontsize=8)
    ax[4].set_title("Qualcomm vs Apple: SoC 性能比較\n(SD8 Elite vs A18 Pro)")
    ax[4].legend(fontsize=9); ax[4].grid(axis="y", alpha=0.25)

    # Panel 5: SoC スペック比較テーブル
    ax[5].axis("off")
    col_labels = ["SoC", "ノード", "面積\n[mm²]", "TDP\n[W]", "NPU\n[TOPS]", "OEM価格\n[USD]"]
    tdata = []
    for s, p in SOC_SPECS.items():
        tdata.append([s.replace("_", " "), p["node"],
                      f"{p['die_area_mm2']:.0f}", f"{p['TDP_W']:.1f}",
                      f"{p['npu_tops_int8']:.0f}", f"${p['price_oem_usd']}"])
    tbl = ax[5].table(cellText=tdata, colLabels=col_labels,
                      cellLoc="center", loc="center", bbox=[0.0, 0.20, 1.0, 0.75])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#003060"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#eef4ff")
    ax[5].set_title("Qualcomm Snapdragon スペック\nSD8 Elite · SD8 Gen3 · X Elite")

    fig.suptitle("Qualcomm — Snapdragon SoC / 5G モバイル モデル\n"
                 "CPU性能 · NPU · スロットリング · 5G パスロス · 大量生産 COO",
                 fontsize=12, fontweight="bold")
    out = os.path.join(out_dir, "qualcomm_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Qualcomm model → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Qualcomm Snapdragon SoC model")
    ap.add_argument("--soc",  action="store_true")
    ap.add_argument("--5g",   dest="mmwave", action="store_true")
    ap.add_argument("--chip", default="SD8_Elite", choices=list(SOC_SPECS.keys()))
    ap.add_argument("--freq", type=float, default=28.0, help="5G frequency [GHz]")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    if args.soc:
        r = soc_mass_production_cost(args.chip, 200.0)
        print(f"\n{args.chip}  CPGD=${r['total_CPGD_usd']:.2f}  "
              f"margin={r['gross_margin_pct']:.1f}%  yield={r['yield_pct']:.1f}%")
    if args.mmwave:
        for d in [50, 100, 200, 500]:
            r = friis_5g_pathloss(args.freq, d, "urban")
            print(f"  {args.freq}GHz  d={d:4d}m  PL={r['total_PL_dB']:.1f}dB  "
                  f"{'接続' if r['connected'] else '圏外'}")
    if not args.no_plot:
        plot_all()

if __name__ == "__main__":
    main()
