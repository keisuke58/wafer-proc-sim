"""
AMD — GPU (RDNA4/CDNA4) + CPU (Zen 5) チップレット モデル
==========================================================
NVIDIA と同じ TSMC / HBM サプライチェーンを使いながら差別化。
3D V-Cache 積層 (SRAM on CPU die)、CDNA4 AI アクセラレーター、
Zen 5 CPU (TSMC N3E) の物理・性能・コストモデル。

Key physics & economics:
  - RDNA4 GPU: クラスター構成 × AI パフォーマンス
  - CDNA4 (MI350X): HBM3E × roofline モデル (vs NVIDIA B200)
  - Zen 5 CPU: 命令レベル IPC モデル
  - 3D V-Cache: SRAM バンプ接続密度 × レイテンシ
  - チップレット設計: ダイコスト最適化 (小ダイ × 多数 vs 大ダイ)
  - AMD vs NVIDIA: Roofline + 価格性能比

実行:
    python fem/amd_model.py
    python fem/amd_model.py --gpu CDNA4
    python fem/amd_model.py --cpu Zen5

参考文献:
    AMD RDNA4 Architecture Guide (2025)
    AMD CDNA4 MI350X White Paper (2025)
    AMD Zen 5 Microarchitecture (2024)
    Williams et al. (2009) CACM — Roofline model
    Naffziger et al. (2021) ISSCC — 3D V-Cache
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
# 1. AMD 製品スペック テーブル
# ════════════════════════════════════════════════════════════════════════════

GPU_PRODUCTS = {
    "RX9070_XT": {
        "type": "RDNA4_gaming",
        "die_area_mm2": 356.0,
        "transistors_B": 53.9,
        "node": "TSMC N4P",
        "n_hbm": 0,
        "mem_type": "GDDR6",
        "mem_bw_GBs": 640.0,
        "tflops_fp16": 96.0,
        "TDP_W": 304.0,
        "price_usd": 599.0,
        "color": "#d62728",
        "ref": "AMD RX 9070 XT (2025)",
    },
    "RX9070": {
        "type": "RDNA4_gaming",
        "die_area_mm2": 356.0,
        "transistors_B": 53.9,
        "node": "TSMC N4P",
        "n_hbm": 0,
        "mem_type": "GDDR6",
        "mem_bw_GBs": 576.0,
        "tflops_fp16": 76.0,
        "TDP_W": 220.0,
        "price_usd": 449.0,
        "color": "#ff7f0e",
        "ref": "AMD RX 9070 (2025)",
    },
    "MI350X": {
        "type": "CDNA4_datacenter",
        "die_area_mm2": 800.0,
        "transistors_B": 153.0,
        "node": "TSMC N3E",
        "n_hbm": 8,
        "mem_type": "HBM3E",
        "mem_bw_GBs": 9830.0,
        "tflops_fp16": 4600.0,
        "TDP_W": 750.0,
        "price_usd": 20000.0,
        "color": "#2166ac",
        "ref": "AMD Instinct MI350X (2025)",
    },
    "MI300X": {
        "type": "CDNA3_datacenter",
        "die_area_mm2": 750.0,
        "transistors_B": 146.0,
        "node": "TSMC N5",
        "n_hbm": 8,
        "mem_type": "HBM3",
        "mem_bw_GBs": 5300.0,
        "tflops_fp16": 2610.0,
        "TDP_W": 750.0,
        "price_usd": 15000.0,
        "color": "#2ca02c",
        "ref": "AMD Instinct MI300X (2024)",
    },
}

CPU_PRODUCTS = {
    "Zen5_Desktop": {
        "cores": 16, "L3_MB": 32, "freq_GHz": 5.7,
        "IPC_vs_Zen4": 1.16, "node": "TSMC N3E",
        "die_area_mm2": 70.0, "TDP_W": 170.0,
        "vcache": False, "color": "#d62728",
        "ref": "AMD Ryzen 9000 Zen 5 (2024)"},
    "Zen5_VCache": {
        "cores": 16, "L3_MB": 96, "freq_GHz": 5.4,
        "IPC_vs_Zen4": 1.16, "node": "TSMC N3E",
        "die_area_mm2": 70.0 + 36.0, "TDP_W": 170.0,
        "vcache": True, "color": "#2166ac",
        "ref": "AMD Ryzen 9000 X3D Zen 5 (2025)"},
    "EPYC_Turin": {
        "cores": 192, "L3_MB": 384, "freq_GHz": 3.7,
        "IPC_vs_Zen4": 1.16, "node": "TSMC N3E",
        "die_area_mm2": 12 * 70.0, "TDP_W": 500.0,
        "vcache": False, "color": "#2ca02c",
        "ref": "AMD EPYC Turin (2024)"},
}

# ════════════════════════════════════════════════════════════════════════════
# 2. Core Physics / Performance Functions
# ════════════════════════════════════════════════════════════════════════════

def ai_roofline(gpu: str = "MI350X", arithmetic_intensity: float = 250.0) -> dict:
    """
    Roofline モデル: attainable TFLOPS = min(peak, BW × AI)
    Ridge point: AI* = peak / BW
    """
    p = GPU_PRODUCTS.get(gpu, GPU_PRODUCTS["MI350X"])
    ridge = p["tflops_fp16"] * 1e3 / p["mem_bw_GBs"]
    attainable = min(p["tflops_fp16"], p["mem_bw_GBs"] * arithmetic_intensity / 1e3)
    eff_pct = attainable / max(p["tflops_fp16"], 1e-9) * 100
    return {
        "gpu": gpu,
        "peak_tflops_fp16": p["tflops_fp16"],
        "mem_bw_GBs": p["mem_bw_GBs"],
        "ridge_AI_FLOP_B": round(ridge, 1),
        "arithmetic_intensity": arithmetic_intensity,
        "attainable_tflops": round(attainable, 1),
        "efficiency_pct": round(eff_pct, 1),
        "bound": "compute" if arithmetic_intensity >= ridge else "memory",
    }


def vcache_latency(sram_type: str = "SRAM_3D", capacity_MB: float = 64.0) -> dict:
    """
    3D V-Cache SRAM スタック: ハイブリッドボンドバンプ密度 vs レイテンシ。
    bump_density: ~9µm ピッチ → ~10,000 bumps/mm²
    SRAM レイテンシ: 4MB L3 ≈ 40ns; 3D V-Cache 追加 ≈ +3ns (フリップ遅延)
    """
    bump_pitch_um = 9.0
    bump_density_per_mm2 = 1e6 / bump_pitch_um ** 2
    L3_latency_ns = 40.0 + math.log2(capacity_MB / 4.0) * 5.0
    vcache_penalty_ns = 3.0   # hybrid bond + TSV
    total_latency_ns = L3_latency_ns + vcache_penalty_ns
    gaming_perf_gain_pct = min(capacity_MB / 8.0 * 3.0, 30.0)  # cache-limited games
    return {
        "sram_type": sram_type,
        "capacity_MB": capacity_MB,
        "bump_pitch_um": bump_pitch_um,
        "bump_density_per_mm2": round(bump_density_per_mm2, 0),
        "L3_latency_ns": round(total_latency_ns, 1),
        "gaming_perf_gain_pct": round(gaming_perf_gain_pct, 1),
    }


def chiplet_cost_optimization(target_transistors_B: float = 100.0,
                               node: str = "TSMC N3E",
                               n_chiplets: int = 4) -> dict:
    """
    チップレット分割コスト最適化:
    monolithic vs multi-chiplet の Murphy 歩留まり × 製造コスト比較。

    D0_N3E = 0.085 /cm²
    Monolithic: 大ダイ → 低歩留まり
    Chiplet: 小ダイ × n_chiplets → 高歩留まり + 追加接続コスト
    """
    D0 = 0.085; wafer_cost = 20000.0
    # Transistor density TSMC N3E
    density_MT_mm2 = 167.0
    mono_area_mm2 = target_transistors_B * 1000.0 / density_MT_mm2
    chiplet_area_mm2 = mono_area_mm2 / n_chiplets
    def murphy_yield(A_mm2, D0):
        A_cm2 = A_mm2 / 100.0
        val = (1 - math.exp(-D0 * A_cm2)) / max(D0 * A_cm2, 1e-12)
        return val ** 2
    Y_mono   = murphy_yield(mono_area_mm2, D0)
    Y_chiplet = murphy_yield(chiplet_area_mm2, D0) ** n_chiplets
    R = 150.0
    dies_mono  = int((math.pi*R**2 - math.pi*R*math.sqrt(2*mono_area_mm2)) / mono_area_mm2)
    dies_chip  = int((math.pi*R**2 - math.pi*R*math.sqrt(2*chiplet_area_mm2)) / chiplet_area_mm2)
    cpgd_mono   = wafer_cost / max(dies_mono * Y_mono, 1)
    intercon_cost = 50.0 * n_chiplets  # packaging interconnect overhead
    cpgd_chiplet  = wafer_cost / max(dies_chip * murphy_yield(chiplet_area_mm2, D0), 1) * n_chiplets + intercon_cost
    return {
        "target_transistors_B": target_transistors_B,
        "n_chiplets": n_chiplets,
        "monolithic_area_mm2": round(mono_area_mm2, 1),
        "chiplet_area_mm2": round(chiplet_area_mm2, 1),
        "monolithic_yield_pct": round(Y_mono * 100, 1),
        "chiplet_yield_pct": round(Y_chiplet * 100, 1),
        "CPGD_monolithic_usd": round(cpgd_mono, 0),
        "CPGD_chiplet_usd": round(cpgd_chiplet, 0),
        "chiplet_wins": cpgd_chiplet < cpgd_mono,
    }


def amd_vs_nvidia(ai_intensity: float = 250.0) -> dict:
    """AMD MI350X vs NVIDIA B200: roofline + price/perf 比較"""
    amd = GPU_PRODUCTS["MI350X"]
    # NVIDIA B200 specs
    nv = {"tflops_fp16": 9000.0, "mem_bw_GBs": 8 * 1228.0, "price_usd": 40000.0,
          "TDP_W": 1000.0, "die_area_mm2": 1040.0}
    amd_rl  = ai_roofline("MI350X", ai_intensity)
    nv_ridge = nv["tflops_fp16"] * 1e3 / nv["mem_bw_GBs"]
    nv_att   = min(nv["tflops_fp16"], nv["mem_bw_GBs"] * ai_intensity / 1e3)
    return {
        "AMD_MI350X": {
            "attainable_tflops": amd_rl["attainable_tflops"],
            "price_usd": amd["price_usd"],
            "perf_per_dollar": round(amd_rl["attainable_tflops"] / amd["price_usd"], 4),
            "perf_per_watt": round(amd_rl["attainable_tflops"] / amd["TDP_W"], 4),
        },
        "NVIDIA_B200": {
            "attainable_tflops": round(nv_att, 1),
            "price_usd": nv["price_usd"],
            "perf_per_dollar": round(nv_att / nv["price_usd"], 4),
            "perf_per_watt": round(nv_att / nv["TDP_W"], 4),
        },
        "AMD_advantage_pct": round((amd_rl["attainable_tflops"] / max(nv_att, 1) - 1) * 100, 1),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. Visualization
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(17, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    ax = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    AI_arr = np.logspace(0, 4, 300)

    # Panel 0: Roofline — AMD vs NVIDIA
    for gpu_n, col, lbl in [
        ("MI350X",  "#2166ac", "AMD MI350X"),
        ("MI300X",  "#2ca02c", "AMD MI300X"),
        ("RX9070_XT","#d62728","AMD RX9070XT (gaming)"),
    ]:
        p = GPU_PRODUCTS[gpu_n]
        perf = [min(p["tflops_fp16"], p["mem_bw_GBs"] * ai / 1e3) for ai in AI_arr]
        ax[0].loglog(AI_arr, perf, color=col, lw=2.2, label=lbl)
    # NVIDIA B200 reference
    nv_bw = 8 * 1228.0
    nv_pk = 9000.0
    nv_p  = [min(nv_pk, nv_bw * ai / 1e3) for ai in AI_arr]
    ax[0].loglog(AI_arr, nv_p, "k--", lw=2.0, label="NVIDIA B200 (ref)")
    ax[0].set_xlabel("Arithmetic Intensity [FLOP/byte]")
    ax[0].set_ylabel("Performance [TFLOPS FP16]")
    ax[0].set_title("AMD GPU Roofline vs NVIDIA B200\nMI350X · MI300X · RX9070XT")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25, which="both")

    # Panel 1: チップレット分割コスト最適化
    n_chip_arr = [1, 2, 4, 6, 8, 12]
    for target_B, col in [(50, "#2166ac"), (100, "#d62728"), (200, "#2ca02c")]:
        cpgd_chip  = [chiplet_cost_optimization(target_B, n_chiplets=n)["CPGD_chiplet_usd"] for n in n_chip_arr]
        cpgd_mono  = chiplet_cost_optimization(target_B, n_chiplets=1)["CPGD_monolithic_usd"]
        ax[1].plot(n_chip_arr, cpgd_chip, "o-", color=col, lw=2.2, ms=7,
                   label=f"{target_B}B トランジスタ")
        ax[1].axhline(cpgd_mono, color=col, ls=":", lw=1.0)
    ax[1].set_xlabel("チップレット分割数"); ax[1].set_ylabel("CPGD [USD]")
    ax[1].set_title("チップレット分割コスト最適化\n(TSMC N3E, 点線=モノリシック)")
    ax[1].legend(fontsize=8.5); ax[1].grid(alpha=0.25)

    # Panel 2: 3D V-Cache レイテンシ vs 容量
    cap_arr = np.linspace(4, 128, 100)
    for has_vcache, col, lbl in [(False, "#d62728", "通常 L3"), (True, "#2166ac", "3D V-Cache")]:
        lat = [vcache_latency(capacity_MB=c)["L3_latency_ns"] +
               (3.0 if has_vcache else 0.0) for c in cap_arr]
        ax[2].plot(cap_arr, lat, color=col, lw=2.2, label=lbl)
    ax[2].set_xlabel("L3 容量 [MB]"); ax[2].set_ylabel("レイテンシ [ns]")
    ax[2].set_title("3D V-Cache レイテンシ vs キャッシュ容量\n(ハイブリッドボンド +3ns)")
    ax[2].legend(fontsize=9); ax[2].grid(alpha=0.25)

    # Panel 3: AMD vs NVIDIA 価格性能比
    comp = amd_vs_nvidia(250.0)
    categories = ["TFLOPS\n(attainable)", "TFLOPS/$\n(×1000)", "TFLOPS/W\n(×100)"]
    amd_vals = [comp["AMD_MI350X"]["attainable_tflops"],
                comp["AMD_MI350X"]["perf_per_dollar"] * 1000,
                comp["AMD_MI350X"]["perf_per_watt"] * 100]
    nv_vals  = [comp["NVIDIA_B200"]["attainable_tflops"],
                comp["NVIDIA_B200"]["perf_per_dollar"] * 1000,
                comp["NVIDIA_B200"]["perf_per_watt"] * 100]
    x = np.arange(len(categories)); w = 0.35
    ax[3].bar(x - w/2, amd_vals, w, color="#d62728", alpha=0.85, label="AMD MI350X")
    ax[3].bar(x + w/2, nv_vals,  w, color="#76b900", alpha=0.85, label="NVIDIA B200")
    ax[3].set_xticks(x); ax[3].set_xticklabels(categories, fontsize=9)
    ax[3].set_title("AMD MI350X vs NVIDIA B200\n価格性能比 (AI=250 FLOP/B)")
    ax[3].legend(fontsize=9); ax[3].grid(axis="y", alpha=0.25)

    # Panel 4: GPU ロードマップ TFLOPS
    products = list(GPU_PRODUCTS.keys())
    tflops = [GPU_PRODUCTS[p]["tflops_fp16"] for p in products]
    prices = [GPU_PRODUCTS[p]["price_usd"] for p in products]
    colors_p = [GPU_PRODUCTS[p]["color"] for p in products]
    ax4b = ax[4].twinx()
    ax[4].bar(range(len(products)), tflops, color=colors_p, width=0.6, alpha=0.85)
    ax4b.plot(range(len(products)), prices, "D--", color="#333333", ms=8, lw=1.8)
    ax[4].set_xticks(range(len(products)))
    ax[4].set_xticklabels([p.replace("_", "\n") for p in products], fontsize=8)
    ax[4].set_ylabel("TFLOPS FP16")
    ax4b.set_ylabel("価格 [USD]")
    ax[4].set_title("AMD GPU 製品比較\nRDNA4 Gaming · CDNA4/3 Datacenter")
    ax[4].grid(axis="y", alpha=0.25)

    # Panel 5: CPU + GPU 仕様比較テーブル
    ax[5].axis("off")
    col_labels = ["製品", "ノード", "TFLOPS\n/コア数", "価格\n[USD]", "TDP\n[W]", "特徴"]
    tdata = []
    for gn, gp in GPU_PRODUCTS.items():
        tdata.append([gn.replace("_", "\n"), gp["node"],
                      f"{gp['tflops_fp16']:.0f}", f"${gp['price_usd']:,}",
                      f"{gp['TDP_W']:.0f}", gp["mem_type"]])
    tbl = ax[5].table(cellText=tdata, colLabels=col_labels,
                      cellLoc="center", loc="center", bbox=[0.0, 0.08, 1.0, 0.88])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#8b0000"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#fff0f0")
    ax[5].set_title("AMD GPU 製品スペック\nRX9070XT · RX9070 · MI350X · MI300X")

    fig.suptitle("AMD — GPU (RDNA4/CDNA4) + CPU Zen 5 チップレット モデル\n"
                 "Roofline · チップレット分割 · 3D V-Cache · vs NVIDIA 比較",
                 fontsize=12, fontweight="bold")
    out = os.path.join(out_dir, "amd_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] AMD model → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="AMD GPU/CPU chiplet model")
    ap.add_argument("--gpu",  default="MI350X", choices=list(GPU_PRODUCTS.keys()))
    ap.add_argument("--cpu",  default="Zen5_Desktop", choices=list(CPU_PRODUCTS.keys()))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    if not args.no_plot:
        plot_all()
    else:
        r = ai_roofline(args.gpu, 250.0)
        print(f"\n{args.gpu} Roofline (AI=250): {r['attainable_tflops']} TFLOPS "
              f"({r['bound']}-bound)")

if __name__ == "__main__":
    main()
