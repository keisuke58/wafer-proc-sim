"""
NVIDIA GPU — Die Architecture / HBM Integration / CoWoS / Roofline Model
==========================================================
Covers H100 SXM (Hopper), B200 (Blackwell), GB200 NVL72 system.

Key physics:
  - Die area vs. transistor density → Murphy yield
  - HBM3/HBM3E effective bandwidth (bus width × frequency × efficiency)
  - CoWoS interposer area estimation (RDL layers, die placement)
  - Thermal: TDP → junction temperature via spreader model
  - AI roofline model: compute-bound vs. memory-bandwidth-bound analysis
  - Supply chain: equipment cost contribution per GPU (Disco/TEL/Lasertec/Advantest)

実行:
    python fem/nvidia_gpu_model.py
    python fem/nvidia_gpu_model.py --gpu H100
    python fem/nvidia_gpu_model.py --roofline --gpu B200

参考文献:
    NVIDIA H100 White Paper (2022) — architecture specs
    NVIDIA B200 Architecture Overview (2024)
    Tsai et al. (2022) ISSCC — CoWoS interposer area model
    Williams et al. (2009) CACM — Roofline model
    Murphy (1964) Proc. IEEE — die yield estimation
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
# 1. GPU Specification Tables
# ════════════════════════════════════════════════════════════════════════════

GPU_SPECS = {
    "H100_SXM": {
        "transistors_B": 80.0,
        "die_area_mm2": 814.0,
        "node": "TSMC 4N",
        "node_nm": 4.0,
        "density_MT_mm2": 98.3,
        "n_hbm_stacks": 6,
        "hbm_type": "HBM3",
        "hbm_bus_width": 1024,
        "hbm_freq_GHz": 3.35,    # 6.7 Gbps per pin → 3.35 GHz DDR
        "hbm_capacity_GB": 80.0,
        "TDP_W": 700.0,
        "tflops_fp16": 1979.0,
        "price_usd": 30000.0,
        "color": "#2166ac",
        "ref": "NVIDIA H100 White Paper (2022)",
    },
    "B200": {
        "transistors_B": 208.0,
        "die_area_mm2": 1040.0,
        "node": "TSMC 4NP",
        "node_nm": 4.0,
        "density_MT_mm2": 200.0,
        "n_hbm_stacks": 8,
        "hbm_type": "HBM3E",
        "hbm_bus_width": 1024,
        "hbm_freq_GHz": 4.8,     # 9.6 Gbps → 4.8 GHz DDR
        "hbm_capacity_GB": 192.0,
        "TDP_W": 1000.0,
        "tflops_fp16": 9000.0,
        "price_usd": 40000.0,
        "color": "#d62728",
        "ref": "NVIDIA Blackwell Architecture Overview (2024)",
    },
    "GB200_NVL72": {
        "transistors_B": 208.0,   # per GPU die
        "die_area_mm2": 1040.0,
        "node": "TSMC 4NP",
        "node_nm": 4.0,
        "density_MT_mm2": 200.0,
        "n_hbm_stacks": 8,
        "hbm_type": "HBM3E",
        "hbm_bus_width": 1024,
        "hbm_freq_GHz": 4.8,
        "hbm_capacity_GB": 192.0,
        "TDP_W": 1200.0,          # per GPU in NVL72 rack
        "tflops_fp16": 18000.0,   # rack-level
        "price_usd": 300000.0,    # NVL72 rack system
        "color": "#2ca02c",
        "ref": "NVIDIA GB200 NVL72 System Brief (2024)",
    },
}

# Equipment cost contribution to each GPU (~% of wafer cost from equipment depreciation)
EQUIPMENT_CONTRIB = {
    "Disco":     {"cost_pct_die": 2.1, "role": "Dicing/Grinding"},
    "TEL":       {"cost_pct_die": 3.8, "role": "CVD/ALD/Clean"},
    "Lasertec":  {"cost_pct_die": 1.5, "role": "Inspection"},
    "Advantest": {"cost_pct_die": 4.2, "role": "ATE Testing"},
    "ASML":      {"cost_pct_die": 12.5, "role": "Lithography"},
    "Lam":       {"cost_pct_die": 4.0, "role": "Etch"},
    "AMAT":      {"cost_pct_die": 5.5, "role": "CVD/CMP/Implant"},
}

# ════════════════════════════════════════════════════════════════════════════
# 2. Core Physics Functions
# ════════════════════════════════════════════════════════════════════════════

def die_area_model(transistors_B: float, density_MT_mm2: float,
                   D0_per_cm2: float = 0.10) -> dict:
    """
    Die area from transistor count + density → Murphy yield.

    A = N_transistors / density
    Y = [(1 - exp(-D0*A)) / (D0*A)]²
    """
    A_mm2 = transistors_B * 1000.0 / density_MT_mm2   # MT/mm² * mm² = MT
    A_cm2 = A_mm2 / 100.0
    D0 = D0_per_cm2
    val = (1.0 - math.exp(-D0 * A_cm2)) / max(D0 * A_cm2, 1e-12)
    Y = val ** 2

    # Dies per 300mm wafer (Nikon formula)
    R = 150.0  # mm
    dies_total = int((math.pi * R ** 2 - math.pi * R * math.sqrt(2 * A_mm2)) / A_mm2)
    good_dies = int(dies_total * Y)

    return {
        "transistors_B": transistors_B,
        "density_MT_mm2": density_MT_mm2,
        "die_area_mm2": round(A_mm2, 1),
        "D0_per_cm2": D0,
        "yield_pct": round(Y * 100, 1),
        "dies_per_300mm": dies_total,
        "good_dies_per_wafer": good_dies,
    }


def hbm_bandwidth(n_stacks: int, bus_width: int = 1024,
                  freq_GHz: float = 3.35, efficiency: float = 0.88) -> dict:
    """
    Effective HBM bandwidth per GPU.

    BW = n_stacks * bus_width * freq_GHz * 2 (DDR) * efficiency / 8  [GB/s]
    """
    BW_Gbps = n_stacks * bus_width * freq_GHz * 2.0 * efficiency
    BW_GBs  = BW_Gbps / 8.0
    BW_TBs  = BW_GBs / 1000.0

    return {
        "n_stacks": n_stacks,
        "bus_width_bits": bus_width,
        "freq_GHz": freq_GHz,
        "efficiency": efficiency,
        "BW_Gbps": round(BW_Gbps, 1),
        "BW_GBs": round(BW_GBs, 1),
        "BW_TBs": round(BW_TBs, 3),
    }


def cowos_interposer(die_count: int, die_area_mm2: float,
                     hbm_stacks: int = 6, rdl_layers: int = 3) -> dict:
    """
    CoWoS interposer area + RDL routing layers.

    Interposer A ≈ (sum die areas + HBM areas) * 1.2 (routing overhead)
    HBM footprint ≈ 104 mm² per stack (12-die × ~8.7 mm²)

    参考: Tsai et al. (2022) ISSCC — CoWoS area budgeting
    """
    HBM_FOOTPRINT_MM2 = 104.0   # per stack (HBM3E 12Hi)
    routing_overhead = 1.20
    A_dies = die_count * die_area_mm2
    A_hbm  = hbm_stacks * HBM_FOOTPRINT_MM2
    A_interposer = (A_dies + A_hbm) * routing_overhead
    rdl_cost_relative = 1.0 + 0.15 * (rdl_layers - 3)  # cost scales with layers

    return {
        "die_count": die_count,
        "die_area_mm2": die_area_mm2,
        "hbm_stacks": hbm_stacks,
        "rdl_layers": rdl_layers,
        "A_interposer_mm2": round(A_interposer, 0),
        "A_interposer_cm2": round(A_interposer / 100.0, 2),
        "rdl_cost_factor": round(rdl_cost_relative, 2),
    }


def tdp_to_junction(TDP_W: float, A_die_mm2: float,
                    TIM_k_W_mK: float = 8.0, spreader_t_mm: float = 3.0) -> dict:
    """
    T_junction from TDP via vapor chamber / spreader thermal model.

    T_j = T_water + R_junction_case * TDP
    R_jc = 1 / (h_spreader * A_die)  where h_spreader ≈ k/t
    """
    T_water_C = 35.0   # water-cooled inlet
    A_m2 = A_die_mm2 * 1e-6
    h_spreader = TIM_k_W_mK / (spreader_t_mm * 1e-3)   # [W/m²·K]
    R_jc = 1.0 / (h_spreader * A_m2)                    # [°C/W]
    T_j  = T_water_C + R_jc * TDP_W
    power_density_W_mm2 = TDP_W / A_die_mm2

    return {
        "TDP_W": TDP_W,
        "A_die_mm2": A_die_mm2,
        "R_jc_C_W": round(R_jc, 4),
        "T_junction_C": round(T_j, 1),
        "power_density_W_mm2": round(power_density_W_mm2, 2),
        "thermal_ok": T_j < 85.0,
    }


def ai_roofline(tflops: float, hbm_bw_GBs: float,
                arithmetic_intensity: float = 100.0) -> dict:
    """
    Roofline model: peak achievable performance.

    Attainable TFLOPS = min(peak_TFLOPS, BW_GBs × AI)
    Ridge point AI* = peak_TFLOPS / BW_GBs  [FLOP/byte]

    参考: Williams et al. (2009) Commun. ACM — Roofline model
    """
    ridge_AI = tflops * 1000.0 / hbm_bw_GBs   # FLOP/byte
    attainable_tflops = min(tflops, hbm_bw_GBs * arithmetic_intensity / 1000.0)
    efficiency_pct = attainable_tflops / max(tflops, 1.0) * 100.0
    bound = "compute" if arithmetic_intensity >= ridge_AI else "memory"

    return {
        "peak_tflops": tflops,
        "hbm_bw_GBs": hbm_bw_GBs,
        "arithmetic_intensity_FLOP_B": arithmetic_intensity,
        "ridge_AI_FLOP_B": round(ridge_AI, 1),
        "attainable_tflops": round(attainable_tflops, 1),
        "efficiency_pct": round(efficiency_pct, 1),
        "bound": bound,
    }


def gpu_supply_chain(gpu_name: str = "H100_SXM") -> dict:
    """Return equipment USD cost contribution embedded in GPU BOM."""
    spec = GPU_SPECS.get(gpu_name, GPU_SPECS["H100_SXM"])
    # Approximate wafer cost for GPU die
    wafer_cost_usd = 20000.0
    die_cost = wafer_cost_usd / max(die_area_model(
        spec["transistors_B"], spec["density_MT_mm2"])["good_dies_per_wafer"], 1)
    equip_usd = {k: round(die_cost * v["cost_pct_die"] / 100.0, 2)
                 for k, v in EQUIPMENT_CONTRIB.items()}
    total_equip = sum(equip_usd.values())
    return {
        "gpu": gpu_name,
        "die_cost_usd": round(die_cost, 2),
        "equipment_usd": equip_usd,
        "total_equipment_usd": round(total_equip, 2),
        "equip_pct_die_cost": round(total_equip / max(die_cost, 1.0) * 100.0, 1),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. Visualization
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(17, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    ax = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    # ── Panel 0: Die area vs. transistor count (yield contours) ──────────
    N_arr = np.linspace(10, 300, 200)
    for density, col, lbl in [(100, "#2166ac", "100 MT/mm² (H100)"),
                               (200, "#d62728", "200 MT/mm² (B200)"),
                               (300, "#2ca02c", "300 MT/mm² (future)")]:
        A_arr = N_arr * 1000.0 / density
        ax[0].plot(N_arr, A_arr, color=col, lw=2.2, label=lbl)
    for gpu_n, spec in GPU_SPECS.items():
        ax[0].scatter([spec["transistors_B"]], [spec["die_area_mm2"]],
                      color=spec["color"], s=80, zorder=5)
        ax[0].annotate(gpu_n.replace("_", " "), (spec["transistors_B"], spec["die_area_mm2"]),
                       fontsize=7.5, xytext=(5, -12), textcoords="offset points")
    ax[0].set_xlabel("Transistors [B]")
    ax[0].set_ylabel("Die Area [mm²]")
    ax[0].set_title("Die Area vs. Transistor Count\n(NVIDIA GPU roadmap, density scaling)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.25)

    # ── Panel 1: HBM bandwidth vs. stack count ────────────────────────────
    stacks_arr = np.arange(1, 13)
    for hbm_type, freq, col in [("HBM3", 3.35, "#2166ac"), ("HBM3E", 4.8, "#d62728"),
                                  ("HBM4*", 6.0, "#2ca02c")]:
        bw = [hbm_bandwidth(n, 1024, freq)["BW_TBs"] for n in stacks_arr]
        ax[1].plot(stacks_arr, bw, "o-", color=col, lw=2.2, ms=6, label=hbm_type)
    ax[1].set_xlabel("HBM Stack Count")
    ax[1].set_ylabel("Bandwidth [TB/s]")
    ax[1].set_title("HBM Bandwidth vs. Stack Count\n(1024-bit bus, 88% efficiency)")
    ax[1].legend(fontsize=8.5)
    ax[1].grid(alpha=0.25)
    ax[1].set_xticks(stacks_arr)

    # ── Panel 2: CoWoS interposer area breakdown ──────────────────────────
    gpus = list(GPU_SPECS.keys())
    cos_areas = []
    die_areas = []
    hbm_areas_list = []
    for g in gpus:
        s = GPU_SPECS[g]
        cos = cowos_interposer(1, s["die_area_mm2"], s["n_hbm_stacks"])
        cos_areas.append(cos["A_interposer_mm2"])
        die_areas.append(s["die_area_mm2"])
        hbm_areas_list.append(s["n_hbm_stacks"] * 104.0)
    x = np.arange(len(gpus))
    w = 0.35
    ax[2].bar(x - w/2, die_areas, w, label="GPU die", color="#2166ac", alpha=0.85)
    ax[2].bar(x + w/2, hbm_areas_list, w, label="HBM footprint", color="#d62728", alpha=0.85)
    ax[2].plot(x, cos_areas, "D--", color="#2ca02c", lw=2.0, ms=8, label="CoWoS total")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels([g.replace("_", "\n") for g in gpus], fontsize=8)
    ax[2].set_ylabel("Area [mm²]")
    ax[2].set_title("CoWoS Interposer Area\n(Die + HBM × 1.2 routing overhead)")
    ax[2].legend(fontsize=8)
    ax[2].grid(axis="y", alpha=0.25)

    # ── Panel 3: Roofline model ───────────────────────────────────────────
    AI_arr = np.logspace(0, 4, 300)
    for gpu_n, col in [("H100_SXM", "#2166ac"), ("B200", "#d62728"), ("GB200_NVL72", "#2ca02c")]:
        s = GPU_SPECS[gpu_n]
        bw_obj = hbm_bandwidth(s["n_hbm_stacks"], s["hbm_bus_width"], s["hbm_freq_GHz"])
        BW = bw_obj["BW_GBs"]
        TFLOPS = s["tflops_fp16"]
        ridge = TFLOPS * 1000.0 / BW
        perf = [min(TFLOPS, BW * ai / 1000.0) for ai in AI_arr]
        ax[3].loglog(AI_arr, perf, color=col, lw=2.2, label=gpu_n.replace("_", " "))
        ax[3].axvline(ridge, color=col, ls=":", lw=1.0)
    ax[3].set_xlabel("Arithmetic Intensity [FLOP/byte]")
    ax[3].set_ylabel("Performance [TFLOPS FP16]")
    ax[3].set_title("AI Roofline Model\n(H100 · B200 · GB200 NVL72)")
    ax[3].legend(fontsize=8)
    ax[3].grid(alpha=0.25, which="both")

    # ── Panel 4: Thermal — T_junction vs. TDP ────────────────────────────
    TDP_arr = np.linspace(100, 1500, 200)
    for gpu_n, col in [("H100_SXM", "#2166ac"), ("B200", "#d62728"), ("GB200_NVL72", "#2ca02c")]:
        A = GPU_SPECS[gpu_n]["die_area_mm2"]
        Tj = [tdp_to_junction(T, A)["T_junction_C"] for T in TDP_arr]
        ax[4].plot(TDP_arr, Tj, color=col, lw=2.2, label=gpu_n.replace("_", " "))
    ax[4].axhline(85.0, color="k", ls="--", lw=1.2, label="85°C spec")
    ax[4].set_xlabel("TDP [W]")
    ax[4].set_ylabel("T_junction [°C]")
    ax[4].set_title("Die Junction Temperature vs. TDP\n(Water-cooled, TIM k=8 W/m·K)")
    ax[4].legend(fontsize=8)
    ax[4].grid(alpha=0.25)

    # ── Panel 5: Supply chain equipment cost table ────────────────────────
    ax[5].axis("off")
    col_labels = ["GPU", "Die Cost\n[USD]", "ASML\n[USD]", "AMAT\n[USD]",
                  "Lam\n[USD]", "Adv.\n[USD]", "Total\n[USD]"]
    tdata = []
    for gpu_n in GPU_SPECS.keys():
        sc = gpu_supply_chain(gpu_n)
        tdata.append([gpu_n.replace("_", " "),
                      f"${sc['die_cost_usd']:.0f}",
                      f"${sc['equipment_usd']['ASML']:.0f}",
                      f"${sc['equipment_usd']['AMAT']:.0f}",
                      f"${sc['equipment_usd']['Lam']:.0f}",
                      f"${sc['equipment_usd']['Advantest']:.0f}",
                      f"${sc['total_equipment_usd']:.0f}"])
    tbl = ax[5].table(cellText=tdata, colLabels=col_labels,
                      cellLoc="center", loc="center",
                      bbox=[0.0, 0.15, 1.0, 0.80])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1a1a2e")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4ff")
    ax[5].set_title("NVIDIA GPU Supply Chain\nEquipment Cost per Die")

    fig.suptitle(
        "NVIDIA GPU — Die Architecture / HBM / CoWoS / AI Roofline\n"
        "H100 SXM (Hopper) · B200 (Blackwell) · GB200 NVL72",
        fontsize=12, fontweight="bold",
    )
    out = os.path.join(out_dir, "nvidia_gpu.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] NVIDIA GPU model → {out}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="NVIDIA GPU die / HBM / roofline model")
    ap.add_argument("--gpu",      default="H100_SXM", choices=list(GPU_SPECS.keys()))
    ap.add_argument("--roofline", action="store_true", help="Print roofline analysis")
    ap.add_argument("--supply",   action="store_true", help="Print supply chain costs")
    ap.add_argument("--no-plot",  action="store_true")
    args = ap.parse_args()

    if args.roofline:
        s = GPU_SPECS[args.gpu]
        bw = hbm_bandwidth(s["n_hbm_stacks"], s["hbm_bus_width"], s["hbm_freq_GHz"])
        print(f"\nRoofline: {args.gpu}")
        print("=" * 40)
        for AI in [10, 50, 100, 250, 1000]:
            r = ai_roofline(s["tflops_fp16"], bw["BW_GBs"], AI)
            print(f"  AI={AI:5d} FLOP/B  att={r['attainable_tflops']:8.1f} TFLOPS  "
                  f"eff={r['efficiency_pct']:5.1f}%  {r['bound']}-bound")

    if args.supply:
        sc = gpu_supply_chain(args.gpu)
        print(f"\nSupply Chain: {args.gpu}")
        print("=" * 40)
        for k, v in sc["equipment_usd"].items():
            role = EQUIPMENT_CONTRIB[k]["role"]
            print(f"  {k:<12} ${v:6.2f}  ({role})")
        print(f"  {'Total':<12} ${sc['total_equipment_usd']:6.2f}")

    if not args.no_plot:
        plot_all()


if __name__ == "__main__":
    main()
