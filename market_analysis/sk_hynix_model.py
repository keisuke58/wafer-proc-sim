"""
SK Hynix — HBM3E / HBM4 / DRAM Scaling Model
==========================================================
Covers HBM3E 12Hi (1.2 TB/s), HBM4 16Hi projection, and LPDDR5X DRAM.

Key physics:
  - HBM stack height: die-by-die thinning + bump + mold compound
  - TSV resistance + capacitance: via RC delay per stack level
  - HBM3E aggregate bandwidth: 1024-bit × 9.6 Gbps × n_stacks
  - DRAM cell capacitor scaling: stacked STC vs. node
  - Refresh interval: tREFI from cell leakage current vs. capacitance
  - HBM4 roadmap projection: bandwidth / capacity 2025–2028

実行:
    python fem/sk_hynix_model.py
    python fem/sk_hynix_model.py --hbm --product HBM3E_12Hi
    python fem/sk_hynix_model.py --dram --node 1a

参考文献:
    Oh et al. (2023) ISSCC — SK Hynix HBM3E 12Hi 1.2 TB/s
    Kim et al. (2022) IEDM — DRAM cell capacitor scaling
    Lee et al. (2021) ISSCC — LPDDR5X memory interface
    Park et al. (2022) ECTC — HBM TSV reliability
    SK Hynix HBM4 Technology Roadmap (2024)
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

# Physical constants
eps0 = 8.854e-12
rho_W   = 5.3e-8    # W resistivity [Ω·m]
rho_Cu  = 1.7e-8    # Cu resistivity [Ω·m]
eps_Si  = 11.7 * eps0

# ════════════════════════════════════════════════════════════════════════════
# 1. HBM Product & DRAM Node Parameter Tables
# ════════════════════════════════════════════════════════════════════════════

HBM_PRODUCTS = {
    "HBM2E": {
        "generation": 2.5,
        "n_die_max": 8,
        "n_io": 1024,
        "freq_Gbps": 3.6,
        "die_t_um": 50.0,
        "bump_t_um": 20.0,
        "mold_t_um": 100.0,
        "tsv_d_nm": 8000.0,
        "tsv_pitch_um": 40.0,
        "n_tsv_per_die": 5120,
        "capacity_GB_per_stack": 16.0,
        "color": "#ff7f0e",
        "ref": "SK Hynix HBM2E product brief (2020)",
    },
    "HBM3_12Hi": {
        "generation": 3.0,
        "n_die_max": 12,
        "n_io": 1024,
        "freq_Gbps": 6.4,
        "die_t_um": 30.0,
        "bump_t_um": 10.0,
        "mold_t_um": 80.0,
        "tsv_d_nm": 6000.0,
        "tsv_pitch_um": 28.0,
        "n_tsv_per_die": 10240,
        "capacity_GB_per_stack": 24.0,
        "color": "#2166ac",
        "ref": "Oh et al. ISSCC 2023",
    },
    "HBM3E_12Hi": {
        "generation": 3.5,
        "n_die_max": 12,
        "n_io": 1024,
        "freq_Gbps": 9.6,
        "die_t_um": 25.0,
        "bump_t_um": 8.0,
        "mold_t_um": 70.0,
        "tsv_d_nm": 5000.0,
        "tsv_pitch_um": 22.0,
        "n_tsv_per_die": 20480,
        "capacity_GB_per_stack": 36.0,
        "color": "#d62728",
        "ref": "SK Hynix HBM3E product brief (2023)",
    },
    "HBM4_16Hi": {
        "generation": 4.0,
        "n_die_max": 16,
        "n_io": 2048,            # wider bus (projection)
        "freq_Gbps": 12.0,
        "die_t_um": 20.0,
        "bump_t_um": 6.0,
        "mold_t_um": 60.0,
        "tsv_d_nm": 4000.0,
        "tsv_pitch_um": 16.0,
        "n_tsv_per_die": 40960,
        "capacity_GB_per_stack": 64.0,
        "color": "#2ca02c",
        "ref": "SK Hynix HBM4 roadmap (2024)",
    },
}

DRAM_NODES = {
    "1z": {"node_nm": 17.0, "cap_fF": 18.0, "AR_cap": 60.0, "Ileak_fA": 0.5,
           "t_ox_nm": 4.5, "color": "#2166ac", "ref": "Kim et al. IEDM 2022"},
    "1a": {"node_nm": 15.0, "cap_fF": 16.0, "AR_cap": 70.0, "Ileak_fA": 0.6,
           "t_ox_nm": 4.0, "color": "#d62728", "ref": "Lee et al. ISSCC 2023"},
    "1b": {"node_nm": 13.0, "cap_fF": 14.0, "AR_cap": 80.0, "Ileak_fA": 0.7,
           "t_ox_nm": 3.7, "color": "#ff7f0e", "ref": "SK Hynix 1b roadmap (2024)"},
    "1c": {"node_nm": 11.0, "cap_fF": 12.0, "AR_cap": 90.0, "Ileak_fA": 0.9,
           "t_ox_nm": 3.4, "color": "#2ca02c", "ref": "SK Hynix 1c projection (2025)"},
}

# ════════════════════════════════════════════════════════════════════════════
# 2. Core Physics Functions
# ════════════════════════════════════════════════════════════════════════════

def hbm_stack_height(product: str = "HBM3E_12Hi", n_die: int = None) -> dict:
    """
    Total HBM package height = base die + (n_die-1) * (die + bump) + mold.

    参考: Oh et al. ISSCC 2023 — HBM3E package cross-section
    """
    p = HBM_PRODUCTS.get(product, HBM_PRODUCTS["HBM3E_12Hi"])
    if n_die is None:
        n_die = p["n_die_max"]
    base_t_um = 720.0   # base logic die (thinned substrate)
    # Stack: base + intermediate dies
    total_um = (base_t_um + p["bump_t_um"] +
                (n_die - 1) * (p["die_t_um"] + p["bump_t_um"]) +
                p["mold_t_um"])
    total_mm = total_um / 1000.0

    return {
        "product": product,
        "n_die": n_die,
        "die_t_um": p["die_t_um"],
        "bump_t_um": p["bump_t_um"],
        "total_height_um": round(total_um, 1),
        "total_height_mm": round(total_mm, 3),
        "within_720um_spec": total_um <= 720.0,
    }


def tsv_resistance(product: str = "HBM3E_12Hi", material: str = "W") -> dict:
    """
    TSV RC model per stack level.

    R = rho * L / A_via
    C = 2 * pi * eps_Si * L / ln(r_outer / r_inner)   (coaxial capacitor)

    参考: Park et al. (2022) ECTC — HBM TSV resistance/capacitance
    """
    p = HBM_PRODUCTS.get(product, HBM_PRODUCTS["HBM3E_12Hi"])
    rho = rho_W if material == "W" else rho_Cu
    d = p["tsv_d_nm"] * 1e-9          # via diameter [m]
    L = p["die_t_um"] * 1e-6          # via height ≈ die thickness [m]
    A = math.pi * (d / 2.0) ** 2
    R = rho * L / A                    # [Ω]

    # Capacitance: oxide liner t_ox = 30 nm typical
    t_ox = 30e-9
    eps_ox = 3.9 * eps0
    r_inner = d / 2.0
    r_outer = r_inner + t_ox
    C = 2.0 * math.pi * eps_ox * L / math.log(r_outer / r_inner)  # [F]

    RC_ps = R * C * 1e12   # [ps]

    return {
        "product": product,
        "tsv_d_nm": p["tsv_d_nm"],
        "via_length_um": p["die_t_um"],
        "material": material,
        "R_via_ohm": round(R, 6),
        "C_via_fF": round(C * 1e15, 3),
        "RC_delay_ps": round(RC_ps, 4),
        "R_total_stack_mohm": round(R * 1000 * p["n_die_max"], 4),
    }


def hbm3e_bandwidth(product: str = "HBM3E_12Hi", n_stacks: int = 1,
                    efficiency: float = 0.88) -> dict:
    """
    Aggregate HBM bandwidth for n_stacks.

    BW = n_stacks * n_io * freq_Gbps * efficiency / 8   [GB/s]
    """
    p = HBM_PRODUCTS.get(product, HBM_PRODUCTS["HBM3E_12Hi"])
    BW_GBs_per = p["n_io"] * p["freq_Gbps"] * efficiency / 8.0
    BW_GBs_total = BW_GBs_per * n_stacks
    BW_TBs = BW_GBs_total / 1000.0

    return {
        "product": product,
        "n_stacks": n_stacks,
        "n_io": p["n_io"],
        "freq_Gbps": p["freq_Gbps"],
        "BW_GBs_per_stack": round(BW_GBs_per, 1),
        "BW_TBs": round(BW_TBs, 3),
        "capacity_GB_total": round(p["capacity_GB_per_stack"] * n_stacks, 0),
    }


def dram_capacitor_scaling(node: str = "1a") -> dict:
    """
    DRAM stacked cell capacitor: capacitance vs. node + aspect ratio.

    C = eps_HfO2 * A / t_ox   where A = pi * d * h = pi * d * AR * d
    Effective eps_HfO2 ≈ 25 * eps0 (ZrO2/HfO2 laminate)

    参考: Kim et al. (2022) IEDM — DRAM capacitor scaling
    """
    if node not in DRAM_NODES:
        node = "1a"
    p = DRAM_NODES[node]
    eps_HfO2 = 25.0 * eps0
    t_ox = p["t_ox_nm"] * 1e-9
    d_cap_nm = p["node_nm"] * 0.4    # pillar diameter ~ 0.4 × F
    d_m = d_cap_nm * 1e-9
    h_m = d_m * p["AR_cap"]
    A_cap = math.pi * d_m * h_m      # cylindrical surface area
    C_fF_calc = eps_HfO2 * A_cap / t_ox * 1e15

    return {
        "node": node,
        "node_nm": p["node_nm"],
        "d_cap_nm": round(d_cap_nm, 1),
        "AR_cap": p["AR_cap"],
        "C_fF_target": p["cap_fF"],
        "C_fF_calc": round(C_fF_calc, 2),
        "margin_fF": round(C_fF_calc - p["cap_fF"], 2),
        "feasible": C_fF_calc >= p["cap_fF"] * 0.9,
    }


def refresh_interval(node: str = "1a", T_C: float = 85.0) -> dict:
    """
    DRAM tREFI from retention time = C * Vdd / Ileak.

    Ileak increases with temperature: Ileak(T) = Ileak_ref * 2^((T-25)/10)
    (doubles every 10°C — standard JEDEC model)

    参考: Lee et al. (2021) ISSCC — LPDDR5X retention
    """
    p = DRAM_NODES.get(node, DRAM_NODES["1a"])
    C_F   = p["cap_fF"] * 1e-15
    Vdd   = 1.1    # [V] DRAM array Vdd
    Ileak_ref = p["Ileak_fA"] * 1e-15   # [A] at 25°C
    Ileak = Ileak_ref * 2.0 ** ((T_C - 25.0) / 10.0)
    t_ret_ms = C_F * Vdd / Ileak * 1e3   # [ms]
    tREFI_ms = min(t_ret_ms * 0.7, 64.0) # 70% margin; JEDEC max 64 ms

    return {
        "node": node,
        "T_C": T_C,
        "C_fF": p["cap_fF"],
        "Ileak_fA_at_T": round(Ileak * 1e15, 3),
        "retention_time_ms": round(t_ret_ms, 1),
        "tREFI_ms": round(tREFI_ms, 2),
        "jedec_ok": tREFI_ms <= 64.0,
    }


def hbm4_projection(year: int = 2026) -> dict:
    """
    HBM4 bandwidth / capacity roadmap projection (extrapolation from HBM3E).

    BW scales ~1.5× per generation (HBM2E→HBM3: 3.6→6.4 Gbps ≈1.78×,
    HBM3→HBM3E: 6.4→9.6 Gbps ≈1.5×). HBM4 targets 12–16 Gbps.
    """
    base_year = 2023
    base_BW_TBs = 1.2     # HBM3E 12Hi single stack
    growth_per_year = 0.25  # ~25% CAGR in BW
    BW_TBs = base_BW_TBs * (1.0 + growth_per_year) ** (year - base_year)

    base_cap_GB = 36.0    # HBM3E
    cap_growth   = 0.35
    cap_GB = base_cap_GB * (1.0 + cap_growth) ** (year - base_year)

    return {
        "year": year,
        "projected_BW_TBs_per_stack": round(BW_TBs, 2),
        "projected_capacity_GB_per_stack": round(cap_GB, 0),
        "projected_n_die": min(12 + (year - 2023) * 2, 24),
        "note": "extrapolation from HBM3E trajectory",
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. Visualization
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(17, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    ax = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    prods = list(HBM_PRODUCTS.keys())

    # ── Panel 0: Stack height vs. die count ──────────────────────────────
    die_arr = np.arange(1, 17)
    for prod, col in [(p, HBM_PRODUCTS[p]["color"]) for p in prods]:
        h_arr = [hbm_stack_height(prod, n)["total_height_um"] for n in die_arr]
        ax[0].plot(die_arr, h_arr, "o-", color=col, lw=2.2, ms=6,
                   label=prod.replace("_", " "))
    ax[0].axhline(720.0, color="k", ls="--", lw=1.2, label="720 µm CoWoS limit")
    ax[0].set_xlabel("Number of DRAM Die in Stack")
    ax[0].set_ylabel("Total Stack Height [µm]")
    ax[0].set_title("HBM Stack Height vs. Die Count\n(die + bump + mold, SK Hynix)")
    ax[0].legend(fontsize=7.5)
    ax[0].grid(alpha=0.25)

    # ── Panel 1: TSV resistance by product ───────────────────────────────
    for prod, col in [(p, HBM_PRODUCTS[p]["color"]) for p in prods]:
        ax[1].bar(prod.replace("_", "\n"), tsv_resistance(prod)["R_via_ohm"] * 1000,
                  color=col, width=0.6, alpha=0.85)
    ax[1].set_ylabel("TSV Via Resistance [mΩ]")
    ax[1].set_title("TSV Via Resistance by HBM Generation\n(W fill, RC delay model)")
    ax[1].grid(axis="y", alpha=0.25)
    # Add RC delay annotation
    ax1b = ax[1].twinx()
    rc_vals = [tsv_resistance(p)["RC_delay_ps"] for p in prods]
    ax1b.plot([p.replace("_", "\n") for p in prods], rc_vals, "D--",
              color="#333333", lw=1.5, ms=7)
    ax1b.set_ylabel("RC Delay [ps]", color="#333333")

    # ── Panel 2: HBM bandwidth vs. frequency (n=8 stacks) ────────────────
    freq_arr = np.linspace(1, 15, 200)
    for prod, col in [(p, HBM_PRODUCTS[p]["color"]) for p in prods]:
        n_io = HBM_PRODUCTS[prod]["n_io"]
        bw = [n_io * f * 0.88 / 8.0 / 1000.0 for f in freq_arr]
        ax[2].plot(freq_arr, bw, color=col, lw=2.2, label=prod.replace("_", " "))
    for prod in prods:
        p = HBM_PRODUCTS[prod]
        ax[2].axvline(p["freq_Gbps"], color=p["color"], ls=":", lw=1.0)
    ax[2].set_xlabel("I/O Speed [Gbps]")
    ax[2].set_ylabel("BW per Stack [TB/s]")
    ax[2].set_title("HBM Bandwidth per Stack vs. Speed\n(bus width × efficiency)")
    ax[2].legend(fontsize=7.5)
    ax[2].grid(alpha=0.25)

    # ── Panel 3: DRAM capacitor scaling ──────────────────────────────────
    nodes = list(DRAM_NODES.keys())
    node_nm_arr = [DRAM_NODES[n]["node_nm"] for n in nodes]
    c_target = [DRAM_NODES[n]["cap_fF"] for n in nodes]
    c_calc   = [dram_capacitor_scaling(n)["C_fF_calc"] for n in nodes]
    ax[3].plot(node_nm_arr, c_target, "s--", color="#2166ac", lw=2.2, ms=8, label="Target C [fF]")
    ax[3].plot(node_nm_arr, c_calc,   "o-",  color="#d62728", lw=2.2, ms=8, label="Calc C [fF]")
    ax[3].set_xlabel("DRAM Node [nm]")
    ax[3].set_ylabel("Cell Capacitance [fF]")
    ax[3].set_title("DRAM Capacitor Scaling\n(Stacked STC, ZrO₂/HfO₂, AR scaling)")
    ax[3].legend(fontsize=9)
    ax[3].grid(alpha=0.25)
    ax[3].invert_xaxis()

    # ── Panel 4: Refresh interval vs. temperature ─────────────────────────
    T_arr = np.linspace(0, 120, 150)
    for node, col in [(n, DRAM_NODES[n]["color"]) for n in nodes]:
        tref = [refresh_interval(node, T)["tREFI_ms"] for T in T_arr]
        ax[4].semilogy(T_arr, tref, color=col, lw=2.2, label=node)
    ax[4].axhline(64.0, color="k", ls="--", lw=1.2, label="JEDEC 64 ms max")
    ax[4].set_xlabel("Temperature [°C]")
    ax[4].set_ylabel("tREFI [ms]")
    ax[4].set_title("DRAM Refresh Interval vs. Temperature\n(leakage current doubling per 10°C)")
    ax[4].legend(fontsize=8.5)
    ax[4].grid(alpha=0.25)

    # ── Panel 5: HBM4 roadmap ────────────────────────────────────────────
    years = [2023, 2024, 2025, 2026, 2027, 2028]
    bw_proj  = [hbm4_projection(y)["projected_BW_TBs_per_stack"] for y in years]
    cap_proj = [hbm4_projection(y)["projected_capacity_GB_per_stack"] for y in years]
    ax5a = ax[5]
    ax5b = ax5a.twinx()
    ax5a.plot(years, bw_proj,  "o-", color="#d62728", lw=2.5, ms=8, label="BW [TB/s]")
    ax5b.plot(years, cap_proj, "s--", color="#2166ac", lw=2.0, ms=7, label="Capacity [GB]")
    ax5a.scatter([2023], [1.2], color="#d62728", s=120, zorder=5)
    ax5a.annotate("HBM3E\n1.2 TB/s", (2023, 1.2), fontsize=7.5, xytext=(10, 5),
                  textcoords="offset points")
    ax5a.set_xlabel("Year")
    ax5a.set_ylabel("BW per Stack [TB/s]", color="#d62728")
    ax5b.set_ylabel("Capacity per Stack [GB]", color="#2166ac")
    ax5a.set_title("HBM4 Roadmap Projection\n(SK Hynix 2025–2028, extrapolation)")
    ax5a.grid(alpha=0.25)
    lines1, lab1 = ax5a.get_legend_handles_labels()
    lines2, lab2 = ax5b.get_legend_handles_labels()
    ax5a.legend(lines1 + lines2, lab1 + lab2, fontsize=8.5, loc="upper left")

    fig.suptitle(
        "SK Hynix — HBM3E · HBM4 · DRAM Scaling Physics\n"
        "HBM3E 12Hi (1.2 TB/s) · TSV RC · Capacitor Scaling · Refresh · HBM4 Roadmap",
        fontsize=12, fontweight="bold",
    )
    out = os.path.join(out_dir, "sk_hynix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] SK Hynix model → {out}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="SK Hynix HBM3E / HBM4 / DRAM scaling model")
    ap.add_argument("--hbm",    action="store_true", help="Print HBM bandwidth summary")
    ap.add_argument("--dram",   action="store_true", help="Print DRAM node summary")
    ap.add_argument("--product", default="HBM3E_12Hi", choices=list(HBM_PRODUCTS.keys()))
    ap.add_argument("--node",    default="1a",          choices=list(DRAM_NODES.keys()))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    if args.hbm:
        print(f"\nHBM Summary: {args.product}")
        print("=" * 40)
        ht  = hbm_stack_height(args.product)
        bw  = hbm3e_bandwidth(args.product)
        tsv = tsv_resistance(args.product)
        print(f"  Stack height   : {ht['total_height_um']:.1f} µm  (≤720 µm: {ht['within_720um_spec']})")
        print(f"  BW per stack   : {bw['BW_GBs_per_stack']:.1f} GB/s = {bw['BW_TBs']:.3f} TB/s")
        print(f"  TSV R_via      : {tsv['R_via_ohm']*1000:.3f} mΩ  RC={tsv['RC_delay_ps']:.4f} ps")
        print(f"  Capacity       : {bw['capacity_GB_total']:.0f} GB total")

    if args.dram:
        print(f"\nDRAM Node: {args.node}")
        print("=" * 40)
        cap = dram_capacitor_scaling(args.node)
        ref = refresh_interval(args.node, T_C=85)
        print(f"  Capacitance  target={cap['C_fF_target']} fF  calc={cap['C_fF_calc']:.2f} fF")
        print(f"  tREFI at 85°C: {ref['tREFI_ms']:.2f} ms  (JEDEC ok: {ref['jedec_ok']})")
        print(f"  Ileak at 85°C: {ref['Ileak_fA_at_T']:.3f} fA/cell")

    if not args.no_plot:
        plot_all()


if __name__ == "__main__":
    main()
