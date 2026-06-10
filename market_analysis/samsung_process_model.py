"""
Samsung — 3nm GAA MBCFET + HBM3 + V-NAND Process Model
==========================================================
Covers Samsung SF3E (3nm MBCFET GAA logic), HBM3 12-layer memory stack,
and 9th-gen V-NAND (236-layer charge trap flash).

Key physics:
  - MBCFET multi-bridge channel FET Id–Vd (Virtual Source model)
  - V-NAND string pass yield: Weibull CD distribution over cell string
  - V-NAND read disturb: threshold voltage drift (charge loss Arrhenius)
  - HBM3 TSV yield: defect rate × TSV count per die
  - HBM3 bandwidth: 1024-bit × 3.35 GHz × n_stacks
  - Samsung product yield integration model

実行:
    python fem/samsung_process_model.py
    python fem/samsung_process_model.py --logic --node SF3E
    python fem/samsung_process_model.py --nand --layers 236

参考文献:
    Samsung SF3E Process Technology Brief (2022)
    Seo et al. (2022) ISSCC — V-NAND 9th gen 236-layer
    Park et al. (2023) ISSCC — Samsung HBM3 12-layer
    Jeong et al. (2013) IEEE TED — Virtual Source model for GAA
    Grupp et al. (2009) IEEE TED — NAND flash threshold voltage statistics
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
q_C  = 1.602e-19
k_B  = 1.381e-23
T_K  = 300.0
eps0 = 8.854e-12

# ════════════════════════════════════════════════════════════════════════════
# 1. Process & Device Parameter Tables
# ════════════════════════════════════════════════════════════════════════════

LOGIC_NODES = {
    "SF3E": {
        "node_nm": 3.0,
        "L_g_nm": 12.0,
        "W_ns_nm": 5.0,
        "n_sheet": 3,
        "t_ox_nm": 0.9,
        "Vdd_V": 0.75,
        "density_MT_mm2": 141.0,
        "D0_per_cm2": 0.09,
        "DIBL_mV_V": 30.0,
        "color": "#1428A0",   # Samsung blue
        "ref": "Samsung SF3E Technology Brief (2022)",
    },
    "SF4X": {
        "node_nm": 4.0,
        "L_g_nm": 18.0,
        "W_ns_nm": 7.0,
        "n_sheet": 3,
        "t_ox_nm": 1.1,
        "Vdd_V": 0.80,
        "density_MT_mm2": 100.0,
        "D0_per_cm2": 0.08,
        "DIBL_mV_V": 35.0,
        "color": "#2166ac",
        "ref": "Samsung SF4X Technology Brief (2022)",
    },
    "SF8": {
        "node_nm": 8.0,
        "L_g_nm": 35.0,
        "W_ns_nm": 0.0,        # FinFET
        "n_sheet": 0,
        "t_ox_nm": 1.4,
        "Vdd_V": 0.90,
        "density_MT_mm2": 50.0,
        "D0_per_cm2": 0.06,
        "DIBL_mV_V": 50.0,
        "color": "#d62728",
        "ref": "Samsung 8nm Process Overview (2020)",
    },
}

VNAND_CONFIGS = {
    "9th_gen_236L": {
        "n_layers": 236,
        "string_len": 236,
        "cell_size_nm2": 6.9 * 4.5,
        "page_size_KB": 64,
        "cell_type": "TLC",
        "Vth_spread_mV": 80.0,    # within-cell Vth distribution σ
        "read_voltage_V": 0.5,
        "color": "#1428A0",
        "ref": "Seo et al. ISSCC 2022",
    },
    "8th_gen_176L": {
        "n_layers": 176,
        "string_len": 176,
        "cell_size_nm2": 7.5 * 5.0,
        "page_size_KB": 48,
        "cell_type": "TLC",
        "Vth_spread_mV": 90.0,
        "read_voltage_V": 0.5,
        "color": "#2166ac",
        "ref": "Kim et al. ISSCC 2020",
    },
    "6th_gen_128L": {
        "n_layers": 128,
        "string_len": 128,
        "cell_size_nm2": 9.0 * 6.0,
        "page_size_KB": 32,
        "cell_type": "TLC",
        "Vth_spread_mV": 100.0,
        "read_voltage_V": 0.5,
        "color": "#d62728",
        "ref": "Kim et al. ISSCC 2018",
    },
}

HBM_CONFIGS = {
    "HBM3_12Hi": {
        "n_die": 12,
        "n_io": 1024,
        "freq_GHz": 3.35,
        "die_t_um": 30.0,
        "tsv_d_nm": 6000.0,
        "tsv_ar": 8.0,
        "n_tsv_per_die": 20480,
        "color": "#1428A0",
        "ref": "Park et al. ISSCC 2023",
    },
    "HBM3E_12Hi": {
        "n_die": 12,
        "n_io": 1024,
        "freq_GHz": 4.8,
        "die_t_um": 25.0,
        "tsv_d_nm": 5000.0,
        "tsv_ar": 10.0,
        "n_tsv_per_die": 20480,
        "color": "#2166ac",
        "ref": "Samsung HBM3E Product Brief (2023)",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 2. Core Physics Functions
# ════════════════════════════════════════════════════════════════════════════

def mbcfet_ids(Vgs: float, Vds: float, node: str = "SF3E") -> dict:
    """
    Multi-Bridge Channel FET (MBCFET) Id–Vd via Virtual Source model.
    Identical mathematical form to RibbonFET; parameterized per Samsung node.

    参考: Jeong et al. (2013) IEEE TED 60, 2541
    """
    p = LOGIC_NODES.get(node, LOGIC_NODES["SF3E"])
    n_s  = max(p["n_sheet"], 1)
    W_ns = p["W_ns_nm"] * 1e-9
    L_g  = p["L_g_nm"] * 1e-9
    t_ox = p["t_ox_nm"] * 1e-9
    DIBL = p["DIBL_mV_V"] * 1e-3   # [V/V]

    eps_ox = 3.9 * eps0
    Cox = eps_ox / t_ox
    Vt = 0.20 - DIBL * Vds          # DIBL shifts Vt
    v_T = 1.1e5                      # [m/s]
    mu_eff = 0.020

    Qinv = Cox * max(Vgs - Vt, 0.0)
    V_dsat = max(Vgs - Vt, 0.01) * 0.5
    I_sat  = n_s * W_ns * Qinv * v_T * 0.5
    I_D    = I_sat * (1.0 - math.exp(-max(Vds, 0.0) / max(V_dsat, 0.01)))
    I_D_uA_um = I_D / max(n_s * W_ns, 1e-15) * 1e6 / 1e-6

    return {
        "node": node,
        "Vgs_V": Vgs,
        "Vds_V": Vds,
        "Vt_eff_V": round(Vt, 4),
        "Id_uA_um": round(I_D_uA_um, 1),
        "Id_uA": round(I_D * 1e6, 3),
        "n_sheet": n_s,
        "on_state": Vgs >= Vt,
    }


def vnand_cell_yield(config: str = "9th_gen_236L", CD_sigma_nm: float = 2.0) -> dict:
    """
    V-NAND string pass probability with Weibull CD distribution.

    P_string_pass = P_cell_pass^string_len
    P_cell_pass = 1 - P(CD too small causes fail) — modeled as normal CDF tail
    Vth spread assumed proportional to CD variation.

    参考: Grupp et al. (2009) IEEE TED — NAND flash Vth statistics
    """
    if config not in VNAND_CONFIGS:
        config = "9th_gen_236L"
    p = VNAND_CONFIGS[config]

    # Vth variation from CD sigma (empirical: dVth/dCD ≈ 5 mV/nm for V-NAND)
    Vth_sigma_from_CD = CD_sigma_nm * 5.0  # mV
    Vth_total_sigma_mV = math.sqrt(p["Vth_spread_mV"] ** 2 + Vth_sigma_from_CD ** 2)

    # Read window margin: assume read voltage ±200 mV from nominal
    margin_sigma = 200.0 / Vth_total_sigma_mV
    from math import erfc
    P_fail_cell = 0.5 * erfc(margin_sigma / math.sqrt(2))
    P_pass_cell = 1.0 - P_fail_cell
    P_pass_string = P_pass_cell ** p["string_len"]
    yield_pct = P_pass_string * 100.0

    return {
        "config": config,
        "n_layers": p["n_layers"],
        "CD_sigma_nm": CD_sigma_nm,
        "Vth_total_sigma_mV": round(Vth_total_sigma_mV, 1),
        "P_pass_cell": round(P_pass_cell, 8),
        "P_pass_string": round(P_pass_string, 6),
        "yield_pct": round(yield_pct, 3),
    }


def vnand_read_disturb(n_cycles: int, T_C: float = 85.0,
                       program_Vth_spread_mV: float = 80.0) -> dict:
    """
    Vth drift from charge loss (read disturb) model.

    dVth = Vth_init * exp(-t_stress / tau)
    tau = tau0 * exp(Ea / kT),  Ea ≈ 0.65 eV for SiN charge trap

    参考: Lee et al. (2013) IEEE TED — Charge trap retention
    """
    Ea_eV = 0.65
    k_B_eV = 8.617e-5
    tau0_s = 1e-3       # pre-exponential [s]
    tau = tau0_s * math.exp(Ea_eV / (k_B_eV * (T_C + 273.15)))
    t_stress_s = n_cycles * 50e-9   # ~50 ns per read cycle
    Vth_init_mV = program_Vth_spread_mV * 5.0   # rough initial Vth
    Vth_drift_mV = Vth_init_mV * (1.0 - math.exp(-t_stress_s / tau))
    window_closed = Vth_drift_mV > program_Vth_spread_mV * 0.5

    return {
        "n_cycles": n_cycles,
        "T_C": T_C,
        "tau_s": round(tau, 2),
        "t_stress_s": round(t_stress_s, 6),
        "Vth_drift_mV": round(Vth_drift_mV, 2),
        "window_closed": window_closed,
        "retention_ok": not window_closed,
    }


def hbm3_tsv_yield(config: str = "HBM3_12Hi", D0_per_cm2: float = 0.05) -> dict:
    """
    HBM3 TSV yield: probability all TSVs in a die stack are defect-free.

    Y_tsv = (1 - D0 * A_tsv)^(n_tsv * n_die)
    A_tsv = pi * (d/2)^2 in cm²

    参考: Park et al. (2023) ISSCC — HBM3 TSV reliability
    """
    if config not in HBM_CONFIGS:
        config = "HBM3_12Hi"
    p = HBM_CONFIGS[config]
    d_cm = p["tsv_d_nm"] * 1e-9 * 100.0   # nm → cm
    A_tsv_cm2 = math.pi * (d_cm / 2.0) ** 2
    n_total_tsv = p["n_tsv_per_die"] * p["n_die"]
    P_tsv_good = (1.0 - D0_per_cm2 * A_tsv_cm2) ** n_total_tsv
    BW_GBs = p["n_io"] * p["freq_GHz"] * 2.0 * 0.88 / 8.0

    return {
        "config": config,
        "n_die": p["n_die"],
        "n_tsv_per_die": p["n_tsv_per_die"],
        "tsv_yield_pct": round(P_tsv_good * 100.0, 2),
        "BW_GBs": round(BW_GBs, 1),
        "BW_TBs": round(BW_GBs / 1000.0, 3),
    }


def samsung_yield(product: str = "SF3E", die_area_mm2: float = 100.0,
                  CD_sigma_nm: float = 1.5) -> dict:
    """Overall Samsung product yield (logic Murphy × NAND string yield)."""
    if product in LOGIC_NODES:
        p = LOGIC_NODES[product]
        D0, A_cm2 = p["D0_per_cm2"], die_area_mm2 / 100.0
        val = (1.0 - math.exp(-D0 * A_cm2)) / max(D0 * A_cm2, 1e-12)
        Y_logic = val ** 2
        return {"product": product, "type": "logic",
                "yield_pct": round(Y_logic * 100.0, 1),
                "die_area_mm2": die_area_mm2, "D0": D0}
    else:
        config = "9th_gen_236L"
        res = vnand_cell_yield(config, CD_sigma_nm)
        return {"product": product, "type": "nand",
                "yield_pct": res["yield_pct"],
                "n_layers": res["n_layers"]}


def full_pipeline(node: str = "SF3E", nand_config: str = "9th_gen_236L",
                  hbm_config: str = "HBM3_12Hi") -> dict:
    """Samsung complete product pipeline summary."""
    logic = mbcfet_ids(Vgs=0.75, Vds=0.7, node=node)
    nand  = vnand_cell_yield(nand_config)
    hbm   = hbm3_tsv_yield(hbm_config)
    rdis  = vnand_read_disturb(n_cycles=100000, T_C=85)

    return {
        "node": node,
        "Id_uA_um": logic["Id_uA_um"],
        "nand_yield_pct": nand["yield_pct"],
        "nand_Vth_sigma_mV": nand["Vth_total_sigma_mV"],
        "hbm_tsv_yield_pct": hbm["tsv_yield_pct"],
        "hbm_BW_TBs": hbm["BW_TBs"],
        "read_disturb_ok": rdis["retention_ok"],
        "Vth_drift_100k_mV": rdis["Vth_drift_mV"],
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. Visualization
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(17, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    ax = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    Vds_arr = np.linspace(0, 0.75, 120)

    # ── Panel 0: MBCFET Id–Vd ────────────────────────────────────────────
    for node_name, col in [("SF3E", "#1428A0"), ("SF4X", "#2166ac"), ("SF8", "#d62728")]:
        nd = LOGIC_NODES[node_name]
        for Vgs, ls in [(nd["Vdd_V"], "-"), (nd["Vdd_V"] * 0.7, "--"), (nd["Vdd_V"] * 0.4, ":")]:
            ids = [mbcfet_ids(Vgs, Vd, node_name)["Id_uA_um"] for Vd in Vds_arr]
            lbl = f"{node_name} Vgs={Vgs:.2f}V" if ls == "-" else None
            ax[0].plot(Vds_arr, ids, color=col, ls=ls, lw=2.0, label=lbl)
    ax[0].set_xlabel("Vds [V]")
    ax[0].set_ylabel("Id [µA/µm]")
    ax[0].set_title("Samsung MBCFET Id–Vd\nSF3E · SF4X · SF8 (Virtual Source)")
    ax[0].legend(fontsize=7.5)
    ax[0].grid(alpha=0.25)

    # ── Panel 1: V-NAND yield vs. layer count ────────────────────────────
    layers_arr = np.arange(64, 301, 8)
    for CD_s, col, lbl in [(1.0, "#1428A0", "σ_CD=1 nm"),
                            (2.0, "#2166ac", "σ_CD=2 nm"),
                            (3.0, "#d62728", "σ_CD=3 nm")]:
        # Compute yield for custom layer count (use 9th gen as template)
        yields = []
        for n_lay in layers_arr:
            Vth_sigma_from_CD = CD_s * 5.0
            Vth_total = math.sqrt(80.0 ** 2 + Vth_sigma_from_CD ** 2)
            margin_sigma = 200.0 / Vth_total
            from math import erfc
            Pf = 0.5 * erfc(margin_sigma / math.sqrt(2))
            yields.append((1.0 - Pf) ** n_lay * 100.0)
        ax[1].plot(layers_arr, yields, color=col, lw=2.2, label=lbl)
    ax[1].axvline(236, color="k", ls="--", lw=1.2, label="9th gen (236L)")
    ax[1].set_xlabel("Layer Count")
    ax[1].set_ylabel("String Yield [%]")
    ax[1].set_title("V-NAND String Yield vs. Layer Count\n(CD variation → Vth spread)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.25)
    ax[1].set_ylim(0, 100)

    # ── Panel 2: V-NAND read disturb vs. cycles ───────────────────────────
    cycles_arr = np.logspace(2, 7, 200).astype(int)
    for T, col in [(25, "#2166ac"), (55, "#ff7f0e"), (85, "#d62728"), (125, "#9467bd")]:
        drift = [vnand_read_disturb(int(c), T)["Vth_drift_mV"] for c in cycles_arr]
        ax[2].semilogx(cycles_arr, drift, color=col, lw=2.2, label=f"{T}°C")
    ax[2].axhline(40.0, color="k", ls="--", lw=1.2, label="40 mV limit")
    ax[2].set_xlabel("Read Cycles")
    ax[2].set_ylabel("Vth Drift [mV]")
    ax[2].set_title("V-NAND Read Disturb\n(SiN charge trap, Ea=0.65 eV)")
    ax[2].legend(fontsize=8.5)
    ax[2].grid(alpha=0.25)

    # ── Panel 3: HBM3 TSV yield vs. D0 ───────────────────────────────────
    D0_arr = np.linspace(0.01, 0.3, 100)
    for config, col in [("HBM3_12Hi", "#1428A0"), ("HBM3E_12Hi", "#2166ac")]:
        y_arr = [hbm3_tsv_yield(config, D0)["tsv_yield_pct"] for D0 in D0_arr]
        ax[3].plot(D0_arr, y_arr, color=col, lw=2.2,
                   label=config.replace("_", " "))
    ax[3].set_xlabel("TSV Defect Density D₀ [/cm²]")
    ax[3].set_ylabel("TSV Stack Yield [%]")
    ax[3].set_title("HBM3 TSV Stack Yield\n(20,480 TSV/die × 12 die)")
    ax[3].legend(fontsize=9)
    ax[3].grid(alpha=0.25)
    ax[3].set_ylim(0, 100)

    # ── Panel 4: HBM3 / HBM3E bandwidth ─────────────────────────────────
    stacks_arr = np.arange(1, 13)
    for conf, col in [("HBM3_12Hi", "#1428A0"), ("HBM3E_12Hi", "#2166ac")]:
        p = HBM_CONFIGS[conf]
        bw_arr = [p["n_io"] * p["freq_GHz"] * 2.0 * 0.88 / 8.0 for _ in stacks_arr]
        # Aggregate: n_stacks independent channels
        agg = [bw * n / 1000.0 for bw, n in zip(bw_arr, stacks_arr)]
        ax[4].plot(stacks_arr, agg, "o-", color=col, lw=2.2, ms=6,
                   label=conf.replace("_", " "))
    ax[4].axhline(1.228, color="#1428A0", ls=":", lw=1.5, label="1.2 TB/s HBM3 spec")
    ax[4].set_xlabel("Number of HBM Stacks")
    ax[4].set_ylabel("Aggregate BW [TB/s]")
    ax[4].set_title("HBM3 / HBM3E Bandwidth\n(Samsung, 1024-bit bus)")
    ax[4].legend(fontsize=8.5)
    ax[4].grid(alpha=0.25)

    # ── Panel 5: Pipeline summary table ─────────────────────────────────
    ax[5].axis("off")
    col_labels = ["Config", "Type", "Key Metric", "Yield\n[%]", "BW\n[TB/s]", "Status"]
    tdata = [
        ["SF3E", "GAA Logic", f"Id={mbcfet_ids(0.75, 0.7, 'SF3E')['Id_uA_um']:.0f} µA/µm",
         f"{samsung_yield('SF3E', 100)['yield_pct']:.1f}", "—", "✓"],
        ["9th gen\n236L", "V-NAND", f"Vth σ={vnand_cell_yield()['Vth_total_sigma_mV']:.0f} mV",
         f"{vnand_cell_yield()['yield_pct']:.2f}", "—", "✓"],
        ["HBM3\n12Hi", "Memory", "20480 TSV",
         f"{hbm3_tsv_yield('HBM3_12Hi')['tsv_yield_pct']:.1f}",
         f"{hbm3_tsv_yield('HBM3_12Hi')['BW_TBs']:.3f}", "✓"],
        ["HBM3E\n12Hi", "Memory", "20480 TSV",
         f"{hbm3_tsv_yield('HBM3E_12Hi')['tsv_yield_pct']:.1f}",
         f"{hbm3_tsv_yield('HBM3E_12Hi')['BW_TBs']:.3f}", "✓"],
    ]
    tbl = ax[5].table(cellText=tdata, colLabels=col_labels,
                      cellLoc="center", loc="center",
                      bbox=[0.0, 0.10, 1.0, 0.85])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1428A0")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#eef0ff")
    ax[5].set_title("Samsung Product Summary\nSF3E · V-NAND 236L · HBM3/3E")

    fig.suptitle(
        "Samsung — 3nm GAA MBCFET · V-NAND 236L · HBM3 Physics\n"
        "SF3E Logic · 9th-gen V-NAND · HBM3 / HBM3E 12-layer",
        fontsize=12, fontweight="bold",
    )
    out = os.path.join(out_dir, "samsung_process.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Samsung process model → {out}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Samsung 3nm GAA / V-NAND / HBM3 model")
    ap.add_argument("--logic",  action="store_true", help="Print MBCFET summary")
    ap.add_argument("--nand",   action="store_true", help="Print V-NAND yield")
    ap.add_argument("--node",   default="SF3E",        choices=list(LOGIC_NODES.keys()))
    ap.add_argument("--layers", type=int, default=236, help="V-NAND layer count")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    if args.logic:
        print(f"\nMBCFET Summary ({args.node})")
        print("=" * 40)
        res = mbcfet_ids(Vgs=LOGIC_NODES[args.node]["Vdd_V"], Vds=0.7, node=args.node)
        for k, v in res.items():
            print(f"  {k:<30} {v}")

    if args.nand:
        print(f"\nV-NAND Yield (9th gen {args.layers}L)")
        print("=" * 40)
        for CD_s in [1.0, 2.0, 3.0]:
            r = vnand_cell_yield("9th_gen_236L", CD_sigma_nm=CD_s)
            print(f"  CD σ={CD_s:.0f} nm  yield={r['yield_pct']:.3f}%  "
                  f"Vth_σ={r['Vth_total_sigma_mV']:.0f} mV")

    if not args.no_plot:
        plot_all()


if __name__ == "__main__":
    main()
