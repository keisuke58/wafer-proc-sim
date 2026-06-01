"""
Micron Technology — DRAM (1β/1γ) / NAND (232L) / HBM3E 物理モデル
==========================================================
唯一の米国系 DRAM/NAND メーカー。SK Hynix / Samsung との比較軸として重要。

Key physics:
  - 1β DRAM セル (13nm世代): セル容量 × リテンション × tREFI
  - 232-layer NAND (B58R): 文字列歩留まり × 書き込み劣化 (wearout)
  - HBM3E 12Hi: TSV 歩留まり × 帯域幅 × 供給量 (SK Hynix との比較)
  - Die コスト モデル: ノード別 MPW スループット → CPGD
  - 地政学的供給リスク: 単一地域集中度 (Micron=米国/日本/台湾分散)

実行:
    python fem/micron_model.py
    python fem/micron_model.py --dram --node 1b
    python fem/micron_model.py --nand --layers 232

参考文献:
    Micron 1β DRAM Technology Brief (2023)
    Micron 232-layer NAND B58R Brief (2023)
    Micron HBM3E Product Brief (2024)
    Lee et al. (2021) ISSCC — 1α DRAM cell capacitor
    Seo et al. (2022) ISSCC — 232-layer NAND
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

eps0 = 8.854e-12

# ════════════════════════════════════════════════════════════════════════════
# 1. Micron DRAM ノード パラメータ
# ════════════════════════════════════════════════════════════════════════════

DRAM_NODES = {
    "1z": {"node_nm": 17.0, "cap_fF": 18.5, "AR_cap": 58.0, "Ileak_fA": 0.50,
           "t_ox_nm": 4.6, "density_Gb_cm2": 1.82, "D0": 0.07, "color": "#ff7f0e",
           "ref": "Micron 1z DRAM (2021)"},
    "1a": {"node_nm": 15.0, "cap_fF": 16.5, "AR_cap": 68.0, "Ileak_fA": 0.60,
           "t_ox_nm": 4.2, "density_Gb_cm2": 2.60, "D0": 0.065, "color": "#2166ac",
           "ref": "Micron 1α DRAM (2022)"},
    "1b": {"node_nm": 13.0, "cap_fF": 15.0, "AR_cap": 78.0, "Ileak_fA": 0.70,
           "t_ox_nm": 3.9, "density_Gb_cm2": 3.50, "D0": 0.06, "color": "#d62728",
           "ref": "Micron 1β DRAM (2023)"},
    "1g": {"node_nm": 11.5, "cap_fF": 13.5, "AR_cap": 88.0, "Ileak_fA": 0.85,
           "t_ox_nm": 3.6, "density_Gb_cm2": 4.60, "D0": 0.055, "color": "#2ca02c",
           "ref": "Micron 1γ DRAM roadmap (2025)"},
}

NAND_CONFIGS = {
    "176L_B47R": {
        "n_layers": 176, "Vth_spread_mV": 90.0, "cell_type": "TLC",
        "bit_density_Gb_cm2": 14.6, "color": "#ff7f0e",
        "ref": "Micron 176-layer B47R (2021)"},
    "232L_B58R": {
        "n_layers": 232, "Vth_spread_mV": 82.0, "cell_type": "TLC",
        "bit_density_Gb_cm2": 19.2, "color": "#2166ac",
        "ref": "Micron 232-layer B58R (2023)"},
    "276L_B68R": {
        "n_layers": 276, "Vth_spread_mV": 78.0, "cell_type": "QLC",
        "bit_density_Gb_cm2": 25.0, "color": "#d62728",
        "ref": "Micron 276-layer B68R roadmap (2025)"},
}

HBM_SPECS = {
    "HBM3_8Hi": {
        "n_die": 8, "n_io": 1024, "freq_Gbps": 6.4, "die_t_um": 30.0,
        "capacity_GB": 24.0, "tsv_per_die": 10240,
        "color": "#ff7f0e", "ref": "Micron HBM3 (2023)"},
    "HBM3E_12Hi": {
        "n_die": 12, "n_io": 1024, "freq_Gbps": 9.6, "die_t_um": 25.0,
        "capacity_GB": 36.0, "tsv_per_die": 20480,
        "color": "#2166ac", "ref": "Micron HBM3E 12Hi (2024)"},
}

# ════════════════════════════════════════════════════════════════════════════
# 2. Core Physics Functions
# ════════════════════════════════════════════════════════════════════════════

def dram_cell_retention(node: str = "1b", T_C: float = 85.0) -> dict:
    """
    DRAM セルリテンション時間とtREFI。
    t_ret = C * Vdd / Ileak(T);  Ileak(T) = Ileak_ref * 2^((T-25)/10)
    """
    p = DRAM_NODES.get(node, DRAM_NODES["1b"])
    C_F   = p["cap_fF"] * 1e-15
    Ileak = p["Ileak_fA"] * 1e-15 * 2.0 ** ((T_C - 25.0) / 10.0)
    Vdd   = 1.1
    t_ret_ms = C_F * Vdd / Ileak * 1e3
    tREFI_ms = min(t_ret_ms * 0.7, 64.0)
    # Murphy yield for DRAM die (8Gb die)
    die_area_mm2 = 8000.0 / (p["density_Gb_cm2"] * 100.0)  # mm²
    A_cm2 = die_area_mm2 / 100.0
    val = (1.0 - math.exp(-p["D0"] * A_cm2)) / max(p["D0"] * A_cm2, 1e-12)
    yield_pct = val ** 2 * 100.0
    return {
        "node": node, "T_C": T_C,
        "cap_fF": p["cap_fF"],
        "Ileak_fA_at_T": round(Ileak * 1e15, 3),
        "retention_ms": round(t_ret_ms, 1),
        "tREFI_ms": round(tREFI_ms, 2),
        "die_area_mm2": round(die_area_mm2, 1),
        "yield_pct": round(yield_pct, 1),
        "jedec_ok": tREFI_ms <= 64.0,
    }


def nand_wearout(config: str = "232L_B58R", n_cycles: int = 1000,
                 T_C: float = 85.0) -> dict:
    """
    NAND P/Eサイクル劣化 (wearout): Vth窓の縮小。

    ΔVth_window = ΔVth0 * (1 - exp(-n_cycles / tau_wear))
    tau_wear: 熱活性化 (Ea ≈ 0.55 eV for SILC)
    Endurance spec: TLC 1000 P/E, QLC 300 P/E

    参考: Compagnoni et al. (2008) IEEE TED — NAND flash wearout
    """
    p = NAND_CONFIGS.get(config, NAND_CONFIGS["232L_B58R"])
    Ea_eV = 0.55; k_B_eV = 8.617e-5
    tau_wear = 1500.0 * math.exp(Ea_eV / (k_B_eV * (T_C + 273.15)) -
                                  Ea_eV / (k_B_eV * 358.15))
    Vth_init_mV = p["Vth_spread_mV"]
    dVth_window = Vth_init_mV * 0.4 * (1.0 - math.exp(-n_cycles / tau_wear))
    window_remaining_pct = max(100.0 * (1.0 - dVth_window / (Vth_init_mV * 0.6)), 0.0)
    endurance_limit = 300 if p["cell_type"] == "QLC" else 1000
    return {
        "config": config, "n_cycles": n_cycles, "T_C": T_C,
        "Vth_initial_mV": Vth_init_mV,
        "Vth_degradation_mV": round(dVth_window, 2),
        "window_remaining_pct": round(window_remaining_pct, 1),
        "endurance_limit": endurance_limit,
        "within_endurance": n_cycles <= endurance_limit,
    }


def nand_string_yield(config: str = "232L_B58R", CD_sigma_nm: float = 2.0) -> dict:
    """V-NAND文字列歩留まり (Micron版、samsung_process_model.py と同物理)"""
    p = NAND_CONFIGS.get(config, NAND_CONFIGS["232L_B58R"])
    Vth_total = math.sqrt(p["Vth_spread_mV"]**2 + (CD_sigma_nm * 5.0)**2)
    from math import erfc
    Pf = 0.5 * erfc(200.0 / (Vth_total * math.sqrt(2)))
    yield_pct = (1.0 - Pf) ** p["n_layers"] * 100.0
    return {
        "config": config, "n_layers": p["n_layers"],
        "Vth_sigma_mV": round(Vth_total, 1),
        "yield_pct": round(yield_pct, 3),
        "bit_density_Gb_cm2": p["bit_density_Gb_cm2"],
    }


def hbm_vs_competitors(hbm_spec: str = "HBM3E_12Hi") -> dict:
    """
    Micron vs SK Hynix vs Samsung HBM帯域幅・容量比較。
    """
    p = HBM_SPECS.get(hbm_spec, HBM_SPECS["HBM3E_12Hi"])
    BW_GBs = p["n_io"] * p["freq_Gbps"] * 0.88 / 8.0
    # TSV yield (簡略版)
    D0 = 0.05; d_cm = 5000e-9 * 100
    A_tsv = math.pi * (d_cm/2)**2
    tsv_yield = (1 - D0 * A_tsv) ** (p["tsv_per_die"] * p["n_die"]) * 100
    return {
        "spec": hbm_spec, "n_die": p["n_die"],
        "BW_GBs": round(BW_GBs, 1),
        "BW_TBs": round(BW_GBs / 1000, 3),
        "capacity_GB": p["capacity_GB"],
        "tsv_yield_pct": round(tsv_yield, 2),
        "freq_Gbps": p["freq_Gbps"],
    }


def supply_concentration_risk() -> dict:
    """
    地政学的供給リスク: HRI (Herfindahl–Hirschman Index) で集中度評価。
    DRAM/NAND市場の地域別シェア → HHI
    Micron: 米/日/台 分散 → 最低リスク
    SK Hynix: 韓国集中
    Samsung: 韓国集中
    """
    suppliers = {
        "Micron":   {"share_pct": 23.0, "regions": ["US", "Japan", "Taiwan"],
                     "region_concentration": 0.38, "color": "#2166ac"},
        "SK Hynix": {"share_pct": 31.0, "regions": ["Korea"],
                     "region_concentration": 0.92, "color": "#d62728"},
        "Samsung":  {"share_pct": 43.0, "regions": ["Korea"],
                     "region_concentration": 0.90, "color": "#ff7f0e"},
    }
    HHI = sum((s["share_pct"] / 100) ** 2 for s in suppliers.values())
    for name, s in suppliers.items():
        s["geopolitical_risk"] = round(s["share_pct"] / 100 * s["region_concentration"], 3)
    return {"suppliers": suppliers, "DRAM_HHI": round(HHI, 4),
            "market_concentrated": HHI > 0.25}


# ════════════════════════════════════════════════════════════════════════════
# 3. Visualization
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(17, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    ax = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    # ── Panel 0: DRAM リテンション vs 温度 ──────────────────────────────
    T_arr = np.linspace(0, 120, 150)
    for node, col in [(n, DRAM_NODES[n]["color"]) for n in DRAM_NODES]:
        tref = [dram_cell_retention(node, T)["tREFI_ms"] for T in T_arr]
        ax[0].semilogy(T_arr, tref, color=col, lw=2.2, label=f"Micron {node}")
    ax[0].axhline(64.0, color="k", ls="--", lw=1.5, label="JEDEC 64ms")
    ax[0].set_xlabel("温度 [°C]"); ax[0].set_ylabel("tREFI [ms]")
    ax[0].set_title("Micron DRAM リテンション vs 温度\n(1z → 1γ ノード進化)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)

    # ── Panel 1: NAND P/E サイクル wearout ────────────────────────────────
    cyc_arr = np.logspace(1, 4, 200).astype(int)
    for cfg, col in [(c, NAND_CONFIGS[c]["color"]) for c in NAND_CONFIGS]:
        win = [nand_wearout(cfg, int(c))["window_remaining_pct"] for c in cyc_arr]
        ax[1].semilogx(cyc_arr, win, color=col, lw=2.2,
                       label=cfg.replace("_", " "))
    ax[1].axhline(10.0, color="k", ls="--", lw=1.5, label="10% window limit")
    ax[1].set_xlabel("P/E サイクル数"); ax[1].set_ylabel("Vth 窓 残存 [%]")
    ax[1].set_title("Micron NAND P/E サイクル wearout\n(SILC モデル, 85°C)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)
    ax[1].set_ylim(0, 105)

    # ── Panel 2: NAND 文字列歩留まり vs 層数 ────────────────────────────
    layers_arr = np.arange(64, 300, 8)
    for CD_s, col in [(1.5, "#2166ac"), (2.5, "#d62728"), (3.5, "#ff7f0e")]:
        from math import erfc, sqrt
        yd = []
        for nl in layers_arr:
            Vts = sqrt(82.0**2 + (CD_s*5)**2)
            Pf = 0.5 * erfc(200.0 / (Vts * sqrt(2)))
            yd.append((1-Pf)**nl * 100)
        ax[2].plot(layers_arr, yd, color=col, lw=2.2, label=f"σ_CD={CD_s}nm")
    ax[2].axvline(232, color="#2166ac", ls=":", lw=1.5, label="232L B58R")
    ax[2].set_xlabel("層数"); ax[2].set_ylabel("文字列歩留まり [%]")
    ax[2].set_title("Micron NAND 文字列歩留まり vs 層数\n(CD ばらつき → Vth spread)")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.25); ax[2].set_ylim(0, 100)

    # ── Panel 3: HBM 帯域幅 比較 ─────────────────────────────────────────
    labels = ["Micron\nHBM3", "Micron\nHBM3E", "SK Hynix\nHBM3E", "Samsung\nHBM3E"]
    bws    = [819, 1228, 1228, 1228]   # GB/s per stack
    caps   = [24, 36, 36, 36]           # GB per stack
    cols   = ["#ff7f0e", "#2166ac", "#d62728", "#1428A0"]
    x = np.arange(len(labels))
    ax[3].bar(x, bws, color=cols, width=0.5, alpha=0.85)
    ax3b = ax[3].twinx()
    ax3b.plot(x, caps, "D--", color="#333333", ms=8, lw=1.8)
    ax3b.set_ylabel("容量 [GB/stack]")
    ax[3].set_xticks(x); ax[3].set_xticklabels(labels, fontsize=8.5)
    ax[3].set_ylabel("帯域幅 [GB/s/stack]")
    ax[3].set_title("HBM3/3E 帯域幅・容量比較\n(Micron vs SK Hynix vs Samsung)")
    ax[3].grid(axis="y", alpha=0.25)

    # ── Panel 4: DRAM 密度 vs ノード ─────────────────────────────────────
    nodes_nm = [DRAM_NODES[n]["node_nm"] for n in DRAM_NODES]
    densities = [DRAM_NODES[n]["density_Gb_cm2"] for n in DRAM_NODES]
    colors4   = [DRAM_NODES[n]["color"] for n in DRAM_NODES]
    ax[4].semilogy(nodes_nm, densities, "o-", color="#2166ac", lw=2.5, ms=8)
    for nm, dn, col in zip(nodes_nm, densities, colors4):
        ax[4].scatter([nm], [dn], color=col, s=100, zorder=5)
    ax[4].set_xlabel("ノード [nm]"); ax[4].set_ylabel("ビット密度 [Gb/cm²]")
    ax[4].set_title("Micron DRAM ビット密度スケーリング\n(1z → 1γ)")
    ax[4].invert_xaxis(); ax[4].grid(alpha=0.25)

    # ── Panel 5: 地政学リスク ────────────────────────────────────────────
    ax[5].axis("off")
    risk = supply_concentration_risk()
    col_labels = ["企業", "市場シェア\n[%]", "生産地域", "地域集中度", "地政学\nリスク"]
    tdata = []
    for name, s in risk["suppliers"].items():
        tdata.append([name, f"{s['share_pct']:.0f}%",
                      "/".join(s["regions"]),
                      f"{s['region_concentration']:.2f}",
                      f"{s['geopolitical_risk']:.3f}"])
    tbl = ax[5].table(cellText=tdata, colLabels=col_labels,
                      cellLoc="center", loc="center", bbox=[0.0, 0.30, 1.0, 0.60])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1b5e8a"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f7ff")
    ax[5].text(0.5, 0.05, f"DRAM市場 HHI={risk['DRAM_HHI']:.3f} "
               f"({'集中' if risk['market_concentrated'] else '競争的'})",
               ha="center", fontsize=9, transform=ax[5].transAxes)
    ax[5].set_title("DRAM 地政学的供給リスク\n(Micron 米/日/台 分散優位)")

    fig.suptitle("Micron Technology — DRAM 1β/1γ · NAND 232L · HBM3E 物理モデル\n"
                 "セルリテンション · P/E劣化 · 文字列歩留まり · 地政学リスク",
                 fontsize=12, fontweight="bold")
    out = os.path.join(out_dir, "micron_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Micron model → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Micron DRAM/NAND/HBM model")
    ap.add_argument("--dram", action="store_true")
    ap.add_argument("--nand", action="store_true")
    ap.add_argument("--node",   default="1b", choices=list(DRAM_NODES.keys()))
    ap.add_argument("--layers", type=int, default=232)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    if args.dram:
        for T in [25, 55, 85, 105]:
            r = dram_cell_retention(args.node, T)
            print(f"  {args.node}  T={T:3d}°C  tREFI={r['tREFI_ms']:.2f}ms  "
                  f"yield={r['yield_pct']:.1f}%")
    if args.nand:
        for cyc in [100, 300, 500, 1000, 3000]:
            r = nand_wearout("232L_B58R", cyc)
            print(f"  232L  {cyc:5d} P/E  window={r['window_remaining_pct']:.1f}%")
    if not args.no_plot:
        plot_all()

if __name__ == "__main__":
    main()
