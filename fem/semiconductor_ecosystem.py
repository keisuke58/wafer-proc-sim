"""
半導体エコシステム モデル
========================
NVIDIA / TSMC / Samsung / Kioxia / ソシオネクスト を中心とした
バリューチェーンと各プロセス装置会社 (Disco / TEL / Lasertec / Advantest) の
相互依存関係を定量化。

1. エコシステム マップ
   各社の役割・依存関係・売上規模

2. ソシオネクスト ADAS SoC 設計制約
   street width 制約 / V_th ばらつき → タイミング余裕
   チップレット接続信頼性 × ダイシング品質

3. TSMC 2nm ロジック プロセス依存
   Lasertec (EUV) → TEL (ALD/etch) → Disco (dicing) → Advantest (test)

4. Samsung / Kioxia NAND メモリ チェーン
   HBM (SK Hynix) → Disco 研削 → Advantest メモリテスト

5. NVIDIA データセンター GPU サプライチェーン
   TSMC (logic) + SK Hynix (HBM) + SiC (電源) → GPU

実行:
    python fem/semiconductor_ecosystem.py
    python fem/semiconductor_ecosystem.py --socionext
    python fem/semiconductor_ecosystem.py --chain
    python fem/semiconductor_ecosystem.py --nvidia
"""

import argparse
import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ════════════════════════════════════════════════════════════════════════════
# 1. 半導体エコシステム データ (2025-2026 推定)
# ════════════════════════════════════════════════════════════════════════════

COMPANIES = {
    # 装置メーカー
    "Disco":       {"cap_B": 7.1,  "type": "equipment", "color": "#d62728",
                    "role": "Dicing/Grinding",    "ticker": "6146.T"},
    "TEL":         {"cap_B": 23.9, "type": "equipment", "color": "#ff7f0e",
                    "role": "CVD/Etch/ALD",        "ticker": "8035.T"},
    "Lasertec":    {"cap_B": 4.2,  "type": "equipment", "color": "#9467bd",
                    "role": "EUV/SiC Inspection",  "ticker": "6920.T"},
    "Advantest":   {"cap_B": 8.0,  "type": "equipment", "color": "#8c564b",
                    "role": "ATE Test",             "ticker": "6857.T"},
    # ファウンドリ/IDM
    "TSMC":        {"cap_B": 900,  "type": "foundry",   "color": "#2166ac",
                    "role": "Logic Foundry",        "ticker": "TSM"},
    "Samsung":     {"cap_B": 350,  "type": "idm",       "color": "#1f77b4",
                    "role": "Memory+Logic",         "ticker": "005930.KS"},
    "SK Hynix":    {"cap_B": 110,  "type": "memory",    "color": "#aec7e8",
                    "role": "DRAM/HBM",             "ticker": "000660.KS"},
    "Kioxia":      {"cap_B": 36.0, "type": "memory",    "color": "#17becf",
                    "role": "NAND Flash",           "ticker": "285A.T"},
    # チップ設計/ユーザー
    "NVIDIA":      {"cap_B": 3000, "type": "fabless",   "color": "#76b900",
                    "role": "GPU/AI Chip",          "ticker": "NVDA"},
    "Socionext":   {"cap_B": 0.46, "type": "fabless",   "color": "#e377c2",
                    "role": "ADAS/Data Center SoC", "ticker": "6526.T"},
    "Infineon":    {"cap_B": 35,   "type": "idm",       "color": "#bcbd22",
                    "role": "SiC Power Device",     "ticker": "IFX.DE"},
}

# バリューチェーン依存関係
DEPENDENCIES = [
    # (from, to, 依存の強さ 1-5, ラベル)
    ("Lasertec", "TSMC",      5, "EUV mask\ninspection"),
    ("Lasertec", "Samsung",   4, "EUV/SiC\ninspect"),
    ("TEL",      "TSMC",      5, "CVD/Etch/ALD"),
    ("TEL",      "Samsung",   4, "CVD/Etch"),
    ("Disco",    "Kioxia",    5, "NAND dicing"),
    ("Disco",    "SK Hynix",  5, "HBM grinding"),
    ("Disco",    "Infineon",  4, "SiC dicing"),
    ("Disco",    "TSMC",      3, "wafer dicing"),
    ("Advantest","Kioxia",    4, "NAND test"),
    ("Advantest","SK Hynix",  4, "HBM test"),
    ("Advantest","Infineon",  3, "power test"),
    ("TSMC",     "NVIDIA",    5, "GPU mfg"),
    ("TSMC",     "Socionext", 4, "SoC mfg"),
    ("SK Hynix", "NVIDIA",    5, "HBM supply"),
    ("Kioxia",   "NVIDIA",    3, "SSD storage"),
    ("Infineon", "NVIDIA",    3, "data center\npower"),
    ("Samsung",  "NVIDIA",    3, "HBM2/DDR"),
]


# ════════════════════════════════════════════════════════════════════════════
# 2. ソシオネクスト ADAS SoC 設計制約モデル
# ════════════════════════════════════════════════════════════════════════════

SOCIONEXT_SOC = {
    "SC2000A":  {"node_nm": 28,  "die_mm": 12.0, "street_um": 80.0,
                 "n_chiplets": 1, "V_dd": 1.0, "freq_GHz": 1.2,
                 "application": "ADAS Level 2", "customer": "Renesas/Toyota"},
    "SC2100":   {"node_nm": 7,   "die_mm": 9.0,  "street_um": 50.0,
                 "n_chiplets": 1, "V_dd": 0.8, "freq_GHz": 2.0,
                 "application": "ADAS Level 3", "customer": "BMW/Mercedes"},
    "SC2300":   {"node_nm": 5,   "die_mm": 8.0,  "street_um": 40.0,
                 "n_chiplets": 2, "V_dd": 0.75,"freq_GHz": 2.5,
                 "application": "ADAS Level 4 (next)", "customer": "Waymo/TBD"},
    "SocioAI1": {"node_nm": 3,   "die_mm": 10.0, "street_um": 30.0,
                 "n_chiplets": 4, "V_dd": 0.7, "freq_GHz": 3.5,
                 "application": "Data Center Inference", "customer": "Cloud"},
}


def socionext_timing_yield(chip: str, sigma_Vth_mV: float = 30.0,
                            sigma_freq_pct: float = 3.0) -> dict:
    """
    V_th ばらつき → タイミング余裕 → 歩留まり。
    Statistical Static Timing Analysis (SSTA) の簡略版。

    Δf/f ≈ -γ × ΔV_th/V_dd  (γ ≈ 0.5-1.0 for sub-threshold slope)
    """
    soc      = SOCIONEXT_SOC[chip]
    f_GHz    = soc["freq_GHz"]
    V_dd     = soc["V_dd"]
    gamma    = 0.7   # 感度係数

    # V_th ばらつきによる周波数ばらつき
    df_frac  = gamma * (sigma_Vth_mV / 1000) / V_dd
    sigma_f  = f_GHz * math.sqrt(df_frac**2 + (sigma_freq_pct/100)**2)

    # タイミング余裕 (setup slack) = 1クロック周期 - 最大遅延
    t_period_ps = 1000 / f_GHz
    t_slack_ps  = t_period_ps * 0.15   # 15% マージン
    # スラック割れ確率
    n_paths    = 1e6   # クリティカルパス数
    z_score    = t_slack_ps / (sigma_f / f_GHz * t_period_ps)
    from scipy.stats import norm
    p_fail_path = 1 - norm.cdf(z_score)
    p_fail_die  = 1 - (1 - p_fail_path) ** n_paths

    # チップレット接続: ダイシング品質 → RDL 信頼性
    chiplet_reliability = 0.999 ** soc["n_chiplets"]

    yield_pct = (1 - p_fail_die) * chiplet_reliability * 100

    return {
        "chip":             chip,
        "node_nm":          soc["node_nm"],
        "street_um":        soc["street_um"],
        "sigma_f_GHz":      round(sigma_f, 4),
        "z_slack":          round(z_score, 2),
        "p_fail_die":       round(p_fail_die * 100, 4),
        "chiplet_rel":      round(chiplet_reliability * 100, 3),
        "timing_yield_pct": round(yield_pct, 3),
        "application":      soc["application"],
        "dicing_critical":  soc["street_um"] < 50,
    }


def socionext_dicing_spec(chip: str) -> dict:
    """
    ソシオネクスト SoC が Disco に要求するダイシング仕様。
    street width が狭いほど → ステルスダイシングが必要。
    """
    soc = SOCIONEXT_SOC[chip]
    sw  = soc["street_um"]

    # 推奨プロセス
    if sw >= 80:
        rec_process = "blade"
        chipping_spec_um = 5.0
    elif sw >= 40:
        rec_process = "laser_ps"
        chipping_spec_um = 1.0
    elif sw >= 20:
        rec_process = "stealth"
        chipping_spec_um = 0.5
    else:
        rec_process = "stealth+plasma"
        chipping_spec_um = 0.2

    return {
        "chip":            chip,
        "street_um":       sw,
        "recommended":     rec_process,
        "chipping_spec":   chipping_spec_um,
        "node_nm":         soc["node_nm"],
        "application":     soc["application"],
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. NVIDIA GPU サプライチェーン
# ════════════════════════════════════════════════════════════════════════════

def nvidia_gpu_bom(gpu_name: str = "Blackwell B200") -> dict:
    """
    NVIDIA GPU の Bill of Materials (BOM) と
    各装置メーカーへの依存度を定量化。
    """
    boms = {
        "Hopper H100": {
            "die_area_mm2": 814,   "node": "TSMC 4N",
            "hbm_stacks": 6,       "hbm_capacity_GB": 80,
            "sic_power_kW": 0.7,   "nand_TB": 0,
            "price_usd": 30000,
        },
        "Blackwell B200": {
            "die_area_mm2": 1000,  "node": "TSMC 3nm",
            "hbm_stacks": 8,       "hbm_capacity_GB": 192,
            "sic_power_kW": 1.2,   "nand_TB": 0,
            "price_usd": 40000,
        },
        "Rubin R100 (est)": {
            "die_area_mm2": 800,   "node": "TSMC 2nm",
            "hbm_stacks": 12,      "hbm_capacity_GB": 288,
            "sic_power_kW": 1.5,   "nand_TB": 0,
            "price_usd": 60000,
        },
    }
    b = boms.get(gpu_name, boms["Blackwell B200"])

    # 装置メーカー依存度推定
    disco_exp    = b["die_area_mm2"] * 0.002 + b["hbm_stacks"] * 5   # USD/GPU
    tel_exp      = b["die_area_mm2"] * 0.010                           # USD/GPU
    lasertec_exp = b["die_area_mm2"] * 0.003                           # USD/GPU
    advantest_exp= b["die_area_mm2"] * 0.005 + b["hbm_stacks"] * 3   # USD/GPU
    total_equip  = disco_exp + tel_exp + lasertec_exp + advantest_exp

    return {
        "gpu":              gpu_name,
        **b,
        "equipment_usd":    {
            "Disco":     round(disco_exp, 1),
            "TEL":       round(tel_exp, 1),
            "Lasertec":  round(lasertec_exp, 1),
            "Advantest": round(advantest_exp, 1),
        },
        "total_equip_usd":  round(total_equip, 1),
        "equip_pct_bom":    round(total_equip / b["price_usd"] * 100, 2),
    }


# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def plot_ecosystem(out_dir=OUT_DIR):
    """バリューチェーン依存関係マップ。"""
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")

    # 各社の表示位置
    positions = {
        "Lasertec":  (1.2, 6.2), "TEL":      (1.2, 4.5),
        "Disco":     (1.2, 2.8), "Advantest":(1.2, 1.2),
        "TSMC":      (4.5, 6.0), "Samsung":  (4.5, 4.2),
        "SK Hynix":  (4.5, 2.7), "Kioxia":   (4.5, 1.2),
        "NVIDIA":    (8.0, 5.5), "Socionext":(8.0, 3.5),
        "Infineon":  (8.0, 1.5),
    }
    type_shapes = {"equipment": "o", "foundry": "s", "memory": "D",
                   "fabless": "^", "idm": "P"}

    # 依存関係の線
    dep_drawn = set()
    for (src, dst, strength, label) in DEPENDENCIES:
        if src not in positions or dst not in positions:
            continue
        key = (src, dst)
        if key in dep_drawn:
            continue
        dep_drawn.add(key)
        x0, y0 = positions[src]
        x1, y1 = positions[dst]
        lw = strength * 0.5
        alpha = 0.3 + strength * 0.1
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=lw,
                                   color="gray", alpha=alpha,
                                   connectionstyle="arc3,rad=0.1"))
        xm, ym = (x0+x1)/2, (y0+y1)/2
        if strength >= 4:
            ax.text(xm, ym, label, fontsize=5.5, ha="center", va="center",
                    color="navy", alpha=0.7)

    # ノード
    type_colors = {"equipment":"#ff9999","foundry":"#99ccff",
                   "memory":"#99ffcc","fabless":"#ffcc99","idm":"#cc99ff"}
    for name, info in COMPANIES.items():
        if name not in positions:
            continue
        x, y = positions[name]
        cap   = info["cap_B"]
        size  = max(300, min(3000, cap * 0.5))
        ax.scatter(x, y, s=size, c=type_colors[info["type"]],
                   edgecolors="k", linewidths=1.5, zorder=5)
        ax.text(x, y+0.35, name, ha="center", va="bottom",
                fontsize=9, fontweight="bold")
        ax.text(x, y-0.38, f"¥{cap:.0f}兆" if cap >= 10 else f"¥{cap:.1f}兆",
                ha="center", va="top", fontsize=7.5, color="navy")

    # 列ラベル
    for xc, lbl in [(1.2,"装置メーカー"), (4.5,"製造"), (8.0,"設計/ユーザー")]:
        ax.text(xc, 6.8, lbl, ha="center", fontsize=11,
                fontweight="bold", color="#333")
        ax.axvline(xc+1.5, color="#ddd", lw=0.8, ls="--")

    # 凡例
    patches = [mpatches.Patch(color=v, label=k) for k, v in type_colors.items()]
    ax.legend(handles=patches, fontsize=8, loc="lower right",
              title="会社タイプ", title_fontsize=8)

    fig.suptitle("半導体エコシステム バリューチェーン (2026)\n"
                 "時価総額スケール表示 — 矢印の太さ = 依存強度",
                 fontsize=12, fontweight="bold")
    out = os.path.join(out_dir, "semiconductor_ecosystem.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Ecosystem map -> {out}")


def plot_socionext(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    chips  = list(SOCIONEXT_SOC.keys())
    colors = ["#d62728","#ff7f0e","#2ca02c","#2166ac"]

    # Panel 1: Timing yield vs V_th ばらつき
    ax = axes[0]
    sigma_arr = np.linspace(10, 80, 80)   # mV
    for chip, col in zip(chips, colors):
        yields = [socionext_timing_yield(chip, s)["timing_yield_pct"]
                  for s in sigma_arr]
        soc = SOCIONEXT_SOC[chip]
        ax.plot(sigma_arr, yields, color=col, lw=2,
                label=f"{chip} ({soc['node_nm']}nm, {soc['application'][:15]})")
    ax.axvline(30, color="gray", ls="--", lw=1.2, label="典型 σ_Vth=30mV")
    ax.axhline(95, color="purple", ls=":", lw=1.2, label="量産目標 95%")
    ax.set_xlabel("σ(V_th) [mV]", fontsize=11)
    ax.set_ylabel("Timing Yield [%]", fontsize=11)
    ax.set_title("ソシオネクスト SoC — V_th ばらつき → タイミング歩留まり\n"
                 "(SSTA 簡略モデル)", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.2)
    ax.set_ylim(0, 102)

    # Panel 2: ダイシング仕様マップ
    ax2 = axes[1]
    ax2.axis("off")
    specs = [socionext_dicing_spec(c) for c in chips]
    rows = [["チップ", "Street [µm]", "推奨プロセス", "チッピング上限", "用途"]]
    proc_colors = {"blade":"#d62728","laser_ps":"#ff7f0e",
                   "stealth":"#2166ac","stealth+plasma":"#9467bd"}
    for sp in specs:
        rows.append([sp["chip"], f"{sp['street_um']:.0f}µm",
                     sp["recommended"],
                     f"{sp['chipping_spec']}µm",
                     sp["application"][:18]])
    tbl = ax2.table(cellText=rows[1:], colLabels=rows[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.1, 2.1)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#333"); cell.set_text_props(color="white")
        elif r <= len(specs):
            proc = rows[r][2]
            cell.set_facecolor(
                {"blade":"#ffcccc","laser_ps":"#fff4cc",
                 "stealth":"#cce4ff","stealth+plasma":"#e8ccff"}.get(proc,"white"))
    ax2.set_title("ソシオネクスト SoC → Disco ダイシング仕様\n"
                  "street width 縮小 → ステルスへ移行", fontsize=10)

    fig.suptitle("ソシオネクスト ADAS/推論 SoC — プロセス制約解析\n"
                 "設計ノード微細化 × ダイシング精度要求", fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "socionext_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Socionext analysis -> {out}")


def plot_nvidia_chain(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    gpus   = list(nvidia_gpu_bom.__doc__ and
                  ["Hopper H100","Blackwell B200","Rubin R100 (est)"])
    gpus   = ["Hopper H100", "Blackwell B200", "Rubin R100 (est)"]
    boms   = [nvidia_gpu_bom(g) for g in gpus]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: 装置メーカー依存度 [USD/GPU]
    ax = axes[0]
    equip_companies = ["Disco","TEL","Lasertec","Advantest"]
    ecolors = {"Disco":"#d62728","TEL":"#ff7f0e",
               "Lasertec":"#9467bd","Advantest":"#8c564b"}
    x  = np.arange(len(gpus))
    w  = 0.18
    for k, eq in enumerate(equip_companies):
        vals = [b["equipment_usd"][eq] for b in boms]
        ax.bar(x + (k-1.5)*w, vals, w, color=ecolors[eq],
               label=eq, alpha=0.85, edgecolor="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace(" (est)","*") for g in gpus], fontsize=9)
    ax.set_ylabel("Equipment Cost Contribution [USD/GPU]", fontsize=10)
    ax.set_title("NVIDIA GPU 1枚あたりの\n装置メーカー貢献コスト", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.2, axis="y")

    # Panel 2: BOM 構成 × GPU 価格推移
    ax2 = axes[1]
    prices = [b["price_usd"] for b in boms]
    equip_totals = [b["total_equip_usd"] for b in boms]
    equip_pcts   = [b["equip_pct_bom"] for b in boms]

    ax2b = ax2.twinx()
    bars = ax2.bar(x, prices, 0.6, color="#76b900", alpha=0.5,
                   label="GPU 価格 [$]", edgecolor="k", lw=0.8)
    ax2b.plot(x, equip_pcts, "ro-", lw=2, ms=8,
              label="装置コスト比率 [%]")
    ax2.set_xticks(x)
    ax2.set_xticklabels([g.replace(" (est)","*") for g in gpus], fontsize=9)
    ax2.set_ylabel("GPU Price [USD]", fontsize=10, color="#76b900")
    ax2b.set_ylabel("Equipment Cost / Price [%]", fontsize=10, color="red")
    ax2.set_title("GPU 価格 × 装置コスト比率\n(*=推定)", fontsize=10)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labels1+labels2, fontsize=8)
    ax2.grid(alpha=0.2, axis="y")

    fig.suptitle("NVIDIA GPU サプライチェーン — Disco/TEL/Lasertec/Advantest の役割\n"
                 "H100→B200→R100: ノード微細化で装置依存が深まる",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "nvidia_supply_chain.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  GPU BOM サマリー:")
    print(f"  {'GPU':>22}  {'Price':>8}  {'Disco':>7}  "
          f"{'TEL':>7}  {'Lasertec':>9}  {'Adv':>7}  {'合計%':>7}")
    print("  " + "─"*70)
    for b in boms:
        print(f"  {b['gpu']:>22}  ${b['price_usd']:>7,}  "
              f"${b['equipment_usd']['Disco']:>5.0f}  "
              f"${b['equipment_usd']['TEL']:>5.0f}  "
              f"${b['equipment_usd']['Lasertec']:>7.0f}  "
              f"${b['equipment_usd']['Advantest']:>5.0f}  "
              f"{b['equip_pct_bom']:>5.1f}%")
    print(f"[✓] NVIDIA chain -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socionext", action="store_true")
    ap.add_argument("--chain",     action="store_true")
    ap.add_argument("--nvidia",    action="store_true")
    args = ap.parse_args()
    run_all = not any([args.socionext, args.chain, args.nvidia])
    if args.chain     or run_all: plot_ecosystem()
    if args.socionext or run_all: plot_socionext()
    if args.nvidia    or run_all: plot_nvidia_chain()


if __name__ == "__main__":
    main()
