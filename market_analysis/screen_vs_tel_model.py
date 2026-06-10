"""
Screen Holdings vs TEL 洗浄装置比較モデル
==========================================
枚葉式ウェーハ洗浄装置の2大メーカーを定量比較。

TEL (東京エレクトロン) CELLESTA:
    - スピン式 + 薬液シャワー
    - SiC 対応: Piranha + H₂水 + Megasonic

Screen Holdings (7735) SS-3200 / SU-3300:
    - SAPS (Spin-spray Alternating Proximity System)
    - 超臨界CO₂洗浄 (SCW) も展開

比較軸:
    1. スループット [wafers/hour]
    2. 粒子除去率 (Particle Removal Efficiency)
    3. 薬液消費量 [mL/wafer]
    4. SiC 対応度
    5. TCO (Total Cost of Ownership)

実行:
    python fem/screen_vs_tel_model.py
"""

import math, os, sys
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── 装置スペック ──────────────────────────────────────────────────────────────
TOOLS = {
    "TEL CELLESTA": {
        "maker": "Tokyo Electron",
        "platform": "Spin + Spray",
        "wafers_hr": 120,
        "chemical_mL_wafer": 80,
        "DIW_L_wafer": 3.5,
        "PRE_Si_pct": 99.5,
        "PRE_SiC_pct": 97.0,     # SiC は化学的安定性から若干低め
        "metal_removal_pct": 99.8,
        "carbon_removal_pct": 95.0,
        "H2_water_support": True,
        "megasonic_support": True,
        "footprint_m2": 4.5,
        "price_MUSD": 4.5,
        "color": "#d62728",
        "note": "SiC プロセス実績豊富、H₂水対応済み",
    },
    "Screen SS-3300": {
        "maker": "Screen Holdings",
        "platform": "SAPS (Proximity Spray)",
        "wafers_hr": 130,
        "chemical_mL_wafer": 60,   # SAPS は薬液消費少ない
        "DIW_L_wafer": 2.8,
        "PRE_Si_pct": 99.8,
        "PRE_SiC_pct": 96.5,
        "metal_removal_pct": 99.7,
        "carbon_removal_pct": 93.0,
        "H2_water_support": True,
        "megasonic_support": False,
        "footprint_m2": 4.2,
        "price_MUSD": 4.2,
        "color": "#2166ac",
        "note": "薬液消費量が少なく TCO 優位",
    },
    "SEMES SC-3000": {
        "maker": "SEMES (Samsung)",
        "platform": "Spin + Spray",
        "wafers_hr": 100,
        "chemical_mL_wafer": 90,
        "DIW_L_wafer": 4.0,
        "PRE_Si_pct": 99.2,
        "PRE_SiC_pct": 94.0,
        "metal_removal_pct": 99.5,
        "carbon_removal_pct": 90.0,
        "H2_water_support": False,
        "megasonic_support": False,
        "footprint_m2": 5.0,
        "price_MUSD": 3.8,
        "color": "#ff7f0e",
        "note": "Samsung 向け最適化、SiC 対応は限定的",
    },
}

# ── TCO 計算パラメータ ────────────────────────────────────────────────────────
TCO_PARAMS = {
    "wafers_yr":     200000,   # 年間処理枚数
    "chemical_cost_per_L": 15, # USD/L  薬液平均単価
    "DIW_cost_per_L":    0.01,
    "depreciation_yr":  7,
    "utilization":       0.85,
    "labor_USD_hr":      50,
    "maintenance_pct":   0.08,  # CAPEX の 8%/年
}


def tco_per_wafer(tool_name: str) -> dict:
    """TCO [USD/wafer] を計算。"""
    t = TOOLS[tool_name]
    p = TCO_PARAMS

    # 年間稼働時間
    op_hr_yr = 8760 * p["utilization"]
    actual_wpy = t["wafers_hr"] * op_hr_yr

    # CAPEX 減価償却
    capex_ann = t["price_MUSD"] * 1e6 / p["depreciation_yr"]
    capex_per_wafer = capex_ann / actual_wpy

    # ランニングコスト
    chem_per_wafer  = t["chemical_mL_wafer"] / 1000 * p["chemical_cost_per_L"]
    diw_per_wafer   = t["DIW_L_wafer"] * p["DIW_cost_per_L"]
    maint_per_wafer = (t["price_MUSD"] * 1e6 * p["maintenance_pct"]) / actual_wpy
    labor_per_wafer = p["labor_USD_hr"] / t["wafers_hr"]

    total = capex_per_wafer + chem_per_wafer + diw_per_wafer + maint_per_wafer + labor_per_wafer
    return {
        "tool":             tool_name,
        "capex_per_wafer":  round(capex_per_wafer, 3),
        "chem_per_wafer":   round(chem_per_wafer, 3),
        "diw_per_wafer":    round(diw_per_wafer, 4),
        "maint_per_wafer":  round(maint_per_wafer, 3),
        "labor_per_wafer":  round(labor_per_wafer, 3),
        "total_USD_wafer":  round(total, 3),
    }


def sic_cleaning_score(tool_name: str) -> float:
    """SiC 洗浄総合スコア [0-100]。"""
    t = TOOLS[tool_name]
    score = (
        t["PRE_SiC_pct"] * 0.35 +
        t["metal_removal_pct"] * 0.25 +
        t["carbon_removal_pct"] * 0.25 +
        (10 if t["H2_water_support"] else 0) * 0.10 +
        (5  if t["megasonic_support"] else 0) * 0.05
    )
    return round(score, 1)


# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    tool_names = list(TOOLS.keys())
    colors = [TOOLS[t]["color"] for t in tool_names]

    # Panel 1: PRE (粒子除去率) Si vs SiC
    ax = axes[0, 0]
    x = np.arange(len(tool_names))
    w = 0.35
    pre_si  = [TOOLS[t]["PRE_Si_pct"]  for t in tool_names]
    pre_sic = [TOOLS[t]["PRE_SiC_pct"] for t in tool_names]
    ax.bar(x-w/2, pre_si,  w, color=colors, alpha=0.5, edgecolor="k", lw=0.8, label="Si")
    ax.bar(x+w/2, pre_sic, w, color=colors, alpha=0.95, edgecolor="k", lw=0.8, label="SiC")
    ax.set_xticks(x); ax.set_xticklabels([t.split()[0]+"\n"+t.split()[1] for t in tool_names], fontsize=9)
    ax.set_ylabel("Particle Removal Efficiency [%]", fontsize=10)
    ax.set_title("PRE: Si vs SiC\n(SiC は化学的安定性から低め)", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.2, axis="y")
    ax.set_ylim(90, 101)

    # Panel 2: 薬液消費量 vs スループット
    ax2 = axes[0, 1]
    for t_name, col in zip(tool_names, colors):
        t = TOOLS[t_name]
        ax2.scatter(t["wafers_hr"], t["chemical_mL_wafer"],
                    s=300, color=col, edgecolors="k", lw=1.5, zorder=5)
        ax2.annotate(t_name, xy=(t["wafers_hr"], t["chemical_mL_wafer"]),
                     xytext=(t["wafers_hr"]+2, t["chemical_mL_wafer"]+1),
                     fontsize=8)
    ax2.set_xlabel("Throughput [wafers/hr]", fontsize=10)
    ax2.set_ylabel("Chemical consumption [mL/wafer]", fontsize=10)
    ax2.set_title("Throughput vs Chemical Use\n(左下が理想)", fontsize=10)
    ax2.grid(alpha=0.2)
    # 理想方向矢印
    ax2.annotate("", xy=(135, 55), xytext=(95, 95),
                 arrowprops={"arrowstyle":"->","color":"green","lw":2})
    ax2.text(110, 72, "理想方向", color="green", fontsize=9, rotation=-40)

    # Panel 3: TCO 内訳
    ax3 = axes[0, 2]
    tco_data = [tco_per_wafer(t) for t in tool_names]
    cats = ["capex_per_wafer","chem_per_wafer","maint_per_wafer","labor_per_wafer"]
    cat_labels = ["CAPEX","Chemical","Maintenance","Labor"]
    cat_colors = ["#2166ac","#d62728","#ff7f0e","#2ca02c"]
    bottom = np.zeros(len(tool_names))
    for cat, col, lbl in zip(cats, cat_colors, cat_labels):
        vals = [tco[cat] for tco in tco_data]
        ax3.bar(tool_names, vals, bottom=bottom, color=col, alpha=0.85, label=lbl, edgecolor="k", lw=0.5)
        bottom += np.array(vals)
    totals = [tco["total_USD_wafer"] for tco in tco_data]
    for i, (t_name, total) in enumerate(zip(tool_names, totals)):
        ax3.text(i, total+0.002, f"${total:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax3.set_ylabel("TCO [USD/wafer]", fontsize=10)
    ax3.set_title("Total Cost of Ownership\nBreakdown per Wafer", fontsize=10)
    ax3.legend(fontsize=8, loc="upper right"); ax3.grid(alpha=0.2, axis="y")
    ax3.set_xticklabels([t.replace(" ","\n") for t in tool_names], fontsize=8)

    # Panel 4: SiC 洗浄総合スコア
    ax4 = axes[1, 0]
    scores = [sic_cleaning_score(t) for t in tool_names]
    bars = ax4.bar(tool_names, scores, color=colors, alpha=0.85, edgecolor="k", lw=0.8)
    for bar, score in zip(bars, scores):
        ax4.text(bar.get_x()+bar.get_width()/2, score+0.1, f"{score:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax4.set_ylabel("SiC Cleaning Score [0-100]", fontsize=10)
    ax4.set_title("SiC Cleaning Overall Score\n(PRE×0.35 + Metal×0.25 + Carbon×0.25 + H₂水+Mega)", fontsize=10)
    ax4.set_xticklabels([t.replace(" ","\n") for t in tool_names], fontsize=9)
    ax4.grid(alpha=0.2, axis="y"); ax4.set_ylim(85, 100)

    # Panel 5: SiC 需要増加 → 市場拡大シミュレーション
    ax5 = axes[1, 1]
    years = np.arange(2023, 2031)
    sic_market_share = np.array([0.05, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32])  # 洗浄装置市場のSiC比率
    total_market_B = np.array([3.5, 3.9, 4.4, 5.0, 5.6, 6.2, 6.9, 7.6])            # 洗浄装置市場規模 [B USD]
    sic_opportunity = total_market_B * sic_market_share
    tel_share = 0.35; screen_share = 0.25
    ax5.fill_between(years, 0, sic_opportunity * tel_share,
                     alpha=0.7, color="#d62728", label="TEL CELLESTA")
    ax5.fill_between(years, sic_opportunity * tel_share,
                     sic_opportunity * (tel_share + screen_share),
                     alpha=0.7, color="#2166ac", label="Screen SS")
    ax5.fill_between(years, sic_opportunity * (tel_share + screen_share),
                     sic_opportunity, alpha=0.4, color="gray", label="Others")
    ax5.set_xlabel("Year", fontsize=10)
    ax5.set_ylabel("SiC cleaning market [B USD]", fontsize=10)
    ax5.set_title("SiC Cleaning Equipment Market\n(TEL vs Screen share)", fontsize=10)
    ax5.legend(fontsize=8); ax5.grid(alpha=0.2)

    # Panel 6: 比較サマリー表
    ax6 = axes[1, 2]
    ax6.axis("off")
    rows = [["", "TEL\nCELLESTA", "Screen\nSS-3300", "SEMES\nSC-3000"]]
    metrics = [
        ("スループット", "wafers_hr", "{} wph"),
        ("PRE (SiC)", "PRE_SiC_pct", "{}%"),
        ("薬液消費", "chemical_mL_wafer", "{} mL"),
        ("H₂水対応", "H2_water_support", "{}"),
        ("Megasonic", "megasonic_support", "{}"),
        ("装置価格", "price_MUSD", "${} M"),
    ]
    for lbl, key, fmt in metrics:
        row = [lbl]
        for t in tool_names:
            val = TOOLS[t][key]
            if isinstance(val, bool):
                row.append("✓" if val else "—")
            else:
                row.append(fmt.format(val))
        rows.append(row)
    tbl = ax6.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.2, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0: cell.set_facecolor("#333"); cell.set_text_props(color="w")
        elif c == 1 and r > 0: cell.set_facecolor("#ffe0e0")
        elif c == 2 and r > 0: cell.set_facecolor("#e0e8ff")
        elif r % 2 == 0: cell.set_facecolor("#f5f5f5")
    ax6.set_title("Spec Comparison\nTEL vs Screen vs SEMES", fontsize=10)

    fig.suptitle("枚葉洗浄装置 競合比較 — TEL CELLESTA vs Screen SS-3300\n"
                 "PRE / TCO / SiC対応スコア / 市場シェア", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(out_dir, "screen_vs_tel.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print("\n  洗浄装置 TCO 比較:")
    for t in tool_names:
        tco = tco_per_wafer(t)
        score = sic_cleaning_score(t)
        print(f"  {t:<20} TCO=${tco['total_USD_wafer']:.3f}/wafer  SiC score={score}")
    print(f"\n[OK] Screen vs TEL -> {out}")


if __name__ == "__main__":
    plot_all()
