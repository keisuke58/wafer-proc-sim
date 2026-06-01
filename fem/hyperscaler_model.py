"""
Hyperscaler + Tesla カスタムシリコン モデル (2026)
====================================================
Microsoft / Google / AWS / Meta / Tesla の自社ASIC開発競争と
半導体サプライチェーンへの影響を定量化。

モデル内容:
  1. カスタムシリコン スペック比較
     Maia 200 / TPU v6e / Trainium3 / MTIA 500 / Tesla AI5
  2. TSMC ウェーハ消費量 モデル
     各社 ASIC + NVIDIA 需要を 2nm/3nm/5nm に分解
  3. TCO 分析 — カスタムASIC vs NVIDIA H100/B200
     TCO 優位性 40〜65% を定量化
  4. Tesla AI5 + Dojo 3
     ウェーハスケール計算機 + FSD/Optimus 推論需要
  5. NVIDIA シェア浸食モデル
     2024→2030 GPU vs ASIC シェア推移
  6. CapEx → 装置需要 波及 (日本株への影響)

実行:
    python fem/hyperscaler_model.py
    python fem/hyperscaler_model.py --specs
    python fem/hyperscaler_model.py --tsmc
    python fem/hyperscaler_model.py --tco
    python fem/hyperscaler_model.py --tesla

参考文献:
    Tom's Hardware (2026) — Custom AI ASIC state of play
    TSMC Q1 2026 Earnings — Wafer revenue mix
    buildmvpfast.com — Hyperscaler CapEx $770B 2026
    applyingai.com (2026) — Tesla AI5 40x performance
    Broadcom 2026 — CoWoS pre-order 200K (Google TPU)
"""

import argparse
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ════════════════════════════════════════════════════════════════════════════
# 1. カスタムシリコン スペックデータ (2026)
# ════════════════════════════════════════════════════════════════════════════

ASIC_SPECS = {
    # name: {tflops_bf16, tdp_W, hbm_GB, process_nm, foundry,
    #         die_area_mm2, year, company, color, wafer_price_k}
    "NVIDIA H100 SXM": {
        "tflops_bf16": 989,  "tdp_W": 700,  "hbm_GB": 80,
        "process_nm": 4.0,   "foundry": "TSMC", "die_area_mm2": 814,
        "year": 2022,        "company": "NVIDIA",  "color": "#76b900",
        "wafer_price_k": 17.0,
    },
    "NVIDIA B200 SXM": {
        "tflops_bf16": 4500, "tdp_W": 1000, "hbm_GB": 192,
        "process_nm": 3.0,   "foundry": "TSMC", "die_area_mm2": 900,
        "year": 2024,        "company": "NVIDIA",  "color": "#5da000",
        "wafer_price_k": 20.0,
    },
    "Google TPU v6e\n(Trillium)": {
        "tflops_bf16": 918,  "tdp_W": 150,  "hbm_GB": 32,
        "process_nm": 3.0,   "foundry": "TSMC", "die_area_mm2": 330,
        "year": 2024,        "company": "Google", "color": "#4285f4",
        "wafer_price_k": 20.0,
    },
    "AWS Trainium3": {
        "tflops_bf16": 1200, "tdp_W": 400,  "hbm_GB": 96,
        "process_nm": 3.0,   "foundry": "TSMC", "die_area_mm2": 450,
        "year": 2025,        "company": "AWS",    "color": "#ff9900",
        "wafer_price_k": 20.0,
    },
    "Microsoft Maia 200": {
        "tflops_bf16": 2100, "tdp_W": 500,  "hbm_GB": 128,
        "process_nm": 3.0,   "foundry": "TSMC", "die_area_mm2": 500,
        "year": 2025,        "company": "Microsoft", "color": "#00a4ef",
        "wafer_price_k": 20.0,
    },
    "Meta MTIA 500": {
        "tflops_bf16": 800,  "tdp_W": 200,  "hbm_GB": 64,
        "process_nm": 3.0,   "foundry": "TSMC", "die_area_mm2": 380,
        "year": 2026,        "company": "Meta",   "color": "#1877f2",
        "wafer_price_k": 20.0,
    },
    "Tesla AI5": {
        "tflops_bf16": 3200, "tdp_W": 300,  "hbm_GB": 128,
        "process_nm": 2.0,   "foundry": "TSMC+Samsung", "die_area_mm2": 420,
        "year": 2026,        "company": "Tesla",  "color": "#cc0000",
        "wafer_price_k": 30.0,  # 2nm wafer ~$30k
    },
    "Tesla Dojo D1\n(wafer-scale)": {
        "tflops_bf16": 362,  "tdp_W": 400,  "hbm_GB": 0,
        "process_nm": 7.0,   "foundry": "TSMC", "die_area_mm2": 645,
        "year": 2023,        "company": "Tesla",  "color": "#e00000",
        "wafer_price_k": 10.0,
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 2. TSMC ウェーハ消費量モデル
# ════════════════════════════════════════════════════════════════════════════

# 各社の 2026 AI CapEx とウェーハへの換算係数
HYPERSCALER_CAPEX = {
    # company: {capex_B, chip_capex_frac, avg_wafer_node_nm, color}
    "Microsoft":  {"capex_B": 80,  "chip_frac": 0.35, "node_nm": 3.0,  "color": "#00a4ef"},
    "Google":     {"capex_B": 75,  "chip_frac": 0.40, "node_nm": 3.0,  "color": "#4285f4"},
    "Meta":       {"capex_B": 125, "chip_frac": 0.30, "node_nm": 3.0,  "color": "#1877f2"},
    "AWS":        {"capex_B": 85,  "chip_frac": 0.35, "node_nm": 3.0,  "color": "#ff9900"},
    "Tesla":      {"capex_B": 15,  "chip_frac": 0.60, "node_nm": 2.5,  "color": "#cc0000"},
    "xAI":        {"capex_B": 20,  "chip_frac": 0.70, "node_nm": 2.0,  "color": "#c0392b"},
    "SpaceX":     {"capex_B": 5,   "chip_frac": 0.50, "node_nm": 1.8,  "color": "#1a1a2e"},
    "OpenAI":     {"capex_B": 30,  "chip_frac": 0.50, "node_nm": 3.0,  "color": "#10a37f"},
}

# TSMC ウェーハ価格 [千ドル/枚] (2026)
WAFER_PRICE_K = {
    1.8:  32.0,  # Intel 18A (Terafab)
    2.0:  30.0,  # N2
    3.0:  20.0,  # N3
    4.0:  17.0,  # N4/N5
    5.0:  14.0,
    7.0:  10.0,
}


def wafer_demand(company: str, spec: dict) -> dict:
    """CapEx → 半導体調達額 → 必要ウェーハ枚数 を推定。"""
    chip_spend_B = spec["capex_B"] * spec["chip_frac"]
    node_nm      = spec["node_nm"]

    # 最も近いウェーハ価格を選択
    closest_nm = min(WAFER_PRICE_K.keys(), key=lambda x: abs(x - node_nm))
    price_k    = WAFER_PRICE_K[closest_nm]

    wafers = int(chip_spend_B * 1e9 / (price_k * 1e3))
    wafers_per_day = math.ceil(wafers / 365)

    return {
        "company":         company,
        "chip_spend_B":    chip_spend_B,
        "wafer_price_k":   price_k,
        "wafers_total":    wafers,
        "wafers_per_day":  wafers_per_day,
        "node_nm":         node_nm,
    }


def tsmc_wafer_mix() -> dict:
    """TSMC Q1 2026 実績ベースの ウェーハ収益構成。"""
    return {
        "3nm":  {"share_pct": 25, "color": "#d73027", "price_k": 20.0},
        "5nm":  {"share_pct": 36, "color": "#f46d43", "price_k": 14.0},
        "7nm":  {"share_pct": 13, "color": "#fdae61", "color2": "#fee090"},
        "10nm+":{"share_pct": 26, "color": "#ffffbf"},
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. TCO 分析 — カスタム ASIC vs NVIDIA
# ════════════════════════════════════════════════════════════════════════════

def tco_analysis(chip_name: str, specs: dict, nvidia_ref: str = "NVIDIA H100 SXM") -> dict:
    """
    TCO (3年所有コスト) = ハード調達 + 電力 + 冷却 + 運用

    TCO指標: $/TFLOPS/year および TFlops/W (演算効率)
    """
    if chip_name not in ASIC_SPECS or nvidia_ref not in ASIC_SPECS:
        return {}

    s     = ASIC_SPECS[chip_name]
    s_ref = ASIC_SPECS[nvidia_ref]

    # ダイあたりコスト (Murphy yield 近似)
    def die_cost(sp):
        A    = sp["die_area_mm2"] * 1e-2  # cm²
        D0   = 0.10
        Y    = (1 - math.exp(-D0 * A)) / (D0 * A)
        Y    = max(Y, 0.05)
        wafer_area_mm2 = math.pi * 150**2 * 0.70
        dpw  = wafer_area_mm2 / sp["die_area_mm2"] * Y
        waf_cost = sp["wafer_price_k"] * 1e3
        return waf_cost / max(dpw, 0.1)

    die_c     = die_cost(s)
    die_c_ref = die_cost(s_ref)

    # 電力コスト (3年, $0.04/kWh データセンター)
    power_cost_kWh = 0.04
    hours_3yr = 3 * 8760
    elec_c     = s["tdp_W"]     / 1000 * hours_3yr * power_cost_kWh
    elec_c_ref = s_ref["tdp_W"] / 1000 * hours_3yr * power_cost_kWh

    tco_total     = die_c     + elec_c
    tco_total_ref = die_c_ref + elec_c_ref

    tco_per_tflops     = tco_total     / s["tflops_bf16"]
    tco_per_tflops_ref = tco_total_ref / s_ref["tflops_bf16"]

    tco_advantage = (tco_per_tflops_ref - tco_per_tflops) / tco_per_tflops_ref * 100

    return {
        "chip":               chip_name,
        "tflops_bf16":        s["tflops_bf16"],
        "tdp_W":              s["tdp_W"],
        "flops_per_W":        s["tflops_bf16"] / s["tdp_W"],
        "die_cost_k":         die_c / 1000,
        "elec_cost_3yr_k":    elec_c / 1000,
        "tco_total_k":        tco_total / 1000,
        "tco_per_tflops":     tco_per_tflops,
        "tco_vs_h100_pct":    tco_advantage,  # 正 = 安い
    }


# ════════════════════════════════════════════════════════════════════════════
# 4. Tesla AI5 + Dojo モデル
# ════════════════════════════════════════════════════════════════════════════

TESLA_CHIPS = {
    "HW3 (2019)": {"tflops": 36,    "tdp_W": 36,  "node_nm": 14, "color": "#ffcccc"},
    "HW4 (2022)": {"tflops": 72,    "tdp_W": 72,  "node_nm": 7,  "color": "#ff9999"},
    "AI4 (2024)": {"tflops": 800,   "tdp_W": 150, "node_nm": 5,  "color": "#ff6666"},
    "AI5 (2026)": {"tflops": 3200,  "tdp_W": 300, "node_nm": 2,  "color": "#cc0000"},
    "AI6 (2028e)":{"tflops": 15000, "tdp_W": 350, "node_nm": 1.8,"color": "#990000"},
}

def tesla_fsd_inference_demand() -> dict:
    """
    Tesla FSD + Optimus の推論需要 → 必要チップ数 → ウェーハ消費

    仮定:
        - Tesla 車両台数: 600万台/年 (2026)
        - Optimus ロボット: 100万台/年 (2027〜)
        - HW5 搭載: 1台 = 2チップ
        - Dojo 3 トレーニング: 1 exaflop クラスタ
    """
    vehicles_per_yr  = 6_000_000
    optimus_per_yr   = 500_000      # 2026 目標
    chips_per_vehicle = 2           # デュアルチップ
    ai5_tflops = TESLA_CHIPS["AI5 (2026)"]["tflops"]

    n_chips_vehicle = vehicles_per_yr * chips_per_vehicle
    n_chips_optimus = optimus_per_yr  * 1   # Optimus 1チップ

    # Dojo 3: 1 exaflop (1e6 TFLOPS) → AI5 換算
    dojo_tflop_target = 1e6   # TFLOPS
    n_dojo_tiles = math.ceil(dojo_tflop_target / ai5_tflops)

    total_chips = n_chips_vehicle + n_chips_optimus + n_dojo_tiles

    # AI5 ウェーハ消費 (2nm, $30k/wafer)
    wafer_area = math.pi * 150**2 * 0.70
    die_area   = ASIC_SPECS["Tesla AI5"]["die_area_mm2"]
    D0         = 0.08
    A          = die_area * 1e-2
    Y          = (1 - math.exp(-D0 * A)) / (D0 * A)
    dpw        = wafer_area / die_area * Y
    wafers_needed = math.ceil(total_chips / dpw)

    return {
        "n_chips_vehicle":  n_chips_vehicle,
        "n_chips_optimus":  n_chips_optimus,
        "n_dojo_tiles":     n_dojo_tiles,
        "total_chips":      total_chips,
        "die_yield":        Y,
        "dies_per_wafer":   int(dpw),
        "wafers_needed":    wafers_needed,
        "wafer_cost_B":     wafers_needed * 30e3 / 1e9,
    }


# ════════════════════════════════════════════════════════════════════════════
# 5. NVIDIA シェア浸食モデル
# ════════════════════════════════════════════════════════════════════════════

NVIDIA_SHARE = {
    # year: (GPU_share_pct, ASIC_share_pct, total_market_B)
    2022: (92, 8,   45),
    2023: (88, 12,  80),
    2024: (80, 20,  160),
    2025: (75, 25,  280),
    2026: (70, 30,  420),  # Tom's Hardware: 70% GPU, 27.8% ASIC 2026
    2027: (62, 38,  600),
    2028: (55, 45,  800),
    2030: (40, 60,  1200),
}


# ════════════════════════════════════════════════════════════════════════════
# 6. Plotting
# ════════════════════════════════════════════════════════════════════════════

def plot_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor("#0a0a0a")
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.50, wspace=0.38,
                           left=0.07, right=0.97,
                           top=0.93, bottom=0.06)

    title_kw = dict(fontsize=9, color="#aaaaaa", pad=6)
    sc = "#333333"

    def style_ax(ax):
        ax.set_facecolor("#111111")
        for sp in ax.spines.values():
            sp.set_color(sc)
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")

    chip_names = list(ASIC_SPECS.keys())
    colors_c   = [ASIC_SPECS[c]["color"] for c in chip_names]

    # ── Panel 1: TFlops/W 演算効率比較 ────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1)
    fw = [ASIC_SPECS[c]["tflops_bf16"] / ASIC_SPECS[c]["tdp_W"]
          for c in chip_names]
    short_names = [c.split("\n")[0] for c in chip_names]
    bars1 = ax1.barh(short_names, fw, color=colors_c, height=0.65)
    for bar, v in zip(bars1, fw):
        ax1.text(v + 0.1, bar.get_y() + bar.get_height()/2,
                 f"{v:.1f}", va="center", fontsize=6.5, color="white")
    ax1.set_xlabel("TFlops/W (BF16)", fontsize=8)
    ax1.set_title("演算効率 (TFlops/W) 比較", **title_kw)

    # ── Panel 2: TFlops BF16 生産性比較 ────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2)
    tfl = [ASIC_SPECS[c]["tflops_bf16"] for c in chip_names]
    bars2 = ax2.barh(short_names, tfl, color=colors_c, height=0.65)
    for bar, v in zip(bars2, tfl):
        ax2.text(v + 10, bar.get_y() + bar.get_height()/2,
                 f"{v}", va="center", fontsize=6.5, color="white")
    ax2.set_xlabel("Peak TFlops (BF16)", fontsize=8)
    ax2.set_title("AI 演算性能 (BF16 TFlops)", **title_kw)

    # ── Panel 3: TCO vs NVIDIA H100 ────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3)
    tco_results = []
    for name in chip_names:
        if "H100" in name:
            continue
        t = tco_analysis(name, {})
        if t:
            tco_results.append((name.split("\n")[0], t))
    names3  = [r[0] for r in tco_results]
    adv3    = [r[1]["tco_vs_h100_pct"] for r in tco_results]
    col3    = [ASIC_SPECS[n if n in ASIC_SPECS else name]["color"]
               if n in ASIC_SPECS else "#888888"
               for n, r in tco_results]
    # 元の名前でカラー取得
    col3 = []
    for n, _ in tco_results:
        found = "#888888"
        for k in ASIC_SPECS:
            if k.split("\n")[0] == n:
                found = ASIC_SPECS[k]["color"]
                break
        col3.append(found)
    bars3 = ax3.barh(names3, adv3,
                     color=[c if a > 0 else "#666666"
                            for c, a in zip(col3, adv3)],
                     height=0.65)
    ax3.axvline(0, color="#aaaaaa", lw=1)
    for bar, v in zip(bars3, adv3):
        ax3.text(v + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{v:+.0f}%", va="center", fontsize=6.5, color="white")
    ax3.set_xlabel("TCO優位性 vs H100 [%] (正=安い)", fontsize=8)
    ax3.set_title("3年 TCO 優位性 vs NVIDIA H100", **title_kw)

    # ── Panel 4: Hyperscaler CapEx 2026 ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    style_ax(ax4)
    hc_names  = list(HYPERSCALER_CAPEX.keys())
    hc_capex  = [HYPERSCALER_CAPEX[c]["capex_B"] for c in hc_names]
    hc_chip   = [HYPERSCALER_CAPEX[c]["capex_B"] * HYPERSCALER_CAPEX[c]["chip_frac"]
                 for c in hc_names]
    hc_colors = [HYPERSCALER_CAPEX[c]["color"] for c in hc_names]
    x4 = np.arange(len(hc_names))
    w4 = 0.4
    ax4.bar(x4 - w4/2, hc_capex, w4, color=hc_colors, alpha=0.9, label="総 CapEx")
    ax4.bar(x4 + w4/2, hc_chip,  w4, color=hc_colors, alpha=0.5, label="チップ調達")
    ax4.set_xticks(x4)
    ax4.set_xticklabels(hc_names, rotation=30, ha="right", fontsize=7)
    ax4.set_ylabel("投資額 [十億ドル]", fontsize=8)
    ax4.legend(fontsize=6, facecolor="#222222", labelcolor="white")
    total_capex = sum(hc_capex)
    ax4.set_title(f"Hyperscaler CapEx 2026 (合計 ${total_capex:.0f}B)", **title_kw)

    # ── Panel 5: TSMC ウェーハ消費量 ────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    style_ax(ax5)
    hc_demands = {c: wafer_demand(c, HYPERSCALER_CAPEX[c]) for c in hc_names}
    wpd_vals   = [hc_demands[c]["wafers_per_day"] for c in hc_names]
    bars5 = ax5.bar(hc_names, wpd_vals, color=hc_colors, width=0.6)
    for bar, v in zip(bars5, wpd_vals):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f"{v:,}", ha="center", fontsize=6, color="white")
    ax5.set_xticklabels(hc_names, rotation=30, ha="right", fontsize=7)
    ax5.set_ylabel("推定ウェーハ消費 [枚/日]", fontsize=8)
    total_wpd = sum(wpd_vals)
    ax5.set_title(f"TSMC ウェーハ需要 (合計 {total_wpd:,}枚/日)", **title_kw)

    # ── Panel 6: TSMC ウェーハ収益 ノード別構成 ─────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor("#111111")
    mix_data = [
        ("3nm (25%)", 25, "#d73027"),
        ("5nm (36%)", 36, "#f46d43"),
        ("7nm (13%)", 13, "#fdae61"),
        ("10nm+ (26%)", 26, "#888888"),
    ]
    wedge_colors = [m[2] for m in mix_data]
    wedge_sizes  = [m[1] for m in mix_data]
    wedge_labels = [m[0] for m in mix_data]
    ax6.pie(wedge_sizes, labels=wedge_labels, colors=wedge_colors,
            autopct="%1.0f%%", startangle=140,
            wedgeprops=dict(edgecolor="#0a0a0a", lw=1.5),
            textprops={"color": "white", "fontsize": 7})
    ax6.set_title("TSMC Q1 2026 ウェーハ収益 ノード別", **title_kw)

    # ── Panel 7: NVIDIA シェア浸食 ────────────────────────────────
    ax7 = fig.add_subplot(gs[2, :2])
    style_ax(ax7)
    yrs  = list(NVIDIA_SHARE.keys())
    gpu_sh  = [NVIDIA_SHARE[y][0] for y in yrs]
    asic_sh = [NVIDIA_SHARE[y][1] for y in yrs]
    mkt_B   = [NVIDIA_SHARE[y][2] for y in yrs]
    ax7.stackplot(yrs, gpu_sh, asic_sh,
                  labels=["NVIDIA GPU", "Custom ASIC (GAFAM+Tesla)"],
                  colors=["#76b900", "#2166ac"], alpha=0.85)
    ax7_r = ax7.twinx()
    ax7_r.set_facecolor("#111111")
    ax7_r.plot(yrs, mkt_B, "o--", color="#fdae61", lw=2, ms=7,
               label="市場規模 [$B]")
    for y, m in zip(yrs, mkt_B):
        ax7_r.text(y, m + 15, f"${m}B", ha="center", fontsize=6, color="#fdae61")
    ax7_r.set_ylabel("AI チップ市場規模 [$B]", fontsize=8, color="#fdae61")
    ax7_r.tick_params(colors="#aaaaaa", labelsize=7)
    ax7.axvline(2026, color="white", ls=":", lw=1.5, alpha=0.5)
    ax7.set_xlabel("年", fontsize=8)
    ax7.set_ylabel("市場シェア [%]", fontsize=8)
    lines1, l1 = ax7.get_legend_handles_labels()
    lines2, l2 = ax7_r.get_legend_handles_labels()
    ax7.legend(lines1 + lines2, l1 + l2,
               fontsize=7, facecolor="#1a1a1a", labelcolor="white", loc="upper left")
    ax7.set_title("AI チップ市場: NVIDIA GPU vs Custom ASIC シェア推移", **title_kw)

    # ── Panel 8: Tesla AI5 需要試算 ────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 2])
    style_ax(ax8)
    tesla = tesla_fsd_inference_demand()
    categories = ["FSD\n車両向け", "Optimus\nロボット向け", "Dojo 3\n学習クラスタ"]
    values     = [tesla["n_chips_vehicle"] / 1e6,
                  tesla["n_chips_optimus"] / 1e6,
                  tesla["n_dojo_tiles"]    / 1e3]
    units      = ["百万枚", "百万枚", "千タイル"]
    ax8_cols   = ["#cc0000", "#ff4444", "#882222"]
    bars8 = ax8.bar(categories, [tesla["n_chips_vehicle"],
                                  tesla["n_chips_optimus"],
                                  tesla["n_dojo_tiles"]],
                    color=ax8_cols, width=0.6)
    for bar, v, u in zip(bars8, values, units):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.03,
                 f"{v:.1f}{u}", ha="center", fontsize=7, color="white")
    ax8.set_ylabel("AI5 チップ需要数", fontsize=8)
    ax8.set_title(f"Tesla AI5 需要試算\n合計 {tesla['total_chips']:,}枚 → "
                  f"{tesla['wafers_needed']:,} ウェーハ\n"
                  f"(${tesla['wafer_cost_B']:.1f}B @ 2nm)",
                  fontsize=8, color="#aaaaaa", pad=4)

    # Tesla チップ世代 ロードマップ (背景に追加テキスト)
    gen_ax = ax8.inset_axes([0.0, -0.45, 1.0, 0.35])
    gen_ax.set_facecolor("#0d0d0d")
    for sp in gen_ax.spines.values():
        sp.set_color(sc)
    gen_names = list(TESLA_CHIPS.keys())
    gen_tfl   = [TESLA_CHIPS[g]["tflops"] for g in gen_names]
    gen_cols  = [TESLA_CHIPS[g]["color"]  for g in gen_names]
    gen_ax.bar(gen_names, gen_tfl, color=gen_cols, width=0.6)
    for i, (n, v) in enumerate(zip(gen_names, gen_tfl)):
        gen_ax.text(i, v * 1.05, f"{v:,}", ha="center", fontsize=5.5,
                    color="white", rotation=45)
    gen_ax.set_ylabel("TFLOPS", fontsize=6, color="#aaaaaa")
    gen_ax.tick_params(colors="#888888", labelsize=5.5)
    gen_ax.set_title("Tesla ASIC ロードマップ", fontsize=7, color="#888888", pad=3)

    fig.suptitle(
        "Hyperscaler + Tesla カスタムシリコン 競争分析 2026 "
        "— CapEx $400B+, TSMC ウェーハ争奪戦",
        fontsize=13, color="white", fontweight="bold", y=0.97,
    )

    out = os.path.join(OUT_DIR, "hyperscaler_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  → {out}")
    return out


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _print_summary():
    print("\n=== Hyperscaler + Tesla カスタムシリコン サマリー ===")
    print("\n[演算効率 TFlops/W]")
    for name, s in ASIC_SPECS.items():
        eff = s["tflops_bf16"] / s["tdp_W"]
        print(f"  {name.split(chr(10))[0]:<25}  {eff:6.1f} TFlops/W  "
              f"({s['tflops_bf16']:5} TFlops  {s['tdp_W']}W  {s['process_nm']}nm)")

    print("\n[TCO 優位性 vs H100 (3年コスト/チップ)]")
    for name in ASIC_SPECS:
        if "H100" in name:
            continue
        t = tco_analysis(name, {})
        if t:
            total_usd = t["tco_total_k"] * 1000
            die_usd   = t["die_cost_k"]  * 1000
            elec_usd  = t["elec_cost_3yr_k"] * 1000
            print(f"  {name.split(chr(10))[0]:<25}  {t['tco_vs_h100_pct']:+5.0f}%  "
                  f"(ダイ ${die_usd:.0f} + 電力 ${elec_usd:.0f} = ${total_usd:.0f} / 3yr)")

    tesla = tesla_fsd_inference_demand()
    print(f"\n[Tesla AI5 年間需要]")
    print(f"  FSD 車両向け:      {tesla['n_chips_vehicle']:,} 枚")
    print(f"  Optimus ロボット:  {tesla['n_chips_optimus']:,} 枚")
    print(f"  Dojo 3 学習:       {tesla['n_dojo_tiles']:,} タイル")
    print(f"  必要 2nm ウェーハ: {tesla['wafers_needed']:,} 枚 (${tesla['wafer_cost_B']:.1f}B)")

    total_capex = sum(v["capex_B"] for v in HYPERSCALER_CAPEX.values())
    print(f"\n[Hyperscaler 総 CapEx 2026: ${total_capex:.0f}B]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperscaler モデル")
    parser.add_argument("--specs",  action="store_true")
    parser.add_argument("--tsmc",   action="store_true")
    parser.add_argument("--tco",    action="store_true")
    parser.add_argument("--tesla",  action="store_true")
    args = parser.parse_args()

    _print_summary()
    out = plot_all()
    print(f"\n全パネル出力: {out}")
