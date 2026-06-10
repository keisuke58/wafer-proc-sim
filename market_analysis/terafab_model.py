"""
Terafab — 垂直統合メガFab モデル (SpaceX × Tesla × xAI × Intel)
==================================================================
2026年3月21日 Elon Musk 発表。Austin, TX に建設予定の
年間1テラワット AIコンピューティング生産を目指す垂直統合Fab。

物理モデル内容:
  1. 年間生産目標モデル    — 1 TW をウェーハ/チップ枚数へ換算
  2. Intel 18A プロセス    — GAA RibbonFET 歩留まり + 密度
  3. 垂直統合 歩留まり連鎖 — 設計→Fab→メモリ→PKG→テスト
  4. CapEx/OpEx 分析      — $119B 投資スプレッドシート
  5. 宇宙向け RAD-hard    — 放射線TID劣化モデル (80%宇宙向け)
  6. 日本装置株 波及試算  — ASML/TEL/Disco/Lasertec 需要増

実行:
    python fem/terafab_model.py
    python fem/terafab_model.py --production
    python fem/terafab_model.py --yield
    python fem/terafab_model.py --capex
    python fem/terafab_model.py --radhard
    python fem/terafab_model.py --japan

参考文献:
    Musk, E. (2026) Terafab announcement — X/Twitter
    Intel 18A IEDM 2023 — Natarajan et al.
    JEDEC JESD57 — Radiation Hardness Assurance
    Murphy (1964) Proc. IEEE — die yield model
    TSMC Annual Report 2025 — Fab productivity benchmarks
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

# 物理定数
q_C  = 1.602e-19
k_B  = 1.381e-23
T_K  = 300.0

# ════════════════════════════════════════════════════════════════════════════
# 1. Terafab プロジェクト定数
# ════════════════════════════════════════════════════════════════════════════

TERAFAB = {
    "target_TW":         1.0,        # 年間生産目標 [テラワット演算]
    "space_fraction":    0.80,       # 宇宙向け比率
    "ground_fraction":   0.20,       # 地上向け比率
    "initial_capex_B":   55.0,       # 初期投資 [十億ドル]
    "total_capex_B":     119.0,      # 総投資 [十億ドル]
    "location":          "Austin, TX",
    "partners":          ["SpaceX", "Tesla", "xAI", "Intel"],
    "process_node_nm":   1.8,        # Intel 18A
    "announced":         "2026-03-21",
    "hvm_start":         "2029",     # 高量産開始予定
    "wafer_size_mm":     300,
    # Intel 18A プロセスパラメータ (IEDM 2023)
    "density_MTr_mm2":   200.0,      # 200 MTr/mm²
    "D0_per_cm2":        0.10,       # 欠陥密度 [/cm²]
    "die_area_mm2":      500.0,      # 大型 AI チップ想定
    "ai_chip_TFlops":    2000.0,     # チップ1枚あたり演算性能 [TFLOPS]
    "chip_TDP_W":        700.0,      # チップ消費電力 [W]
}

# ════════════════════════════════════════════════════════════════════════════
# 2. 年間生産目標モデル
# ════════════════════════════════════════════════════════════════════════════

def production_target(target_TW: float = 1.0) -> dict:
    """
    Terafab 年間チップ生産量モデル。

    Musk の "1 TW of compute" は新造語で電力単位ではなく演算性能の比喩。
    本モデルは CapEx $119B → 設備規模 → ウェーハ/日 から逆算し、
    target_TW を CapEx スケールファクターとして扱う。

    宇宙向け (80% CapEx 配分):
        SpaceX Starship / 軌道インフラ向け RAD-hard AI チップ
        小型 80mm²、Intel 18A ベース
    地上向け (20% CapEx 配分):
        xAI Grok クラスタ / Tesla Dojo 向け大型 AI アクセラレータ

    設備費 $3B = 10,000 wafers/月 キャパシティ (業界標準)
    参考: TSMC Q1 2026 CapEx 開示 / Intel 18A Fab コスト試算
    """
    total_capex_B  = 119.0 * target_TW   # スケールファクター
    equip_B        = total_capex_B * 0.60   # 60% が装置費 (業界標準)

    # $3B 装置費 = 10k wafers/month の capacity
    capex_per_10kwpm = 3.0
    wpd_total = equip_B / capex_per_10kwpm * 10_000 / 30

    D0             = 0.10
    wafer_area_mm2 = math.pi * (300 / 2) ** 2 * 0.70

    def segment(wpd, die_area_mm2):
        A   = die_area_mm2 * 1e-2
        Y   = max((1 - math.exp(-D0 * A)) / (D0 * A), 0.01)
        dpw = math.floor(wafer_area_mm2 / die_area_mm2) * Y
        chips_yr = math.floor(wpd * dpw * 365)
        return {"n_chips": chips_yr, "die_yield": Y, "dpw": dpw,
                "wpd": wpd}

    # 宇宙向け ライン (CapEx 80%)
    space_area, space_tflops, space_tdp = 80.0, 50.0, 50.0
    sp = segment(wpd_total * 0.80, space_area)
    sp["tflops_total"] = sp["n_chips"] * space_tflops
    sp["desc"] = f"宇宙向け RAD-hard (50W, 50T, {space_area:.0f}mm²)"

    # 地上向け ライン (CapEx 20%)
    ground_area, ground_tflops, ground_tdp = 450.0, 3000.0, 500.0
    gr = segment(wpd_total * 0.20, ground_area)
    gr["tflops_total"] = gr["n_chips"] * ground_tflops
    gr["desc"] = f"地上向け AI Accel (500W, 3000T, {ground_area:.0f}mm²)"

    total_tflops_B = (sp["tflops_total"] + gr["tflops_total"]) / 1e9  # B-TFLOPS

    # "1 TW of compute" ≒ 全チップ合算ピーク演算 [EFlops相当]
    # 1 B-TFLOPS = 1e9 × 1e12 FLOPS/s = 1e21 FLOPS/s = 1 ZettaFLOPS/s
    # → Musk 単位では 1 TW ≒ 221 B-TFLOPS (発表時の目標スケール感)
    implied_TW_note = f"{total_tflops_B:.1f} B-TFLOPS (≈ EFlops 規模)"

    return {
        "target_TW":        target_TW,
        "space":            sp,
        "ground":           gr,
        "n_chips_needed":   sp["n_chips"] + gr["n_chips"],
        "die_yield":        (sp["die_yield"] * 0.8 + gr["die_yield"] * 0.2),
        "dies_per_wafer":   (sp["dpw"] * 0.8 + gr["dpw"] * 0.2),
        "wafers_needed":    math.ceil((sp["wpd"] + gr["wpd"]) * 365),
        "wafers_per_day":   math.ceil(wpd_total),
        "total_tflops_B":   total_tflops_B,   # B-TFLOPS (ピーク合算)
        "implied_TW_note":  implied_TW_note,
        "perf_per_W":       ground_tflops / ground_tdp,
    }

# ════════════════════════════════════════════════════════════════════════════
# 3. 垂直統合 歩留まり連鎖モデル
# ════════════════════════════════════════════════════════════════════════════

PROCESS_STEPS = {
    "Litho (EUV High-NA)":   {"Y": 0.985, "step": 1, "color": "#2166ac"},
    "Etch/Depo (TEL)":       {"Y": 0.992, "step": 2, "color": "#4393c3"},
    "CMP (Planarization)":   {"Y": 0.990, "step": 3, "color": "#74add1"},
    "Die Fab (18A Logic)":   {"Y": 0.820, "step": 4, "color": "#f4a582"},
    "Memory (HBM4)":         {"Y": 0.880, "step": 5, "color": "#d6604d"},
    "Hybrid Bonding (PKG)":  {"Y": 0.950, "step": 6, "color": "#b2182b"},
    "Final Test (ATE)":      {"Y": 0.975, "step": 7, "color": "#67001f"},
}

def yield_chain(steps: dict = PROCESS_STEPS) -> dict:
    """各プロセスステップの歩留まりを積算してシステム歩留まりを計算。"""
    cumulative = 1.0
    results = {}
    for name, v in steps.items():
        cumulative *= v["Y"]
        results[name] = {
            "step_yield":  v["Y"],
            "cumulative":  cumulative,
            "color":       v["color"],
        }
    return results

# ════════════════════════════════════════════════════════════════════════════
# 4. CapEx / OpEx 分析
# ════════════════════════════════════════════════════════════════════════════

CAPEX_BREAKDOWN = {
    # カテゴリ: [金額(十億ドル), 色]
    "EUV / Litho 装置 (ASML)":     [18.0, "#2166ac"],
    "Etch / Depo 装置 (TEL/LAM)":  [14.0, "#4393c3"],
    "CMP / 洗浄装置":              [6.0,  "#74add1"],
    "検査・計測 (Lasertec/KLA)":    [5.0,  "#abd9e9"],
    "テスト装置 (Advantest)":       [4.0,  "#e0f3f8"],
    "メモリ Fab 設備":              [22.0, "#d73027"],
    "先端パッケージング":           [15.0, "#f46d43"],
    "建屋・インフラ・電力":         [20.0, "#808080"],
    "ソフト・設計 (EDA/IP)":        [5.0,  "#b8b8b8"],
    "予備費・その他":               [10.0, "#d8d8d8"],
}

def capex_analysis(breakdown: dict = CAPEX_BREAKDOWN) -> dict:
    total = sum(v[0] for v in breakdown.values())
    japan_equip = (breakdown["EUV / Litho 装置 (ASML)"][0] +
                   breakdown["Etch / Depo 装置 (TEL/LAM)"][0] +
                   breakdown["検査・計測 (Lasertec/KLA)"][0] +
                   breakdown["テスト装置 (Advantest)"][0])
    return {
        "total_B":         total,
        "japan_equip_B":   japan_equip,
        "japan_share_pct": japan_equip / total * 100,
        "breakdown":       breakdown,
    }

# ════════════════════════════════════════════════════════════════════════════
# 5. 宇宙向け RAD-hard 劣化モデル
# ════════════════════════════════════════════════════════════════════════════

def rad_hard_tid_model(tid_krad_range: np.ndarray,
                       Vth_shift_per_krad: float = 5e-3) -> dict:
    """
    Total Ionizing Dose (TID) によるトランジスタ特性劣化。

    Vth シフト: ΔVth = α × TID  (α = Vth_shift_per_krad [V/krad])
    Intel 18A RAD-hard 設計: α ≈ 2 mV/krad (特殊ゲート酸化膜)
    標準 CMOS:               α ≈ 5 mV/krad
    """
    # 標準 CMOS
    dVth_std  = Vth_shift_per_krad * tid_krad_range         # [V]
    # RAD-hard (Intel 18A enhanced)
    dVth_rhc  = 0.002 * tid_krad_range                      # α=2mV/krad

    # スペース環境: MEO で ~100 krad/year, GEO で ~20 krad/year
    leo_tid   = 10.0    # krad/year (ISS)
    meo_tid   = 100.0   # krad/year
    geo_tid   = 20.0    # krad/year

    return {
        "tid_krad":       tid_krad_range,
        "dVth_std_V":     dVth_std,
        "dVth_rhc_V":     dVth_rhc,
        "leo_tid_krad":   leo_tid,
        "meo_tid_krad":   meo_tid,
        "geo_tid_krad":   geo_tid,
        "lifetime_meo_yr": 300.0 / meo_tid,  # 300 krad 耐性 @ MEO
        "lifetime_geo_yr": 300.0 / geo_tid,
    }

# ════════════════════════════════════════════════════════════════════════════
# 6. 日本装置株 波及試算
# ════════════════════════════════════════════════════════════════════════════

JAPAN_IMPACT = {
    "ASML":       {"demand_B": 18.0, "ticker": "ASML",   "note": "EUV High-NA 独占"},
    "TEL":        {"demand_B": 8.0,  "ticker": "8035.T", "note": "CVD/Etch/ALD主要サプライヤー"},
    "Lam Research":{"demand_B": 6.0, "ticker": "LRCX",   "note": "Plasma Etch/ALD"},
    "Lasertec":   {"demand_B": 2.5,  "ticker": "6920.T", "note": "EUV マスク検査"},
    "Advantest":  {"demand_B": 4.0,  "ticker": "6857.T", "note": "ATE テスト装置"},
    "Disco":      {"demand_B": 1.5,  "ticker": "6146.T", "note": "ウェーハ研削/ダイシング"},
    "KLA":        {"demand_B": 2.5,  "ticker": "KLAC",   "note": "計測/検査"},
    "AMAT":       {"demand_B": 6.0,  "ticker": "AMAT",   "note": "CVD/PVD/CMP/Implant"},
}

def japan_impact_analysis(impact: dict = JAPAN_IMPACT) -> dict:
    total = sum(v["demand_B"] for v in impact.values())
    sorted_items = sorted(impact.items(), key=lambda x: x[1]["demand_B"], reverse=True)
    return {
        "total_demand_B": total,
        "items":          sorted_items,
        "impact":         impact,
    }

# ════════════════════════════════════════════════════════════════════════════
# 7. Plotting
# ════════════════════════════════════════════════════════════════════════════

def plot_all():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#0a0a0a")
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.45, wspace=0.38,
                           left=0.07, right=0.97,
                           top=0.92, bottom=0.07)

    title_kw = dict(fontsize=9, color="#aaaaaa", pad=6)
    spine_color = "#333333"

    def style_ax(ax):
        ax.set_facecolor("#111111")
        for sp in ax.spines.values():
            sp.set_color(spine_color)
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")

    # ── Panel 1: 生産目標スケール感 ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1)
    targets = [0.1, 0.5, 1.0, 2.0]
    results_list = [production_target(t) for t in targets]
    n_chips = [r["n_chips_needed"] / 1e6 for r in results_list]
    colors1 = ["#4393c3", "#74add1", "#d73027", "#b2182b"]
    bars = ax1.bar([f"{t} TW" for t in targets], n_chips, color=colors1, width=0.6)
    for bar, v in zip(bars, n_chips):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{v:.1f}M", ha="center", va="bottom", fontsize=7, color="white")
    ax1.set_ylabel("必要チップ数 [百万枚]", fontsize=8)
    ax1.set_title("生産目標 vs 必要チップ数", **title_kw)

    # ── Panel 2: 歩留まり連鎖 ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2)
    chain = yield_chain()
    steps = list(chain.keys())
    cumY  = [chain[s]["cumulative"] * 100 for s in steps]
    stepY = [chain[s]["step_yield"] * 100 for s in steps]
    x = np.arange(len(steps))
    ax2.bar(x, stepY, 0.4, color="#4393c3", label="ステップ歩留まり")
    ax2.plot(x, cumY, "o-", color="#d73027", lw=2, ms=5, label="累積歩留まり")
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.split("(")[0].strip() for s in steps],
                        rotation=35, ha="right", fontsize=6)
    ax2.set_ylabel("歩留まり [%]", fontsize=8)
    ax2.set_ylim(0, 105)
    ax2.axhline(cumY[-1], color="#ff7f0e", ls="--", lw=1)
    ax2.text(len(steps)-1, cumY[-1]+1, f"総歩留まり {cumY[-1]:.1f}%",
             fontsize=7, color="#ff7f0e", ha="right")
    ax2.legend(fontsize=6, facecolor="#222222", labelcolor="white")
    ax2.set_title("垂直統合 歩留まり連鎖", **title_kw)

    # ── Panel 3: CapEx 内訳 パイチャート ──────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#111111")
    capex = capex_analysis()
    labels = list(capex["breakdown"].keys())
    sizes  = [v[0] for v in capex["breakdown"].values()]
    colors3= [v[1] for v in capex["breakdown"].values()]
    wedges, texts, autotexts = ax3.pie(
        sizes, labels=None, colors=colors3,
        autopct=lambda p: f"{p:.0f}%" if p > 4 else "",
        pctdistance=0.8, startangle=140,
        wedgeprops=dict(edgecolor="#0a0a0a", linewidth=1.2))
    for at in autotexts:
        at.set_fontsize(6)
        at.set_color("white")
    ax3.set_title(f"CapEx $119B 内訳\n(日本装置 {capex['japan_share_pct']:.0f}%)",
                  **title_kw)
    legend_patches = [mpatches.Patch(color=c, label=f"{l}: ${s:.0f}B")
                      for l, s, c in zip(labels, sizes, colors3)]
    ax3.legend(handles=legend_patches, fontsize=5, loc="lower left",
               facecolor="#1a1a1a", labelcolor="white",
               bbox_to_anchor=(-0.3, -0.05))

    # ── Panel 4: RAD-hard TID 劣化 ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    style_ax(ax4)
    tid_range = np.linspace(0, 300, 300)
    rad = rad_hard_tid_model(tid_range)
    ax4.plot(tid_range, rad["dVth_std_V"] * 1e3, color="#d73027",
             lw=2, label="標準 CMOS (5mV/krad)")
    ax4.plot(tid_range, rad["dVth_rhc_V"] * 1e3, color="#4393c3",
             lw=2, label="Intel 18A RAD-hard (2mV/krad)")
    for env, tid_val, col in [("LEO", rad["leo_tid_krad"], "#74c476"),
                               ("GEO", rad["geo_tid_krad"], "#fdae61"),
                               ("MEO", rad["meo_tid_krad"], "#f03b20")]:
        ax4.axvline(tid_val, color=col, ls=":", lw=1.2,
                    label=f"{env} {tid_val:.0f} krad/yr")
    ax4.set_xlabel("累積 TID [krad]", fontsize=8)
    ax4.set_ylabel("ΔVth [mV]", fontsize=8)
    ax4.legend(fontsize=6, facecolor="#222222", labelcolor="white")
    ax4.set_title("宇宙向け RAD-hard — TID 劣化モデル", **title_kw)

    # ── Panel 5: 日本装置株 需要試算 ─────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    style_ax(ax5)
    imp = japan_impact_analysis()
    companies = [k for k, _ in imp["items"]]
    demands   = [v["demand_B"] for _, v in imp["items"]]
    colors5   = plt.cm.Blues(np.linspace(0.4, 0.9, len(companies)))
    bars5 = ax5.barh(companies, demands, color=colors5, height=0.6)
    for bar, d in zip(bars5, demands):
        ax5.text(d + 0.1, bar.get_y() + bar.get_height()/2,
                 f"${d:.1f}B", va="center", fontsize=7, color="white")
    ax5.set_xlabel("Terafab 向け推定需要 [十億ドル]", fontsize=8)
    ax5.set_title(f"日本・装置株 波及試算 (合計 ${imp['total_demand_B']:.0f}B)",
                  **title_kw)
    ax5.set_xlim(0, max(demands) * 1.3)

    # ── Panel 6: 宇宙 vs 地上 比率と用途 ─────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    style_ax(ax6)
    categories = ["宇宙向け (80%)\nStarship / Optimus衛星\nRAD-hard AI チップ",
                  "地上向け (20%)\nxAI Grok クラスター\nTesla Dojo / FSD"]
    fracs = [0.80, 0.20]
    colors6 = ["#2166ac", "#d73027"]
    bars6 = ax6.bar(categories, [f * 100 for f in fracs], color=colors6, width=0.5)
    for bar, f in zip(bars6, fracs):
        ax6.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() / 2,
                 f"{f*100:.0f}%",
                 ha="center", va="center", fontsize=18,
                 fontweight="bold", color="white")
    ax6.set_ylabel("生産比率 [%]", fontsize=8)
    ax6.set_ylim(0, 100)
    ax6.set_title("1 TW: 宇宙 vs 地上 用途比率", **title_kw)

    # ── Panel 7: ウェーハ生産スケール ──────────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    style_ax(ax7)
    prod = production_target()
    # TSMC 比較
    tsmc_wpd = 60000   # TSMC 現在の 300mm ウェーハ/日 (推定)
    terafab_wpd = prod["wafers_per_day"]
    companies7 = ["TSMC\n(現在)", "Samsung\n(現在)", "Intel\n(現在)",
                  "Terafab\n(目標)"]
    wpd7 = [60000, 35000, 20000, terafab_wpd]
    cols7 = ["#2166ac", "#4393c3", "#74add1", "#d73027"]
    bars7 = ax7.bar(companies7, [w/1000 for w in wpd7], color=cols7, width=0.6)
    for bar, w in zip(bars7, wpd7):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{w/1000:.0f}k", ha="center", fontsize=7, color="white")
    ax7.set_ylabel("ウェーハ生産 [千枚/日]", fontsize=8)
    ax7.set_title("300mm ウェーハ生産規模比較", **title_kw)

    # ── Panel 8: 投資フェーズ タイムライン ─────────────────────────
    ax8 = fig.add_subplot(gs[2, 1:])
    style_ax(ax8)
    phases = [
        ("発表",           2026.2, 0,    "#808080"),
        ("Phase 1 着工",   2026.5, 5,    "#4393c3"),
        ("Intel 18A 搬入", 2027.0, 15,   "#2166ac"),
        ("パイロット生産", 2028.0, 35,   "#74add1"),
        ("HVM 開始",       2029.0, 55,   "#d73027"),
        ("フル生産 (1TW)", 2031.0, 119,  "#b2182b"),
    ]
    years  = [p[1] for p in phases]
    capexs = [p[2] for p in phases]
    labels8= [p[0] for p in phases]
    cols8  = [p[3] for p in phases]

    ax8.fill_between(years, capexs, alpha=0.3, color="#d73027")
    ax8.plot(years, capexs, "o-", color="#d73027", lw=2, ms=8)
    for y, c, l, col in zip(years, capexs, labels8, cols8):
        ax8.annotate(f"{l}\n${c}B", xy=(y, c),
                     xytext=(y, c + 8),
                     arrowprops=dict(arrowstyle="-", color=col, lw=1),
                     ha="center", fontsize=7, color=col)
    ax8.set_xlabel("年", fontsize=8)
    ax8.set_ylabel("累積投資 [十億ドル]", fontsize=8)
    ax8.set_xlim(2025.8, 2031.8)
    ax8.set_ylim(-5, 140)
    ax8.set_title("Terafab 投資フェーズ タイムライン ($119B)", **title_kw)

    fig.suptitle(
        "Terafab (SpaceX × Tesla × xAI × Intel) — 垂直統合メガFab 分析",
        fontsize=14, color="white", fontweight="bold", y=0.97,
    )

    out = os.path.join(OUT_DIR, "terafab_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  → {out}")
    return out


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _print_summary():
    prod = production_target()
    capex = capex_analysis()
    imp = japan_impact_analysis()

    print("\n=== Terafab プロジェクト サマリー ===")
    print(f"  目標:        {TERAFAB['target_TW']} テラワット / 年")
    print(f"  プロセス:    Intel {TERAFAB['process_node_nm']}nm (18A)")
    print(f"  場所:        {TERAFAB['location']}")
    print(f"  総投資:      ${TERAFAB['total_capex_B']:.0f}B")
    print(f"  HVM 開始:    {TERAFAB['hvm_start']}")
    print()
    print("=== 生産規模 ===")
    print(f"  ウェーハ/日:       {prod['wafers_per_day']:,} 枚 (CapEx $119B 試算)")
    print(f"  宇宙向けチップ:    {prod['space']['n_chips']:,} 枚/年  "
          f"({prod['space']['desc']})")
    print(f"  地上向けチップ:    {prod['ground']['n_chips']:,} 枚/年  "
          f"({prod['ground']['desc']})")
    print(f"  合算ピーク演算:    {prod['total_tflops_B']:.1f} B-TFLOPS "
          f"(≈ {prod['implied_TW_note']})")
    print(f"  注: Musk '1TW' = 演算性能の比喩、電力単位ではない")
    print()
    print("=== 装置需要 ===")
    print(f"  日本装置 推定需要: ${capex['japan_equip_B']:.0f}B ({capex['japan_share_pct']:.0f}%)")
    print(f"  ASML / TEL / Lasertec / Advantest + Disco が主要受益")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Terafab モデル")
    parser.add_argument("--production", action="store_true", help="生産目標分析")
    parser.add_argument("--yield",      action="store_true", dest="yield_",
                        help="歩留まり連鎖")
    parser.add_argument("--capex",      action="store_true", help="CapEx 分析")
    parser.add_argument("--radhard",    action="store_true", help="RAD-hard モデル")
    parser.add_argument("--japan",      action="store_true", help="日本装置株波及")
    args = parser.parse_args()

    _print_summary()
    out = plot_all()
    print(f"\n全パネル出力: {out}")
