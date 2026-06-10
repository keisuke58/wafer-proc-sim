"""
パワーデバイス新興5市場 × Disco 加工需要モデル
================================================
ヒューマノイドロボット / 軍用ドローン / 系統BESS / LEO衛星 / EV商用車
材料棲み分け: GaN (≤ 650 V)  vs  SiC (> 650 V)

Disco 視点:
  市場規模 → 年間ダイ数 → ウェーハ枚数 → Disco 装置台数 → TAM
  GaN: ステルスレーザー/DAD,  SiC: ダイヤモンドブレード+CBN 研削

References:
  Goldman Sachs "Humanoid Robots — 300万台/年 by 2030" (2024)
  BloombergNEF EV Outlook 2024 — commercial vehicle SiC ramp
  IRENA "Utility-Scale Batteries" 2024
  SpaceX Starlink v2 power specs (2023)
  Yole Développement "GaN & SiC Power" 2024 (TAM figures)
  Wolfspeed AN-2x, Infineon SiC EAN
  Lawn & Evans (1977) lateral crack chipping model
  Levinshtein et al. (2001) — GaN fracture toughness K_Ic
"""

import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ════════════════════════════════════════════════════════════════════════════
# 1. 材料物性 (Disco 加工関連のみ)
# ════════════════════════════════════════════════════════════════════════════
MAT = {
    "GaN": {
        "K_Ic":      0.9,    # MPa√m  fracture toughness
        "H_vickers": 1300,   # HV
        "E_GPa":     295,    # Young's modulus
        "color":     "#e45756",
        "label":     "GaN",
    },
    "SiC_4H": {
        "K_Ic":      2.8,
        "H_vickers": 2580,
        "E_GPa":     448,
        "color":     "#4c78a8",
        "label":     "4H-SiC",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 2. 5大市場スペック
# ════════════════════════════════════════════════════════════════════════════
MARKETS = {
    "humanoid_robot": {
        "label":            "ヒューマノイドロボット",
        "material":         "GaN",
        "voltage_V":        48,
        "market_2030_B":    4.0,    # $B (パワー半導体のみ)
        "cagr_pct":         150,
        "die_area_mm2":     2.0,    # GaN 関節 DC/DC
        "die_price_usd":    5.0,
        "wafer_mm":         200,
        "process":          "GaN-on-Si + ステルスレーザー",
        "street_um":        50,     # ストリート幅 [µm]
        "blade_W_um":       30,     # ブレード幅 (Disco Z09)
        "cut_depth_um":     120,    # GaN epi + Si 基板 thin
        "feed_mm_s":        60,     # レーザーはより高速
        "color":            "#e45756",
        "disco_tool":       "DFL7161 (ステルスレーザー)",
    },
    "military_drone": {
        "label":            "軍用ドローン / UAS",
        "material":         "GaN",
        "voltage_V":        48,
        "market_2030_B":    5.0,
        "cagr_pct":         58,
        "die_area_mm2":     3.0,    # ESC / PA チップ
        "die_price_usd":    8.0,
        "wafer_mm":         150,
        "process":          "GaN-on-SiC + DAD",
        "street_um":        60,
        "blade_W_um":       25,
        "cut_depth_um":     350,
        "feed_mm_s":        4.0,
        "color":            "#54a24b",
        "disco_tool":       "DFD6361 (ブレード 精密)",
    },
    "grid_bess": {
        "label":            "系統用 BESS",
        "material":         "SiC_4H",
        "voltage_V":        1500,
        "market_2030_B":    8.5,
        "cagr_pct":         47,
        "die_area_mm2":     15.0,   # 大電流 SiC MOSFET
        "die_price_usd":    50.0,
        "wafer_mm":         150,
        "process":          "4H-SiC + ダイヤモンドブレード",
        "street_um":        120,
        "blade_W_um":       80,
        "cut_depth_um":     350,
        "feed_mm_s":        2.5,
        "color":            "#f58518",
        "disco_tool":       "DAD3240 (SiC 専用)",
    },
    "leo_satellite": {
        "label":            "LEO 衛星",
        "material":         "SiC_4H",
        "voltage_V":        100,
        "market_2030_B":    3.5,
        "cagr_pct":         63,
        "die_area_mm2":     8.0,    # 宇宙グレード SiC スイッチ
        "die_price_usd":    200.0,  # 宇宙グレード プレミアム
        "wafer_mm":         150,
        "process":          "4H-SiC + ステルス+研削",
        "street_um":        100,
        "blade_W_um":       60,
        "cut_depth_um":     300,
        "feed_mm_s":        1.5,
        "color":            "#001f3f",
        "disco_tool":       "DFD6361 + DGP8761 (研削)",
    },
    "ev_commercial": {
        "label":            "EV 商用車 (トラック/バス)",
        "material":         "SiC_4H",
        "voltage_V":        800,
        "market_2030_B":    7.0,
        "cagr_pct":         37,
        "die_area_mm2":     25.0,   # 大電流インバータ用
        "die_price_usd":    30.0,
        "wafer_mm":         200,    # 200mm SiC 移行中
        "process":          "4H-SiC 200mm + ブレード",
        "street_um":        150,
        "blade_W_um":       100,
        "cut_depth_um":     450,
        "feed_mm_s":        2.0,
        "color":            "#7e7e7e",
        "disco_tool":       "DAD3240 + DGP8761 200mm",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 3. ウェーハ枚数・Disco TAM 計算
# ════════════════════════════════════════════════════════════════════════════

# ダイシングソー単価 (Disco)
DISCO_PRICE = {
    "dicing_saw":   1_500_000,  # $1.5M / 台
    "laser_saw":    3_000_000,  # $3.0M / 台 (DFL)
    "grinder":      1_200_000,  # $1.2M / 台
    "service_ratio": 0.15,      # 年間 MRO/売上 比率
}

# Disco 収益/ウェーハ = 装置償却 + 消耗品 (ブレード/砥石)
# SiC: 高硬度 → ブレード消耗大 → $120-200/w; GaN-on-Si: レーザー → $30-50/w
DISCO_REVENUE_PER_WAFER = {
    "GaN":      45,   # $45/wafer (ステルスレーザー + ブレード消耗品)
    "SiC_4H":  160,  # $160/wafer (ダイヤモンドブレード + CBN 研削砥石)
}

# 1台あたりのスループット: ウェーハ/年
THROUGHPUT_WAFER_PER_SAW = {
    "dicing_saw":  50_000,
    "laser_saw":   80_000,
    "grinder":    100_000,
}


def _dies_per_wafer(wafer_mm: float, die_area_mm2: float,
                    street_um: float = 100) -> int:
    """ウェーハ1枚あたりダイ数 (Bingel 近似)。"""
    r      = wafer_mm / 2
    pitch  = math.sqrt(die_area_mm2) + street_um / 1000  # mm
    t      = 5  # edge exclusion mm
    n_eff  = int(math.pi * (r - t) ** 2 / pitch ** 2)
    return max(n_eff, 1)


def compute_market_metrics(m: dict) -> dict:
    """ウェーハ枚数 × Disco 収益/ウェーハ で TAM を算出。"""
    dies_per_w      = _dies_per_wafer(m["wafer_mm"], m["die_area_mm2"], m["street_um"])
    dies_per_year   = m["market_2030_B"] * 1e9 / m["die_price_usd"]
    wafers_per_year = dies_per_year / dies_per_w

    # Disco TAM = 年間ウェーハ × 収益/ウェーハ (装置償却 + 消耗品)
    rev_per_w = DISCO_REVENUE_PER_WAFER[m["material"]]
    tam_total = wafers_per_year * rev_per_w

    # 必要装置台数 (参考値)
    if "ステルス" in m["process"] or "DFL" in m["disco_tool"]:
        tool_type = "laser_saw"
    else:
        tool_type = "dicing_saw"
    n_saws = max(1, math.ceil(wafers_per_year / THROUGHPUT_WAFER_PER_SAW[tool_type]))

    return {
        "dies_per_wafer":    dies_per_w,
        "dies_per_year_M":   dies_per_year / 1e6,
        "wafers_per_year_k": wafers_per_year / 1e3,
        "tool_type":         tool_type,
        "n_saws":            n_saws,
        "rev_per_wafer":     rev_per_w,
        "tam_total_M":       tam_total / 1e6,
    }

# ════════════════════════════════════════════════════════════════════════════
# 4. Lawn-Evans チッピングモデル — Disco ブレード条件
# ════════════════════════════════════════════════════════════════════════════

def chipping_model(market_key: str) -> dict:
    """
    Lawn-Evans 横亀裂モデルによるチッピング深さ推定。
    c_l ∝ (E/H)^0.4 × (P/K_Ic)^0.5
    校正定数 C=5e-4 は Micro2026 SiC 実験値 (5µm@d_eff=40µm) でフィット。
    """
    m = MARKETS[market_key]
    mat = MAT[m["material"]]

    E_Pa = mat["E_GPa"] * 1e9          # Pa
    # 1 HV = 9.81 N/mm² = 9.81 MPa = 9.81e6 Pa
    H_Pa = mat["H_vickers"] * 9.81e6   # Pa
    K_Ic = mat["K_Ic"] * 1e6           # MPa√m → Pa√m

    d_eff_um = m["blade_W_um"] / 2               # 有効グリット押し込み深さ [µm]
    P_grit   = 0.5 * H_Pa * (d_eff_um * 1e-6) ** 2  # [N]

    # Calibrated Lawn-Evans: C=5e-4 gives ~5µm for SiC @ d_eff=40µm
    C_cal = 5e-4
    c_l_m   = C_cal * (E_Pa / H_Pa) ** 0.4 * (P_grit / K_Ic) ** 0.5  # [m]
    chip_um = c_l_m * 1e6
    Ra_um   = chip_um / 10

    # 4H-SiC {0001} 劈開異方性補正
    anisotropy = 1.20 if m["material"] == "SiC_4H" else 1.05

    return {
        "chip_um":    chip_um * anisotropy,
        "Ra_um":      Ra_um   * anisotropy,
        "P_grit_mN":  P_grit  * 1e3,
        "c_l_m":      c_l_m,
    }


# ════════════════════════════════════════════════════════════════════════════
# 5. 成長曲線 (2024–2030)
# ════════════════════════════════════════════════════════════════════════════

def market_growth(key: str) -> tuple[np.ndarray, np.ndarray]:
    m = MARKETS[key]
    years = np.arange(2024, 2031)
    # 2030 値から逆算して 2024 基準値を求め、CAGR で展開
    cagr  = m["cagr_pct"] / 100
    v2030 = m["market_2030_B"]
    v2024 = v2030 / (1 + cagr) ** 6
    vals  = v2024 * (1 + cagr) ** (years - 2024)
    return years, vals


# ════════════════════════════════════════════════════════════════════════════
# 6. Visualization
# ════════════════════════════════════════════════════════════════════════════

def plot_all(save: bool = True) -> str:
    fig = plt.figure(figsize=(22, 16))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    ax_growth  = fig.add_subplot(gs[0, :2])   # 市場成長曲線
    ax_tam     = fig.add_subplot(gs[0, 2])    # Disco TAM 棒グラフ
    ax_chip    = fig.add_subplot(gs[1, 0])    # チッピング比較
    ax_wafer   = fig.add_subplot(gs[1, 1])    # ウェーハ枚数/年
    ax_radar   = fig.add_subplot(gs[1, 2], polar=True)  # レーダー
    ax_flow    = fig.add_subplot(gs[2, :])    # プロセスフロー表

    # ── (A) 市場成長曲線 ─────────────────────────────────────────────────────
    for key, m in MARKETS.items():
        yrs, vals = market_growth(key)
        ax_growth.plot(yrs, vals, "o-", color=m["color"], lw=2,
                       label=f"{m['label']} (CAGR {m['cagr_pct']}%)")
    ax_growth.set_xlabel("Year", fontsize=11)
    ax_growth.set_ylabel("パワー半導体 市場規模 [$B]", fontsize=11)
    ax_growth.set_title("5大新興市場 成長曲線 (2024–2030)", fontsize=12, fontweight="bold")
    ax_growth.legend(fontsize=8, loc="upper left")
    ax_growth.grid(alpha=0.25)
    ax_growth.axvline(2026, color="gray", ls="--", alpha=0.5, lw=1)
    ax_growth.text(2026.05, ax_growth.get_ylim()[1]*0.95, "2026", color="gray", fontsize=8)

    # ── (B) Disco TAM 棒グラフ ───────────────────────────────────────────────
    keys   = list(MARKETS.keys())
    labels = [MARKETS[k]["label"].replace(" ", "\n") for k in keys]
    colors = [MARKETS[k]["color"] for k in keys]
    metrics_all = {k: compute_market_metrics(MARKETS[k]) for k in keys}
    tams   = [metrics_all[k]["tam_total_M"] for k in keys]

    bars = ax_tam.bar(range(len(keys)), tams, color=colors, edgecolor="k", lw=0.5)
    ax_tam.set_xticks(range(len(keys)))
    ax_tam.set_xticklabels(labels, fontsize=7)
    ax_tam.set_ylabel("Disco TAM [$M]", fontsize=10)
    ax_tam.set_title("Disco 装置 + MRO TAM\n(2030 市場 フル稼働時)", fontsize=10, fontweight="bold")
    ax_tam.grid(axis="y", alpha=0.25)
    for bar, v in zip(bars, tams):
        ax_tam.text(bar.get_x() + bar.get_width()/2, v + 2,
                    f"${v:.0f}M", ha="center", va="bottom", fontsize=7)

    # ── (C) チッピング [µm] ───────────────────────────────────────────────────
    chips = []
    for k in keys:
        try:
            chips.append(chipping_model(k)["chip_um"])
        except Exception:
            chips.append(0)
    bars2 = ax_chip.bar(range(len(keys)), chips, color=colors, edgecolor="k", lw=0.5)
    ax_chip.set_xticks(range(len(keys)))
    ax_chip.set_xticklabels(labels, fontsize=7)
    ax_chip.axhline(15, color="red", ls="--", lw=1.2, label="上限 15 µm")
    ax_chip.set_ylabel("チッピング [µm]", fontsize=10)
    ax_chip.set_title("Lawn-Evans 推定チッピング\n(ブレードダイシング)", fontsize=10, fontweight="bold")
    ax_chip.legend(fontsize=8)
    ax_chip.grid(axis="y", alpha=0.25)
    for bar, v in zip(bars2, chips):
        ax_chip.text(bar.get_x() + bar.get_width()/2, v + 0.1,
                     f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    # ── (D) ウェーハ枚数/年 ───────────────────────────────────────────────────
    wpy = [metrics_all[k]["wafers_per_year_k"] for k in keys]
    bars3 = ax_wafer.bar(range(len(keys)), wpy, color=colors, edgecolor="k", lw=0.5)
    ax_wafer.set_xticks(range(len(keys)))
    ax_wafer.set_xticklabels(labels, fontsize=7)
    ax_wafer.set_ylabel("年間ウェーハ枚数 [千枚]", fontsize=10)
    ax_wafer.set_title("Disco 装置需要の根拠\n(ウェーハ枚数/年, 2030)", fontsize=10, fontweight="bold")
    ax_wafer.grid(axis="y", alpha=0.25)
    for bar, v in zip(bars3, wpy):
        ax_wafer.text(bar.get_x() + bar.get_width()/2, v + 1,
                      f"{v:.0f}k", ha="center", va="bottom", fontsize=7)

    # ── (E) レーダーチャート ─────────────────────────────────────────────────
    categories = ["市場規模", "CAGR", "TAM", "チッピング\n難易度", "ウェーハ需要"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # 正規化 (0–1)
    norm_vals = {
        "市場規模":     [MARKETS[k]["market_2030_B"] / 8.5 for k in keys],
        "CAGR":         [MARKETS[k]["cagr_pct"] / 150 for k in keys],
        "TAM":          [metrics_all[k]["tam_total_M"] / max(tams) for k in keys],
        "チッピング\n難易度": [min(chips[i] / 15, 1.0) for i in range(len(keys))],
        "ウェーハ需要": [wpy[i] / max(wpy) for i in range(len(keys))],
    }

    for i, k in enumerate(keys):
        vals_r = [norm_vals[c][i] for c in categories]
        vals_r += vals_r[:1]
        ax_radar.plot(angles, vals_r, color=MARKETS[k]["color"], lw=1.5)
        ax_radar.fill(angles, vals_r, color=MARKETS[k]["color"], alpha=0.08)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories, fontsize=8)
    ax_radar.set_title("総合スコア (正規化)", fontsize=10, fontweight="bold", pad=15)
    ax_radar.set_ylim(0, 1)

    # ── (F) プロセスフロー表 ─────────────────────────────────────────────────
    ax_flow.axis("off")
    col_labels = ["市場", "材料", "電圧", "ウェーハ径",
                  "ダイ面積", "Disco 装置", "年間装置台数", "TAM ($M)"]
    table_data = []
    for k in keys:
        m  = MARKETS[k]
        mt = metrics_all[k]
        table_data.append([
            m["label"],
            m["material"].replace("_4H", ""),
            f"{m['voltage_V']} V",
            f"{m['wafer_mm']} mm",
            f"{m['die_area_mm2']:.0f} mm²",
            m["disco_tool"],
            f"{mt['n_saws']}台",
            f"${mt['tam_total_M']:.0f}",
        ])
    tbl = ax_flow.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#ecf0f1")
        cell.set_edgecolor("#bdc3c7")
    ax_flow.set_title("Disco パワーデバイス5大市場 — 装置需要サマリー",
                      fontsize=11, fontweight="bold", pad=10)

    fig.suptitle(
        "パワーデバイス新興5市場 × Disco 加工需要\n"
        "GaN (≤650V: ロボット/ドローン)  vs  SiC (>650V: BESS/衛星/EV商用車)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "power_device_disco_market.png")
    if save:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[✓] → {out_path}")
    plt.close()
    return out_path


# ════════════════════════════════════════════════════════════════════════════
# 7. テキストサマリー
# ════════════════════════════════════════════════════════════════════════════

def print_summary():
    print("\n" + "="*70)
    print("  パワーデバイス新興5市場 × Disco 加工需要  (2030 フル稼働)")
    print("="*70)

    total_tam = 0.0
    for key, m in MARKETS.items():
        mt    = compute_market_metrics(m)
        chip  = chipping_model(key)
        total_tam += mt["tam_total_M"]

        print(f"\n【{m['label']}】  {m['material']}  {m['voltage_V']}V")
        print(f"  市場規模(2030): ${m['market_2030_B']:.1f}B   CAGR: {m['cagr_pct']}%")
        print(f"  ダイ: {m['die_area_mm2']:.0f}mm²  @{m['die_price_usd']:.0f}$/die  "
              f"→ {mt['dies_per_year_M']:.0f}M dies/yr  "
              f"({mt['wafers_per_year_k']:.0f}k wafers/yr)")
        print(f"  Disco装置: {m['disco_tool']}")
        print(f"  必要台数 : {mt['n_saws']}台  |  TAM: ${mt['tam_total_M']:.0f}M")
        print(f"  チッピング(Lawn-Evans): {chip['chip_um']:.1f} µm  "
              f"Ra: {chip['Ra_um']:.2f} µm")

    print(f"\n{'='*70}")
    print(f"  5市場合計 Disco TAM (2030): ${total_tam:.0f}M = ${total_tam/1e3:.2f}B")
    print("="*70)


# ════════════════════════════════════════════════════════════════════════════
# 8. Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print_summary()
    if not args.no_plot:
        plot_all()
