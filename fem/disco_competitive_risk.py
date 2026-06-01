"""
Disco 競合リスク定量シミュレーション: ブレード → レーザー移行
===============================================================
Research findings (June 2026):
  1. Disco 消耗品 = 売上の約 32% (FY2024: ¥393B → 消耗品 ¥126B)
  2. レーザーソー累計出荷 4,000台達成 (2026/02) — 2020比2倍; 加速中
  3. ステルスダイシング特許: 浜松ホトニクス保有 (約600件), Disco は非独占ライセンシー
     → 韓国・中国メーカーも同条件でライセンス取得可能
  4. 3D-Micromac TLS-Dicing: SiC向け別技術 (浜松特許外) で参入済み
  5. SiC 市場: 2025-2027 過剰設備調整, 2028- 8インチ移行で再加速 → $10B/2030 (Yole)
  6. レーザーダイシング装置市場: $397M (2024) → $1,265M (2035), CAGR 11%

Key physics (revenue model):
  ブレード: 装置 $30/w + 消耗品 $130/w = $160/w (SiC 高硬度 → ブレード激消耗)
  Disco DFL:  装置 $80/w + 消耗品 $10/w  = $90/w  (レーザー → 消耗品ほぼゼロ)
  非 Disco レーザー: Disco 収益 $0/w

Simulation:
  Module 1: Bass 拡散モデル (レーザー普及曲線)
  Module 2: 市場シェア争奪モデル (Disco vs 新規参入)
  Module 3: 売上インパクト (3シナリオ: Bull / Base / Bear)
"""

import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ════════════════════════════════════════════════════════════════════════════
# 0. 実績データ (研究結果)
# ════════════════════════════════════════════════════════════════════════════
DISCO_FY2024 = {
    "total_revenue_B_JPY":    393.3,
    "consumable_ratio":        0.32,   # 32% = ¥125.9B
    "equipment_ratio":         0.68,
    "laser_saw_cumulative":  4_000,    # 2026/02 時点
    "blade_saw_cumulative": 30_000,    # 推定: Disco 総装置出荷
    "sic_share_of_consumable": 0.12,   # SiC 向け消耗品 ≈ 12% (推定)
}
# Disco の SiC 消耗品ビジネス (2024 baseline)
SIC_CONSUMABLE_BASELINE_B_JPY = (
    DISCO_FY2024["total_revenue_B_JPY"]
    * DISCO_FY2024["consumable_ratio"]
    * DISCO_FY2024["sic_share_of_consumable"]
)  # ≈ ¥15.1B

# ── 収益/ウェーハ (USD, SiC 向け) ─────────────────────────────────────────────
REV_PER_WAFER = {
    "blade_total":       160,   # ブレード装置 + 消耗品
    "blade_equipment":    30,
    "blade_consumable":  130,
    "disco_laser_total":  90,   # Disco DFL
    "disco_laser_equip":  80,
    "disco_laser_cons":   10,
    "rival_laser":          0,   # 非Disco レーザー → Disco 収益 0
}

# ════════════════════════════════════════════════════════════════════════════
# 1. シナリオ定義
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Scenario:
    name: str
    color: str
    # Bass 拡散パラメータ (レーザー普及)
    p_bass: float   # innovation coeff
    q_bass: float   # imitation coeff
    # Disco DFL の レーザー市場シェア
    disco_laser_share_2028: float
    disco_laser_share_2032: float
    # SiC 市場成長 (wafer 枚数 index, 2024=1.0)
    sic_volume_2030: float

SCENARIOS = {
    "bull": Scenario(
        name="Bull: Disco DFL 独走",
        color="#2166ac",
        p_bass=0.025, q_bass=0.30,      # 普及ゆっくり
        disco_laser_share_2028=0.80,
        disco_laser_share_2032=0.75,     # 競合参入でやや低下
        sic_volume_2030=4.5,
    ),
    "base": Scenario(
        name="Base: 新規参入あり",
        color="#f58518",
        p_bass=0.035, q_bass=0.42,
        disco_laser_share_2028=0.65,
        disco_laser_share_2032=0.55,
        sic_volume_2030=3.8,
    ),
    "bear": Scenario(
        name="Bear: 韓中ライセンス参入 + TLS台頭",
        color="#d62728",
        p_bass=0.050, q_bass=0.55,      # 普及速い
        disco_laser_share_2028=0.50,
        disco_laser_share_2032=0.35,
        sic_volume_2030=3.2,
    ),
}

# ════════════════════════════════════════════════════════════════════════════
# 2. Bass 拡散モデル
# ════════════════════════════════════════════════════════════════════════════

def bass_diffusion(p: float, q: float, years: np.ndarray,
                   t0: float = 2024.0) -> np.ndarray:
    """
    Bass モデルによるレーザー採用率 (0–1)。
    f(t) = (p+q)² / p × exp(-(p+q)(t-t0)) / (1 + q/p × exp(-(p+q)(t-t0)))²
    累積: F(t) = (1 - exp(-(p+q)(t-t0))) / (1 + q/p × exp(-(p+q)(t-t0)))
    """
    tau = years - t0
    tau = np.maximum(tau, 0)
    F   = (1 - np.exp(-(p + q) * tau)) / (1 + (q / p) * np.exp(-(p + q) * tau))
    return np.clip(F, 0, 1)


def _interp_disco_laser_share(sc: Scenario, laser_pct: float,
                               year: float) -> float:
    """Disco の レーザー市場シェアを年で線形補間。"""
    if year <= 2026:
        return 0.90  # DFL が先行, 競合まだ少ない
    elif year <= 2028:
        t = (year - 2026) / 2
        return 0.90 + t * (sc.disco_laser_share_2028 - 0.90)
    elif year <= 2032:
        t = (year - 2028) / 4
        return sc.disco_laser_share_2028 + t * (sc.disco_laser_share_2032 - sc.disco_laser_share_2028)
    else:
        return sc.disco_laser_share_2032

# ════════════════════════════════════════════════════════════════════════════
# 3. 売上インパクト計算
# ════════════════════════════════════════════════════════════════════════════

def compute_revenue_index(sc: Scenario,
                          years: np.ndarray) -> dict:
    """
    Disco の SiC ダイシング売上インデックス (2024=100) を年ごとに計算。

    Revenue components:
      blade_rev   = blade_share × blade_total
      laser_rev   = laser_share × disco_laser_share × disco_laser_total
      total       = (blade_rev + laser_rev) × volume_index
    """
    laser_pct = bass_diffusion(sc.p_bass, sc.q_bass, years)

    # SiC 市場: 2025-2027 補正, 2028- 再加速
    base_volume = np.ones_like(years)
    for i, yr in enumerate(years):
        if yr <= 2024:
            base_volume[i] = 1.0
        elif yr <= 2027:
            # 過剰設備調整 → 成長鈍化
            t = (yr - 2024) / 3
            base_volume[i] = 1.0 + t * 0.3   # 3年で +30% のみ
        else:
            # 8インチ移行で急加速
            t = (yr - 2027) / 3
            vol_2027 = 1.3
            base_volume[i] = vol_2027 * (sc.sic_volume_2030 / vol_2027) ** (t)

    # 2024 baseline での 正規化係数
    rev_2024 = 1.0 * REV_PER_WAFER["blade_total"]  # 2024: blade 100%

    total_idx  = np.zeros_like(years, dtype=float)
    equip_idx  = np.zeros_like(years, dtype=float)
    cons_idx   = np.zeros_like(years, dtype=float)

    for i, (yr, lp, vol) in enumerate(zip(years, laser_pct, base_volume)):
        blade_share = 1 - lp
        laser_share = lp
        dl_share    = _interp_disco_laser_share(sc, lp, yr)

        rev_blade        = blade_share * REV_PER_WAFER["blade_total"]
        rev_disco_laser  = laser_share * dl_share * REV_PER_WAFER["disco_laser_total"]
        rev_total        = (rev_blade + rev_disco_laser) * vol

        equip = (blade_share * REV_PER_WAFER["blade_equipment"]
                 + laser_share * dl_share * REV_PER_WAFER["disco_laser_equip"]) * vol
        cons  = (blade_share * REV_PER_WAFER["blade_consumable"]
                 + laser_share * dl_share * REV_PER_WAFER["disco_laser_cons"]) * vol

        total_idx[i] = rev_total / rev_2024 * 100
        equip_idx[i] = equip    / rev_2024 * 100
        cons_idx[i]  = cons     / rev_2024 * 100

    return {
        "years":     years,
        "laser_pct": laser_pct * 100,
        "volume":    base_volume,
        "total_idx": total_idx,
        "equip_idx": equip_idx,
        "cons_idx":  cons_idx,
    }

# ════════════════════════════════════════════════════════════════════════════
# 4. 競合エントリーマップ
# ════════════════════════════════════════════════════════════════════════════

COMPETITORS = {
    "Disco DFL":        {"tech": "ステルスレーザー (浜松 OEM)", "moat": 9,
                         "entry": 2002, "threat": 0, "color": "#001f3f"},
    "3D-Micromac":      {"tech": "TLS-Dicing (熱誘起分離, 独自特許)", "moat": 5,
                         "entry": 2018, "threat": 6, "color": "#d62728"},
    "Hamamatsu 直販":   {"tech": "SD エンジン + 他社統合", "moat": 4,
                         "entry": 2020, "threat": 5, "color": "#f58518"},
    "韓国 (Ultratech等)": {"tech": "浜松ライセンス + 低価格", "moat": 3,
                           "entry": 2024, "threat": 7, "color": "#9ecae1"},
    "中国 (LUMI等)":    {"tech": "浜松ライセンス / 独自開発", "moat": 2,
                          "entry": 2025, "threat": 8, "color": "#fc8d59"},
    "プラズマ (SPTS)":  {"tech": "プラズマダイシング", "moat": 4,
                          "entry": 2016, "threat": 3, "color": "#74c476"},
}

# ════════════════════════════════════════════════════════════════════════════
# 5. Visualization
# ════════════════════════════════════════════════════════════════════════════

def plot_all(save: bool = True) -> str:
    years = np.linspace(2024, 2035, 200)

    fig = plt.figure(figsize=(22, 16))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    ax_laser  = fig.add_subplot(gs[0, 0])    # Bass 普及曲線
    ax_rev    = fig.add_subplot(gs[0, 1])    # 売上インデックス
    ax_mix    = fig.add_subplot(gs[0, 2])    # 装置 vs 消耗品 構成
    ax_threat = fig.add_subplot(gs[1, 0])    # 競合脅威マップ
    ax_moat   = fig.add_subplot(gs[1, 1], polar=True)  # Disco の堀 分析
    ax_cons   = fig.add_subplot(gs[1, 2])    # 消耗品 消滅リスク
    ax_table  = fig.add_subplot(gs[2, :])    # サマリーテーブル

    results = {k: compute_revenue_index(v, years) for k, v in SCENARIOS.items()}

    # ── (A) Bass 普及曲線 ─────────────────────────────────────────────────────
    for key, sc in SCENARIOS.items():
        r = results[key]
        ax_laser.plot(years, r["laser_pct"], color=sc.color, lw=2.2, label=sc.name)
    ax_laser.axhline(50, color="gray", ls=":", lw=1)
    ax_laser.text(2024.3, 52, "50% 普及", fontsize=8, color="gray")
    ax_laser.axvspan(2025, 2028, alpha=0.06, color="orange",
                     label="SiC 過剰在庫調整期")
    ax_laser.set_xlabel("Year"); ax_laser.set_ylabel("Laser Adoption [%]")
    ax_laser.set_title("SiC Laser Dicing Adoption\n(Bass Diffusion Model)", fontweight="bold")
    ax_laser.legend(fontsize=7.5); ax_laser.grid(alpha=0.25)
    ax_laser.set_ylim(0, 100)

    # ── (B) 売上インデックス ─────────────────────────────────────────────────
    for key, sc in SCENARIOS.items():
        r = results[key]
        ax_rev.plot(years, r["total_idx"], color=sc.color, lw=2.2, label=sc.name)
    ax_rev.axhline(100, color="black", ls="--", lw=1, alpha=0.5)
    ax_rev.text(2024.2, 102, "2024 base", fontsize=8, color="black", alpha=0.7)
    ax_rev.set_xlabel("Year"); ax_rev.set_ylabel("Revenue Index (2024=100)")
    ax_rev.set_title("Disco SiC Dicing Revenue Index\n(Equipment + Consumable, volume growth)", fontweight="bold")
    ax_rev.legend(fontsize=7.5); ax_rev.grid(alpha=0.25)

    # ── (C) 装置 vs 消耗品 mix (Base シナリオ) ───────────────────────────────
    r = results["base"]
    ax_mix.stackplot(
        years,
        r["cons_idx"],
        r["equip_idx"],
        labels=["Consumable (blades)", "Equipment"],
        colors=["#d62728", "#4c78a8"],
        alpha=0.75,
    )
    ax_mix.set_xlabel("Year"); ax_mix.set_ylabel("Revenue Index (2024=100)")
    ax_mix.set_title("Revenue Mix Shift (Base scenario)\nBlade consumable erosion", fontweight="bold")
    ax_mix.legend(fontsize=8, loc="upper left"); ax_mix.grid(alpha=0.15)

    # ── (D) 競合脅威マップ (threat vs moat) ──────────────────────────────────
    for name, comp in COMPETITORS.items():
        ax_threat.scatter(comp["moat"], comp["threat"],
                          color=comp["color"], s=180, zorder=5,
                          edgecolors="k", linewidths=0.6)
        ax_threat.annotate(
            name, (comp["moat"], comp["threat"]),
            textcoords="offset points", xytext=(6, 4), fontsize=7.5,
        )
    ax_threat.axvline(5, color="gray", ls="--", lw=1, alpha=0.5)
    ax_threat.axhline(5, color="gray", ls="--", lw=1, alpha=0.5)
    ax_threat.set_xlabel("Competitive Moat (1=low, 10=high)")
    ax_threat.set_ylabel("Threat to Disco (1=low, 10=high)")
    ax_threat.set_title("Competitor Map\n(moat vs threat)", fontweight="bold")
    ax_threat.set_xlim(0, 11); ax_threat.set_ylim(0, 11)
    ax_threat.fill_between([0, 5], [5, 5], [11, 11],
                            alpha=0.05, color="red")
    ax_threat.text(1, 9.5, "High threat\nLow moat", fontsize=8, color="red", alpha=0.7)
    ax_threat.grid(alpha=0.2)

    # ── (E) Disco の堀 vs 弱点 ───────────────────────────────────────────────
    categories = ["特許\n(HP 依存)", "装置精度", "顧客関係\n(MRO)", "消耗品\n安定収益", "8\" SiC\n移行対応"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    disco_vals  = [4, 9, 8, 9, 7]  # 2024
    disco_2030  = [5, 8, 7, 4, 8]  # 2030 (消耗品が落ちる)
    disco_vals  += disco_vals[:1]
    disco_2030  += disco_2030[:1]

    ax_moat.set_theta_offset(np.pi / 2)
    ax_moat.set_theta_direction(-1)
    ax_moat.plot(angles, disco_vals,  color="#001f3f", lw=2, label="2024")
    ax_moat.fill(angles, disco_vals,  color="#001f3f", alpha=0.20)
    ax_moat.plot(angles, disco_2030,  color="#d62728", lw=2, ls="--", label="2030 (Base)")
    ax_moat.fill(angles, disco_2030,  color="#d62728", alpha=0.12)
    ax_moat.set_xticks(angles[:-1])
    ax_moat.set_xticklabels(categories, fontsize=8)
    ax_moat.set_ylim(0, 10)
    ax_moat.set_title("Disco Competitive Moat\n(radar, 10=strong)", fontweight="bold", pad=12)
    ax_moat.legend(fontsize=8, loc="upper right")

    # ── (F) 消耗品消滅リスク — 単独グラフ ────────────────────────────────────
    for key, sc in SCENARIOS.items():
        r = results[key]
        ax_cons.plot(years, r["cons_idx"], color=sc.color, lw=2.2, label=sc.name)
    ax_cons.axhline(100, color="black", ls="--", lw=1, alpha=0.5)
    ax_cons.set_xlabel("Year")
    ax_cons.set_ylabel("Consumable Revenue Index (2024=100)")
    ax_cons.set_title("Blade Consumable Revenue Erosion\n(SiC segment, scenario comparison)", fontweight="bold")
    ax_cons.legend(fontsize=7.5); ax_cons.grid(alpha=0.25)
    ax_cons.fill_between(years,
                         results["bear"]["cons_idx"],
                         results["bull"]["cons_idx"],
                         alpha=0.1, color="#d62728", label="scenario range")

    # ── (G) サマリーテーブル ─────────────────────────────────────────────────
    ax_table.axis("off")
    col_labels = ["シナリオ", "2028 Laser普及", "2032 Laser普及",
                  "Disco DFL シェア (2032)", "消耗品収益 (2032)", "総収益 (2032)", "主なリスク"]
    table_data = []
    for key, sc in SCENARIOS.items():
        r = results[key]
        idx_2032 = np.argmin(np.abs(years - 2032))
        lp_2028  = bass_diffusion(sc.p_bass, sc.q_bass, np.array([2028.0]))[0]
        lp_2032  = bass_diffusion(sc.p_bass, sc.q_bass, np.array([2032.0]))[0]

        risks = {
            "bull": "競合参入遅延, DFL 技術的優位維持",
            "base": "韓/中ライセンス参入, 消耗品侵食 -50%",
            "bear": "TLS + 中国コピー + 浜松直販 三重苦",
        }
        table_data.append([
            sc.name,
            f"{lp_2028*100:.0f}%",
            f"{lp_2032*100:.0f}%",
            f"{sc.disco_laser_share_2032*100:.0f}%",
            f"{r['cons_idx'][idx_2032]:.0f}",
            f"{r['total_idx'][idx_2032]:.0f}",
            risks[key],
        ])

    tbl = ax_table.table(
        cellText=table_data, colLabels=col_labels,
        cellLoc="center", loc="center", bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#ecf0f1")
        cell.set_edgecolor("#bdc3c7")
    ax_table.set_title(
        "Disco SiC Dicing Competitive Risk — Scenario Summary (2032)",
        fontsize=11, fontweight="bold",
    )

    fig.suptitle(
        "Disco ブレード → レーザー移行 競合リスク定量シミュレーション\n"
        "浜松特許(非独占) + 3D-Micromac TLS + 韓中参入 × Bass拡散モデル",
        fontsize=13, fontweight="bold", y=0.98,
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "disco_competitive_risk.png")
    if save:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[V] -> {out_path}")
    plt.close()
    return out_path


# ════════════════════════════════════════════════════════════════════════════
# 6. テキストサマリー
# ════════════════════════════════════════════════════════════════════════════

def print_summary():
    years = np.linspace(2024, 2035, 200)
    print("\n" + "="*72)
    print("  Disco 競合リスク定量サマリー  (ブレード→レーザー移行)")
    print("="*72)
    print(f"\n  Disco FY2024 売上: ¥{DISCO_FY2024['total_revenue_B_JPY']:.1f}B")
    print(f"  消耗品 (32%):  ¥{DISCO_FY2024['total_revenue_B_JPY']*DISCO_FY2024['consumable_ratio']:.1f}B")
    print(f"  SiC 消耗品推定: ¥{SIC_CONSUMABLE_BASELINE_B_JPY:.1f}B (全消耗品の12%)")
    print(f"\n  収益/ウェーハ: ブレード $160 → Disco レーザー $90 (-44%) → 非Disco $0")
    print(f"  ステルス特許: 浜松保有 600件, 非独占ライセンス → 競合参入障壁は低い")

    print(f"\n{'  シナリオ':<28} {'Laser@2030':>10} {'DFL Share@2032':>16} "
          f"{'消耗品idx@2032':>14} {'総収益idx@2032':>14}")
    print("  " + "-"*68)
    for key, sc in SCENARIOS.items():
        r = compute_revenue_index(sc, years)
        idx_2030 = np.argmin(np.abs(years - 2030))
        idx_2032 = np.argmin(np.abs(years - 2032))
        lp_2030  = bass_diffusion(sc.p_bass, sc.q_bass, np.array([2030.0]))[0]
        print(f"  {sc.name:<28} {lp_2030*100:>9.0f}%  {sc.disco_laser_share_2032*100:>14.0f}%  "
              f"{r['cons_idx'][idx_2032]:>13.0f}  {r['total_idx'][idx_2032]:>13.0f}")

    print("\n  KEY RISKS:")
    print("  [1] ブレード消耗品は SiC レーザー化で構造的に消滅 (不可逆)")
    print("  [2] 浜松特許 非独占 → 韓国 Ultratech / 中国 LUMI が同技術で参入可")
    print("  [3] 3D-Micromac TLS は浜松特許外 → Disco DFL も対抗困難")
    print("  [4] SiC 過剰調整 2025-2027 → 直近は影響小, 2028 以降に顕在化")
    print(f"\n  HEDGE:")
    print("  [+] Disco は DFL レーザーソーを自社保有 (装置売上で一部補完)")
    print("  [+] 8\" SiC 移行でウェーハ枚数は増加 → volume で消耗品減少を部分補完")
    print("  [+] 研削砥石 (SiC 薄化) は構造的に残存 → 別収益源")
    print("="*72)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    print_summary()
    if not args.no_plot:
        plot_all()
