"""
パワー半導体 応用領域 × 材料 全景分析
=========================================
EV / AI DC / ロボット / ドローン / 衛星 / 鉄道 / 再エネ / 防衛 / 5G / 医療
× SiC / GaN / Ga2O3 / Si-IGBT / Diamond

どの産業が伸びるか、どの材料が勝つかを定量的に評価する。

References:
  - Yole Développement "Power SiC 2024"
  - IRENA "World Energy Transitions Outlook 2024"
  - Goldman Sachs "Humanoid Robots" report 2024
  - BloombergNEF "New Energy Outlook 2024"
  - Precedence Research "GaN Power Device Market 2024"
"""

import math

# ════════════════════════════════════════════════════════════════════════════
# 1. 応用領域マスターデータ
# ════════════════════════════════════════════════════════════════════════════

APPLICATIONS = {

    # ── 輸送系 ────────────────────────────────────────────────────────────
    "EV_passenger": {
        "label":        "EV 乗用車",
        "category":     "transport",
        "units_2026M":  15.0,       # 百万台/年
        "units_2030M":  35.0,
        "sic_chips_per_unit": 24,   # インバータ+OBC
        "voltage_V":    800,
        "power_kW":     150,
        "primary_mat":  "SiC",
        "alt_mat":      "GaN",      # 将来の競合
        "market_2026_B": 4.5,
        "market_2030_B": 12.0,
        "cagr_pct":     27,
        "moat":         "high",     # SiC の参入障壁
        "color":        "#4c78a8",
    },
    "EV_commercial": {
        "label":        "EV 商用車・バス",
        "category":     "transport",
        "units_2026M":  1.5,
        "units_2030M":  5.0,
        "sic_chips_per_unit": 48,   # 高出力 → チップ多い
        "voltage_V":    800,
        "power_kW":     400,
        "primary_mat":  "SiC",
        "alt_mat":      None,
        "market_2026_B": 2.0,
        "market_2030_B": 7.0,
        "cagr_pct":     37,
        "moat":         "very_high",
        "color":        "#2196F3",
    },
    "Railway": {
        "label":        "鉄道インバータ",
        "category":     "transport",
        "units_2026M":  0.05,
        "units_2030M":  0.08,
        "sic_chips_per_unit": 96,
        "voltage_V":    3300,
        "power_kW":     2000,
        "primary_mat":  "SiC",
        "alt_mat":      None,
        "market_2026_B": 1.5,
        "market_2030_B": 2.8,
        "cagr_pct":     17,
        "moat":         "high",
        "color":        "#9C27B0",
    },

    # ── エネルギー系 ──────────────────────────────────────────────────────
    "Solar_inverter": {
        "label":        "太陽光インバータ",
        "category":     "energy",
        "units_2026M":  50.0,       # ユニット数
        "units_2030M":  120.0,
        "sic_chips_per_unit": 4,
        "voltage_V":    1500,
        "power_kW":     10,
        "primary_mat":  "SiC",
        "alt_mat":      "GaN",
        "market_2026_B": 2.5,
        "market_2030_B": 6.5,
        "cagr_pct":     27,
        "moat":         "medium",
        "color":        "#FF9800",
    },
    "Grid_BESS": {
        "label":        "系統用蓄電池 (BESS)",
        "category":     "energy",
        "units_2026M":  5.0,        # GW 相当
        "units_2030M":  20.0,
        "sic_chips_per_unit": 200,  # MW スケール → 大量
        "voltage_V":    1500,
        "power_kW":     1000,
        "primary_mat":  "SiC",
        "alt_mat":      "Si_IGBT",
        "market_2026_B": 1.8,
        "market_2030_B": 8.5,
        "cagr_pct":     47,         # 最高成長率の一つ
        "moat":         "high",
        "color":        "#4CAF50",
    },

    # ── AI / データ ────────────────────────────────────────────────────────
    "AI_datacenter": {
        "label":        "AI データセンター PSU",
        "category":     "ai_compute",
        "units_2026M":  2.0,        # M サーバー
        "units_2030M":  10.0,
        "sic_chips_per_unit": 8,
        "voltage_V":    400,
        "power_kW":     6,
        "primary_mat":  "GaN",      # 低電圧 → GaN 有利
        "alt_mat":      "SiC",
        "market_2026_B": 1.2,
        "market_2030_B": 6.0,
        "cagr_pct":     50,
        "moat":         "medium",
        "color":        "#e45756",
    },
    "5G_basestation": {
        "label":        "5G 基地局 RF パワー",
        "category":     "ai_compute",
        "units_2026M":  3.0,
        "units_2030M":  8.0,
        "sic_chips_per_unit": 6,
        "voltage_V":    50,         # GaN-on-SiC 領域
        "power_kW":     0.2,
        "primary_mat":  "GaN_on_SiC",
        "alt_mat":      "GaN",
        "market_2026_B": 2.0,
        "market_2030_B": 4.5,
        "cagr_pct":     22,
        "moat":         "high",
        "color":        "#f58518",
    },

    # ── 🤖 ロボット ────────────────────────────────────────────────────────
    "Humanoid_robot": {
        "label":        "ヒューマノイドロボット",
        "category":     "robot",
        "units_2026M":  0.05,       # 5万台 (黎明期)
        "units_2030M":  3.0,        # 300万台 (Goldman Sachs 予測)
        "sic_chips_per_unit": 36,   # 関節×12 × 3チップ
        "voltage_V":    48,         # モーター駆動 → GaN
        "power_kW":     2.0,
        "primary_mat":  "GaN",
        "alt_mat":      "SiC",
        "market_2026_B": 0.1,
        "market_2030_B": 4.0,       # 急成長
        "cagr_pct":     150,        # 最高 CAGR
        "moat":         "low",      # まだ確立していない
        "color":        "#72B7B2",
    },
    "Industrial_robot": {
        "label":        "産業用ロボット",
        "category":     "robot",
        "units_2026M":  0.8,
        "units_2030M":  1.5,
        "sic_chips_per_unit": 12,
        "voltage_V":    400,
        "power_kW":     10,
        "primary_mat":  "SiC",
        "alt_mat":      "GaN",
        "market_2026_B": 1.5,
        "market_2030_B": 3.2,
        "cagr_pct":     21,
        "moat":         "high",
        "color":        "#54A24B",
    },

    # ── ✈️ ドローン ────────────────────────────────────────────────────────
    "Drone_commercial": {
        "label":        "商業ドローン (配送・測量)",
        "category":     "drone",
        "units_2026M":  5.0,
        "units_2030M":  20.0,
        "sic_chips_per_unit": 4,
        "voltage_V":    48,
        "power_kW":     3,
        "primary_mat":  "GaN",
        "alt_mat":      "SiC",
        "market_2026_B": 0.5,
        "market_2030_B": 2.5,
        "cagr_pct":     50,
        "moat":         "low",
        "color":        "#B279A2",
    },
    "Drone_military": {
        "label":        "軍用ドローン (FPV/MALE)",
        "category":     "drone",
        "units_2026M":  3.0,        # ウクライナ戦争が示す規模
        "units_2030M":  15.0,
        "sic_chips_per_unit": 2,
        "voltage_V":    48,
        "power_kW":     5,
        "primary_mat":  "GaN",
        "alt_mat":      "SiC",
        "market_2026_B": 0.8,
        "market_2030_B": 5.0,
        "cagr_pct":     58,
        "moat":         "very_high",  # 防衛調達
        "color":        "#FF4136",
    },
    "eVTOL": {
        "label":        "空飛ぶクルマ (eVTOL)",
        "category":     "drone",
        "units_2026M":  0.002,      # 2千機
        "units_2030M":  0.05,
        "sic_chips_per_unit": 60,   # 高信頼性 SiC 必須
        "voltage_V":    800,
        "power_kW":     250,
        "primary_mat":  "SiC",
        "alt_mat":      None,
        "market_2026_B": 0.3,
        "market_2030_B": 2.0,
        "cagr_pct":     61,
        "moat":         "very_high",
        "color":        "#FFDC00",
    },

    # ── 🛰️ 衛星・宇宙 ─────────────────────────────────────────────────────
    "LEO_satellite": {
        "label":        "LEO 衛星 (Starlink 等)",
        "category":     "space",
        "units_2026M":  0.01,       # 1万機/年
        "units_2030M":  0.05,
        "sic_chips_per_unit": 20,   # RAD-hard SiC
        "voltage_V":    100,
        "power_kW":     5,
        "primary_mat":  "SiC",      # 放射線耐性 → SiC 一択
        "alt_mat":      "GaN_on_SiC",
        "market_2026_B": 0.5,
        "market_2030_B": 3.5,
        "cagr_pct":     63,
        "moat":         "very_high",
        "color":        "#001f3f",
    },
    "Satellite_power": {
        "label":        "衛星電力系 (深宇宙・GEO)",
        "category":     "space",
        "units_2026M":  0.001,
        "units_2030M":  0.003,
        "sic_chips_per_unit": 50,
        "voltage_V":    100,
        "power_kW":     20,
        "primary_mat":  "SiC",
        "alt_mat":      None,
        "market_2026_B": 0.3,
        "market_2030_B": 0.8,
        "cagr_pct":     28,
        "moat":         "very_high",
        "color":        "#2ECC40",
    },

    # ── 産業・医療 ─────────────────────────────────────────────────────────
    "Industrial_motor": {
        "label":        "産業モーター駆動",
        "category":     "industrial",
        "units_2026M":  200.0,
        "units_2030M":  350.0,
        "sic_chips_per_unit": 2,
        "voltage_V":    690,
        "power_kW":     75,
        "primary_mat":  "SiC",
        "alt_mat":      "Si_IGBT",
        "market_2026_B": 3.0,
        "market_2030_B": 7.0,
        "cagr_pct":     24,
        "moat":         "medium",
        "color":        "#607D8B",
    },
    "Medical": {
        "label":        "医療 (MRI / 陽子線治療)",
        "category":     "industrial",
        "units_2026M":  0.01,
        "units_2030M":  0.02,
        "sic_chips_per_unit": 30,
        "voltage_V":    1700,
        "power_kW":     500,
        "primary_mat":  "SiC",
        "alt_mat":      None,
        "market_2026_B": 0.4,
        "market_2030_B": 0.9,
        "cagr_pct":     22,
        "moat":         "very_high",
        "color":        "#E91E63",
    },
}


# ════════════════════════════════════════════════════════════════════════════
# 2. 材料スペック比較
# ════════════════════════════════════════════════════════════════════════════

MATERIALS = {
    "SiC": {
        "name":            "4H-SiC",
        "bandgap_eV":      3.26,
        "breakdown_MV_cm": 2.5,
        "mobility_cm2Vs":  900,
        "thermal_W_mK":    490,
        "max_voltage_V":   15_000,
        "freq_limit_kHz":  100,
        "temp_max_C":      600,
        "wafer_cost_6in":  800,    # USD/wafer (6インチ SiC)
        "maturity":        "high",
        "color":           "#2166ac",
        "sweet_spot":      "650V–3300V, EV/鉄道/工業",
    },
    "GaN": {
        "name":            "GaN-on-Si",
        "bandgap_eV":      3.40,
        "breakdown_MV_cm": 3.3,
        "mobility_cm2Vs":  2000,   # 2DEG
        "thermal_W_mK":    130,
        "max_voltage_V":   650,
        "freq_limit_kHz":  10_000,
        "temp_max_C":      300,
        "wafer_cost_6in":  150,    # Si 基板流用
        "maturity":        "medium",
        "color":           "#d73027",
        "sweet_spot":      "<650V, 高周波, DC/DC, ドローン, 5G",
    },
    "GaN_on_SiC": {
        "name":            "GaN-on-SiC",
        "bandgap_eV":      3.40,
        "breakdown_MV_cm": 3.3,
        "mobility_cm2Vs":  2000,
        "thermal_W_mK":    400,    # SiC 基板の高熱伝導
        "max_voltage_V":   1000,
        "freq_limit_kHz":  50_000,
        "temp_max_C":      400,
        "wafer_cost_6in":  1500,
        "maturity":        "medium",
        "color":           "#f4a582",
        "sweet_spot":      "RF/5G/軍事, 高電力密度",
    },
    "Ga2O3": {
        "name":            "β-Ga₂O₃",
        "bandgap_eV":      4.80,
        "breakdown_MV_cm": 8.0,    # 理論最高クラス
        "mobility_cm2Vs":  200,    # 低い (課題)
        "thermal_W_mK":    11,     # 低い (最大課題)
        "max_voltage_V":   100_000,
        "freq_limit_kHz":  10,
        "temp_max_C":      300,
        "wafer_cost_6in":  3000,   # まだ高い
        "maturity":        "low",
        "color":           "#7b2d8b",
        "sweet_spot":      "超高圧 (>3kV), 2030年以降",
    },
    "Diamond": {
        "name":            "Diamond (CVD)",
        "bandgap_eV":      5.47,
        "breakdown_MV_cm": 10.0,
        "mobility_cm2Vs":  4500,
        "thermal_W_mK":    2200,   # 最高
        "max_voltage_V":   1_000_000,
        "freq_limit_kHz":  1_000,
        "temp_max_C":      1000,
        "wafer_cost_6in":  100_000, # 量産不可
        "maturity":        "research",
        "color":           "#aec7e8",
        "sweet_spot":      "研究段階 (量産 2040年以降?)",
    },
    "Si_IGBT": {
        "name":            "Si IGBT",
        "bandgap_eV":      1.12,
        "breakdown_MV_cm": 0.3,
        "mobility_cm2Vs":  1500,
        "thermal_W_mK":    150,
        "max_voltage_V":   6500,
        "freq_limit_kHz":  20,
        "temp_max_C":      150,
        "wafer_cost_6in":  30,
        "maturity":        "mature",
        "color":           "#969696",
        "sweet_spot":      "コスト重視, 既存インフラ置き換え中",
    },
}


# ════════════════════════════════════════════════════════════════════════════
# 3. 市場規模・成長率 スコアリング
# ════════════════════════════════════════════════════════════════════════════

def score_applications() -> list[dict]:
    """各応用領域を市場規模 × 成長率 × 戦略的価値でスコアリング。"""
    results = []
    for key, a in APPLICATIONS.items():
        # スコア = log(market_2030) × CAGR × moat_weight
        moat_w = {"very_high": 1.5, "high": 1.2, "medium": 1.0, "low": 0.7}
        w = moat_w.get(a["moat"], 1.0)

        raw_score = math.log10(max(a["market_2030_B"], 0.1)) * (a["cagr_pct"] / 10) * w
        growth_ratio = a["market_2030_B"] / max(a["market_2026_B"], 0.01)

        results.append({
            "key":           key,
            "label":         a["label"],
            "category":      a["category"],
            "market_2026_B": a["market_2026_B"],
            "market_2030_B": a["market_2030_B"],
            "cagr_pct":      a["cagr_pct"],
            "growth_ratio":  round(growth_ratio, 1),
            "primary_mat":   a["primary_mat"],
            "score":         round(raw_score, 2),
            "moat":          a["moat"],
            "voltage_V":     a["voltage_V"],
            "color":         a["color"],
        })
    return sorted(results, key=lambda x: x["score"], reverse=True)


def material_battle() -> list[dict]:
    """2026→2030 での材料別 市場シェア推定。"""
    mat_market = {}
    for a in APPLICATIONS.values():
        pm = a["primary_mat"]
        am = a.get("alt_mat")
        # プライマリ: 70%, 代替材料: 10% (将来侵食を考慮)
        mat_market[pm] = mat_market.get(pm, 0) + a["market_2030_B"] * 0.70
        if am:
            mat_market[am] = mat_market.get(am, 0) + a["market_2030_B"] * 0.10

    total = sum(mat_market.values())
    results = []
    for mat, mkt in sorted(mat_market.items(), key=lambda x: x[1], reverse=True):
        m = MATERIALS.get(mat, {})
        results.append({
            "material":    mat,
            "name":        m.get("name", mat),
            "market_2030_B": round(mkt, 1),
            "share_pct":   round(mkt / total * 100, 1),
            "sweet_spot":  m.get("sweet_spot", "-"),
            "maturity":    m.get("maturity", "-"),
            "wafer_cost":  m.get("wafer_cost_6in", 0),
            "bandgap_eV":  m.get("bandgap_eV", 0),
            "color":       m.get("color", "#999"),
        })
    return results


def top_picks() -> dict:
    """総合判定: 産業 × 材料の勝者。"""
    scores = score_applications()
    mats   = material_battle()

    top3_industries = scores[:3]
    winner_mat      = mats[0]
    dark_horse      = next(
        (m for m in mats if m["maturity"] in ("low", "medium") and m["market_2030_B"] > 1),
        mats[1]
    )

    # 総合 市場規模 2030
    total_2030 = sum(a["market_2030_B"] for a in APPLICATIONS.values())
    total_2026 = sum(a["market_2026_B"] for a in APPLICATIONS.values())

    return {
        "top3_industries":   top3_industries,
        "material_winner":   winner_mat,
        "dark_horse_mat":    dark_horse,
        "total_market_2026_B": round(total_2026, 1),
        "total_market_2030_B": round(total_2030, 1),
        "total_cagr_pct":    round(
            (total_2030 / total_2026) ** (1/4) * 100 - 100, 1),
        "all_scores":        scores,
        "all_materials":     mats,
    }


# ════════════════════════════════════════════════════════════════════════════
# Quick demo
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    picks = top_picks()

    print("=" * 60)
    print("  パワー半導体 応用領域 スコアランキング (Top 10)")
    print("=" * 60)
    for i, r in enumerate(picks["all_scores"][:10], 1):
        print(f"  {i:2}. {r['label']:<25} "
              f"CAGR={r['cagr_pct']:3}%  "
              f"2030=${r['market_2030_B']:5.1f}B  "
              f"材料={r['primary_mat']}")

    print()
    print("  材料別 市場シェア 2030")
    print("-" * 50)
    for m in picks["all_materials"]:
        bar = "█" * int(m["share_pct"] / 3)
        print(f"  {m['name']:<18} {bar:<20} {m['share_pct']:5.1f}%  ${m['market_2030_B']:.1f}B")

    print()
    print("  ★ 総合判定")
    print(f"  産業 #1: {picks['top3_industries'][0]['label']} "
          f"(CAGR {picks['top3_industries'][0]['cagr_pct']}%)")
    print(f"  産業 #2: {picks['top3_industries'][1]['label']} "
          f"(CAGR {picks['top3_industries'][1]['cagr_pct']}%)")
    print(f"  産業 #3: {picks['top3_industries'][2]['label']} "
          f"(CAGR {picks['top3_industries'][2]['cagr_pct']}%)")
    print(f"  材料 勝者:  {picks['material_winner']['name']} "
          f"({picks['material_winner']['share_pct']:.1f}%)")
    print(f"  ダークホース: {picks['dark_horse_mat']['name']}")
    print(f"  全市場: ${picks['total_market_2026_B']:.0f}B → "
          f"${picks['total_market_2030_B']:.0f}B "
          f"(CAGR {picks['total_cagr_pct']:.1f}%/年)")
