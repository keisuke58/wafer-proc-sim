"""
Disco 最強ポジション分析 — 全応用領域の集約
=============================================
「チップを作るなら全部 Disco を通る」を定量化。

Disco の収益構造:
  装置 (ダイサー・グラインダー・ポリッシャー)  ~50%
  消耗品 (ダイシングブレード)                ~50%  ← リカーリング収益

SiC は Si より硬い (モース硬度 9.5 vs 7) → ブレード消耗が速い → 消耗品売上↑
GaN / Ga₂O₃ も SiC より硬い or 脆い → Disco の特殊技術が必要

References:
  - Disco 有価証券報告書 2023
  - Yole Développement "Dicing Equipment Market" 2024
  - 各応用モデル (ai_datacenter, ev_sic, robot_drone, space_compute 等)
"""

import math

# ════════════════════════════════════════════════════════════════════════════
# 1. Disco 装置・消耗品の材料別単価
# ════════════════════════════════════════════════════════════════════════════

DISCO_BLADE_ECONOMICS = {
    "Si": {
        "material":          "シリコン (Si)",
        "mohs_hardness":     7.0,
        "blade_cost_usd":    80,        # $/枚
        "blade_life_wafers": 200,       # 1枚で何ウェーハ切れるか
        "blade_cost_per_wafer": 0.40,
        "dicing_method":     "blade",
        "saw_price_M$":      1.2,       # 装置単価
        "color":             "#969696",
    },
    "SiC": {
        "material":          "炭化ケイ素 (SiC)",
        "mohs_hardness":     9.5,
        "blade_cost_usd":    400,       # ダイヤモンドブレード
        "blade_life_wafers": 8,         # SiC はすぐ摩耗
        "blade_cost_per_wafer": 50.0,   # Si の 125倍!
        "dicing_method":     "stealth_or_blade",
        "saw_price_M$":      4.5,       # SiC 対応 高価
        "color":             "#2166ac",
    },
    "GaN": {
        "material":          "窒化ガリウム (GaN)",
        "mohs_hardness":     9.0,
        "blade_cost_usd":    250,
        "blade_life_wafers": 20,
        "blade_cost_per_wafer": 12.5,
        "dicing_method":     "blade_or_laser",
        "saw_price_M$":      2.5,
        "color":             "#d73027",
    },
    "Ga2O3": {
        "material":          "酸化ガリウム (Ga₂O₃)",
        "mohs_hardness":     7.5,       # 脆性が高く特殊技術が必要
        "blade_cost_usd":    300,       # 脆性で割れやすい → 特殊ブレード
        "blade_life_wafers": 15,
        "blade_cost_per_wafer": 20.0,
        "dicing_method":     "stealth_laser",  # Disco ステルス独壇場
        "saw_price_M$":      5.0,       # ステルス対応
        "color":             "#7b2d8b",
    },
    "GaAs": {
        "material":          "ヒ化ガリウム (GaAs) — SBSP 太陽電池",
        "mohs_hardness":     4.5,
        "blade_cost_usd":    120,
        "blade_life_wafers": 80,
        "blade_cost_per_wafer": 1.5,
        "dicing_method":     "blade",
        "saw_price_M$":      1.5,
        "color":             "#f5a623",
    },
    "Diamond": {
        "material":          "ダイヤモンド (核融合センサー等)",
        "mohs_hardness":     10.0,
        "blade_cost_usd":    2000,      # レーザー以外不可
        "blade_life_wafers": 1,
        "blade_cost_per_wafer": 2000.0,
        "dicing_method":     "laser_only",
        "saw_price_M$":      8.0,
        "color":             "#aec7e8",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 2. 全応用領域 → Disco ウェーハ需要マッピング
# ════════════════════════════════════════════════════════════════════════════

APPLICATION_TO_DISCO = {
    # ── 既存市場 ──────────────────────────────────────────────────────────
    "Consumer_Si": {
        "label":         "スマホ・PC (従来 Si)",
        "material":      "Si",
        "wafers_day":    3_000,
        "trend":         "stable",
        "category":      "existing",
        "color":         "#969696",
    },
    "Automotive_Si": {
        "label":         "自動車 ECU (Si)",
        "material":      "Si",
        "wafers_day":    500,
        "trend":         "slow_growth",
        "category":      "existing",
        "color":         "#607D8B",
    },
    # ── EV × SiC ─────────────────────────────────────────────────────────
    "EV_passenger_SiC": {
        "label":         "EV 乗用車 SiC インバータ",
        "material":      "SiC",
        "wafers_day":    207,            # ev_sic_model の計算値
        "trend":         "fast_growth",
        "category":      "ev",
        "color":         "#4c78a8",
    },
    "EV_commercial_SiC": {
        "label":         "EV 商用車・バス SiC",
        "material":      "SiC",
        "wafers_day":    80,
        "trend":         "fast_growth",
        "category":      "ev",
        "color":         "#2196F3",
    },
    # ── AI データセンター ──────────────────────────────────────────────────
    "AI_GPU_Si": {
        "label":         "AI GPU (H100/B200) Si",
        "material":      "Si",
        "wafers_day":    500,
        "trend":         "very_fast",
        "category":      "ai",
        "color":         "#e45756",
    },
    "AI_DC_SiC_PSU": {
        "label":         "AI DC SiC 電源",
        "material":      "SiC",
        "wafers_day":    1,              # ai_datacenter_model の計算値
        "trend":         "fast_growth",
        "category":      "ai",
        "color":         "#f58518",
    },
    # ── ロボット × GaN ────────────────────────────────────────────────────
    "Humanoid_GaN": {
        "label":         "ヒューマノイド GaN ESC",
        "material":      "GaN",
        "wafers_day":    5,              # 2026年推定: 5万台 × 50chips / 3000 good/wafer
        "trend":         "explosive",    # CAGR 150%
        "category":      "robot",
        "color":         "#72B7B2",
    },
    "Industrial_Robot_SiC": {
        "label":         "産業ロボット SiC",
        "material":      "SiC",
        "wafers_day":    30,
        "trend":         "growth",
        "category":      "robot",
        "color":         "#54A24B",
    },
    # ── ドローン ──────────────────────────────────────────────────────────
    "Military_Drone_GaN": {
        "label":         "軍用ドローン GaN ESC",
        "material":      "GaN",
        "wafers_day":    20,
        "trend":         "explosive",
        "category":      "drone",
        "color":         "#FF4136",
    },
    "eVTOL_SiC": {
        "label":         "eVTOL SiC インバータ",
        "material":      "SiC",
        "wafers_day":    2,
        "trend":         "growth",
        "category":      "drone",
        "color":         "#FFDC00",
    },
    # ── 衛星・宇宙 ────────────────────────────────────────────────────────
    "Starlink_ASIC_Si": {
        "label":         "Starlink ビームフォーミング ASIC (Si)",
        "material":      "Si",
        "wafers_day":    20,
        "trend":         "growth",
        "category":      "space",
        "color":         "#001f3f",
    },
    "Satellite_SiC_power": {
        "label":         "衛星 SiC 電源モジュール",
        "material":      "SiC",
        "wafers_day":    0.01,
        "trend":         "growth",
        "category":      "space",
        "color":         "#2166ac",
    },
    "SBSP_GaAs": {
        "label":         "SBSP GaAs 太陽電池 (2035年以降)",
        "material":      "GaAs",
        "wafers_day":    0,              # まだゼロ (将来)
        "trend":         "future",
        "category":      "space",
        "color":         "#f5a623",
    },
    # ── エネルギー ────────────────────────────────────────────────────────
    "BESS_SiC": {
        "label":         "系統用 BESS SiC インバータ",
        "material":      "SiC",
        "wafers_day":    10,
        "trend":         "fast_growth",
        "category":      "energy",
        "color":         "#4CAF50",
    },
    "Solar_SiC": {
        "label":         "太陽光インバータ SiC",
        "material":      "SiC",
        "wafers_day":    50,
        "trend":         "fast_growth",
        "category":      "energy",
        "color":         "#FF9800",
    },
    # ── 核融合 ────────────────────────────────────────────────────────────
    "Fusion_SiC": {
        "label":         "核融合炉 SiC 電源 (2030年代)",
        "material":      "SiC",
        "wafers_day":    0.1,
        "trend":         "future",
        "category":      "fusion",
        "color":         "#9C27B0",
    },
    "Fusion_Ga2O3": {
        "label":         "核融合 Ga₂O₃ ジャイロトロン電源",
        "material":      "Ga2O3",
        "wafers_day":    0.05,
        "trend":         "future",
        "category":      "fusion",
        "color":         "#7b2d8b",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 3. Disco 収益モデル
# ════════════════════════════════════════════════════════════════════════════

DISCO_MARKET_SHARE = 0.70    # ダイシング装置 世界シェア 70%
DISCO_BLADE_SHARE  = 0.65    # ブレード消耗品シェア 65%


def disco_revenue_from_application(app_key: str) -> dict:
    """
    1応用領域から Disco が得る年間収益の試算。
    装置収益 + 消耗品 (ブレード) 収益
    """
    app   = APPLICATION_TO_DISCO[app_key]
    blade = DISCO_BLADE_ECONOMICS[app["material"]]

    wpd = app["wafers_day"]
    wpy = wpd * 365

    # ブレード収益 (消耗品)
    blade_rev_yr = wpy * blade["blade_cost_per_wafer"] * DISCO_BLADE_SHARE

    # 装置収益 (新規装置: 1台で 10wafers/day を処理と仮定)
    saws_needed   = math.ceil(wpd / 10)
    saw_rev_yr    = saws_needed * blade["saw_price_M$"] * 1e6 * DISCO_MARKET_SHARE / 5  # 5年償却

    total_rev_yr = blade_rev_yr + saw_rev_yr

    return {
        "app":            app_key,
        "label":          app["label"],
        "material":       app["material"],
        "wafers_day":     wpd,
        "wafers_yr":      wpy,
        "blade_rev_M$":   round(blade_rev_yr / 1e6, 2),
        "saw_rev_M$":     round(saw_rev_yr / 1e6, 2),
        "total_rev_M$":   round(total_rev_yr / 1e6, 2),
        "blade_cost_per_wafer": blade["blade_cost_per_wafer"],
        "dicing_method":  blade["dicing_method"],
        "trend":          app["trend"],
        "category":       app["category"],
        "color":          app["color"],
    }


def disco_total_tam() -> dict:
    """全応用領域の Disco TAM 合計。"""
    results = []
    for key in APPLICATION_TO_DISCO:
        results.append(disco_revenue_from_application(key))

    total = sum(r["total_rev_M$"] for r in results)
    blade_total = sum(r["blade_rev_M$"] for r in results)
    saw_total   = sum(r["saw_rev_M$"]   for r in results)

    # カテゴリ別
    by_category = {}
    by_material = {}
    for r in results:
        by_category[r["category"]] = by_category.get(r["category"], 0) + r["total_rev_M$"]
        by_material[r["material"]]  = by_material.get(r["material"], 0) + r["total_rev_M$"]

    # SiC プレミアム (Si比)
    sic_premium = (
        DISCO_BLADE_ECONOMICS["SiC"]["blade_cost_per_wafer"] /
        DISCO_BLADE_ECONOMICS["Si"]["blade_cost_per_wafer"]
    )

    return {
        "total_M$":      round(total, 1),
        "blade_M$":      round(blade_total, 1),
        "saw_M$":        round(saw_total, 1),
        "blade_ratio":   round(blade_total / max(total, 1) * 100, 1),
        "by_category":   {k: round(v, 1) for k, v in by_category.items()},
        "by_material":   {k: round(v, 1) for k, v in by_material.items()},
        "sic_premium_x": sic_premium,
        "applications":  sorted(results, key=lambda x: x["total_rev_M$"], reverse=True),
    }


# ════════════════════════════════════════════════════════════════════════════
# 4. Disco のモート (競争優位) 分析
# ════════════════════════════════════════════════════════════════════════════

DISCO_MOAT = {
    "stealth_dicing": {
        "title":   "ステルスダイシング特許",
        "detail":  "レーザーを内部に集光 → SiC/GaN を無傷で分割。競合なし。",
        "patents":  200,
        "moat":    "very_high",
    },
    "consumables": {
        "title":   "消耗品ロックイン (替え刃モデル)",
        "detail":  "プリンターと同じ。装置を買ったら Disco のブレードを買い続ける。",
        "patents":  80,
        "moat":    "high",
    },
    "sic_dominance": {
        "title":   "SiC ダイシング技術",
        "detail":  "SiC は Si の 125倍のブレードコスト → 高収益。競合は量産対応不可。",
        "patents":  120,
        "moat":    "very_high",
    },
    "process_know_how": {
        "title":   "プロセスノウハウ蓄積",
        "detail":  "70年以上の精密加工データ。顧客の歩留まりをサポートする技術者派遣。",
        "patents":  0,
        "moat":    "high",
    },
}


def moat_score() -> dict:
    """Disco の競争優位スコア (Warren Buffett 的な経済的堀)。"""
    moat_weights = {"very_high": 1.5, "high": 1.0, "medium": 0.7, "low": 0.4}
    total_score = sum(
        moat_weights.get(m["moat"], 1.0) for m in DISCO_MOAT.values()
    )
    max_score = len(DISCO_MOAT) * 1.5

    return {
        "moat_score":        round(total_score / max_score * 10, 1),
        "total_patents":     sum(m["patents"] for m in DISCO_MOAT.values()),
        "stealth_monopoly":  True,   # 競合なし
        "sic_premium_x":     DISCO_BLADE_ECONOMICS["SiC"]["blade_cost_per_wafer"] /
                             DISCO_BLADE_ECONOMICS["Si"]["blade_cost_per_wafer"],
        "recurring_revenue_pct": 50,
        "verdict": "世界最強クラスのモートを持つニッチ独占企業",
    }


# ════════════════════════════════════════════════════════════════════════════
# Quick demo
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    tam = disco_total_tam()
    moat = moat_score()

    print("=" * 65)
    print("  Disco — 全応用領域からの収益分析")
    print("=" * 65)

    print(f"\n総 TAM: ${tam['total_M$']:.0f}M/年")
    print(f"  消耗品: ${tam['blade_M$']:.0f}M ({tam['blade_ratio']:.0f}%)")
    print(f"  装置:   ${tam['saw_M$']:.0f}M")

    print(f"\n上位 応用領域 (Disco 収益):")
    for r in tam["applications"][:8]:
        trend_icon = {"explosive": "🚀", "very_fast": "⚡", "fast_growth": "📈",
                      "growth": "↗", "stable": "→", "future": "🔮"}.get(r["trend"], "")
        print(f"  {trend_icon} {r['label']:<35} "
              f"${r['total_rev_M$']:>6.1f}M/yr  "
              f"({r['material']}, {r['dicing_method']})")

    print(f"\n材料別 収益:")
    for mat, rev in sorted(tam["by_material"].items(), key=lambda x: x[1], reverse=True):
        blade = DISCO_BLADE_ECONOMICS.get(mat, {})
        print(f"  {mat:<8} ${rev:>6.1f}M  "
              f"(ブレード ${blade.get('blade_cost_per_wafer', 0):.1f}/wafer)")

    print(f"\nモート分析:")
    print(f"  スコア: {moat['moat_score']}/10")
    print(f"  特許数: {moat['total_patents']}")
    print(f"  SiC ブレードプレミアム: Si 比 {moat['sic_premium_x']:.0f}倍")
    print(f"  判定: {moat['verdict']}")
