"""
宇宙安全保障 × 半導体モデル
=============================
ASAT攻撃耐性 / ケスラー症候群 / 軍事衛星チップ需要 /
Disco の地政学的リスク / 日本の宇宙安全保障

現実の宇宙安全保障は半導体なしには存在しない:
  GPS精度 → SiC クロック回路
  偵察衛星 → RAD-hard イメージングチップ
  Starshield → TID 100krad SiC
  ASAT迎撃 → GaN-on-SiC レーダー
  量子暗号通信 → フォトニックチップ

References:
  - Kessler & Cour-Palais (1978) "Collision frequency of artificial satellites"
  - DoD Space Threat Assessment 2024
  - 日本 宇宙安全保障構想 (2023年6月)
  - Starshield program documentation
  - USSPACECOM orbital debris statistics
"""

import math

# ════════════════════════════════════════════════════════════════════════════
# 1. ケスラー症候群 物理モデル
# ════════════════════════════════════════════════════════════════════════════

# 軌道別の衛星・デブリ密度 (2024年時点)
ORBITAL_ENVIRONMENT = {
    "LEO_500km": {
        "label":          "LEO 500km (Starlink 軌道)",
        "altitude_km":    550,
        "n_active":       6_000,    # Starlink 等
        "n_debris_10cm":  23_000,   # 追跡可能なデブリ
        "n_debris_1cm":   500_000,  # 追跡不可
        "collision_prob_yr": 0.001,  # 衛星1機あたりの年間衝突確率
        "kessler_threshold_sats": 20_000,  # ケスラー臨界密度
        "current_fraction": 0.30,   # 臨界の何%か
    },
    "LEO_1200km": {
        "label":          "LEO 1200km (OneWeb / Iridium)",
        "altitude_km":    1200,
        "n_active":       600,
        "n_debris_10cm":  3_000,
        "n_debris_1cm":   100_000,
        "collision_prob_yr": 0.003,
        "kessler_threshold_sats": 5_000,
        "current_fraction": 0.12,
    },
    "GEO_36000km": {
        "label":          "GEO 36,000km (通信・気象)",
        "altitude_km":    35_786,
        "n_active":       500,
        "n_debris_10cm":  300,
        "n_debris_1cm":   5_000,
        "collision_prob_yr": 0.0001,
        "kessler_threshold_sats": 2_000,
        "current_fraction": 0.25,
    },
}


def kessler_cascade_model(orbit: str = "LEO_500km",
                           asat_shots: int = 0) -> dict:
    """
    ケスラー症候群の進行モデル。
    ASAT攻撃 n 発が引き起こすデブリ連鎖を定量化。

    1機の衛星破壊 → 平均 2000個の 10cm+ デブリ生成
    各デブリは追加衝突を引き起こす (連鎖)
    """
    env = ORBITAL_ENVIRONMENT[orbit]

    # ASAT攻撃によるデブリ追加
    debris_per_asat = 2000   # Fengyun-1C 破壊で ~2800個生成 (実測)
    new_debris      = asat_shots * debris_per_asat

    total_debris = env["n_debris_10cm"] + new_debris

    # 臨界密度比
    critical_ratio = (env["n_active"] + total_debris * 0.01) / \
                     env["kessler_threshold_sats"]

    # 連鎖反応時定数 (年)
    if critical_ratio < 0.5:
        cascade_years = 999   # 安定
    elif critical_ratio < 0.8:
        cascade_years = 200
    elif critical_ratio < 1.0:
        cascade_years = 50
    else:
        cascade_years = 10   # 臨界超え → 急速連鎖

    # 年間衝突確率 (デブリ増加後)
    new_collision_prob = env["collision_prob_yr"] * (1 + new_debris / env["n_debris_10cm"])

    # GDP 損失 (GPS停止 = 世界GDPの1%/日)
    gdp_loss_per_day_B = 1000   # $1T/day = $1000B
    if critical_ratio >= 1.0:
        gdp_loss_yr_T = gdp_loss_per_day_B * 365 / 1000   # $T/年
    else:
        gdp_loss_yr_T = 0

    return {
        "orbit":            orbit,
        "asat_shots":       asat_shots,
        "total_debris":     total_debris,
        "critical_ratio":   round(critical_ratio, 3),
        "cascade_years":    cascade_years,
        "collision_prob_yr": round(new_collision_prob, 5),
        "kessler_risk":     "🔴 臨界超え" if critical_ratio >= 1.0 else
                            "🟡 警戒" if critical_ratio >= 0.7 else "🟢 安全",
        "gdp_loss_yr_T$":   round(gdp_loss_yr_T, 1),
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. 軍事衛星 チップ需要モデル
# ════════════════════════════════════════════════════════════════════════════

MILITARY_SATELLITE_PROGRAMS = {
    "Starshield_US": {
        "label":            "Starshield (米 DoD × SpaceX)",
        "n_sats":           1_000,
        "purpose":          "通信・ISR・宇宙状況把握",
        "tid_krad":         100,    # 民間 Starlink の 3倍以上
        "sic_per_sat":      20,     # 追加 RAD-hard SiC チップ
        "gan_per_sat":      8,      # GaN-on-SiC RF アンプ
        "contract_B$":      1.8,    # 推定
        "chip_premium_x":   10,     # 軍事グレード = 民間の10倍単価
        "color":            "#001f3f",
    },
    "China_Guowang": {
        "label":            "国網 (中国版 Starlink)",
        "n_sats":           13_000,
        "purpose":          "商業通信 + 軍事デュアルユース",
        "tid_krad":         50,
        "sic_per_sat":      12,
        "gan_per_sat":      6,
        "contract_B$":      None,   # 非公開
        "chip_premium_x":   5,
        "color":            "#e41a1c",
    },
    "Japan_QZS_Next": {
        "label":            "みちびき後継 + 宇宙状況把握衛星",
        "n_sats":           30,
        "purpose":          "測位精度向上 + 軍民共用",
        "tid_krad":         30,
        "sic_per_sat":      10,
        "gan_per_sat":      4,
        "contract_B$":      0.3,
        "chip_premium_x":   5,
        "color":            "#ff7f00",
    },
    "EU_Iris2": {
        "label":            "EU IRIS² (欧州自律通信衛星)",
        "n_sats":           290,
        "purpose":          "欧州のStarlink依存脱却",
        "tid_krad":         40,
        "sic_per_sat":      12,
        "gan_per_sat":      5,
        "contract_B$":      6.0,
        "chip_premium_x":   6,
        "color":            "#003399",
    },
}


def military_chip_demand(program: str = "Starshield_US") -> dict:
    """軍事衛星プログラムの半導体需要計算。"""
    p = MILITARY_SATELLITE_PROGRAMS[program]

    total_sic = p["n_sats"] * p["sic_per_sat"]
    total_gan = p["n_sats"] * p["gan_per_sat"]

    # SiC ウェーハ換算
    die_sic_mm2 = 50   # RAD-hard 小型ダイ
    dpw_sic     = int(math.pi * 150**2 / die_sic_mm2 * 0.70)
    wafers_sic  = math.ceil(total_sic / (dpw_sic * 0.85))

    # GaN ウェーハ換算 (GaN-on-SiC 4インチ基板)
    die_gan_mm2 = 8
    dpw_gan     = int(math.pi * 50**2 / die_gan_mm2 * 0.75)
    wafers_gan  = math.ceil(total_gan / (dpw_gan * 0.80))

    # 経済的価値
    sic_unit_price = 2000 * p["chip_premium_x"]   # 宇宙グレード
    gan_unit_price = 500  * p["chip_premium_x"]
    chip_value_M = (total_sic * sic_unit_price + total_gan * gan_unit_price) / 1e6

    return {
        "program":       program,
        "label":         p["label"],
        "n_sats":        p["n_sats"],
        "total_sic":     total_sic,
        "total_gan":     total_gan,
        "wafers_sic":    wafers_sic,
        "wafers_gan":    wafers_gan,
        "chip_value_M$": round(chip_value_M, 0),
        "tid_krad":      p["tid_krad"],
        "premium_x":     p["chip_premium_x"],
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. Disco の地政学的リスク × 戦略的重要性
# ════════════════════════════════════════════════════════════════════════════

DISCO_GEOPOLITICS = {
    "market_position": {
        "dicing_share_pct": 70,
        "grinding_share_pct": 60,
        "country": "Japan",
        "hq": "Tokyo",
        "classified_as": "戦略的重要インフラ",
    },
    "supply_chain_risk": {
        "single_point_failure": True,   # Disco なしでは世界の半導体製造が停止
        "alternative_suppliers": ["Accretech (日本, 15%)", "DISCO 代替なし (SiC)"],
        "lead_time_weeks":       52,     # 装置の納期: 1年待ち
        "stockpile_weeks":       8,      # 典型的な顧客在庫
    },
    "japan_strategic_value": {
        "export_control": "外為法: 武器転用可能装置は経産省許可",
        "ally_dependency": "米 TSMC (台湾) と同様の「唯一の供給者」問題",
        "military_use":    "Starshield, 各国軍事衛星のチップ全て Disco を通る",
        "china_access":    "現在: 一部製品は輸出制限。2023年から強化",
    },
}

ASAT_THREAT_MATRIX = {
    "China_DN-3": {
        "type":      "Direct Ascent ASAT",
        "target_alt_km": 36_000,   # GEO を狙える
        "demonstrated": True,      # 2007年 Fengyun-1C 破壊
        "debris_generated": 2800,
        "semiconductor_target": "GPS IIF / 偵察衛星",
    },
    "Russia_Nudol": {
        "type":      "Direct Ascent ASAT",
        "target_alt_km": 1200,
        "demonstrated": True,      # 2021年 Kosmos-1408 破壊
        "debris_generated": 1500,
        "semiconductor_target": "ISS / LEO 軍事衛星",
    },
    "Cyber_GPS_spoofing": {
        "type":      "サイバー攻撃 (GPS スプーフィング)",
        "target_alt_km": 0,        # 地上からの電波攻撃
        "demonstrated": True,      # 中東で頻発
        "debris_generated": 0,
        "semiconductor_target": "GPS 受信機チップ (SiC 量子時計が対策)",
    },
    "Co_orbital_sat": {
        "type":      "共軌道型 ASAT",
        "target_alt_km": 550,
        "demonstrated": False,     # 中国が開発中とされる
        "debris_generated": 500,
        "semiconductor_target": "Starlink / 通信衛星",
    },
}


def disco_strategic_shutdown_impact() -> dict:
    """
    Disco が輸出停止・工場停止した場合の世界半導体への影響試算。
    """
    # 世界のダイシング装置需要
    global_dicing_units_yr = 3000   # 台/年
    disco_units_yr         = global_dicing_units_yr * 0.70
    lead_time_yr           = 1.0    # 代替調達に1年

    # 影響を受けるウェーハ (Disco 装置がないと切れない)
    affected_wafers_day    = 8000   # Disco 処理分 (SiC 重点)
    gdp_impact_B_per_day   = affected_wafers_day * 50_000 / 1e9  # 平均 $5万/wafer の価値

    # SiC への影響が最大
    sic_wafers_day_disco   = 400    # SiC の Disco 依存度ほぼ100%
    ev_impact_pct          = 80     # EV SiC の 80% が Disco 依存

    return {
        "disco_units_yr":       disco_units_yr,
        "stockpile_weeks":      8,
        "lead_time_yr":         lead_time_yr,
        "affected_wafers_day":  affected_wafers_day,
        "gdp_impact_B_day":     round(gdp_impact_B_per_day, 2),
        "sic_affected_pct":     95,
        "ev_production_loss_pct": ev_impact_pct,
        "semiconductor_gdp_loss_yr_T$": round(gdp_impact_B_per_day * 365 / 1000, 1),
        "strategic_classification": "半導体版 TSMC リスク",
        "japan_leverage":       "Disco を持つ日本は半導体安全保障の鍵",
    }


# ════════════════════════════════════════════════════════════════════════════
# 4. 日本の宇宙安全保障 × 半導体戦略
# ════════════════════════════════════════════════════════════════════════════

JAPAN_SPACE_SECURITY = {
    "2023_strategy": {
        "title":   "宇宙安全保障構想 (2023年6月)",
        "key_points": [
            "宇宙状況把握 (SSA) 衛星の整備",
            "JAXA と防衛省の連携強化",
            "衛星攻撃への対処能力 (抑止力)",
            "Starlink への依存リスク認識",
            "国産衛星の強靭化 (RAD-hard SiC 採用)",
        ],
        "budget_B_yen": 1.0,   # 概算
    },
    "semiconductor_angle": {
        "disco_as_leverage":   "Disco 輸出制限 = 中国の軍事衛星製造を阻害",
        "sic_restriction":     "SiC ウェーハ/装置の対中輸出規制 (2023〜)",
        "jaxa_sic_adoption":   "JAXA 次期衛星に SiC パワー系採用予定",
        "raketen_sic":         "三菱電機の防衛衛星に SiC 電源管理 IC",
    },
    "vulnerabilities": {
        "gps_dependency":      "自衛隊の精密誘導兵器はほぼ GPS 依存",
        "no_asat_capability":  "現行法: 日本は ASAT 能力を保有しない",
        "starlink_dependency": "陸自の通信 Starlink 依存 (マスクリスク)",
        "disco_single_point":  "Disco が攻撃されると同盟国の衛星も製造停止",
    },
}


# ════════════════════════════════════════════════════════════════════════════
# Quick demo
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  宇宙安全保障 × 半導体モデル")
    print("=" * 65)

    print("\n【ケスラー症候群リスク】")
    for orbit in ORBITAL_ENVIRONMENT:
        for asat in [0, 5, 20]:
            r = kessler_cascade_model(orbit, asat)
            if asat == 0 or orbit == "LEO_500km":
                print(f"  {r['orbit']:<20} ASAT×{asat:2}発 "
                      f"→ 臨界比={r['critical_ratio']:.2f}  "
                      f"{r['kessler_risk']}  "
                      f"連鎖まで{r['cascade_years']}年")

    print("\n【軍事衛星 半導体需要】")
    for prog in MILITARY_SATELLITE_PROGRAMS:
        r = military_chip_demand(prog)
        print(f"  {r['label'][:30]:<32} "
              f"SiC={r['total_sic']:>6,}個  "
              f"GaN={r['total_gan']:>5,}個  "
              f"${r['chip_value_M$']:.0f}M")

    print("\n【Disco 輸出停止インパクト】")
    d = disco_strategic_shutdown_impact()
    print(f"  影響ウェーハ/日: {d['affected_wafers_day']:,}")
    print(f"  GDP損失/日:    ${d['gdp_impact_B_day']:.2f}B")
    print(f"  年間損失:      ${d['semiconductor_gdp_loss_yr_T$']:.1f}T")
    print(f"  SiC影響率:     {d['sic_affected_pct']}%")
    print(f"  戦略的位置付け: {d['strategic_classification']}")

    print("\n【日本の半導体安全保障 カード】")
    for k, v in JAPAN_SPACE_SECURITY["semiconductor_angle"].items():
        print(f"  {k}: {v}")
