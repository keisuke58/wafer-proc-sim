"""
SpaceX × 宇宙コンピューティング × 宇宙太陽光発電モデル
=========================================================
① Starlink / SpaceX カスタムチップ (ビームフォーミング ASIC, Raptor ECU)
② 宇宙データセンター (軌道上 GPU クラスター)
③ 宇宙太陽光発電 SBSP (Space-Based Solar Power)

References:
  - SpaceX Starlink v2 technical briefing (2023)
  - JAXA SBSP ロードマップ (2023)
  - ESA SOLARIS initiative (2022)
  - NRL / AFRL wireless power transmission experiments (2023)
  - Microsoft Azure Space (2023)
  - Frazer-Nash SBSP feasibility study (2021, UK BEIS)
  - Starship IFU payload specs (2024): $1000/kg → $100/kg 目標
"""

import math

# ════════════════════════════════════════════════════════════════════════════
# 1. SpaceX / Starlink カスタムチップ
# ════════════════════════════════════════════════════════════════════════════

STARLINK_CHIP_SPECS = {
    # 衛星搭載チップ
    "Beamforming_ASIC": {
        "label":          "衛星フェーズドアレイ ASIC",
        "function":       "1024素子アンテナのビームフォーミング演算",
        "process_node":   "7nm (TSMC)",
        "die_area_mm2":   80,
        "freq_GHz":       28,       # Ku/Ka バンド
        "power_W":        8,
        "chips_per_sat":  4,        # v2 Mini: 4 ASIC
        "rad_hard":       True,
        "tid_krad":       30,       # 5年 LEO
        "price_usd":      500,
        "color":          "#001f3f",
    },
    "Satellite_SoC": {
        "label":          "衛星メイン SoC",
        "function":       "軌道制御・スラスタ管理・暗号化・ルーティング",
        "process_node":   "12nm (Samsung)",
        "die_area_mm2":   150,
        "freq_GHz":       1.5,
        "power_W":        25,
        "chips_per_sat":  2,
        "rad_hard":       True,
        "tid_krad":       30,
        "price_usd":      2000,
        "color":          "#2166ac",
    },
    "Power_Management": {
        "label":          "衛星電源管理 IC (SiC ベース)",
        "function":       "太陽電池 MPPT + バッテリー充放電 + DC/DC",
        "process_node":   "SiC 1200V",
        "die_area_mm2":   25,
        "freq_GHz":       0,
        "power_W":        5,        # 自身の消費
        "chips_per_sat":  4,
        "rad_hard":       True,
        "tid_krad":       30,
        "price_usd":      800,
        "color":          "#4CAF50",
    },
    # ユーザー端末チップ (地上)
    "User_Terminal_ASIC": {
        "label":          "ユーザー端末 フェーズドアレイ ASIC",
        "function":       "平面アンテナ 1280素子 ビームステアリング",
        "process_node":   "16nm (TSMC)",
        "die_area_mm2":   200,
        "freq_GHz":       12,       # Ku バンド受信
        "power_W":        100,      # 端末全体
        "chips_per_sat":  0,        # 地上端末
        "chips_per_terminal": 8,
        "rad_hard":       False,
        "tid_krad":       0,
        "price_usd":      150,      # 地上は安い
        "color":          "#d73027",
    },
    # Raptor エンジン制御
    "Raptor_ECU": {
        "label":          "Raptor エンジン制御ユニット",
        "function":       "燃料流量・燃焼制御 (33台/Starship)",
        "process_node":   "28nm (耐熱性重視)",
        "die_area_mm2":   60,
        "freq_GHz":       0.3,
        "power_W":        15,
        "chips_per_sat":  0,
        "chips_per_vehicle": 33,   # Starship: 33 Raptor
        "rad_hard":       False,   # 大気圏内 → 放射線より熱・振動
        "tid_krad":       0,
        "price_usd":      5000,
        "color":          "#FF4136",
    },
}

STARLINK_CONSTELLATION = {
    "current_2024":  6_000,
    "target_v2":    12_000,
    "ultimate":     42_000,    # FCC 申請上限
    "starshield":    1_000,    # 軍事版 (US DoD 契約)
    "launch_per_yr":  1_500,   # Falcon 9 + Starship
}


def starlink_chip_demand() -> dict:
    """Starlink コンステレーション全体のチップ需要。"""
    n_sat = STARLINK_CONSTELLATION["target_v2"]
    n_new_per_yr = STARLINK_CONSTELLATION["launch_per_yr"]

    total_chips = {}
    annual_chips = {}
    for key, spec in STARLINK_CHIP_SPECS.items():
        if spec.get("chips_per_sat", 0) > 0:
            total = n_sat * spec["chips_per_sat"]
            annual = n_new_per_yr * spec["chips_per_sat"]
            total_chips[key]  = total
            annual_chips[key] = annual

    # ウェーハ換算 (7nm → 300mm ウェーハ)
    # TSMC N7: die_per_wafer ≈ 300mm²_wafer / die_area * 0.85
    wafer_demand = {}
    for key, spec in STARLINK_CHIP_SPECS.items():
        if spec.get("chips_per_sat", 0) > 0:
            wafer_area = math.pi * 150 ** 2
            dpw = int(wafer_area / spec["die_area_mm2"] * 0.85)
            n = annual_chips.get(key, 0)
            wafer_demand[key] = math.ceil(n / (dpw * 0.90))

    # ユーザー端末: 月100万台と仮定
    terminals_per_yr = 1_200_000
    ut_chips = terminals_per_yr * STARLINK_CHIP_SPECS["User_Terminal_ASIC"]["chips_per_terminal"]

    return {
        "n_satellites":       n_sat,
        "annual_new_sats":    n_new_per_yr,
        "total_chips":        total_chips,
        "annual_chips":       annual_chips,
        "wafer_demand_yr":    wafer_demand,
        "user_terminal_chips_yr": ut_chips,
        "total_chip_value_M$": sum(
            annual_chips.get(k, 0) * STARLINK_CHIP_SPECS[k]["price_usd"]
            for k in annual_chips
        ) / 1e6,
    }


def starshield_military_value() -> dict:
    """Starshield (軍事 Starlink) の付加価値。"""
    n = STARLINK_CONSTELLATION["starshield"]
    # 軍事グレード: 追加放射線耐性処理 + 暗号化 + AESA アップグレード
    mil_premium_per_sat = 500_000   # $500k/衛星 (民間比 ×10)
    total_contract_B = n * mil_premium_per_sat / 1e9

    # 軍事グレード SiC 電源: TID 100krad 仕様
    extra_sic_per_sat = 8   # 追加 SiC チップ (耐放射線強化)

    return {
        "n_starshield_sats": n,
        "contract_B$":       round(total_contract_B, 1),
        "sic_chips_total":   n * extra_sic_per_sat,
        "tid_spec_krad":     100,
        "encryption":        "AES-256 + QKD 対応",
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. 宇宙データセンター
# ════════════════════════════════════════════════════════════════════════════

SPACE_DC_CONCEPTS = {
    "Microsoft_Azure_Orbital": {
        "label":         "Microsoft Azure Orbital",
        "status":        "実証段階 (地上連携)",
        "orbit_km":      550,
        "power_kW":      20,
        "compute":       "FPGA + GPU クラスター",
        "cooling":       "ラジエーター (放射冷却)",
        "latency_ms":    20,         # LEO 往復遅延
        "advantage":     "地上 DC 不要地域への即時展開",
        "sic_chips":     40,
        "color":         "#0078d4",
    },
    "SpaceX_Starlink_Edge": {
        "label":         "Starlink 衛星エッジコンピューティング",
        "status":        "v2 衛星に GPU 搭載計画",
        "orbit_km":      550,
        "power_kW":      5,          # 衛星電力の一部を計算に
        "compute":       "NVIDIA Jetson Orin 相当",
        "cooling":       "機体放熱",
        "latency_ms":    2,          # 衛星間レーザーリンク
        "advantage":     "超低遅延、海上・極地カバー",
        "sic_chips":     10,
        "color":         "#001f3f",
    },
    "Orbital_Reef_DC": {
        "label":         "Orbital Reef 商業宇宙ステーション DC",
        "status":        "2030年代計画 (Blue Origin + Sierra Space)",
        "orbit_km":      500,
        "power_kW":      100,
        "compute":       "専用 GPU サーバー (放射線耐性)",
        "cooling":       "大型ラジエーターパネル",
        "latency_ms":    50,
        "advantage":     "量子通信ノード、物理的隔離セキュリティ",
        "sic_chips":     200,
        "color":         "#4c78a8",
    },
}


def space_dc_physics(concept: str = "SpaceX_Starlink_Edge") -> dict:
    """
    宇宙 DC の物理的制約モデル。
    最大の課題: 廃熱処理 (大気がないため空冷不可)
    """
    dc = SPACE_DC_CONCEPTS[concept]
    P_kW = dc["power_kW"]

    # ステファン-ボルツマン則: P = σ × ε × A × T^4
    sigma = 5.67e-8     # W/m²K⁴
    epsilon = 0.9       # 黒体放射率
    T_radiator_K = 320  # 放射板温度 (GPU 冷却後)
    T_space_K    = 4    # 宇宙背景温度

    A_radiator = P_kW * 1000 / (sigma * epsilon * (T_radiator_K**4 - T_space_K**4))

    # GPU 密度 (放射線耐性版): 通常の 1/5 (ECC + シールド分)
    P_per_gpu_W = 300    # 放射線耐性 GPU (H100 の 1/3 性能)
    n_gpu = max(1, int(P_kW * 1000 / P_per_gpu_W))
    tflops = n_gpu * 600   # 600 TFLOPS/GPU (rad-hard 版)

    # 地上 DC との比較
    pue_ground = 1.15
    power_equiv_ground_kW = P_kW / pue_ground
    cooling_ground_kW = power_equiv_ground_kW * (pue_ground - 1)

    # 宇宙の優位: 太陽光は 24h/day (影の時間以外)
    solar_availability = 0.60   # LEO: 60% 日照
    solar_power_kW = 22 * 0.30 * 1.361 * solar_availability   # Starlink v2 Mini 換算

    return {
        "concept":            concept,
        "label":              dc["label"],
        "P_compute_kW":       P_kW,
        "A_radiator_m2":      round(A_radiator, 1),
        "n_gpu_equiv":        n_gpu,
        "tflops":             tflops,
        "T_radiator_K":       T_radiator_K,
        "latency_ms":         dc["latency_ms"],
        "solar_power_kW":     round(solar_power_kW, 1),
        "status":             dc["status"],
        "sic_chips":          dc["sic_chips"],
    }


def space_dc_vs_ground(power_kW: float = 10.0) -> dict:
    """宇宙 DC vs 地上 DC: コスト・効率・レイテンシ比較。"""
    # 地上 DC
    pue_ground      = 1.15
    elec_cost_kwh   = 0.05
    opex_ground_yr  = power_kW * pue_ground * 8760 * elec_cost_kwh / 1e3   # $M/yr
    capex_ground_M  = power_kW * 2.0   # $2M/kW 地上 DC 建設

    # 宇宙 DC
    launch_cost_kg  = 1000   # Starship 目標 $1000/kg
    dc_mass_kg_kW   = 50    # 50 kg/kW (放射線耐性機器)
    launch_cost_M   = power_kW * dc_mass_kg_kW * launch_cost_kg / 1e6
    opex_space_yr   = 0.1   # 無人 → 電気代ほぼゼロ
    life_yr         = 5
    total_cost_M    = launch_cost_M + opex_space_yr * life_yr

    return {
        "power_kW":          power_kW,
        "ground_capex_M$":   round(capex_ground_M, 2),
        "ground_opex_yr_M$": round(opex_ground_yr, 3),
        "ground_total_M$":   round(capex_ground_M + opex_ground_yr * life_yr, 2),
        "space_launch_M$":   round(launch_cost_M, 2),
        "space_total_M$":    round(total_cost_M, 2),
        "breakeven_$/kg":    round(capex_ground_M * 1e6 / (power_kW * dc_mass_kg_kW), 0),
        "advantage":         "宇宙有利" if total_cost_M < capex_ground_M else "地上有利",
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. 宇宙太陽光発電 SBSP
# ════════════════════════════════════════════════════════════════════════════

SBSP_PROJECTS = {
    "JAXA_2050": {
        "label":          "JAXA SBSP (2050年 1GW 目標)",
        "power_target_GW": 1.0,
        "orbit_km":       36_000,    # GEO
        "solar_eff":      0.35,      # 宇宙用 GaAs 多接合
        "dc_rf_eff":      0.85,      # DC → マイクロ波変換 (SiC 使用)
        "transmission_eff": 0.75,    # 大気透過
        "rectenna_eff":   0.85,      # 地上整流アンテナ
        "freq_GHz":       2.45,      # マイクロ波 (ISMバンド)
        "satellite_mass_t": 10_000,  # 1GW 衛星: 1万トン
        "launch_cost_B$": 100,       # 現時点
        "target_cost_B$": 10,        # Starship 活用後
        "status":         "2025年 軌道実証機",
        "color":          "#FF4136",
    },
    "ESA_SOLARIS": {
        "label":          "ESA SOLARIS (2040年代)",
        "power_target_GW": 2.0,
        "orbit_km":       36_000,
        "solar_eff":      0.38,
        "dc_rf_eff":      0.90,      # GaN/SiC 進化後
        "transmission_eff": 0.78,
        "rectenna_eff":   0.87,
        "freq_GHz":       5.8,       # 高周波数 → 小さい整流アンテナ
        "satellite_mass_t": 8_000,
        "launch_cost_B$": 50,
        "target_cost_B$": 8,
        "status":         "可能性調査完了 (2022)",
        "color":          "#003399",
    },
    "China_OMEGA": {
        "label":          "中国 OMEGA (2035年 1MW 実証)",
        "power_target_GW": 0.001,    # まず 1MW
        "orbit_km":       36_000,
        "solar_eff":      0.30,
        "dc_rf_eff":      0.80,
        "transmission_eff": 0.70,
        "rectenna_eff":   0.82,
        "freq_GHz":       2.45,
        "satellite_mass_t": 10,      # 1MW 規模
        "launch_cost_B$": 0.5,
        "target_cost_B$": 0.5,
        "status":         "2035年 軌道実証目標",
        "color":          "#e41a1c",
    },
    "Caltech_SSPP": {
        "label":          "Caltech SSPP (2023年 LEO 実証成功)",
        "power_target_GW": 0.000001, # 1W (実証実験)
        "orbit_km":       550,
        "solar_eff":      0.33,
        "dc_rf_eff":      0.70,
        "transmission_eff": 0.60,
        "rectenna_eff":   0.80,
        "freq_GHz":       2.45,
        "satellite_mass_t": 0.000005,  # 5kg キューブサット
        "launch_cost_B$": 0.001,
        "target_cost_B$": 0.001,
        "status":         "2023年 軌道実証 SUCCESS ✓",
        "color":          "#4CAF50",
    },
}


def sbsp_power_chain(project: str = "JAXA_2050") -> dict:
    """
    SBSP 電力チェーン効率計算。
    太陽光 (宇宙) → DC → マイクロ波 → 大気 → 整流アンテナ → AC グリッド

    各ステップで SiC / GaN / GaAs の役割が異なる。
    """
    p = SBSP_PROJECTS[project]
    solar_irr = 1361   # W/m² (GEO, 遮蔽なし)

    # 太陽電池 面積
    P_target_W = p["power_target_GW"] * 1e9
    A_solar_m2 = P_target_W / (solar_irr * p["solar_eff"]
                                * p["dc_rf_eff"] * p["transmission_eff"] * p["rectenna_eff"])

    # 総合効率
    eta_total = (p["solar_eff"] * p["dc_rf_eff"]
                 * p["transmission_eff"] * p["rectenna_eff"])

    # 必要 SiC DC/DC 変換器 (1MW ユニット)
    P_dc_W = P_target_W / (p["dc_rf_eff"] * p["transmission_eff"] * p["rectenna_eff"])
    n_sic_units = math.ceil(P_dc_W / 1e6)   # 1MW/ユニット

    # 宇宙打ち上げコスト
    mass_kg = p["satellite_mass_t"] * 1000
    launch_now_B  = p["launch_cost_B$"]
    launch_star_B = mass_kg * 1000 / 1e9   # Starship $1000/kg

    # 地上等価物との比較 (LCOE)
    lcoe_sbsp  = launch_now_B * 1e9 / (P_target_W * 8760 * 30)   # 30年運用
    lcoe_solar = 0.04   # $/kWh 地上太陽光

    return {
        "project":          project,
        "label":            p["label"],
        "P_target_GW":      p["power_target_GW"],
        "eta_total_pct":    round(eta_total * 100, 1),
        "A_solar_km2":      round(A_solar_m2 / 1e6, 2),
        "n_sic_units":      n_sic_units,
        "mass_t":           p["satellite_mass_t"],
        "launch_now_B$":    launch_now_B,
        "launch_starship_B$": round(launch_star_B, 2),
        "lcoe_sbsp_$/kWh":  round(lcoe_sbsp * 1e6, 2),
        "lcoe_solar_$/kWh": lcoe_solar,
        "status":           p["status"],
        "freq_GHz":         p["freq_GHz"],
        "eta_dc_rf_pct":    p["dc_rf_eff"] * 100,
    }


def sbsp_semiconductor_role() -> list[dict]:
    """SBSP の各構成要素と必要半導体の対応。"""
    return [
        {
            "component":    "太陽電池アレイ",
            "material":     "GaAs 三接合",
            "reason":       "宇宙放射線耐性 + 効率 35% (Si 22%)",
            "scale":        "km²オーダー",
            "readiness":    "実用",
        },
        {
            "component":    "DC/DC 変換器 (衛星内)",
            "material":     "SiC 1200V",
            "reason":       "高放射線耐性・高温動作・高効率 (98.5%)",
            "scale":        "1MW/ユニット × 数千台",
            "readiness":    "実用",
        },
        {
            "component":    "マイクロ波送信アレイ (RF)",
            "material":     "GaN-on-SiC",
            "reason":       "2.45/5.8GHz 高電力 RF → GaN が最適",
            "scale":        "数百万素子 フェーズドアレイ",
            "readiness":    "研究→実証段階",
        },
        {
            "component":    "電力管理 IC (温度 -170〜+120°C)",
            "material":     "SiC / Ga₂O₃",
            "reason":       "宇宙の極端な温度サイクル (-170〜+120°C) に耐える",
            "scale":        "衛星あたり 10-50 個",
            "readiness":    "SiC実用/Ga₂O₃研究",
        },
        {
            "component":    "地上整流アンテナ (レクテナ)",
            "material":     "GaAs Schottky ダイオード",
            "reason":       "2.45GHz 整流効率 85% → GaAs が最高",
            "scale":        "km²アレイ",
            "readiness":    "実証済み",
        },
        {
            "component":    "軌道間電力伝送 (レーザー版)",
            "material":     "AlGaAs / InGaAs レーザー",
            "reason":       "1064nm レーザー → 高強度ビーム",
            "scale":        "実証実験段階",
            "readiness":    "Caltech 2023年実証",
        },
    ]


def sbsp_market_forecast() -> list[dict]:
    """SBSP 関連半導体市場予測 (Starship によるコスト革命前提)。"""
    return [
        {"year": 2025, "stage": "実証",   "sbsp_GW": 0.001, "sic_M$": 10,   "gan_M$": 5},
        {"year": 2030, "stage": "原型",   "sbsp_GW": 0.1,   "sic_M$": 500,  "gan_M$": 200},
        {"year": 2035, "stage": "商業初", "sbsp_GW": 1.0,   "sic_M$": 5000, "gan_M$": 2000},
        {"year": 2040, "stage": "商業",   "sbsp_GW": 10.0,  "sic_M$": 40000,"gan_M$": 15000},
        {"year": 2050, "stage": "主力",   "sbsp_GW": 100.0, "sic_M$": 300000,"gan_M$": 100000},
    ]


# ════════════════════════════════════════════════════════════════════════════
# Quick demo
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  SpaceX × 宇宙コンピューティング × 宇宙太陽光発電")
    print("=" * 65)

    # 1. Starlink チップ
    demand = starlink_chip_demand()
    print(f"\n【Starlink チップ需要】({demand['n_satellites']:,}機コンステ)")
    for key, n in demand["annual_chips"].items():
        spec = STARLINK_CHIP_SPECS[key]
        w = demand["wafer_demand_yr"].get(key, 0)
        print(f"  {spec['label'][:25]:<26} {n:>8,} chips/yr  {w:>4} wafers/yr")
    print(f"  年間チップ価値: ${demand['total_chip_value_M$']:.0f}M")
    print(f"  ユーザー端末 ASIC: {demand['user_terminal_chips_yr']/1e6:.1f}M chips/yr")

    # 2. 宇宙 DC
    print(f"\n【宇宙データセンター 物理制約】")
    for concept in SPACE_DC_CONCEPTS:
        r = space_dc_physics(concept)
        print(f"  {r['label'][:30]:<32} "
              f"放熱板={r['A_radiator_m2']:.1f}m²  "
              f"GPU×{r['n_gpu_equiv']}  "
              f"{r['tflops']}TFLOPS")

    vs = space_dc_vs_ground(10.0)
    print(f"\n  10kW DC: 地上=${vs['ground_total_M$']:.1f}M "
          f"vs 宇宙=${vs['space_total_M$']:.1f}M  "
          f"(打上 $1000/kg 仮定)")
    print(f"  損益分岐点: ${vs['breakeven_$/kg']:.0f}/kg → Starship 目標 $100/kg なら宇宙有利")

    # 3. SBSP
    print(f"\n【宇宙太陽光発電 SBSP】")
    for proj in ["JAXA_2050", "Caltech_SSPP"]:
        r = sbsp_power_chain(proj)
        print(f"  {r['label'][:30]:<32} "
              f"総合効率={r['eta_total_pct']:.1f}%  "
              f"太陽電池={r['A_solar_km2']:.3f}km²  "
              f"LCOE=${r['lcoe_sbsp_$/kWh']:.2f}/kWh")

    print(f"\n  半導体の役割:")
    for row in sbsp_semiconductor_role():
        print(f"  {row['component']:<20} → {row['material']:<20} ({row['readiness']})")
