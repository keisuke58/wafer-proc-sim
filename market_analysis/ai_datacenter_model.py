"""
AI Data Center × SiC Power Electronics Model
============================================
AI チップ消費電力 → データセンター電力インフラ → SiC パワーモジュール需要 → ウェーハ需要

Chain:
  GPU/TPU スペック → サーバー消費電力 → ラック → データセンター
      → SiC PSU/UPS/PDU 需要 → SiC チップ面積 → ウェーハ枚数

References:
  - IEA "Electricity 2024" — AI data center power forecasts
  - NVIDIA H100/B200 product specs
  - Wolfspeed / STMicro SiC power module specs
  - Lawrence Berkeley National Lab "US Data Center Energy Use 2023"
"""

import math

# ════════════════════════════════════════════════════════════════════════════
# AI チップ消費電力スペック
# ════════════════════════════════════════════════════════════════════════════
AI_CHIP_POWER_W = {
    "H100_SXM5":   700,   # NVIDIA H100 SXM5 (Hopper, 4nm)
    "B200_SXM":   1000,   # NVIDIA B200 SXM (Blackwell, 4nm)
    "B300_SXM":   1200,   # NVIDIA B300 (Blackwell Ultra, 2025)
    "TPU_v6e":     200,   # Google TPU v6e (per chip, 5nm)
    "MI300X":      750,   # AMD Instinct MI300X (5nm)
    "Trainium3":   500,   # AWS Trainium3
    "Tesla_AI5":   600,   # Tesla AI5 (2nm, 2026)
    "Gaudi3":      900,   # Intel Gaudi3
}

# データセンター内サーバー構成 (chips per server)
CHIPS_PER_SERVER = {
    "H100_SXM5":  8,
    "B200_SXM":   8,
    "B300_SXM":   8,
    "TPU_v6e":   16,   # Google TPU Pod スライス
    "MI300X":     8,
    "Trainium3":  8,
    "Tesla_AI5":  8,
    "Gaudi3":     8,
}

# 大手 AI クラスター GPU 台数 (2026年 推定)
AI_CLUSTER_GPUS = {
    "Microsoft_Azure":  400_000,   # OpenAI 向け含む
    "Google_Cloud":     350_000,   # TPU + GPU 混在
    "Meta_AI":          350_000,   # Llama 系トレーニング
    "AWS":              300_000,   # Trainium + H100
    "Tesla_Dojo":       100_000,   # FSD + Optimus
    "xAI_Colossus":     100_000,   # Grok
    "CoreWeave":        150_000,
    "Oracle_OCI":        80_000,
}

# SiC パワーモジュールスペック (Wolfspeed / STMicro)
SIC_MODULE = {
    "rating_kW":       50.0,    # 1モジュール定格 [kW]
    "efficiency":       0.985,  # PSU 変換効率 (SiC: 98.5% vs Si: 94%)
    "die_area_mm2":   120.0,    # SiC チップ面積 (1200V, 300A モジュール)
    "price_usd":      800.0,    # $/モジュール (2026)
    "wafer_yield":      0.80,   # 300mm SiC ウェーハからの良品率
    "die_per_wafer":   400,     # 300mm ウェーハ / 120mm² ダイ
}

# PUE (Power Usage Effectiveness): 実際の電力 / IT 電力
PUE = {
    "hyperscaler":  1.12,   # Google/Microsoft 最新 DC
    "colocation":   1.45,
    "enterprise":   1.58,
    "average":      1.30,
}


# ════════════════════════════════════════════════════════════════════════════
# 1. サーバー / クラスター 電力モデル
# ════════════════════════════════════════════════════════════════════════════
def server_power(chip: str = "H100_SXM5", utilization: float = 0.85) -> dict:
    """
    1サーバーの消費電力モデル。
    GPU 電力 + CPU + メモリ + ネットワーク + HBM を含む。

    utilization: GPU 使用率 (学習: 0.85, 推論: 0.60)
    """
    n_chips = CHIPS_PER_SERVER.get(chip, 8)
    p_chip  = AI_CHIP_POWER_W.get(chip, 700)

    p_gpu_total = n_chips * p_chip * utilization
    p_cpu       = 400.0          # 2× CPU (Xeon / EPYC)
    p_memory    = 300.0          # DDR5 / HBM
    p_network   = 200.0          # 400GbE InfiniBand NIC
    p_storage   = 100.0          # NVMe SSD

    p_it_total  = p_gpu_total + p_cpu + p_memory + p_network + p_storage

    # SiC PSU: AC→DC 変換
    p_input = p_it_total / SIC_MODULE["efficiency"]   # 入力電力 (損失込み)
    p_loss  = p_input - p_it_total

    return {
        "chip":          chip,
        "n_chips":       n_chips,
        "p_gpu_W":       round(p_gpu_total, 0),
        "p_server_it_W": round(p_it_total, 0),
        "p_input_W":     round(p_input, 0),      # PSU 入力
        "p_loss_W":      round(p_loss, 1),
        "psu_efficiency_pct": SIC_MODULE["efficiency"] * 100,
    }


def cluster_power(company: str = "Microsoft_Azure",
                  chip: str = "H100_SXM5",
                  pue_type: str = "hyperscaler") -> dict:
    """
    クラスター全体の電力 + SiC モジュール需要。
    """
    n_gpu    = AI_CLUSTER_GPUS.get(company, 100_000)
    sv       = server_power(chip)
    n_chips  = CHIPS_PER_SERVER.get(chip, 8)
    n_server = math.ceil(n_gpu / n_chips)

    it_mw    = n_server * sv["p_server_it_W"] / 1e6
    total_mw = it_mw * PUE[pue_type]

    # SiC モジュール数 (PSU 用): 1サーバー 1台の PSU が 50kW モジュールを使用
    sic_per_server = math.ceil(sv["p_input_W"] / (SIC_MODULE["rating_kW"] * 1000))
    sic_total      = n_server * sic_per_server
    sic_cost_m     = sic_total * SIC_MODULE["price_usd"] / 1e6

    return {
        "company":        company,
        "n_gpu":          n_gpu,
        "n_server":       n_server,
        "it_power_MW":    round(it_mw, 1),
        "total_power_MW": round(total_mw, 1),
        "sic_modules":    sic_total,
        "sic_cost_M_usd": round(sic_cost_m, 1),
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. SiC ウェーハ需要換算
# ════════════════════════════════════════════════════════════════════════════
def sic_wafer_demand_from_ai(annual_growth_pct: float = 40.0) -> dict:
    """
    全 AI クラスター合計の SiC ウェーハ需要推定。

    annual_growth_pct: 年間設備増強率 (AI DC は ~40%/年 拡張中)
    SiC モジュール需要 = 新規導入分 + 更新分 (10%/年)
    """
    fleet_modules = 0
    breakdown = {}
    for company in AI_CLUSTER_GPUS:
        r = cluster_power(company)
        fleet_modules += r["sic_modules"]
        breakdown[company] = r["sic_modules"]

    # 年間需要 = 新規拡張 + リプレース
    annual_new    = fleet_modules * annual_growth_pct / 100
    annual_replace = fleet_modules * 0.10
    annual_modules = annual_new + annual_replace

    # ウェーハ換算 (1モジュール = 1ダイ 簡略化)
    good_per_wafer = SIC_MODULE["die_per_wafer"] * SIC_MODULE["wafer_yield"]
    wafers_per_year = math.ceil(annual_modules / good_per_wafer)

    ev_wafers_day = 8_000
    ai_wpd        = round(wafers_per_year / 365, 0)

    return {
        "fleet_sic_modules":  fleet_modules,
        "annual_sic_modules": round(annual_modules, 0),
        "wafers_per_year":    wafers_per_year,
        "wafers_per_day":     ai_wpd,
        "ev_wafers_per_day":  ev_wafers_day,
        "ai_fraction_pct":    round(ai_wpd / (ai_wpd + ev_wafers_day) * 100, 1),
        "breakdown":          breakdown,
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. Si IGBT vs SiC 効率・コスト比較
# ════════════════════════════════════════════════════════════════════════════
def sic_vs_si_tco(cluster_power_mw: float = 100.0,
                  years: float = 5.0,
                  electricity_usd_kwh: float = 0.05) -> dict:
    """
    SiC PSU vs Si IGBT PSU の TCO (Total Cost of Ownership) 比較。
    データセンター規模での導入判断指標。
    """
    efficiency_sic = 0.985
    efficiency_si  = 0.940

    # 年間電力消費 [kWh]
    hours_per_year = 8_760
    energy_sic_kwh = cluster_power_mw * 1000 / efficiency_sic * hours_per_year * years
    energy_si_kwh  = cluster_power_mw * 1000 / efficiency_si  * hours_per_year * years

    # 電気代
    cost_elec_sic = energy_sic_kwh * electricity_usd_kwh
    cost_elec_si  = energy_si_kwh  * electricity_usd_kwh
    elec_saving   = cost_elec_si - cost_elec_sic

    # 初期投資差 (SiC は Si の約 2.5× 高い)
    n_modules     = math.ceil(cluster_power_mw * 1000 / SIC_MODULE["rating_kW"])
    capex_sic     = n_modules * SIC_MODULE["price_usd"]
    capex_si      = n_modules * SIC_MODULE["price_usd"] / 2.5
    capex_diff    = capex_sic - capex_si

    # 冷却コスト削減 (電力損失が減る → 冷却設備も小さくなる)
    loss_reduction_mw = cluster_power_mw * (1/efficiency_si - 1/efficiency_sic)
    cooling_saving    = loss_reduction_mw * 1000 * hours_per_year * years * electricity_usd_kwh * 0.4

    net_saving = elec_saving + cooling_saving - capex_diff
    payback_yr = capex_diff / ((elec_saving + cooling_saving) / years)

    return {
        "cluster_MW":          cluster_power_mw,
        "elec_saving_M$":      round(elec_saving / 1e6, 2),
        "cooling_saving_M$":   round(cooling_saving / 1e6, 2),
        "capex_premium_M$":    round(capex_diff / 1e6, 2),
        "net_saving_M$":       round(net_saving / 1e6, 2),
        "payback_years":       round(payback_yr, 2),
        "efficiency_sic_pct":  efficiency_sic * 100,
        "efficiency_si_pct":   efficiency_si  * 100,
        "loss_reduction_MW":   round(loss_reduction_mw, 2),
        "co2_reduction_tpa":   round(loss_reduction_mw * 8760 * 0.4 / 1000, 0),  # tCO2/yr @ 0.4kg/kWh
    }


# ════════════════════════════════════════════════════════════════════════════
# 4. AI チップ → SiC 製造プロセス接続
# ════════════════════════════════════════════════════════════════════════════
AI_PROCESS_NODE = {
    "H100_SXM5":  "4nm (TSMC N4)",
    "B200_SXM":   "4nm (TSMC N4P)",
    "B300_SXM":   "3nm (TSMC N3E)",
    "TPU_v6e":    "5nm (TSMC N5)",
    "MI300X":     "5nm (TSMC N5)",
    "Trainium3":  "3nm (TSMC N3)",
    "Tesla_AI5":  "2nm (TSMC N2, 2026)",
    "Gaudi3":     "5nm (TSMC N5)",
}

# AI チップ die サイズ [mm²]
AI_DIE_AREA_MM2 = {
    "H100_SXM5":  814,
    "B200_SXM":   904,
    "B300_SXM":   950,
    "TPU_v6e":    595,
    "MI300X":     696,   # per chiplet ×3
    "Trainium3":  600,
    "Tesla_AI5":  450,
    "Gaudi3":     660,
}


def ai_chip_yield_and_wafer_demand(chip: str = "H100_SXM5",
                                    n_chips_needed: int = 100_000) -> dict:
    """
    AI チップの収率モデル + 必要ウェーハ枚数。
    Murphy の収率モデル: Y = ((1 - exp(-D0·A)) / (D0·A))²
    """
    D0 = {
        "4nm (TSMC N4)":   0.15,  # defects/cm²
        "4nm (TSMC N4P)":  0.12,
        "3nm (TSMC N3E)":  0.18,
        "5nm (TSMC N5)":   0.10,
        "2nm (TSMC N2, 2026)": 0.25,  # 新プロセス初期
        "3nm (TSMC N3)":   0.16,
        "5nm (TSMC N6)":   0.09,
    }
    node     = AI_PROCESS_NODE.get(chip, "4nm (TSMC N4)")
    d0       = D0.get(node, 0.15)
    area_cm2 = AI_DIE_AREA_MM2.get(chip, 700) / 100.0

    # Murphy yield
    x = d0 * area_cm2
    Y = ((1 - math.exp(-x)) / x) ** 2

    # 300mm ウェーハ 1枚あたりのダイ数 (近似)
    wafer_area_cm2 = math.pi * 15.0 ** 2
    edge_loss      = math.pi * 15.0 * 2.0 / area_cm2  # 端効果
    dies_per_wafer = max(1, int((wafer_area_cm2 / area_cm2) - edge_loss))
    good_per_wafer = int(dies_per_wafer * Y)

    wafers_needed  = math.ceil(n_chips_needed / max(good_per_wafer, 1))

    return {
        "chip":             chip,
        "node":             node,
        "die_area_mm2":     AI_DIE_AREA_MM2.get(chip, 700),
        "D0_defects_cm2":   d0,
        "murphy_yield_pct": round(Y * 100, 1),
        "dies_per_wafer":   dies_per_wafer,
        "good_per_wafer":   good_per_wafer,
        "wafers_needed":    wafers_needed,
        "n_chips_needed":   n_chips_needed,
    }


# ════════════════════════════════════════════════════════════════════════════
# 5. AI 電力需要 マクロ予測 (IEA 2024 ベース)
# ════════════════════════════════════════════════════════════════════════════
AI_POWER_FORECAST_TWH = {
    2023: 460,    # 全データセンター消費電力 [TWh/年]
    2024: 600,
    2025: 800,
    2026: 1050,
    2027: 1300,
    2028: 1600,
    2030: 2300,
}

# AI 向け比率 (%) と SiC 普及率 (%)
AI_FRACTION_PCT = {2023: 10, 2024: 18, 2025: 28, 2026: 38, 2027: 48, 2028: 58, 2030: 70}
SIC_ADOPTION_PSU_PCT = {2023: 15, 2024: 22, 2025: 35, 2026: 50, 2027: 65, 2028: 78, 2030: 90}


def ai_power_and_sic_market_forecast() -> list[dict]:
    """
    AI データセンター電力需要 × SiC 市場規模の中期予測。
    """
    results = []
    for year in sorted(AI_POWER_FORECAST_TWH):
        total_twh = AI_POWER_FORECAST_TWH[year]
        ai_frac   = AI_FRACTION_PCT.get(year, 50) / 100
        sic_frac  = SIC_ADOPTION_PSU_PCT.get(year, 50) / 100

        ai_twh    = total_twh * ai_frac
        # SiC 市場規模: 電力 × SiC 普及率 × モジュール単価密度 ($15M/MW 相当)
        sic_market_b = ai_twh * 1e9 / 8760 * sic_frac * 15 / 1e9  # $B

        results.append({
            "year":             year,
            "dc_total_TWh":     total_twh,
            "ai_dc_TWh":        round(ai_twh, 0),
            "ai_avg_power_GW":  round(ai_twh * 1e9 / 8760 / 1e9, 1),
            "sic_adoption_pct": round(sic_frac * 100, 0),
            "sic_market_B$":    round(sic_market_b, 2),
        })
    return results


# ════════════════════════════════════════════════════════════════════════════
# Quick demo
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== AI Data Center × SiC Power Model ===\n")

    sv = server_power("H100_SXM5")
    print(f"H100 サーバー (8GPU): IT={sv['p_server_it_W']/1000:.1f}kW, "
          f"入力={sv['p_input_W']/1000:.1f}kW, 損失={sv['p_loss_W']:.0f}W")

    cl = cluster_power("Microsoft_Azure", "H100_SXM5")
    print(f"\nMicrosoft Azure クラスター ({cl['n_gpu']:,} GPU):")
    print(f"  IT 電力: {cl['it_power_MW']:.0f} MW")
    print(f"  総電力:  {cl['total_power_MW']:.0f} MW")
    print(f"  SiC モジュール: {cl['sic_modules']:,} 個 (${cl['sic_cost_M_usd']:.0f}M)")

    wd = sic_wafer_demand_from_ai()
    print(f"\n全 AI クラスター SiC ウェーハ需要:")
    print(f"  合計: {wd['wafers_per_day']:,.0f} 枚/日")
    print(f"  EV 向け: {wd['ev_wafers_per_day']:,} 枚/日")
    print(f"  AI 比率: {wd['ai_fraction_pct']:.1f}%")

    tco = sic_vs_si_tco(100, years=5)
    print(f"\n100MW クラスター SiC vs Si 5年 TCO:")
    print(f"  電気代節約: ${tco['elec_saving_M$']:.1f}M")
    print(f"  冷却節約:   ${tco['cooling_saving_M$']:.1f}M")
    print(f"  初期コスト増: ${tco['capex_premium_M$']:.1f}M")
    print(f"  回収期間:   {tco['payback_years']:.1f}年")
    print(f"  CO2 削減:   {tco['co2_reduction_tpa']:,.0f} t/年")

    yld = ai_chip_yield_and_wafer_demand("B200_SXM", 50_000)
    print(f"\nB200 ({yld['die_area_mm2']}mm², {yld['node']}):")
    print(f"  収率: {yld['murphy_yield_pct']:.1f}%")
    print(f"  良品/ウェーハ: {yld['good_per_wafer']}")
    print(f"  50k チップに必要なウェーハ: {yld['wafers_needed']:,} 枚")
