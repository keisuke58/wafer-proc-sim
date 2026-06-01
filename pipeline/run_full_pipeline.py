"""
完全統合パイプライン実行スクリプト
====================================
ASML → Lasertec → TEL CMP → Disco → Lasertec 後検査
→ TEL ALD → Advantest ATE → Socionext SoC → NVIDIA GPU

1コマンドで全モジュールを順次実行し、
各ステップの結果を次ステップへ受け渡す。

実行:
    python pipeline/run_full_pipeline.py
    python pipeline/run_full_pipeline.py --process stealth
    python pipeline/run_full_pipeline.py --process blade --plots
"""

import argparse
import os
import sys
import time

import numpy as np

# パス設定
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ─── カラー出力ヘルパー ──────────────────────────────────────────────────
BOLD  = "\033[1m";  RESET = "\033[0m"
GREEN = "\033[92m"; CYAN  = "\033[96m"
YELLOW= "\033[93m"; RED   = "\033[91m"

def step(n, title):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  Step {n}: {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def info(msg): print(f"  {YELLOW}→{RESET} {msg}")
def warn(msg): print(f"  {RED}⚠{RESET}  {msg}")


def run(dicing_process: str = "stealth", make_plots: bool = False) -> dict:
    t0 = time.time()
    results = {"process": dicing_process}

    # ════════════════════════════════════════════════════════════════════
    step(1, "ASML EUV リソグラフィー (2nm ノード)")
    # ════════════════════════════════════════════════════════════════════
    from fem.asml_model import aerial_image, process_window, cd_to_vth_sigma

    node_nm   = 2.0
    system    = "EXE:5000 (High-NA)"
    ai = aerial_image(pitch_nm=4.0, CD_nm=node_nm, NA=0.55,
                      wavelength_nm=13.5, sigma=0.90)
    pw = process_window(system, CD_target_nm=node_nm)

    sigma_Vth = cd_to_vth_sigma(pw["sigma_CD_nm"], node_nm)
    ok(f"k1={ai['k1']:.3f}  NILS={ai['NILS']:.2f}  "
       f"DOF={pw['DOF_nm']:.0f}nm  EL={pw['EL_pct']}%")
    ok(f"σ_CD={pw['sigma_CD_nm']:.2f}nm → σ_Vth={sigma_Vth:.1f}mV")
    results.update({"k1": ai["k1"], "sigma_Vth_mV": sigma_Vth,
                    "DOF_nm": pw["DOF_nm"]})

    # ════════════════════════════════════════════════════════════════════
    step(2, "Lasertec ACTIS — EUV マスク検査")
    # ════════════════════════════════════════════════════════════════════
    from fem.tel_cmp_lasertec import euv_mask_defect_density, euv_inspection_roi

    euv = euv_mask_defect_density(node_nm, fab="TSMC")
    roi = euv_inspection_roi(node_nm)
    ok(f"欠陥密度={euv['defect_density']:.4f}/cm²  "
       f"歩留まり損失={euv['yield_loss_pct']:.3f}%")
    ok(f"ACTIS ROI={roi['ROI']:.0f}×  ($検査 ${roi['actis_cost_k']:.0f}k "
       f"vs リスク ${roi['cost_per_defect_M']:.0f}M)")
    results.update({"euv_yield_loss_pct": euv["yield_loss_pct"],
                    "actis_roi": roi["ROI"]})

    # ════════════════════════════════════════════════════════════════════
    step(3, "TEL CMP — SiC 表面平坦化")
    # ════════════════════════════════════════════════════════════════════
    from fem.tel_cmp_lasertec import cmp_time, preston_mrr

    cmp = cmp_time(target_depth_nm=50, pressure_kPa=40,
                   velocity_m_s=0.5, material="4H-SiC")
    ok(f"MRR={cmp['MRR_nm_min']:.2f} nm/min  "
       f"研磨時間={cmp['time_min']:.1f} min  "
       f"Si比={cmp['cost_factor_vs_Si']}× コスト")
    ok(f"Post-CMP Ra={cmp['Ra_after_nm']:.2f} nm")
    results["cmp_time_min"] = cmp["time_min"]
    results["Ra_precmp_nm"] = cmp["Ra_after_nm"]

    # ════════════════════════════════════════════════════════════════════
    step(4, f"Disco ダイシング ({dicing_process})")
    # ════════════════════════════════════════════════════════════════════
    from ml.quantum_defect_model import (process_induced_trap_density,
                                          shockley_read_hall_lifetime)

    HAZ_map = {"blade":2.5,"laser_ns":2.0,"laser_ps":0.5,
               "laser_fs":0.2,"stealth":0.05,"plasma":0.8}
    Ra_map  = {"blade":80,"laser_ns":60,"laser_ps":30,
               "laser_fs":15,"stealth":5,"plasma":20}
    HAZ = HAZ_map.get(dicing_process, 2.5)
    Ra  = Ra_map.get(dicing_process, 80)

    traps = process_induced_trap_density(HAZ, Ra, dicing_process)
    tau   = shockley_read_hall_lifetime(traps["V_C (Z1/2)"])
    ok(f"HAZ={HAZ}µm  Ra={Ra}nm")
    ok(f"V_C 密度={traps['V_C (Z1/2)']:.2e} cm⁻³  "
       f"キャリア寿命={tau:.1f}µs")
    results.update({"HAZ_um": HAZ, "Ra_nm": Ra,
                    "N_VC": traps["V_C (Z1/2)"], "tau_us": tau})

    # ════════════════════════════════════════════════════════════════════
    step(5, "Lasertec SiC 後検査")
    # ════════════════════════════════════════════════════════════════════
    from fem.tel_cmp_lasertec import sic_postdice_inspection

    sic_insp = sic_postdice_inspection(dicing_process)
    grade_ok = sic_insp["grade"] in ["A", "B"]
    (ok if grade_ok else warn)(
        f"Grade {sic_insp['grade']}  "
        f"Ra={sic_insp['Ra_nm']:.1f}nm  "
        f"チッピング={sic_insp['chip_mean']:.3f}µm")
    info(sic_insp["feedback"])
    results.update({"inspection_grade": sic_insp["grade"],
                    "chip_mean_um": sic_insp["chip_mean"]})
    if not grade_ok:
        warn("Disco にプロセス改善フィードバック送信")

    # ════════════════════════════════════════════════════════════════════
    step(6, "TEL ALD + 熱酸化 → SiC MOSFET 製造")
    # ════════════════════════════════════════════════════════════════════
    from fem.tel_process_model import full_pipeline as tel_full

    tel = tel_full(dicing_process=dicing_process, ald_material="HfO2",
                   ald_cycles=50, ox_T_C=1150, ox_t_hr=2.0, anneal_T_C=950)
    ok(f"µ_ch={tel['mu_ch']:.1f} cm²/Vs  "
       f"EOT={tel['total_EOT_nm']:.2f}nm  "
       f"R_on={tel['R_on_mohm_mm2']:.3f} mΩ·mm²")
    results.update({"mu_ch": tel["mu_ch"], "EOT_nm": tel["total_EOT_nm"],
                    "R_on": tel["R_on_mohm_mm2"], "Dit": tel["Dit_cm2eV"]})

    # ════════════════════════════════════════════════════════════════════
    step(7, "Advantest ATE テスト → 歩留まり")
    # ════════════════════════════════════════════════════════════════════
    from fem.advantest_model import (simulate_device_params, run_ate_test,
                                      wafer_map, test_economics)

    wm   = wafer_map(tel["mu_ch"], tel["Dit_cm2eV"])
    econ = test_economics(wm["yield_pct"], wm["n_die"])
    n_good = int(wm["n_die"] * wm["yield_pct"] / 100)
    ok(f"ATE 歩留まり={wm['yield_pct']:.2f}%  "
       f"良品数={n_good}  "
       f"CPGD=${econ['cpgd_full_usd']:.2f}")
    ok(f"テストコスト比={econ['test_cost_pct']:.1f}%")
    results.update({"yield_pct": wm["yield_pct"], "n_good": n_good,
                    "cpgd_usd": econ["cpgd_full_usd"],
                    "test_cost_pct": econ["test_cost_pct"]})

    # ════════════════════════════════════════════════════════════════════
    step(8, "ソシオネクスト SoC タイミング解析")
    # ════════════════════════════════════════════════════════════════════
    from fem.semiconductor_ecosystem import socionext_timing_yield

    soc_yield = socionext_timing_yield("SC2300", sigma_Vth_mV=sigma_Vth)
    ok(f"SC2300 ADAS SoC  node={soc_yield['node_nm']}nm  "
       f"street={soc_yield['street_um']}µm")
    ok(f"タイミング歩留まり={soc_yield['timing_yield_pct']:.2f}%  "
       f"z_slack={soc_yield['z_slack']:.2f}")
    results.update({"soc_timing_yield": soc_yield["timing_yield_pct"],
                    "z_slack": soc_yield["z_slack"]})

    # ════════════════════════════════════════════════════════════════════
    step(9, "NVIDIA GPU サプライチェーン影響")
    # ════════════════════════════════════════════════════════════════════
    from fem.semiconductor_ecosystem import nvidia_gpu_bom

    gpu = nvidia_gpu_bom("Blackwell B200")
    ok(f"B200 GPU装置コスト: Disco ${gpu['equipment_usd']['Disco']}/GPU  "
       f"TEL ${gpu['equipment_usd']['TEL']}/GPU")
    ok(f"装置コスト計 ${gpu['total_equip_usd']}/GPU = "
       f"{gpu['equip_pct_bom']:.1f}% of BOM")
    results.update({"gpu_disco_cost": gpu["equipment_usd"]["Disco"],
                    "gpu_equip_pct": gpu["equip_pct_bom"]})

    # ════════════════════════════════════════════════════════════════════
    step(10, "Lam Research プラズマエッチ — CD → プロファイル → ALD ライナー")
    # ════════════════════════════════════════════════════════════════════
    from fem.lam_research_model import full_pipeline as lam_full

    lam = lam_full(feature_nm=node_nm * 7.0, aspect_ratio=12.0,
                   material="Si", precursor="TMA")
    ok(f"ER={lam['ER_nm_min']:.0f} nm/min  "
       f"ARDE 抑制={lam['ard_suppression_pct']:.1f}%  "
       f"CD バイアス={lam['CD_bias_nm']:.2f} nm")
    ok(f"ALD GPC={lam['ALD_GPC_A_cycle']:.3f} Å/cycle  "
       f"ライナー 2nm = {lam['liner_cycles_for_2nm']} cycles")
    results.update({"lam_ER_nm_min": lam["ER_nm_min"],
                    "lam_CD_bias_nm": lam["CD_bias_nm"],
                    "lam_ALD_GPC": lam["ALD_GPC_A_cycle"]})

    # ════════════════════════════════════════════════════════════════════
    step(11, "AMAT CMP + イオン注入 → 活性化")
    # ════════════════════════════════════════════════════════════════════
    from fem.amat_model import full_pipeline as amat_full

    amat = amat_full(node_nm=node_nm, mat_stack="SiO2")
    ok(f"PECVD dep={amat['pecvd_dep_rate_nm_min']:.0f} nm/min  "
       f"CMP MRR={amat['cmp_mrr_nm_min']:.0f} nm/min  "
       f"ディッシング={amat['dishing_nm']:.1f} nm")
    ok(f"注入 Rp={amat['implant_Rp_nm']:.1f} nm  "
       f"活性化率={amat['activation_fraction']*100:.2f}%")
    results.update({"amat_cmp_mrr": amat["cmp_mrr_nm_min"],
                    "amat_dishing_nm": amat["dishing_nm"],
                    "amat_activation": amat["activation_fraction"]})

    # ════════════════════════════════════════════════════════════════════
    step(12, "Samsung / SK Hynix HBM → NVIDIA GPU ルーフライン")
    # ════════════════════════════════════════════════════════════════════
    from fem.sk_hynix_model import hbm3e_bandwidth, hbm_stack_height
    from fem.samsung_process_model import hbm3_tsv_yield
    from fem.nvidia_gpu_model import ai_roofline, GPU_SPECS

    hbm_bw  = hbm3e_bandwidth("HBM3E_12Hi", n_stacks=6)
    hbm_ht  = hbm_stack_height("HBM3E_12Hi")
    tsv_yld = hbm3_tsv_yield("HBM3E_12Hi")
    roofline= ai_roofline(tflops=GPU_SPECS["H100_SXM"]["tflops_fp16"],
                           hbm_bw_GBs=hbm_bw["BW_GBs_per_stack"] * 6,
                           arithmetic_intensity=250.0)
    ok(f"HBM3E 6×スタック BW={hbm_bw['BW_GBs_per_stack']*6:.0f} GB/s "
       f"= {hbm_bw['BW_TBs']*6:.2f} TB/s  "
       f"スタック高={hbm_ht['total_height_um']:.0f} µm")
    ok(f"TSV 歩留まり={tsv_yld['tsv_yield_pct']:.2f}%  "
       f"ルーフライン AI=250 FLOP/B → {roofline['attainable_tflops']:.0f} TFLOPS "
       f"({roofline['bound']}-bound)")
    results.update({"hbm_BW_TBs": hbm_bw["BW_TBs"] * 6,
                    "tsv_yield_pct": tsv_yld["tsv_yield_pct"],
                    "roofline_tflops": roofline["attainable_tflops"],
                    "roofline_bound": roofline["bound"]})

    # ════════════════════════════════════════════════════════════════════
    step(13, "Terafab — 垂直統合メガFab 生産目標 + 日本装置波及")
    # ════════════════════════════════════════════════════════════════════
    from fem.terafab_model import production_target, yield_chain, japan_impact_analysis

    tf_prod  = production_target()
    tf_chain = yield_chain()
    tf_final_yield = list(tf_chain.values())[-1]["cumulative"]
    tf_imp   = japan_impact_analysis()
    ok(f"CapEx $119B → ウェーハ: {tf_prod['wafers_per_day']:,} 枚/日  "
       f"(TSMC比 {tf_prod['wafers_per_day']/60000*100:.0f}%)")
    ok(f"宇宙向け: {tf_prod['space']['n_chips']/1e6:.0f}M 枚  "
       f"地上向け: {tf_prod['ground']['n_chips']/1e6:.0f}M 枚  "
       f"合算: {tf_prod['total_tflops_B']:.0f} B-TFLOPS")
    ok(f"垂直統合 総歩留まり: {tf_final_yield*100:.1f}%  "
       f"日本装置需要: ${tf_imp['total_demand_B']:.0f}B")
    results.update({
        "terafab_chips_needed": tf_prod["n_chips_needed"],
        "terafab_wpd":          tf_prod["wafers_per_day"],
        "terafab_total_yield":  tf_final_yield,
        "terafab_japan_B":      tf_imp["total_demand_B"],
    })

    # ════════════════════════════════════════════════════════════════════
    step(14, "量子コンピュータ — Si Qubit + Surface Code QEC")
    # ════════════════════════════════════════════════════════════════════
    from fem.quantum_computing_model import (si_spin_qubit_coherence,
                                              transmon_qubit,
                                              surface_code_overhead,
                                              si_qubit_wafer_integration)

    si_q  = si_spin_qubit_coherence(rms_nm=0.1, n_imp_cm2=1e10, B_mT=1.0)
    tm_q  = transmon_qubit(EJ_GHz=25.0, EC_GHz=0.25)
    qec   = surface_code_overhead(p_phys=0.001)
    integ = si_qubit_wafer_integration(qubit_pitch_um=50.0)
    mc    = qec.get("min_config", {})
    ok(f"Si spin qubit: T1={si_q['T1_us']:.0f}µs  T2={si_q['T2_us']:.0f}µs  "
       f"f_q={si_q['f_q_GHz']:.3f} GHz")
    ok(f"Transmon: f01={tm_q['f_01_GHz']:.2f} GHz  "
       f"|α|={tm_q['alpha_MHz']:.0f} MHz  T1={tm_q['T1_us']:.0f}µs")
    if mc:
        ok(f"Surface Code (p=0.1%): d={mc['d']}  "
           f"物理/論理={mc['n_phys']} qubit  p_L={mc['p_L']:.2e}")
    ok(f"300mm ウェーハ Si qubit (50µm ピッチ): {integ['n_qubits_yield']:,} qubit")
    results.update({
        "si_qubit_T1_us":   si_q["T1_us"],
        "si_qubit_T2_us":   si_q["T2_us"],
        "transmon_f01_GHz": tm_q["f_01_GHz"],
        "qec_d":            mc.get("d", 0),
        "qec_n_phys":       mc.get("n_phys", 0),
        "wafer_qubits":     integ["n_qubits_yield"],
    })

    # ════════════════════════════════════════════════════════════════════
    step(15, "Hyperscaler + Tesla — カスタムシリコン競争 & ウェーハ需要")
    # ════════════════════════════════════════════════════════════════════
    from fem.hyperscaler_model import (ASIC_SPECS, HYPERSCALER_CAPEX,
                                        wafer_demand, tco_analysis,
                                        tesla_fsd_inference_demand)

    total_capex_B = sum(v["capex_B"] for v in HYPERSCALER_CAPEX.values())
    total_wpd     = sum(wafer_demand(c, v)["wafers_per_day"]
                        for c, v in HYPERSCALER_CAPEX.items())
    # Tesla AI5 最高効率
    ai5_eff = (ASIC_SPECS["Tesla AI5"]["tflops_bf16"] /
               ASIC_SPECS["Tesla AI5"]["tdp_W"])
    tpu_eff = (ASIC_SPECS["Google TPU v6e\n(Trillium)"]["tflops_bf16"] /
               ASIC_SPECS["Google TPU v6e\n(Trillium)"]["tdp_W"])
    # Tesla FSD 需要
    tesla_dem = tesla_fsd_inference_demand()
    ok(f"Hyperscaler 総 CapEx: ${total_capex_B:.0f}B  "
       f"TSMC 推定需要: {total_wpd:,} ウェーハ/日")
    ok(f"Tesla AI5: {ai5_eff:.1f} TFlops/W  "
       f"Google TPU v6e: {tpu_eff:.1f} TFlops/W  "
       f"(NVIDIA H100: 1.4 TFlops/W)")
    ok(f"Tesla AI5 年間需要: {tesla_dem['total_chips']:,} 枚 → "
       f"{tesla_dem['wafers_needed']:,} 枚 2nm ウェーハ (${tesla_dem['wafer_cost_B']:.1f}B)")
    results.update({
        "hyperscaler_capex_B":   total_capex_B,
        "hyperscaler_tsmc_wpd":  total_wpd,
        "tesla_ai5_eff":         ai5_eff,
        "tesla_ai5_wafers":      tesla_dem["wafers_needed"],
        "tesla_fsd_chips":       tesla_dem["n_chips_vehicle"],
    })

    # ── Step 16: AI DC × SiC ─────────────────────────────────────────
    step(16, "AI DC × SiC パワーモデル")
    from fem.ai_datacenter_model import (
        server_power, sic_wafer_demand_from_ai, sic_vs_si_tco,
        ai_power_and_sic_market_forecast,
    )
    sv_h100 = server_power("H100_SXM5")
    ai_wd   = sic_wafer_demand_from_ai()
    tco_100 = sic_vs_si_tco(100.0, years=5)
    fc_2030 = next(r for r in ai_power_and_sic_market_forecast() if r["year"] == 2030)
    ok(f"H100 サーバー IT={sv_h100['p_server_it_W']/1000:.1f}kW  "
       f"PSU損失={sv_h100['p_loss_W']:.0f}W  "
       f"AI SiC需要={ai_wd['wafers_per_day']:.0f} wafers/day")
    ok(f"100MW DC SiC vs Si: 回収{tco_100['payback_years']:.1f}年  "
       f"CO2削減{tco_100['co2_reduction_tpa']:,.0f}t/年")
    ok(f"2030 AI DC予測: {fc_2030['ai_avg_power_GW']}GW  "
       f"SiC市場 ${fc_2030['sic_market_B$']:.1f}B")
    results.update({
        "ai_server_it_kw":     sv_h100["p_server_it_W"] / 1000,
        "ai_sic_wpd":          ai_wd["wafers_per_day"],
        "ai_dc_tco_payback":   tco_100["payback_years"],
        "ai_sic_2030_B":       fc_2030["sic_market_B$"],
    })

    # ── Step 17: EV × SiC ─────────────────────────────────────────────
    step(17, "EV × SiC パワーエレクトロニクス")
    from fem.ev_sic_model import switching_loss, thermal_model, ev_sic_demand
    r_sic = switching_loss("SiC_MOSFET_1200V", "BEV_800V")
    r_si  = switching_loss("Si_IGBT_1200V",    "BEV_800V")
    th    = thermal_model(r_sic["P_total_W"])
    dem   = ev_sic_demand()
    ok(f"BEV 800V インバータ: SiC={r_sic['P_total_W']:.0f}W  "
       f"Si={r_si['P_total_W']:.0f}W  "
       f"Tj(SiC)={th['Tj_C']:.0f}°C ({'Safe' if th['safe'] else 'Over'})")
    ok(f"EV SiC 需要: {dem['total_chips_M']:.0f}M チップ/年  "
       f"→ {dem['wafers_per_day']:.0f} wafers/day")
    results.update({
        "ev_sic_loss_W":    r_sic["P_total_W"],
        "ev_si_loss_W":     r_si["P_total_W"],
        "ev_Tj_C":          th["Tj_C"],
        "ev_sic_wpd":       dem["wafers_per_day"],
    })

    # ════════════════════════════════════════════════════════════════════
    # サマリー
    # ════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  完全パイプライン結果 — プロセス: {dicing_process}{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")

    summary = [
        ("ASML k1",              f"{results['k1']:.3f}"),
        ("σ_Vth (ASML→回路)",   f"{results['sigma_Vth_mV']:.1f} mV"),
        ("CMP 研磨時間",          f"{results['cmp_time_min']:.1f} min"),
        ("Disco HAZ",             f"{results['HAZ_um']:.2f} µm"),
        ("キャリア寿命",           f"{results['tau_us']:.1f} µs"),
        ("Lasertec Grade",        results["inspection_grade"]),
        ("チャンネル移動度",       f"{results['mu_ch']:.1f} cm²/Vs"),
        ("ATE 歩留まり",          f"{results['yield_pct']:.2f} %"),
        ("CPGD",                 f"$ {results['cpgd_usd']:.2f}"),
        ("SoC タイミング yield",  f"{results['soc_timing_yield']:.2f} %"),
        ("GPU 装置コスト比",      f"{results['gpu_equip_pct']:.1f} %"),
        ("Lam CD バイアス",        f"{results.get('lam_CD_bias_nm', 0):.2f} nm  ER={results.get('lam_ER_nm_min', 0):.0f} nm/min"),
        ("AMAT 活性化率",         f"{results.get('amat_activation', 0)*100:.2f} %"),
        ("HBM3E BW (6スタック)", f"{results.get('hbm_BW_TBs', 0):.2f} TB/s"),
        ("GPU ルーフライン",      f"{results.get('roofline_tflops', 0):.0f} TFLOPS ({results.get('roofline_bound', '-')})"),
        ("Terafab 必要ウェーハ/日", f"{results.get('terafab_wpd', 0):,} 枚"),
        ("Terafab 総歩留まり",    f"{results.get('terafab_total_yield', 0)*100:.1f} %"),
        ("日本装置 波及需要",     f"${results.get('terafab_japan_B', 0):.0f}B"),
        ("Si Qubit T2",          f"{results.get('si_qubit_T2_us', 0):.0f} µs"),
        ("QEC 物理/論理 qubit",  f"{results.get('qec_n_phys', 0)} (d={results.get('qec_d', 0)})"),
        ("300mm ウェーハ Qubit", f"{results.get('wafer_qubits', 0):,} qubit"),
        ("Hyperscaler CapEx",    f"${results.get('hyperscaler_capex_B', 0):.0f}B"),
        ("TSMC 推定需要",         f"{results.get('hyperscaler_tsmc_wpd', 0):,} ウェーハ/日"),
        ("Tesla AI5 効率",        f"{results.get('tesla_ai5_eff', 0):.1f} TFlops/W"),
        ("Tesla FSD 年間チップ",  f"{results.get('tesla_fsd_chips', 0):,} 枚"),
        ("AI DC SiC ウェーハ/日", f"{results.get('ai_sic_wpd', 0):.0f} 枚"),
        ("AI DC SiC 回収期間",   f"{results.get('ai_dc_tco_payback', 0):.1f} 年"),
        ("EV SiC 損失",          f"{results.get('ev_sic_loss_W', 0):.0f} W (Tj={results.get('ev_Tj_C', 0):.0f}°C)"),
        ("EV SiC ウェーハ/日",   f"{results.get('ev_sic_wpd', 0):.0f} 枚"),
    ]
    for k, v in summary:
        print(f"  {k:<24} {BOLD}{v}{RESET}")

    print(f"\n  {GREEN}実行時間: {elapsed:.1f}s{RESET}")
    return results


def compare_all(make_plots: bool = False):
    """全プロセスを比較実行。"""
    processes = ["blade", "laser_ps", "laser_fs", "stealth"]
    all_results = {}
    for proc in processes:
        print(f"\n{'#'*60}")
        print(f"# プロセス: {proc}")
        print(f"{'#'*60}")
        all_results[proc] = run(proc, make_plots=False)

    # 比較表
    print(f"\n{BOLD}{'═'*80}{RESET}")
    print(f"{BOLD}  プロセス比較サマリー{RESET}")
    print(f"{BOLD}{'═'*80}{RESET}")
    print(f"  {'指標':<24} {'ブレード':>12} {'レーザーps':>12} "
          f"{'レーザーfs':>12} {'ステルス':>12}")
    print(f"  {'─'*72}")

    metrics = [
        ("HAZ [µm]",        "HAZ_um",          ".2f"),
        ("寿命 [µs]",       "tau_us",           ".1f"),
        ("Grade",           "inspection_grade", "s"),
        ("µ_ch [cm²/Vs]",  "mu_ch",            ".1f"),
        ("Yield [%]",       "yield_pct",        ".2f"),
        ("CPGD [$]",        "cpgd_usd",         ".2f"),
        ("SoC yield [%]",   "soc_timing_yield", ".2f"),
    ]
    for label, key, fmt in metrics:
        vals = [all_results[p][key] for p in processes]
        row = f"  {label:<24}"
        for v in vals:
            if fmt == "s":
                row += f" {str(v):>12}"
            else:
                row += f" {v:{'>12' + fmt}}"
        print(row)

    if make_plots:
        _plot_comparison(all_results, processes)

    return all_results


def _plot_comparison(all_results: dict, processes: list):
    """比較バーチャートを生成。"""
    import matplotlib.pyplot as plt
    import os

    labels = {"blade":"ブレード","laser_ps":"レーザーps",
              "laser_fs":"レーザーfs","stealth":"ステルス"}
    colors = {"blade":"#d62728","laser_ps":"#ff7f0e",
              "laser_fs":"#2ca02c","stealth":"#2166ac"}

    metrics = [
        ("HAZ_um",          "HAZ [µm]",         False),
        ("tau_us",          "寿命 τ [µs]",       True),
        ("mu_ch",           "µ_ch [cm²/Vs]",    False),
        ("yield_pct",       "Yield [%]",         False),
        ("cpgd_usd",        "CPGD [$]",          False),
        ("soc_timing_yield","SoC Timing Yield%", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (key, ylabel, log_scale) in zip(axes.flat, metrics):
        vals = [all_results[p][key] for p in processes]
        bars = ax.bar([labels[p] for p in processes], vals,
                      color=[colors[p] for p in processes],
                      alpha=0.85, edgecolor="k", lw=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    v + max(vals)*0.01, f"{v:.2f}",
                    ha="center", fontsize=8, fontweight="bold")
        if log_scale:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11)
        ax.grid(alpha=0.2, axis="y")

    fig.suptitle("完全パイプライン比較\n"
                 "ASML→Lasertec→TEL CMP→Disco→TEL ALD→Advantest→ソシオネクスト",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(ROOT, "results", "full_pipeline_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[✓] 比較チャート -> {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="半導体製造 完全パイプライン実行")
    ap.add_argument("--process", default="stealth",
                    choices=["blade","laser_ns","laser_ps","laser_fs","stealth","plasma"],
                    help="ダイシングプロセス")
    ap.add_argument("--all",    action="store_true",
                    help="全プロセスを比較実行")
    ap.add_argument("--plots",  action="store_true",
                    help="比較チャートを生成")
    args = ap.parse_args()

    if args.all:
        compare_all(make_plots=args.plots)
    else:
        run(args.process, make_plots=args.plots)


if __name__ == "__main__":
    main()
