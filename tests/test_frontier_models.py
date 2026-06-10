"""
フロンティア モデル ユニットテスト
=====================================
terafab_model / quantum_computing_model / hyperscaler_model の
物理的整合性・数値範囲・モノトニシティを検証。

参考文献:
    Musk (2026) Terafab announcement
    Veldhorst et al. (2014) Nature Nanotechnology — Si spin qubit T1/T2
    Takeda et al. (2022) Nature Electronics — ²⁸Si spin qubit ms-range T1
    Koch et al. (2007) PRA — Transmon qubit design
    Fowler et al. (2012) PRA — Surface code threshold 1%
"""
import math
import pytest
import numpy as np


# ════════════════════════════════════════════════════════════════════════════
# Terafab Model
# ════════════════════════════════════════════════════════════════════════════

class TestTerafabProduction:
    from market_analysis.terafab_model import production_target, yield_chain, capex_analysis

    def test_production_target_1TW(self):
        """1 TW → 少なくとも 1億枚以上のチップが必要。"""
        from market_analysis.terafab_model import production_target
        r = production_target(target_TW=1.0)
        assert r["n_chips_needed"] > 1e8, \
            f"1 TW needs > 100M chips, got {r['n_chips_needed']:.0f}"

    def test_production_target_scales_linearly(self):
        """目標 TW を 2倍にすると必要チップ数も 2倍になる。"""
        from market_analysis.terafab_model import production_target
        r1 = production_target(target_TW=0.5)
        r2 = production_target(target_TW=1.0)
        ratio = r2["n_chips_needed"] / r1["n_chips_needed"]
        assert 1.9 <= ratio <= 2.1, f"2x target should give ~2x chips, got {ratio:.2f}"

    def test_die_yield_between_0_and_1(self):
        """Murphy 歩留まりは 0〜1 の範囲内。"""
        from market_analysis.terafab_model import production_target
        r = production_target()
        assert 0.0 < r["die_yield"] < 1.0, f"Yield out of range: {r['die_yield']}"

    def test_yield_chain_cumulative_monotone_decreasing(self):
        """各ステップを経るごとに累積歩留まりは単調減少する。"""
        from market_analysis.terafab_model import yield_chain
        chain = yield_chain()
        values = [v["cumulative"] for v in chain.values()]
        for i in range(len(values) - 1):
            assert values[i] >= values[i+1], \
                f"Cumulative yield not monotone at step {i}: {values[i]:.4f} < {values[i+1]:.4f}"

    def test_final_yield_above_50pct(self):
        """垂直統合でも総歩留まりは 50% 超えること。"""
        from market_analysis.terafab_model import yield_chain
        chain = yield_chain()
        final = list(chain.values())[-1]["cumulative"]
        assert final > 0.50, f"Total yield too low: {final*100:.1f}%"

    def test_capex_japan_share_reasonable(self):
        """日本・装置メーカーの CapEx シェアは 20〜50% の範囲。"""
        from market_analysis.terafab_model import capex_analysis
        c = capex_analysis()
        assert 20 <= c["japan_share_pct"] <= 50, \
            f"Japan share out of range: {c['japan_share_pct']:.1f}%"

    def test_wafers_per_day_positive(self):
        """必要ウェーハ枚数/日は正の整数。"""
        from market_analysis.terafab_model import production_target
        r = production_target()
        assert r["wafers_per_day"] > 0

    def test_rad_hard_tid_model_monotone(self):
        """TID が増えるにつれ ΔVth は単調増加。"""
        from market_analysis.terafab_model import rad_hard_tid_model
        tid = np.linspace(0, 300, 50)
        r = rad_hard_tid_model(tid)
        diffs = np.diff(r["dVth_rhc_V"])
        assert all(d >= 0 for d in diffs), "ΔVth should be monotone increasing with TID"


# ════════════════════════════════════════════════════════════════════════════
# Quantum Computing Model
# ════════════════════════════════════════════════════════════════════════════

class TestSiSpinQubit:
    def test_T1_order_of_magnitude(self):
        """
        良好な界面 (rms=0.1nm, n_imp=1e10) で T1 > 1ms (= 1000µs)。
        Ref: Takeda et al. (2022) Nature Electronics — T1 ~ 10ms with ²⁸Si
        """
        from market_analysis.quantum_computing_model import si_spin_qubit_coherence
        r = si_spin_qubit_coherence(rms_nm=0.1, n_imp_cm2=1e10, B_mT=1.0)
        assert r["T1_us"] > 1000, f"T1 should exceed 1ms, got {r['T1_us']:.1f} µs"

    def test_T2_le_2T1(self):
        """T2 ≤ 2T1 (量子力学的上限)。"""
        from market_analysis.quantum_computing_model import si_spin_qubit_coherence
        r = si_spin_qubit_coherence(rms_nm=0.1, n_imp_cm2=1e10, B_mT=1.0)
        assert r["T2_us"] <= 2 * r["T1_us"] + 1e-6, \
            f"T2={r['T2_us']:.1f} > 2*T1={2*r['T1_us']:.1f} (violates quantum limit)"

    def test_T1_decreases_with_interface_roughness(self):
        """界面粗さが大きいほど T1 が短い (単調減少)。"""
        from market_analysis.quantum_computing_model import si_spin_qubit_coherence
        T1s = [si_spin_qubit_coherence(rms_nm=r)["T1_us"]
               for r in [0.05, 0.1, 0.5, 1.0]]
        assert T1s == sorted(T1s, reverse=True), \
            f"T1 not monotone with roughness: {T1s}"

    def test_T1_decreases_with_impurity_density(self):
        """不純物密度が高いほど T1 が短い。"""
        from market_analysis.quantum_computing_model import si_spin_qubit_coherence
        T1s = [si_spin_qubit_coherence(rms_nm=0.1, n_imp_cm2=n)["T1_us"]
               for n in [1e9, 1e10, 1e11, 1e12]]
        assert T1s == sorted(T1s, reverse=True), \
            f"T1 not monotone with impurity: {T1s}"

    def test_qubit_frequency_positive(self):
        """qubit 周波数 > 0。"""
        from market_analysis.quantum_computing_model import si_spin_qubit_coherence
        r = si_spin_qubit_coherence(rms_nm=0.1)
        assert r["f_q_GHz"] > 0


class TestTransmonQubit:
    def test_qubit_frequency_in_GHz_range(self):
        """Transmon qubit 周波数は 4〜8 GHz の範囲 (Koch 2007)。"""
        from market_analysis.quantum_computing_model import transmon_qubit
        r = transmon_qubit(EJ_GHz=25.0, EC_GHz=0.25)
        assert 4.0 <= r["f_01_GHz"] <= 8.0, \
            f"Transmon freq out of range: {r['f_01_GHz']:.2f} GHz"

    def test_anharmonicity_negative(self):
        """アンハーモニシティ α < 0 (transmon では必ず負)。"""
        from market_analysis.quantum_computing_model import transmon_qubit
        r = transmon_qubit(EJ_GHz=25.0, EC_GHz=0.25)
        assert r["alpha_MHz"] < 0, f"|α| should be negative, got {r['alpha_MHz']}"

    def test_EJ_EC_ratio_above_50_is_immune(self):
        """EJ/EC > 50 で charge noise 免疫フラグが True。"""
        from market_analysis.quantum_computing_model import transmon_qubit
        r = transmon_qubit(EJ_GHz=15.0, EC_GHz=0.25)  # ratio=60
        assert r["charge_noise_immune"] is True

    def test_EJ_EC_ratio_below_50_not_immune(self):
        """EJ/EC < 50 で charge noise 免疫フラグが False。"""
        from market_analysis.quantum_computing_model import transmon_qubit
        r = transmon_qubit(EJ_GHz=10.0, EC_GHz=0.25)  # ratio=40
        assert r["charge_noise_immune"] is False

    def test_T1_positive(self):
        from market_analysis.quantum_computing_model import transmon_qubit
        r = transmon_qubit(EJ_GHz=25.0, EC_GHz=0.25)
        assert r["T1_us"] > 0


class TestSurfaceCodeQEC:
    def test_below_threshold_returns_results(self):
        """p_phys < p_th なら results リストが返る。"""
        from market_analysis.quantum_computing_model import surface_code_overhead
        qec = surface_code_overhead(p_phys=0.001, p_th=0.01)
        assert "results" in qec and len(qec["results"]) > 0

    def test_above_threshold_returns_error(self):
        """p_phys >= p_th なら 'error' キーが返る。"""
        from market_analysis.quantum_computing_model import surface_code_overhead
        qec = surface_code_overhead(p_phys=0.02, p_th=0.01)
        assert "error" in qec

    def test_logical_error_decreases_with_distance(self):
        """距離 d が大きいほど論理エラー率 p_L が小さくなる (単調減少)。"""
        from market_analysis.quantum_computing_model import surface_code_overhead
        qec = surface_code_overhead(p_phys=0.001)
        pLs = [r["p_L"] for r in qec["results"]]
        # p_L は d が増えるにつれ減少 (降順)
        assert pLs == sorted(pLs, reverse=True), \
            "p_L should monotone decrease with distance d"

    def test_physical_qubit_count_scales_as_2d_squared(self):
        """n_phys = 2d² の公式確認。"""
        from market_analysis.quantum_computing_model import surface_code_overhead
        qec = surface_code_overhead(p_phys=0.001)
        for r in qec["results"]:
            expected = 2 * r["d"] ** 2
            assert r["n_phys"] == expected, \
                f"n_phys={r['n_phys']} != 2d²={expected} for d={r['d']}"

    def test_wafer_integration_qubit_count_positive(self):
        from market_analysis.quantum_computing_model import si_qubit_wafer_integration
        r = si_qubit_wafer_integration(qubit_pitch_um=50.0)
        assert r["n_qubits_yield"] > 0
        assert r["n_qubits_yield"] < r["n_qubits_raw"]


# ════════════════════════════════════════════════════════════════════════════
# Hyperscaler Model
# ════════════════════════════════════════════════════════════════════════════

class TestHyperscalerModel:
    def test_tesla_ai5_highest_efficiency(self):
        """Tesla AI5 (2nm) が全チップ中最高 TFlops/W であること。"""
        from market_analysis.hyperscaler_model import ASIC_SPECS
        efficiencies = {
            k: v["tflops_bf16"] / v["tdp_W"]
            for k, v in ASIC_SPECS.items()
        }
        max_chip = max(efficiencies, key=efficiencies.get)
        assert "Tesla AI5" in max_chip, \
            f"Expected Tesla AI5 to be most efficient, got {max_chip}"

    def test_all_efficiencies_positive(self):
        """全チップの TFlops/W > 0。"""
        from market_analysis.hyperscaler_model import ASIC_SPECS
        for name, s in ASIC_SPECS.items():
            eff = s["tflops_bf16"] / s["tdp_W"]
            assert eff > 0, f"{name} efficiency <= 0"

    def test_tco_advantage_custom_asic_positive(self):
        """Google/AWS/Microsoft/Meta ASIC は H100 対比 TCO 優位 > 0%。"""
        from market_analysis.hyperscaler_model import tco_analysis
        for name in ["Google TPU v6e\n(Trillium)", "AWS Trainium3",
                     "Microsoft Maia 200", "Meta MTIA 500"]:
            t = tco_analysis(name, {})
            assert t["tco_vs_h100_pct"] > 0, \
                f"{name} should have positive TCO advantage vs H100"

    def test_tesla_ai5_tco_highest_advantage(self):
        """Tesla AI5 が H100 対比 TCO 優位性で最高 (> 70%)。"""
        from market_analysis.hyperscaler_model import tco_analysis
        t = tco_analysis("Tesla AI5", {})
        assert t["tco_vs_h100_pct"] > 70, \
            f"Tesla AI5 TCO advantage should be > 70%, got {t['tco_vs_h100_pct']:.1f}%"

    def test_dojo_d1_tco_negative(self):
        """Dojo D1 (7nm, 旧世代) は H100 対比 TCO 不利 (< 0%)。"""
        from market_analysis.hyperscaler_model import tco_analysis
        t = tco_analysis("Tesla Dojo D1\n(wafer-scale)", {})
        assert t["tco_vs_h100_pct"] < 0, \
            f"Old Dojo D1 should be worse TCO than H100, got {t['tco_vs_h100_pct']:.1f}%"

    def test_wafer_demand_positive(self):
        """各 hyperscaler のウェーハ需要 > 0。"""
        from market_analysis.hyperscaler_model import HYPERSCALER_CAPEX, wafer_demand
        for company, spec in HYPERSCALER_CAPEX.items():
            r = wafer_demand(company, spec)
            assert r["wafers_per_day"] > 0, f"{company} wafer demand is 0"

    def test_tesla_fsd_chip_demand_millions(self):
        """Tesla FSD 年間需要は 500万枚超 (車両600万台×2チップ以上)。"""
        from market_analysis.hyperscaler_model import tesla_fsd_inference_demand
        r = tesla_fsd_inference_demand()
        assert r["n_chips_vehicle"] > 5_000_000, \
            f"Tesla FSD chip demand too low: {r['n_chips_vehicle']:,}"

    def test_nvidia_share_decreasing_over_time(self):
        """2022→2030 で NVIDIA GPU シェアは単調減少。"""
        from market_analysis.hyperscaler_model import NVIDIA_SHARE
        years = sorted(NVIDIA_SHARE.keys())
        gpu_shares = [NVIDIA_SHARE[y][0] for y in years]
        for i in range(len(gpu_shares) - 1):
            assert gpu_shares[i] >= gpu_shares[i+1], \
                f"GPU share not decreasing: {years[i]}={gpu_shares[i]} < {years[i+1]}={gpu_shares[i+1]}"

    def test_total_hyperscaler_capex_above_400B(self):
        """全 hyperscaler 合計 CapEx は $400B 超。"""
        from market_analysis.hyperscaler_model import HYPERSCALER_CAPEX
        total = sum(v["capex_B"] for v in HYPERSCALER_CAPEX.values())
        assert total > 400, f"Total capex too low: ${total}B"
