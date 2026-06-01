"""
TEL Process Model — ALD / 熱酸化 / RIE × SiC デバイス性能
===========================================================
Tokyo Electron (TEL) の主要プロセスを物理モデル化。
Disco ダイシング後の後工程として SiC MOSFET 製造プロセスを記述。

完全パイプライン:
    Disco DAD/DFL ダイシング
        ↓ HAZ, 欠陥密度 (quantum_defect_model.py)
    TEL RIE 表面処理
        ↓ 表面粗さ改善 / 追加損傷
    TEL 熱酸化 (Deal-Grove)
        ↓ SiO₂ 界面形成 + Dit
    TEL ALD ゲート誘電体 (Al₂O₃/HfO₂)
        ↓ 等価酸化膜厚 EOT, Dit 低減
    SiC MOSFET 性能 (µ_ch, V_th, I_on)

1. ALD モデル (Atomic Layer Deposition)
   自己制限成長 → GPC (Growth Per Cycle)
   ALD ウィンドウ / 核生成遅延 / 膜均一性

2. Deal-Grove 熱酸化 (SiC 修正版)
   SiC + O₂ → SiO₂ + CO↑
   Si の約 100 倍遅い酸化: Arrhenius 活性化エネルギー

3. RIE 表面損傷モデル
   イオンエネルギー × フラックス → 損傷深さ
   CF₄/O₂, SF₆, Cl₂ 各ガスの SiC エッチング

4. Dit (界面トラップ密度) → チャンネル移動度
   Coulomb 散乱モデル: µ = µ_bulk / (1 + α·Dit·e/Cox)

5. MOSFET 性能統合モデル
   V_th, µ_ch, I_on/I_off vs プロセス条件

実行:
    python fem/tel_process_model.py
    python fem/tel_process_model.py --ald --material HfO2
    python fem/tel_process_model.py --oxidation --temp 1200
    python fem/tel_process_model.py --pipeline    # Disco→TEL→デバイス完全統合

参考文献:
    George (2010) Chem Rev — ALD review
    Deal & Grove (1965) J Appl Phys — thermal oxidation
    Kimoto & Cooper (2014) — SiC MOSFET technology
    Saks et al. (1999) Appl Phys Lett — SiC/SiO₂ interface traps
    TEL CLEAN TRACK / Tactras / Probus spec sheets
"""

import argparse
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

kB  = 8.617e-5   # eV/K
eps0 = 8.854e-12  # F/m
eV   = 1.602e-19  # J


# ════════════════════════════════════════════════════════════════════════════
# 1. ALD モデル — 自己制限成長
# ════════════════════════════════════════════════════════════════════════════

ALD_MATERIALS = {
    "Al2O3": {
        "GPC_A":        1.1,        # Å/cycle (飽和成長速度)
        "T_window":     (150, 300),  # °C ALD ウィンドウ
        "k_dielectric": 9.0,        # 比誘電率
        "rho_g_cm3":    3.2,        # 密度 g/cm³
        "E_act_eV":     0.30,       # 表面反応活性化エネルギー
        "nucleation_cycles": 5,     # 核生成遅延サイクル数
        "WER_nm_s":     0.12,       # ウェットエッチング速度 (希 HF)
        "color":        "#2166ac",
        "precursor":    "TMA / H₂O",
        "ref": "Kim et al. (2003) Appl Phys Lett",
    },
    "HfO2": {
        "GPC_A":        1.0,
        "T_window":     (200, 350),
        "k_dielectric": 22.0,       # High-k: 2nm ノードで重要
        "rho_g_cm3":    9.68,
        "E_act_eV":     0.45,
        "nucleation_cycles": 8,
        "WER_nm_s":     0.05,
        "color":        "#d62728",
        "precursor":    "HfCl₄ / H₂O",
        "ref": "Gusev et al. (2001) Microelectron Eng",
    },
    "SiO2 (ALD)": {
        "GPC_A":        0.6,
        "T_window":     (300, 500),
        "k_dielectric": 3.9,
        "rho_g_cm3":    2.2,
        "E_act_eV":     0.55,
        "nucleation_cycles": 3,
        "WER_nm_s":     0.40,
        "color":        "#2ca02c",
        "precursor":    "SiCl₄ / H₂O",
        "ref": "Morishita et al. (2005) Appl Surf Sci",
    },
}


def ald_gpc(T_C: float, material: str = "Al2O3") -> float:
    """
    温度依存 GPC (Growth Per Cycle) [Å/cycle]:
        GPC(T) = GPC_sat × exp(-E_act/(kT))  (低温: 反応律速)
             = GPC_sat                        (ALD ウィンドウ内)
             → 0                              (高温: 脱離律速)
    """
    d  = ALD_MATERIALS[material]
    T1, T2 = d["T_window"]
    if T_C < T1:
        # 低温: 活性化律速
        return d["GPC_A"] * math.exp(-d["E_act_eV"] / (kB * (T_C + 273)))
    elif T_C <= T2:
        # ALD ウィンドウ: 自己制限
        return d["GPC_A"]
    else:
        # 高温: CVD モード (脱離が追いつかない)
        return d["GPC_A"] * (1 + 0.005 * (T_C - T2))


def ald_film(n_cycles: int, T_C: float = 250.0,
              material: str = "Al2O3") -> dict:
    """
    ALD 成膜: n_cycles サイクル → 膜厚・EOT・核生成遅延

    EOT (Equivalent Oxide Thickness):
        EOT = t_film × (k_SiO2 / k_material)
    """
    d      = ALD_MATERIALS[material]
    nu     = d["nucleation_cycles"]
    gpc    = ald_gpc(T_C, material)

    # 核生成遅延: 最初の nu サイクルは GPC が線形に立ち上がる
    thickness = 0.0
    for cyc in range(1, n_cycles + 1):
        if cyc <= nu:
            thickness += gpc * (cyc / nu)
        else:
            thickness += gpc
    thickness_nm = thickness / 10.0   # Å → nm

    EOT = thickness_nm * 3.9 / d["k_dielectric"]

    # 均一性 (TEL 装置の強み: ±0.3%)
    uniformity_pct = 0.3 + 0.01 * max(0, T_C - d["T_window"][1])

    return {
        "material":       material,
        "n_cycles":       n_cycles,
        "T_C":            T_C,
        "gpc_A":          round(gpc, 3),
        "thickness_nm":   round(thickness_nm, 3),
        "EOT_nm":         round(EOT, 3),
        "k":              d["k_dielectric"],
        "uniformity_pct": round(uniformity_pct, 2),
        "Cox_fF_um2":     round(d["k_dielectric"] * eps0 / (thickness_nm * 1e-9) * 1e3, 3),
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. Deal-Grove 熱酸化 (SiC 修正版)
# ════════════════════════════════════════════════════════════════════════════

# SiC 酸化の Rate Constants (Arrhenius パラメータ)
# 参考: Kimoto & Cooper (2014) Table 6.1
OXIDATION_PARAMS = {
    "4H-SiC": {
        "A_A":    2.0e-3,   # µm  (linear rate const prefactor)
        "E_A_eV": 2.55,     # eV  (linear rate activation energy)
        "A_B":    0.15,     # µm²/hr (parabolic rate const prefactor)
        "E_B_eV": 2.80,     # eV
        "remark": "C face ~10× faster than Si face",
        "color":  "#2166ac",
    },
    "Si": {
        "A_A":    6.23e6,
        "E_A_eV": 2.00,
        "A_B":    7.72e2,
        "E_B_eV": 1.23,
        "remark": "standard dry oxidation",
        "color":  "#d62728",
    },
}


def deal_grove(t_hr: float, T_C: float,
                material: str = "4H-SiC",
                x_i_nm: float = 0.0) -> dict:
    """
    Deal-Grove モデル:
        x² + A·x = B·(t + τ)
        τ = (x_i² + A·x_i) / B  (初期酸化膜厚から)

    返り値: 酸化膜厚 [nm], 酸化速度 [nm/hr]
    """
    p  = OXIDATION_PARAMS[material]
    T  = T_C + 273.15

    B_over_A = p["A_A"] * math.exp(-p["E_A_eV"] / (kB * T))   # µm/hr (linear)
    B        = p["A_B"] * math.exp(-p["E_B_eV"] / (kB * T))    # µm²/hr (parabolic)
    A        = B / B_over_A if B_over_A > 0 else 1e10

    x_i = x_i_nm * 1e-3   # nm → µm
    tau = (x_i**2 + A * x_i) / B if B > 0 else 0

    # 二次方程式の解
    disc = A**2 / 4 + B * (t_hr + tau)
    x    = -A/2 + math.sqrt(max(disc, 0))   # µm

    rate_nm_hr = B / (2*x + A) * 1e3 if (2*x + A) > 0 else 0

    return {
        "material":    material,
        "T_C":         T_C,
        "t_hr":        t_hr,
        "x_ox_nm":     round(x * 1e3, 2),
        "rate_nm_hr":  round(rate_nm_hr, 3),
        "B_over_A":    round(B_over_A * 1e3, 4),   # nm/hr
        "B":           round(B * 1e6, 6),           # nm²/hr
    }


def dit_from_oxidation(T_C: float, t_hr: float,
                        surface: str = "Si-face",
                        anneal_T_C: float = None) -> float:
    """
    SiC/SiO₂ 界面トラップ密度 Dit [cm⁻²·eV⁻¹]:
    Saks et al. (1999) の経験式に基づく推定。

    近傍バンドギャップに C クラスター欠陥が生成 → Dit
    NO アニール (900-1050°C) で 10× 低減可能。
    """
    # 基準 Dit (Si 面, 1150°C 乾燥酸化, アニールなし)
    Dit_ref = 5e12   # cm⁻²eV⁻¹

    # 温度依存: 高温ほど界面が荒れる
    T_factor = math.exp(0.005 * (T_C - 1150))

    # 時間依存: 長時間ほど C クラスター蓄積
    t_factor = 1.0 + 0.2 * math.log10(max(t_hr, 0.1))

    # C 面 vs Si 面
    face_factor = 3.0 if surface == "C-face" else 1.0

    Dit = Dit_ref * T_factor * t_factor * face_factor

    # NO アニール効果 (900-1050°C で最適)
    if anneal_T_C is not None:
        if 900 <= anneal_T_C <= 1050:
            Dit *= 0.1   # 10× 低減
        elif anneal_T_C > 1050:
            Dit *= 0.3

    return Dit


# ════════════════════════════════════════════════════════════════════════════
# 3. RIE 表面損傷モデル
# ════════════════════════════════════════════════════════════════════════════

RIE_GASES = {
    "CF4/O2": {
        "etch_rate_nm_min": 80.0,    # SiC エッチング速度 [nm/min]
        "damage_depth_nm":  5.0,     # イオン損傷深さ
        "selectivity_SiO2": 1.2,
        "roughness_factor": 1.5,     # Ra 増加倍率
        "color": "#2166ac",
        "ref": "Wang et al. (2020) J Vac Sci",
    },
    "SF6": {
        "etch_rate_nm_min": 150.0,
        "damage_depth_nm":  8.0,
        "selectivity_SiO2": 0.8,
        "roughness_factor": 2.0,
        "color": "#d62728",
        "ref": "Shul et al. (1998)",
    },
    "Cl2": {
        "etch_rate_nm_min": 40.0,
        "damage_depth_nm":  3.0,     # 低損傷
        "selectivity_SiO2": 5.0,
        "roughness_factor": 1.2,     # 平滑
        "color": "#2ca02c",
        "ref": "Tanaka et al. (2015)",
    },
}


def rie_damage(t_min: float, gas: str = "CF4/O2",
               power_W: float = 200.0,
               pressure_mTorr: float = 10.0) -> dict:
    """
    RIE プロセス → エッチング深さ・損傷・表面粗さ
    イオンエネルギー E_ion ∝ power/pressure → 損傷深さをスケール
    """
    g = RIE_GASES[gas]
    # イオンエネルギーの規格化係数
    E_ion_factor = math.sqrt(power_W / pressure_mTorr) / math.sqrt(200 / 10)

    depth_nm  = g["etch_rate_nm_min"] * t_min
    damage_nm = g["damage_depth_nm"]  * E_ion_factor

    return {
        "gas":         gas,
        "etch_nm":     round(depth_nm, 2),
        "damage_nm":   round(damage_nm, 2),
        "Ra_factor":   round(g["roughness_factor"] * E_ion_factor, 2),
    }


# ════════════════════════════════════════════════════════════════════════════
# 4. Dit → チャンネル移動度 → MOSFET 性能
# ════════════════════════════════════════════════════════════════════════════

MU_BULK_SIC = 900.0   # cm²/Vs  4H-SiC バルク電子移動度

# ── 反転層移動度 Matthiessen パラメータ ──────────────────────────────────────
# Kimoto & Cooper (2014) + Chung et al. (2001 IEEE EDL) キャリブレーション
# NO アニール後 Dit~1e11 → µ_inv~35-70 cm²/Vs を再現
_MU_SP   = 120.0    # cm²/Vs  表面フォノン散乱限界 (SiC/SiO₂ 界面)
_MU_SR   = 200.0    # cm²/Vs  表面粗さ散乱限界
_B_COUL  = 6.56e13  # cm⁻²eV⁻¹ × cm²/Vs  Coulomb 散乱係数


def channel_mobility(Dit: float, Cox_fF_um2: float,
                      mu_bulk: float = MU_BULK_SIC) -> float:
    """後方互換用ラッパー。新規コードは channel_mobility_inv を使用。"""
    return channel_mobility_inv(Dit, Cox_fF_um2)


def channel_mobility_inv(Dit: float, Cox_fF_um2: float,
                          anneal: str = "none") -> float:
    """
    SiC MOSFET 反転チャンネル移動度 [cm²/Vs] — Matthiessen 則。

    1/µ_inv = 1/µ_phonon + 1/µ_coulomb + 1/µ_roughness

    µ_phonon  : 表面フォノン散乱 (~120 cm²/Vs, SiC/SiO₂)
    µ_coulomb : Coulomb 散乱 = B/Dit (Dit 依存)
    µ_roughness: 表面粗さ散乱 (~200 cm²/Vs)

    キャリブレーション (Chung et al. 2001, IEEE EDL):
        NO アニール後 Dit~1e11 → µ_inv ≈ 67 cm²/Vs  ✓
        無処理       Dit~1e13 → µ_inv ≈  6 cm²/Vs   ✓
        標準         Dit~1e12 → µ_inv ≈ 35 cm²/Vs   ✓

    anneal: "none" | "NO" | "N2O" | "POCl3"
        NO/N₂O アニールは Dit を直接下げる（ここでは追加補正なし）
        POCl₃ は Dit と界面粗さを同時改善するため µ_sr 補正を適用
    """
    mu_coulomb = min(_B_COUL / max(Dit, 1e8), 1e4)
    mu_sr = _MU_SR * (1.5 if anneal == "POCl3" else 1.0)
    return 1.0 / (1.0 / _MU_SP + 1.0 / mu_coulomb + 1.0 / mu_sr)


def mosfet_performance(mu_ch: float, Cox_fF_um2: float,
                        V_th: float, V_gs: float = 15.0,
                        W_um: float = 100.0, L_um: float = 5.0) -> dict:
    """
    SiC MOSFET 性能 (長チャンネル近似):
        I_on = Cox·µ_ch·(W/L)·(Vgs-Vth)²/2
        R_on = L/(Cox·µ_ch·W·(Vgs-Vth))
    """
    Cox  = max(Cox_fF_um2, 1e-6) * 1e-15 / (1e-4)**2   # F/cm²  (1µm=1e-4cm)
    WL   = (W_um * 1e-4) / (L_um * 1e-4)    # cm/cm
    Vod  = max(V_gs - V_th, 0)

    I_on_mA    = Cox * mu_ch * WL * Vod**2 / 2 * 1e3   # mA
    R_on_mohm  = 1.0 / (Cox * mu_ch * WL * max(Vod, 1e-6)) * 1e3

    return {
        "mu_ch_cm2Vs":  round(mu_ch, 1),
        "V_th_V":       round(V_th, 2),
        "I_on_mA":      round(I_on_mA, 2),
        "R_on_mohm_mm2": round(R_on_mohm * (W_um*L_um*1e-8), 3),
    }


# ════════════════════════════════════════════════════════════════════════════
# 5. 完全パイプライン: Disco → TEL → MOSFET
# ════════════════════════════════════════════════════════════════════════════

def full_pipeline(dicing_process: str = "blade",
                  ald_material:    str = "Al2O3",
                  ald_cycles:      int = 100,
                  ox_T_C:          float = 1150.0,
                  ox_t_hr:         float = 2.0,
                  anneal_T_C:      float = 950.0,
                  rie_gas:         str = "CF4/O2") -> dict:
    """
    Disco ダイシング → TEL RIE → TEL 熱酸化 → TEL ALD → SiC MOSFET 性能

    プロセス順:
        1. Disco ダイシング (HAZ, 欠陥密度)
        2. TEL RIE 表面クリーニング
        3. TEL 熱酸化 (ゲート酸化膜)
        4. TEL ALD ゲート誘電体
        5. NO ポストアニール
        6. MOSFET 性能評価
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from ml.quantum_defect_model import process_induced_trap_density, shockley_read_hall_lifetime

    # ── Step 1: Disco ダイシング ──────────────────────────────────
    haz_by_proc = {
        "blade":    2.5, "laser_ns": 2.0, "laser_ps": 0.5,
        "laser_fs": 0.2, "stealth":  0.05, "plasma":  0.8,
    }
    ra_by_proc = {
        "blade":    80.0, "laser_ns": 60.0, "laser_ps": 30.0,
        "laser_fs": 15.0, "stealth":  5.0,  "plasma":   20.0,
    }
    HAZ_um = haz_by_proc.get(dicing_process, 2.5)
    Ra_nm  = ra_by_proc.get(dicing_process, 80.0)

    trap_dens = process_induced_trap_density(HAZ_um, Ra_nm, dicing_process)
    tau_us    = shockley_read_hall_lifetime(trap_dens["V_C (Z1/2)"])

    # ── Step 2: TEL RIE 表面クリーニング ─────────────────────────
    rie_res = rie_damage(2.0, rie_gas, power_W=100)
    # RIE 後の表面粗さ改善 (軽い RIE で機械損傷層を除去)
    HAZ_after_rie = max(0.0, HAZ_um - rie_res["etch_nm"]*1e-3 * 0.5)
    Ra_after_rie  = Ra_nm / rie_res["Ra_factor"] * 0.8   # RIE で平滑化

    # ── Step 3: TEL 熱酸化 ───────────────────────────────────────
    ox = deal_grove(ox_t_hr, ox_T_C, "4H-SiC")
    Dit_ox = dit_from_oxidation(ox_T_C, ox_t_hr, anneal_T_C=anneal_T_C)

    # ── Step 4: TEL ALD ゲート誘電体 ─────────────────────────────
    ald = ald_film(ald_cycles, 250.0, ald_material)
    # ダイシング由来の追加 Dit: HAZ 深さに比例した表面欠陥
    Dit_dicing = 1e11 * (HAZ_after_rie / 0.1) ** 0.5   # cm⁻²eV⁻¹
    # ALD 後: 熱酸化 Dit の 20% + ダイシング由来 Dit (ALD では除去しきれない)
    Dit_ald = (Dit_ox + Dit_dicing) * (0.2 if ald_material in ["Al2O3","HfO2"] else 0.8)

    # ── Step 5: MOSFET 性能 ───────────────────────────────────────
    mu_ch  = channel_mobility(Dit_ald, ald["Cox_fF_um2"])
    # V_th: 3.0V + Dit による閾値シフト (小さいが定性的に正しい)
    EOT_nm   = ox["x_ox_nm"] + ald["EOT_nm"]
    Cox_Fcm2 = 3.9 * eps0 / max(EOT_nm, 0.1) * 1e9 * 1e-4  # F/cm²
    # ΔV_th = q × N_it / Cox  (N_it = Dit × ΔE, ΔE=0.1eV)
    N_it  = Dit_ald * 0.1   # cm⁻²
    V_th  = 3.0 + eV * N_it / max(Cox_Fcm2, 1e-9)

    perf = mosfet_performance(mu_ch, ald["Cox_fF_um2"], V_th)

    return {
        "dicing_process": dicing_process,
        "HAZ_um":         round(HAZ_um, 3),
        "Ra_nm":          round(Ra_nm, 1),
        "tau_SRH_us":     round(tau_us, 2),
        "RIE_etch_nm":    rie_res["etch_nm"],
        "ox_thickness_nm":ox["x_ox_nm"],
        "Dit_cm2eV":      float(Dit_ald),
        "ALD_material":   ald_material,
        "ALD_EOT_nm":     ald["EOT_nm"],
        "total_EOT_nm":   round(EOT_nm, 2),
        "mu_ch":          perf["mu_ch_cm2Vs"],
        "V_th":           perf["V_th_V"],
        "I_on_mA":        perf["I_on_mA"],
        "R_on_mohm_mm2":  perf["R_on_mohm_mm2"],
    }


# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def plot_ald(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: GPC vs 温度 (ALD ウィンドウ)
    ax = axes[0]
    T_arr = np.linspace(50, 500, 200)
    for mat, d in ALD_MATERIALS.items():
        gpcs = [ald_gpc(T, mat) for T in T_arr]
        ax.plot(T_arr, gpcs, color=d["color"], lw=2.5, label=f"{mat} ({d['precursor']})")
        ax.axvspan(*d["T_window"], alpha=0.08, color=d["color"])
    ax.set_xlabel("Temperature [°C]", fontsize=11)
    ax.set_ylabel("GPC [Å/cycle]", fontsize=11)
    ax.set_title("ALD Growth Per Cycle\nALD ウィンドウ (shaded)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Panel 2: 膜厚 vs サイクル数 (核生成遅延)
    ax2 = axes[1]
    cycles = np.arange(1, 201)
    for mat, d in ALD_MATERIALS.items():
        res = [ald_film(int(n), 250, mat)["thickness_nm"] for n in cycles]
        ax2.plot(cycles, res, color=d["color"], lw=2, label=mat)
        ax2.axvline(d["nucleation_cycles"], color=d["color"],
                    ls=":", lw=1.0, alpha=0.7)
    ax2.set_xlabel("ALD Cycles", fontsize=11)
    ax2.set_ylabel("Film Thickness [nm]", fontsize=11)
    ax2.set_title("ALD 膜厚 vs サイクル数\n点線: 核生成遅延", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    # Panel 3: EOT vs 膜厚
    ax3 = axes[2]
    t_arr = np.linspace(0, 20, 100)   # nm
    for mat, d in ALD_MATERIALS.items():
        eots = [t * 3.9 / d["k_dielectric"] for t in t_arr]
        ax3.plot(t_arr, eots, color=d["color"], lw=2,
                 label=f"{mat} (k={d['k_dielectric']})")
    ax3.plot([0,20],[0,20], "k--", lw=1, label="SiO₂ (k=3.9)")
    ax3.axhline(0.5, color="purple", ls=":", lw=1.5, label="EOT 0.5nm (2nm node)")
    ax3.set_xlabel("Physical Thickness [nm]", fontsize=11)
    ax3.set_ylabel("EOT [nm]", fontsize=11)
    ax3.set_title("EOT vs 物理膜厚\nHigh-k の優位性", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.25)

    fig.suptitle("TEL ALD — Atomic Layer Deposition モデル\n"
                 "SiC MOSFET ゲート誘電体 (CLEAN TRACK / Probus 相当)",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "tel_ald.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] ALD model -> {out}")


def plot_oxidation(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: 酸化膜厚 vs 時間 (SiC vs Si)
    ax = axes[0]
    t_arr = np.linspace(0.01, 10, 100)
    for mat, col, ls in [("4H-SiC","#2166ac","-"),
                          ("Si",    "#d62728","--")]:
        for T_C, alpha in [(1150,1.0),(1050,0.6),(950,0.3)]:
            res = [deal_grove(t, T_C, mat)["x_ox_nm"] for t in t_arr]
            ax.semilogy(t_arr, res, color=col, lw=2*alpha, ls=ls,
                        label=f"{mat} {T_C}°C" if alpha==1.0 else None)

    ax.set_xlabel("Oxidation Time [hr]", fontsize=11)
    ax.set_ylabel("Oxide Thickness [nm]", fontsize=11)
    ax.set_title("Deal-Grove 熱酸化\nSiC は Si の ~100 倍遅い", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, which="both")

    # Panel 2: Dit vs 酸化条件 + NO アニール効果
    ax2 = axes[1]
    T_arr = np.linspace(900, 1300, 60)
    for surface, ls in [("Si-face","-"),("C-face","--")]:
        for anneal, col, lbl in [(None, "#d62728", "酸化のみ"),
                                  (950,  "#2ca02c", "+NO 950°C"),
                                  (1000, "#2166ac", "+NO 1000°C")]:
            dits = [dit_from_oxidation(T, 2.0, surface, anneal) for T in T_arr]
            ax2.semilogy(T_arr, dits, color=col, lw=2, ls=ls,
                         label=f"{surface} {lbl}" if surface=="Si-face" else None)

    ax2.axhline(1e12, color="purple", ls=":", lw=1.5, label="デバイス目標 Dit < 1e12")
    ax2.set_xlabel("Oxidation Temperature [°C]", fontsize=11)
    ax2.set_ylabel("Dit [cm⁻²eV⁻¹]", fontsize=11)
    ax2.set_title("SiC/SiO₂ 界面トラップ密度\nNO アニール効果 (TEL RTP)", fontsize=10)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.2, which="both")

    fig.suptitle("TEL 熱酸化 — Deal-Grove (SiC 修正版)\nSiC MOSFET ゲート酸化膜品質",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "tel_oxidation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Thermal oxidation -> {out}")


def plot_pipeline(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    procs = ["blade", "laser_ps", "stealth"]
    labels = {"blade": "ブレード", "laser_ps": "レーザー(ps)", "stealth": "ステルス"}
    colors = {"blade": "#d62728", "laser_ps": "#2ca02c", "stealth": "#2166ac"}

    results = {}
    for proc in procs:
        for mat in ["Al2O3", "HfO2"]:
            key = f"{proc}_{mat}"
            results[key] = full_pipeline(
                dicing_process=proc, ald_material=mat,
                ald_cycles=50, ox_T_C=1150, ox_t_hr=2.0, anneal_T_C=950)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    metrics = [
        ("mu_ch",     "Channel Mobility [cm²/Vs]",      "移動度"),
        ("I_on_mA",   "I_on [mA]",                       "オン電流"),
        ("R_on_mohm_mm2", "R_on [mΩ·mm²]",              "オン抵抗"),
        ("total_EOT_nm",  "Total EOT [nm]",               "等価酸化膜厚"),
    ]

    for ax, (key, ylabel, title) in zip(axes.flat, metrics):
        x    = np.arange(len(procs))
        w    = 0.35
        for k, (mat, off, hatch) in enumerate([("Al2O3", -w/2, ""), ("HfO2", w/2, "//")]):
            vals = [results[f"{proc}_{mat}"][key] for proc in procs]
            bars = ax.bar(x + off, vals, w, color=[colors[p] for p in procs],
                          alpha=0.85, hatch=hatch, edgecolor="k", linewidth=0.5,
                          label=mat)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[p] for p in procs], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(["Al₂O₃", "HfO₂"], fontsize=8)
        ax.grid(alpha=0.2, axis="y")

    fig.suptitle("完全プロセスパイプライン\n"
                 "Disco ダイシング → TEL RIE → TEL 熱酸化 → TEL ALD → SiC MOSFET",
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(out_dir, "tel_full_pipeline.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    # テキストサマリー
    print(f"\n  {'プロセス':>10}  {'誘電体':>6}  {'µ_ch':>8}  {'I_on[mA]':>10}  "
          f"{'R_on':>8}  {'EOT[nm]':>8}")
    print("  " + "-"*60)
    for proc in procs:
        for mat in ["Al2O3", "HfO2"]:
            r = results[f"{proc}_{mat}"]
            print(f"  {labels[proc]:>10}  {mat:>6}  "
                  f"{r['mu_ch']:>8.1f}  {r['I_on_mA']:>10.2f}  "
                  f"{r['R_on_mohm_mm2']:>8.3f}  {r['total_EOT_nm']:>8.2f}")
    print(f"[✓] Full pipeline -> {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ald",        action="store_true")
    ap.add_argument("--oxidation",  action="store_true")
    ap.add_argument("--pipeline",   action="store_true")
    ap.add_argument("--material",   default="Al2O3")
    ap.add_argument("--temp",       type=float, default=1150.0)
    args = ap.parse_args()

    run_all = not any([args.ald, args.oxidation, args.pipeline])

    if args.ald       or run_all: plot_ald()
    if args.oxidation or run_all: plot_oxidation()
    if args.pipeline  or run_all: plot_pipeline()


if __name__ == "__main__":
    main()
