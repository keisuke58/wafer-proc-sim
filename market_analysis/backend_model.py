"""
後工程モデル — ワイヤーボンディング × パッケージ応力 × 信頼性
================================================================
SiC パワーデバイス後工程 (OSAT) を物理モデル化。

1. ワイヤーボンディング (サーモソニック)
   - Au / Cu / Al ワイヤー材料比較
   - 金属間化合物 (IMC) 成長 — Arrhenius (Au-Al 系)
   - ボンドプル強度 — Weibull 統計
   - ヒールクラック疲労 — Coffin-Manson
   - 寄生インダクタンス — ループ形状依存

2. パッケージ応力
   - CTE ミスマッチ熱応力 (Suhir モデル)
   - Timoshenko バイメタル反り (ΔT 依存)
   - ダイアタッチ層 von Mises 応力
   - はんだ接合疲労寿命 — Engelmaier モデル

3. 完全後工程パイプライン
   洗浄済みダイ → ダイアタッチ → ワイヤーボンディング
   → モールド → 熱サイクル → 信頼性予測 (MTTF)

実行:
    python fem/backend_model.py
    python fem/backend_model.py --wire-only
    python fem/backend_model.py --package-only

参考文献:
    Harman (1997) Wire Bonding in Microelectronics, McGraw-Hill
    Timoshenko (1925) J Opt Soc Am — bimaterial beam
    Engelmaier (1993) ASME J Electron Packag — solder fatigue
    Suhir (1989) J Appl Mech — CTE mismatch stress
    MIL-STD-883 Method 2023 — wire bond pull test
    Kimoto & Cooper (2014) — SiC MOSFET reliability
"""

import argparse
import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

kB = 8.617e-5  # eV/K
mu0 = 4 * math.pi * 1e-7  # H/m

# ── ワイヤー材料パラメータ ────────────────────────────────────────────────────
WIRE_MATERIALS = {
    "Au": {
        "r_um": 12.5,         # ワイヤー半径 [µm]
        "E_GPa": 79.0,        # ヤング率
        "nu": 0.44,
        "CTE_ppm": 14.2,      # 熱膨張係数
        "rho_gcm3": 19.3,
        "T_bond_C": 175,      # 標準ボンディング温度
        "eta_g": 7.0,         # Weibull スケール (プル強度 [g])
        "beta": 8.5,          # Weibull 形状
        "color": "#ffd700",
        # IMC: Breach et al. (2004) Au-Al 系キャリブレーション
        # d=0.29µm @ 125°C/1000h → A = d²/(t*exp(-Q/kT)) = 0.94 µm²/s
        "Q_imc_eV": 0.60,
        "A_imc": 0.94,
        # Coffin-Manson: ΔT=100K → N_f~50,000 サイクル (Harman 1997)
        # C = N_f * eps_p^m, eps_p = ΔαΔT = 10.2e-6*100 = 1.02e-3
        "coffin_C": 0.052,
        "coffin_m": 2.0,
    },
    "Cu": {
        "r_um": 12.5,
        "E_GPa": 120.0,
        "nu": 0.34,
        "CTE_ppm": 17.0,
        "rho_gcm3": 8.96,
        "T_bond_C": 230,
        "eta_g": 9.5,
        "beta": 10.0,
        "color": "#b87333",
        # Cu-Al IMC: Q=0.80eV, Cu は Au より低温での成長が遅い
        "Q_imc_eV": 0.80,
        "A_imc": 180.0,       # Cu-Al より高い前指数 (高活性化エネルギー補正)
        # Cu は Au より高強度 → N_f~100,000 @ ΔT=100K
        "coffin_C": 0.13,
        "coffin_m": 2.0,
    },
    "Al": {
        "r_um": 12.5,
        "E_GPa": 70.0,
        "nu": 0.33,
        "CTE_ppm": 23.1,
        "rho_gcm3": 2.70,
        "T_bond_C": 25,       # 超音波ボンディング (加熱不要)
        "eta_g": 4.5,
        "beta": 6.0,
        "color": "#9e9e9e",
        "Q_imc_eV": 0.85,
        "A_imc": 350.0,
        # Al は延性低く疲労寿命短め → N_f~20,000 @ ΔT=100K
        "coffin_C": 0.021,
        "coffin_m": 1.8,
    },
}

# ── パッケージ材料 CTE / 弾性定数 ────────────────────────────────────────────
PACKAGE_MATERIALS = {
    "4H-SiC":     {"CTE": 4.0,  "E_GPa": 450, "nu": 0.21, "h_mm": 0.35},
    "AlN_sub":    {"CTE": 4.5,  "E_GPa": 320, "nu": 0.24, "h_mm": 0.64},
    "Al2O3_sub":  {"CTE": 7.0,  "E_GPa": 370, "nu": 0.22, "h_mm": 0.63},
    "Cu_DBC":     {"CTE": 17.0, "E_GPa": 117, "nu": 0.34, "h_mm": 0.30},
    "Cu_frame":   {"CTE": 17.0, "E_GPa": 117, "nu": 0.34, "h_mm": 0.20},
    "SAC305":     {"CTE": 22.0, "E_GPa": 50,  "nu": 0.40, "h_mm": 0.10},
    "Ag_sinter":  {"CTE": 20.0, "E_GPa": 9.0, "nu": 0.37, "h_mm": 0.05},
    "EMC":        {"CTE": 15.0, "E_GPa": 20,  "nu": 0.30, "h_mm": 0.80},
}


# ════════════════════════════════════════════════════════════════════════════
# 1. ワイヤーボンディング
# ════════════════════════════════════════════════════════════════════════════

def pull_strength_samples(wire: str, n: int = 500, rng=None) -> np.ndarray:
    """Weibull 分布からプル強度サンプルを生成 [g]。"""
    rng = rng or np.random.default_rng(42)
    m = WIRE_MATERIALS[wire]
    return weibull_min.rvs(m["beta"], scale=m["eta_g"], size=n, random_state=rng)


def imc_thickness(wire: str, T_C: float, t_h: float) -> float:
    """
    IMC 成長厚さ [µm]。拡散律速: d = sqrt(A * t * exp(-Q/kT))
    wire: "Au" | "Cu" | "Al"
    T_C : 保存/動作温度 [°C]
    t_h : 時間 [h]
    """
    m = WIRE_MATERIALS[wire]
    t_s = t_h * 3600
    T_K = T_C + 273.15
    D = m["A_imc"] * math.exp(-m["Q_imc_eV"] / (kB * T_K))
    return math.sqrt(D * t_s)


def heel_crack_life(wire: str, delta_T: float, loop_h_um: float = 200) -> float:
    """
    ヒールクラック疲労寿命 [サイクル] — Coffin-Manson。
    delta_T: 熱サイクル温度幅 [K]
    loop_h_um: ワイヤーループ高さ [µm] (大きいほど応力緩和)
    Harman (1997) キャリブレーション: Au @ ΔT=100K → ~50,000 cycles
    """
    m = WIRE_MATERIALS[wire]
    # 塑性ひずみ振幅: CTE ミスマッチ × ΔT × ループ緩和係数
    delta_cte = abs(m["CTE_ppm"] - PACKAGE_MATERIALS["4H-SiC"]["CTE"]) * 1e-6
    loop_factor = (200.0 / loop_h_um) ** 0.5  # ループ高さ増加で応力緩和
    eps_p = delta_cte * delta_T * loop_factor
    eps_p = max(eps_p, 1e-9)
    return m["coffin_C"] * eps_p ** (-m["coffin_m"])


def wire_inductance(loop_h_um: float, wire_len_um: float, r_um: float = 12.5) -> float:
    """
    ワイヤーボンド寄生インダクタンス [nH]。
    Rosa (1908) 近似: L = (µ₀*l/2π)*(ln(2h/r) - 1)
    """
    h_m = loop_h_um * 1e-6
    l_m = wire_len_um * 1e-6
    r_m = r_um * 1e-6
    if h_m <= r_m:
        h_m = r_m * 2
    L_H = (mu0 * l_m / (2 * math.pi)) * (math.log(2 * h_m / r_m) - 1)
    return L_H * 1e9  # nH


# ════════════════════════════════════════════════════════════════════════════
# 2. パッケージ応力
# ════════════════════════════════════════════════════════════════════════════

def cte_mismatch_stress(mat1: str, mat2: str, delta_T: float) -> dict:
    """
    Suhir モデル: CTE ミスマッチ熱応力 [MPa]。
    biaxial: σ = E1/(1-ν1) * (α2-α1) * ΔT (薄膜近似)
    """
    m1 = PACKAGE_MATERIALS[mat1]
    m2 = PACKAGE_MATERIALS[mat2]
    da = (m2["CTE"] - m1["CTE"]) * 1e-6  # ppm/K → 1/K
    sigma1 = m1["E_GPa"] * 1e3 / (1 - m1["nu"]) * da * delta_T  # MPa
    sigma2 = m2["E_GPa"] * 1e3 / (1 - m2["nu"]) * (-da) * delta_T
    return {
        "mat1": mat1, "mat2": mat2, "delta_T": delta_T,
        "sigma_mat1_MPa": sigma1,
        "sigma_mat2_MPa": sigma2,
        "delta_CTE_ppm": (m2["CTE"] - m1["CTE"]),
    }


def timoshenko_warpage(mat1: str, mat2: str,
                       delta_T: float, L_mm: float = 10.0) -> dict:
    """
    Timoshenko バイメタルビーム反り公式 (1925)。
    κ = 6*(α₁-α₂)*ΔT*(1+m)² /
        [h*(3*(1+m)²+(1+mn)*(m²+1/(mn)))]
    warpage δ = κ*L²/8 [µm]
    """
    m1 = PACKAGE_MATERIALS[mat1]
    m2 = PACKAGE_MATERIALS[mat2]
    h1 = m1["h_mm"]
    h2 = m2["h_mm"]
    E1 = m1["E_GPa"]
    E2 = m2["E_GPa"]
    a1 = m1["CTE"] * 1e-6
    a2 = m2["CTE"] * 1e-6

    m_ratio = h1 / h2          # 厚さ比
    n_ratio = E1 / E2          # 剛性比
    h_tot = h1 + h2

    denom = (3 * (1 + m_ratio) ** 2
             + (1 + m_ratio * n_ratio) * (m_ratio ** 2 + 1 / (m_ratio * n_ratio)))
    kappa = (6 * (a1 - a2) * delta_T * (1 + m_ratio) ** 2) / (h_tot * denom)  # 1/mm

    warpage_um = abs(kappa) * L_mm ** 2 / 8 * 1e3  # µm

    return {
        "mat1": mat1, "mat2": mat2, "delta_T": delta_T,
        "curvature_per_mm": kappa,
        "warpage_um": warpage_um,
        "delta_CTE_ppm": (m1["CTE"] - m2["CTE"]),
    }


def engelmaier_fatigue(mat_sub: str, mat_die: str,
                       delta_T: float, L_die_mm: float = 5.0,
                       h_attach_mm: float = 0.10,
                       T_mean_C: float = 60, t_dwell_min: float = 15) -> dict:
    """
    Engelmaier モデル: はんだ/ダイアタッチ接合疲労寿命 [サイクル]。
    N_f = 0.5 * (Δγ/(2ε_f'))^(1/c)
    c   = -0.442 - 6e-4*T_mean + 1.74e-2*ln(1+360/t_d)
    """
    sub = PACKAGE_MATERIALS[mat_sub]
    die = PACKAGE_MATERIALS[mat_die]
    da = abs(sub["CTE"] - die["CTE"]) * 1e-6  # 1/K
    LD = L_die_mm / math.sqrt(2)              # 対角半分

    F = 1.0                   # 経験因子
    delta_gamma = F * da * delta_T * LD / h_attach_mm

    c = -0.442 - 6e-4 * T_mean_C + 1.74e-2 * math.log(1 + 360 / t_dwell_min)
    eps_f = 0.325             # SAC305 疲労延性係数

    N_f = 0.5 * (delta_gamma / (2 * eps_f)) ** (1 / c)
    N_f = max(N_f, 1)

    return {
        "mat_sub": mat_sub, "mat_die": mat_die,
        "delta_gamma": delta_gamma,
        "N_f_cycles": round(N_f),
        "c_exponent": round(c, 4),
        "delta_T": delta_T,
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. 完全後工程パイプライン
# ════════════════════════════════════════════════════════════════════════════

def backend_pipeline(wire: str = "Au",
                     substrate: str = "AlN_sub",
                     die_attach: str = "Ag_sinter",
                     delta_T_op: float = 100,
                     delta_T_pkg: float = 150,
                     L_die_mm: float = 5.0) -> dict:
    """
    後工程全体: ダイアタッチ → ワイヤーボンド → パッケージ応力 → 信頼性。
    """
    pull_samples = pull_strength_samples(wire)
    pull_mean = float(np.mean(pull_samples))
    pull_b10  = float(np.percentile(pull_samples, 10))

    imc_125 = imc_thickness(wire, T_C=125, t_h=1000)
    imc_175 = imc_thickness(wire, T_C=175, t_h=1000)

    heel_life = heel_crack_life(wire, delta_T=delta_T_op)

    inductance = wire_inductance(loop_h_um=200, wire_len_um=2000)

    stress     = cte_mismatch_stress("4H-SiC", substrate, delta_T_pkg)
    warpage    = timoshenko_warpage("4H-SiC", substrate, delta_T_pkg, L_die_mm * 2)
    fatigue    = engelmaier_fatigue(substrate, "4H-SiC", delta_T_op, L_die_mm)

    return {
        "wire": wire,
        "substrate": substrate,
        "die_attach": die_attach,
        # ワイヤーボンド
        "pull_mean_g": round(pull_mean, 2),
        "pull_b10_g":  round(pull_b10,  2),
        "imc_125C_1000h_um": round(imc_125, 3),
        "imc_175C_1000h_um": round(imc_175, 3),
        "heel_crack_cycles": round(heel_life),
        "inductance_nH": round(inductance, 3),
        # パッケージ応力
        "die_stress_MPa": round(abs(stress["sigma_mat1_MPa"]), 1),
        "warpage_um": round(warpage["warpage_um"], 1),
        "solder_fatigue_cycles": fatigue["N_f_cycles"],
    }


# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    rng = np.random.default_rng(42)

    wires   = ["Au", "Cu", "Al"]
    w_colors = {w: WIRE_MATERIALS[w]["color"] for w in wires}

    # ── Panel 1: プル強度 Weibull 分布 ───────────────────────────────────
    ax = axes[0, 0]
    for wire in wires:
        samples = pull_strength_samples(wire, n=2000, rng=rng)
        x = np.linspace(0, samples.max() * 1.1, 300)
        m = WIRE_MATERIALS[wire]
        pdf = weibull_min.pdf(x, m["beta"], scale=m["eta_g"])
        ax.plot(x, pdf, color=w_colors[wire], lw=2.5, label=f"{wire} wire")
        ax.axvline(m["eta_g"], color=w_colors[wire], ls="--", lw=1, alpha=0.6)
    ax.axvline(3.0, color="red", ls=":", lw=1.5, label="MIL-STD-883 min (3g)")
    ax.set_xlabel("Pull strength [g]", fontsize=10)
    ax.set_ylabel("Probability density", fontsize=10)
    ax.set_title("Wire Bond Pull Strength\nWeibull Distribution", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.2)

    # ── Panel 2: IMC 成長 vs 時間 (Au/Cu/Al @ 125°C) ────────────────────
    ax2 = axes[0, 1]
    t_arr = np.linspace(0, 3000, 200)
    for wire in wires:
        imc_vals = [imc_thickness(wire, T_C=125, t_h=t) for t in t_arr]
        ax2.plot(t_arr, imc_vals, color=w_colors[wire], lw=2.5, label=f"{wire}")
    ax2.axhline(1.0, color="red", ls="--", lw=1.5, label="Reliability limit (1µm)")
    ax2.set_xlabel("Time [h]", fontsize=10)
    ax2.set_ylabel("IMC thickness [µm]", fontsize=10)
    ax2.set_title("Intermetallic Compound Growth\n@ 125°C (ATE burn-in)", fontsize=10)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.2)

    # ── Panel 3: ヒールクラック疲労寿命 vs ΔT ────────────────────────────
    ax3 = axes[0, 2]
    dT_arr = np.linspace(20, 200, 100)
    for wire in wires:
        lives = [heel_crack_life(wire, dT) for dT in dT_arr]
        ax3.semilogy(dT_arr, lives, color=w_colors[wire], lw=2.5, label=f"{wire}")
    ax3.axhline(1e5, color="purple", ls="--", lw=1.5, label="Target >1e5 cycles")
    ax3.set_xlabel("Thermal cycle ΔT [K]", fontsize=10)
    ax3.set_ylabel("Fatigue life [cycles]", fontsize=10)
    ax3.set_title("Heel Crack Fatigue\nCoffin-Manson Model", fontsize=10)
    ax3.legend(fontsize=9); ax3.grid(alpha=0.2, which="both")

    # ── Panel 4: CTE ミスマッチ応力 (基板別) ─────────────────────────────
    ax4 = axes[1, 0]
    substrates = ["AlN_sub", "Al2O3_sub", "Cu_DBC", "Cu_frame"]
    sub_labels = {"AlN_sub": "AlN", "Al2O3_sub": "Al₂O₃",
                  "Cu_DBC": "Cu/DBC", "Cu_frame": "Cu frame"}
    sub_colors = ["#2166ac", "#d62728", "#ff7f0e", "#2ca02c"]
    dT_arr2 = np.linspace(0, 200, 80)
    for sub, col in zip(substrates, sub_colors):
        stresses = [abs(cte_mismatch_stress("4H-SiC", sub, dT)["sigma_mat1_MPa"])
                    for dT in dT_arr2]
        ax4.plot(dT_arr2, stresses, color=col, lw=2.5, label=sub_labels[sub])
    ax4.axhline(400, color="red", ls="--", lw=1.5, label="SiC fracture ~400 MPa")
    ax4.set_xlabel("ΔT [K]", fontsize=10)
    ax4.set_ylabel("Die stress [MPa]", fontsize=10)
    ax4.set_title("CTE Mismatch Stress (Suhir)\nvs Substrate Material", fontsize=10)
    ax4.legend(fontsize=8); ax4.grid(alpha=0.2)

    # ── Panel 5: パッケージ反り vs ダイサイズ ────────────────────────────
    ax5 = axes[1, 1]
    L_arr = np.linspace(2, 15, 60)
    for sub, col in zip(substrates, sub_colors):
        warpages = [timoshenko_warpage("4H-SiC", sub, 150, L * 2)["warpage_um"]
                    for L in L_arr]
        ax5.plot(L_arr, warpages, color=col, lw=2.5, label=sub_labels[sub])
    ax5.axhline(200, color="red", ls="--", lw=1.5, label="JEDEC limit 200µm")
    ax5.set_xlabel("Die size [mm]", fontsize=10)
    ax5.set_ylabel("Warpage [µm]", fontsize=10)
    ax5.set_title("Package Warpage (Timoshenko)\nvs Die Size, ΔT=150K", fontsize=10)
    ax5.legend(fontsize=8); ax5.grid(alpha=0.2)

    # ── Panel 6: 後工程パイプライン サマリー表 ───────────────────────────
    ax6 = axes[1, 2]
    ax6.axis("off")
    rows_data = []
    for wire in wires:
        for sub in ["AlN_sub", "Al2O3_sub"]:
            r = backend_pipeline(wire=wire, substrate=sub)
            rows_data.append([
                f"{wire}/{sub_labels[sub]}",
                f"{r['pull_mean_g']:.1f} g",
                f"{r['imc_125C_1000h_um']:.3f}",
                f"{r['heel_crack_cycles']:.0e}",
                f"{r['warpage_um']:.0f}",
                f"{r['solder_fatigue_cycles']:.0e}",
            ])
    col_labels = ["Wire/Sub", "Pull", "IMC\n1000h", "Heel\ncycles", "Warp\nµm", "Solder\ncycles"]
    tbl = ax6.table(cellText=rows_data, colLabels=col_labels,
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    tbl.scale(1.1, 1.9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#333"); cell.set_text_props(color="w")
        elif r % 2 == 1:
            cell.set_facecolor("#f5f5f5")
    ax6.set_title("Backend Pipeline Summary\n(SiC MOSFET Package)", fontsize=10)

    fig.suptitle("TEL/OSAT Backend Model — Wire Bonding × Package Stress\n"
                 "Thermosonic (Au/Cu/Al) + Timoshenko Warpage + Engelmaier Fatigue",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "backend_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    # ターミナルサマリー
    print("\n  後工程パイプライン サマリー (Au wire / AlN substrate):")
    r = backend_pipeline(wire="Au", substrate="AlN_sub")
    for k, v in r.items():
        print(f"    {k:<30} {v}")
    print(f"\n[OK] backend model -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wire-only",    action="store_true")
    ap.add_argument("--package-only", action="store_true")
    args = ap.parse_args()
    plot_all()


if __name__ == "__main__":
    main()
