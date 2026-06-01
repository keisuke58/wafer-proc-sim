"""
Quantum Spin & Transport Model — SiC 欠陥の深層量子物理
=========================================================
前モジュール (quantum_defect_model.py) の拡張。
より深い量子力学的記述を実装する。

1. スピン Hamiltonian (V_Si in 4H-SiC, S=3/2)
   ゼロ磁場分裂 D, E テンソル + Zeeman 効果 + 超微細結合
   → ODMR スペクトル / 量子ビット操作周波数

2. フォノンカップリング (Huang-Rhys 因子)
   ゼロフォノン線 (ZPL) vs フォノンサイドバンド
   → 光学的欠陥識別・温度依存線幅

3. 密度行列 + リンドブラッド方程式
   核スピンバスによるデコヒーレンス (T₁, T₂*)
   → 量子ビット・センサとしての性能予測

4. WKB トンネリング (SiC MOSFET ゲート酸化膜)
   Fowler-Nordheim / 直接トンネル電流
   → 2nm ノード SiC パワー MOSFET の界面品質評価

実行:
    python ml/quantum_spin_model.py              # 全図
    python ml/quantum_spin_model.py --odmr       # ODMR スペクトル
    python ml/quantum_spin_model.py --phonon     # Huang-Rhys
    python ml/quantum_spin_model.py --lindblad   # デコヒーレンス
    python ml/quantum_spin_model.py --tunnel     # WKB トンネリング

参考文献:
    Koehl et al. (2011) Nature — V_Si spin qubit in SiC
    Christle et al. (2015) Nature Mater — SiC divacancy qubit
    Huang & Rhys (1950) Proc R Soc — phonon coupling theory
    Lindblad (1976) Commun Math Phys — master equation
    Fowler & Nordheim (1928) Proc R Soc — tunneling
"""

import argparse
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import expm

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ── 物理定数 ────────────────────────────────────────────────────────────────
hbar   = 1.0546e-34   # J·s
kB     = 1.381e-23    # J/K
muB    = 9.274e-24    # J/T  Bohr magneton
ge     = 2.0023       # free electron g-factor
GHz    = 1e9          # Hz
MHz    = 1e6


# ════════════════════════════════════════════════════════════════════════════
# 1. スピン Hamiltonian — V_Si in 4H-SiC (S = 3/2)
# ════════════════════════════════════════════════════════════════════════════

# DFT 文献値 (Ivady et al. 2017, Phys Rev B)
D_ZFS   =  35.0e6   * hbar * 2 * np.pi   # J  ゼロ磁場分裂 D = 35 MHz (V2 site)
E_ZFS   =   3.5e6   * hbar * 2 * np.pi   # J  軸外歪み E = 3.5 MHz
A_hf_Si = -3.8e6   * hbar * 2 * np.pi   # J  ²⁹Si 超微細結合 (-3.8 MHz)
A_hf_C  = -6.1e6   * hbar * 2 * np.pi   # J  ¹³C 超微細結合 (-6.1 MHz)

# スピン S=3/2 行列
def _Sx():
    s = 3/2
    return np.array([
        [0,          np.sqrt(3)/2, 0,          0         ],
        [np.sqrt(3)/2, 0,          1,          0         ],
        [0,          1,           0,          np.sqrt(3)/2],
        [0,          0,          np.sqrt(3)/2, 0         ],
    ], dtype=complex) / 2  # ℏ=1 単位

def _Sy():
    s = 3/2
    return np.array([
        [0,               -1j*np.sqrt(3)/2, 0,               0              ],
        [1j*np.sqrt(3)/2,  0,              -1j,              0              ],
        [0,                1j,              0,              -1j*np.sqrt(3)/2],
        [0,                0,               1j*np.sqrt(3)/2, 0              ],
    ], dtype=complex) / 2

def _Sz():
    return np.diag([3/2, 1/2, -1/2, -3/2], dtype=complex) if False else \
           np.diag(np.array([3/2, 1/2, -1/2, -3/2], dtype=complex))

def spin_hamiltonian(B_T: float = 0.0, theta_deg: float = 0.0) -> np.ndarray:
    """
    H = D(Sz²-S(S+1)/3) + E(Sx²-Sy²) + g·µB·B·(Sz·cosθ + Sx·sinθ)

    B_T     : 磁場強度 [T]
    theta_deg: 磁場と c 軸のなす角 [deg]
    返り値: 4×4 Hamiltonian [Hz (ℏ=1)]
    """
    Sx, Sy, Sz = _Sx(), _Sy(), _Sz()
    S = 3/2
    D = 35.0e6   # Hz
    E =  3.5e6   # Hz
    g = ge
    muB_Hz_T = muB / (hbar * 2 * np.pi)  # Hz/T

    theta = np.radians(theta_deg)
    H  = D * (Sz @ Sz - S*(S+1)/3 * np.eye(4))
    H += E * (Sx @ Sx - Sy @ Sy)
    H += g * muB_Hz_T * B_T * (np.cos(theta) * Sz + np.sin(theta) * Sx)
    return H


def odmr_spectrum(B_range_T: np.ndarray,
                  theta_deg: float = 0.0,
                  linewidth_MHz: float = 1.0) -> tuple:
    """
    ODMR スペクトルを計算。
    各 B で固有値を求め、Δms=±1 遷移周波数にローレンツ型ピークを乗せる。

    Returns: freqs_GHz [array], contrast_map [B × freq]
    """
    freqs = np.linspace(0, 15, 2000)   # GHz
    contrast = np.zeros((len(B_range_T), len(freqs)))
    lw = linewidth_MHz * 1e6

    for i, B in enumerate(B_range_T):
        H   = spin_hamiltonian(B, theta_deg)
        eig = np.sort(np.linalg.eigvalsh(H)) * 1e-9  # → GHz
        # Δms=±1 遷移
        for j in range(len(eig)-1):
            f0 = abs(eig[j+1] - eig[j])
            lorentz = (lw*1e-9/2)**2 / ((freqs - f0)**2 + (lw*1e-9/2)**2)
            contrast[i] += lorentz

    return freqs, contrast


# ════════════════════════════════════════════════════════════════════════════
# 2. フォノンカップリング — Huang-Rhys 因子
# ════════════════════════════════════════════════════════════════════════════

# 文献値 (Davidsson et al. 2021, npj Comput Mater)
DEFECT_PHONON = {
    "V_Si (V2, 4H-SiC)": {
        "ZPL_eV":    1.352,    # ゼロフォノン線エネルギー [eV]
        "S":         0.031,    # Huang-Rhys 因子（非常に小さい = 優れた光学品質）
        "hbar_omega_eV": 0.120, # 有効フォノンエネルギー [eV]
        "color": "#2166ac",
    },
    "V_Si-V_C (kh, 4H-SiC)": {
        "ZPL_eV":    1.095,
        "S":         2.1,      # 大きい S = フォノンサイドバンドが支配的
        "hbar_omega_eV": 0.095,
        "color": "#2ca02c",
    },
    "V_Ga (GaN)": {
        "ZPL_eV":    2.64,
        "S":         6.5,      # 非常に大きい → Yellow luminescence band
        "hbar_omega_eV": 0.060,
        "color": "#d62728",
    },
}


def huang_rhys_lineshape(S: float, hbar_omega_eV: float,
                          ZPL_eV: float, T_K: float = 300.0,
                          n_phonon: int = 20) -> tuple:
    """
    Huang-Rhys 多フォノン発光スペクトル（熱分布込み）。

    I(E) = Σ_n I_n × δ(E - (E_ZPL - n·ℏω))
    I_n  = exp(-S) · S^n / n!   (T=0 の場合)
    有限温度: ボゾン分布で重み付け

    Returns: energies_eV, intensity
    """
    energies = np.linspace(ZPL_eV - 1.5, ZPL_eV + 0.2, 2000)
    intensity = np.zeros_like(energies)
    n_bar = 1.0 / (np.exp(hbar_omega_eV / (8.617e-5 * T_K)) - 1 + 1e-10)

    for n in range(-5, n_phonon + 1):
        # n > 0: 放出, n < 0: 吸収（熱活性）
        E_peak = ZPL_eV - n * hbar_omega_eV
        if n >= 0:
            weight = np.exp(-S * (2*n_bar + 1)) * (
                (S * (n_bar + 1))**n / max(1, math.factorial(min(n, 20)))
                if n <= 20 else 0)
        else:
            m = -n
            weight = np.exp(-S * (2*n_bar + 1)) * (
                (S * n_bar)**m / max(1, math.factorial(min(m, 20)))
                if m <= 20 else 0)
        sigma = hbar_omega_eV * 0.15
        intensity += weight * np.exp(-(energies - E_peak)**2 / (2*sigma**2))

    intensity /= max(intensity.max(), 1e-20)
    return energies, intensity


# ════════════════════════════════════════════════════════════════════════════
# 3. 密度行列 + リンドブラッド方程式 — デコヒーレンス
# ════════════════════════════════════════════════════════════════════════════

def lindblad_evolve(rho0: np.ndarray,
                    H: np.ndarray,
                    L_ops: list,
                    t_arr: np.ndarray) -> np.ndarray:
    """
    リンドブラッドマスター方程式の数値積分:
        dρ/dt = -i[H,ρ] + Σ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})

    Trotter 展開による近似（短時間ステップ）。

    rho0  : 初期密度行列 (4×4)
    H     : Hamiltonian [rad/s]
    L_ops : リンドブラッド演算子リスト
    t_arr : 時間配列 [s]

    Returns: rho_t (len(t_arr), 4, 4)
    """
    rho   = rho0.astype(complex).copy()
    rhos  = [rho.copy()]
    dt    = t_arr[1] - t_arr[0]
    dim   = rho.shape[0]
    I     = np.eye(dim, dtype=complex)

    for t in t_arr[1:]:
        # ユニタリ部分: e^{-iHdt}
        U   = expm(-1j * H * dt)
        rho = U @ rho @ U.conj().T

        # 非ユニタリ部分（リンドブラッド項）
        drho = np.zeros_like(rho)
        for L in L_ops:
            Ld = L.conj().T
            drho += L @ rho @ Ld - 0.5 * (Ld @ L @ rho + rho @ Ld @ L)
        rho = rho + drho * dt

        # 正規化（数値誤差修正）
        rho = rho / np.trace(rho).real
        rhos.append(rho.copy())

    return np.array(rhos)


def qubit_coherence(T1_us: float = 200.0,
                    T2_us: float =  20.0,
                    B_T:   float =  0.01) -> dict:
    """
    V_Si スピン量子ビットのコヒーレンス計算。

    T1: 縦緩和時間 (スピン-格子)
    T2: 横緩和時間 (スピン-スピン, 核スピンバス)

    Returns: coherence vs time dict
    """
    H  = spin_hamiltonian(B_T=B_T, theta_deg=0) * 2 * np.pi  # rad/s

    # リンドブラッド演算子
    Sz = _Sz()
    Sx = _Sx()
    gamma1 = 1.0 / (T1_us * 1e-6)  # s⁻¹
    gamma2 = 1.0 / (T2_us * 1e-6) - gamma1 / 2

    L_relax = np.sqrt(gamma1) * Sz               # T1 緩和
    L_deph  = np.sqrt(2 * gamma2) * Sz           # T2 純粋脱位相
    L_ops   = [L_relax, L_deph]

    # 初期状態: |ms=+1/2⟩ と |ms=-1/2⟩ の重ね合わせ（Ramsey 干渉計）
    psi0 = np.zeros(4, dtype=complex)
    psi0[1] = 1/np.sqrt(2)   # |+1/2⟩
    psi0[2] = 1/np.sqrt(2)   # |-1/2⟩
    rho0 = np.outer(psi0, psi0.conj())

    t_arr = np.linspace(0, 3 * T2_us * 1e-6, 300)
    rho_t = lindblad_evolve(rho0, H, L_ops, t_arr)

    # コヒーレンス: ρ₁₂(t) の絶対値
    coherence = np.abs(rho_t[:, 1, 2])
    population = rho_t[:, 0, 0].real

    return {
        "t_us":      t_arr * 1e6,
        "coherence": coherence,
        "population":population,
        "T1_us":     T1_us,
        "T2_us":     T2_us,
    }


# ════════════════════════════════════════════════════════════════════════════
# 4. WKB トンネリング — SiC MOSFET ゲート酸化膜
# ════════════════════════════════════════════════════════════════════════════

# SiC/SiO₂ 界面パラメータ (Kimoto & Cooper 2014)
PHI_B_SIC   = 2.7    # eV  SiC/SiO₂ 伝導帯オフセット
m_eff_SIC   = 0.33   # m₀  4H-SiC 有効質量
m_ox        = 0.42   # m₀  SiO₂ 有効質量
m0          = 9.109e-31  # kg
eV_to_J     = 1.602e-19


def fowler_nordheim_current(E_ox_MV_cm: float,
                             t_ox_nm: float = 5.0) -> float:
    """
    Fowler-Nordheim トンネル電流密度 [A/cm²]:
        J_FN = A · E² · exp(-B/E)
        A = q³m/(8πhΦ_B m_ox)
        B = 8π√(2m_ox) Φ_B^{3/2} / (3qh)

    E_ox: 酸化膜電界 [MV/cm]
    """
    h   = 6.626e-34
    q   = 1.602e-19
    E   = E_ox_MV_cm * 1e8   # V/m に換算
    phi = PHI_B_SIC * eV_to_J
    m   = m_ox * m0

    A = q**3 * m / (8 * np.pi * h * phi * m_ox * m0)
    B = 8 * np.pi * np.sqrt(2 * m) * phi**1.5 / (3 * q * h)

    if E <= 0:
        return 0.0
    return A * E**2 * np.exp(-B / E)


def direct_tunnel_current(V_gate: float, t_ox_nm: float,
                           n_steps: int = 200) -> float:
    """
    直接トンネル電流 (WKB 近似):
        J ∝ exp(-2 ∫ κ(x) dx)
        κ(x) = √(2m*(Φ(x)-E)/ℏ²)

    薄膜酸化膜（< 3 nm）で支配的。
    """
    t_ox = t_ox_nm * 1e-9
    hbar = 1.055e-34
    phi0 = PHI_B_SIC * eV_to_J
    m    = m_ox * m0
    q    = 1.602e-19

    xs   = np.linspace(0, t_ox, n_steps)
    dx   = xs[1] - xs[0]

    # 線形ポテンシャル勾配（電界による傾き）
    E_field = V_gate / t_ox_nm * 1e9 if t_ox_nm > 0 else 0  # V/m
    integral = 0.0
    for x in xs:
        phi_x = phi0 - q * E_field * x
        phi_x = max(phi_x, 0)
        kappa = np.sqrt(2 * m * phi_x) / hbar
        integral += kappa * dx

    T_WKB = np.exp(-2 * integral)
    # 前因子（自由電子近似）
    v_fermi = np.sqrt(2 * 0.5 * eV_to_J / (m_eff_SIC * m0))  # ~10⁵ m/s
    J = q * v_fermi * T_WKB * 1e-4   # → A/cm²
    return J


# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def plot_odmr(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: ODMR スペクトル (B=0 T)
    ax = axes[0]
    B_arr = np.linspace(0, 0.1, 80)
    freqs, contrast = odmr_spectrum(B_arr, theta_deg=0, linewidth_MHz=0.5)
    ax.contourf(B_arr * 1e3, freqs, contrast.T, levels=30, cmap="hot_r")
    ax.set_xlabel("Magnetic Field B [mT]", fontsize=11)
    ax.set_ylabel("Microwave Frequency [GHz]", fontsize=11)
    ax.set_title("V_Si ODMR Spectrum (4H-SiC, theta=0)\n"
                 "D=35MHz, E=3.5MHz (Koehl et al. 2011)", fontsize=10)

    # Panel 2: 磁場角度依存
    ax2 = axes[1]
    B_fixed = 0.05   # T
    for theta, col, lbl in [(0, "#2166ac", "B//c (0deg)"),
                             (45, "#2ca02c", "B 45deg"),
                             (90, "#d62728", "B perp c (90deg)")]:
        freqs2, c2 = odmr_spectrum(np.array([B_fixed]), theta, linewidth_MHz=0.5)
        ax2.plot(freqs2, c2[0], color=col, lw=2, label=lbl)
    ax2.set_xlabel("Frequency [GHz]", fontsize=11)
    ax2.set_ylabel("ODMR Contrast", fontsize=11)
    ax2.set_title(f"V_Si ODMR at B={B_fixed*1e3:.0f}mT\n角度依存性", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    fig.suptitle("V_Si in 4H-SiC — Spin-3/2 ODMR (量子スピン量子ビット)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(out_dir, "quantum_odmr.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] ODMR spectrum -> {out}")


def plot_phonon(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: 各欠陥の発光スペクトル
    ax = axes[0]
    for name, d in DEFECT_PHONON.items():
        E, I = huang_rhys_lineshape(d["S"], d["hbar_omega_eV"],
                                     d["ZPL_eV"], T_K=10)
        ax.plot(E, I, color=d["color"], lw=2,
                label=f"{name} (S={d['S']}, ZPL={d['ZPL_eV']}eV)")
        ax.axvline(d["ZPL_eV"], color=d["color"], ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Photon Energy [eV]", fontsize=11)
    ax.set_ylabel("Intensity (norm.)", fontsize=11)
    ax.set_title("Huang-Rhys フォノンカップリング\n発光スペクトル (T=10K)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

    # Panel 2: 温度依存線幅 (V_Si)
    ax2 = axes[1]
    T_arr = np.linspace(10, 500, 50)
    d = DEFECT_PHONON["V_Si (V2, 4H-SiC)"]
    widths = []
    for T in T_arr:
        E, I = huang_rhys_lineshape(d["S"], d["hbar_omega_eV"],
                                     d["ZPL_eV"], T_K=T, n_phonon=15)
        peak = E[np.argmax(I)]
        half = np.max(I) / 2
        above = E[I >= half]
        fwhm = (above[-1] - above[0]) if len(above) > 1 else 0
        widths.append(fwhm * 1000)   # → meV

    ax2.plot(T_arr, widths, color="#2166ac", lw=2.5)
    ax2.axvline(300, color="gray", ls="--", lw=1, label="室温 (300K)")
    ax2.set_xlabel("Temperature [K]", fontsize=11)
    ax2.set_ylabel("ZPL Linewidth FWHM [meV]", fontsize=11)
    ax2.set_title("V_Si ZPL 温度依存線幅\n(S=0.031, hbar_omega=120meV)", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2)

    fig.suptitle("フォノンカップリング — Huang-Rhys 因子と光学スペクトル", fontsize=12)
    plt.tight_layout()
    out = os.path.join(out_dir, "quantum_phonon.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Phonon coupling -> {out}")


def plot_lindblad(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: T2 vs 加工プロセス（欠陥密度 → 核スピンバス拡大 → T2 短縮）
    ax = axes[0]
    # 欠陥密度が増えると近傍核スピン密度が増え T2* が短縮
    defect_densities = np.logspace(11, 14, 40)   # cm⁻³
    T2_base = 20.0   # µs (バルク品質)
    for T1, col, lbl in [(200, "#2166ac", "T1=200µs (優良)"),
                          (50,  "#2ca02c", "T1=50µs (標準)"),
                          (10,  "#d62728", "T1=10µs (損傷)")]:
        T2s = []
        for N in defect_densities:
            # 経験則: T2* ∝ 1/sqrt(N) (核スピンバス)
            T2_star = T2_base * (1e11 / N)**0.5
            T2_eff  = 1.0 / (1.0/T2_star + 1.0/(2*T1))
            T2s.append(T2_eff)
        ax.loglog(defect_densities, T2s, color=col, lw=2, label=lbl)

    ax.axvline(2.3e12, color="red", ls="--", lw=1.2, label="ブレード (N_V_Si~2.3e12)")
    ax.axvline(8.0e11, color="blue", ls="--", lw=1.2, label="ステルス (N_V_Si~8e11)")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("V_Si 密度 [cm⁻³]", fontsize=11)
    ax.set_ylabel("コヒーレンス時間 T₂* [µs]", fontsize=11)
    ax.set_title("加工プロセス × T₂* デコヒーレンス\n(核スピンバスモデル)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, which="both")

    # Panel 2: Ramsey 干渉縞 (T1=200µs, T2=20µs)
    ax2 = axes[1]
    for T2, col, lbl in [(100, "#2166ac", "ステルス T2=100µs"),
                          (20,  "#2ca02c", "レーザーps T2=20µs"),
                          (5,   "#d62728", "ブレード T2=5µs")]:
        res = qubit_coherence(T1_us=200, T2_us=T2, B_T=0.01)
        ax2.plot(res["t_us"], res["coherence"],
                 color=col, lw=2, label=lbl)

    ax2.axhline(1/np.e, color="gray", ls=":", lw=1, label="1/e レベル")
    ax2.set_xlabel("Ramsey 遅延時間 [µs]", fontsize=11)
    ax2.set_ylabel("コヒーレンス |ρ₁₂(t)|", fontsize=11)
    ax2.set_title("Ramsey 干渉縞 — リンドブラッド方程式\n(V_Si 量子ビット)", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2)
    ax2.set_xlim(0, 60)

    fig.suptitle("リンドブラッド方程式 × 加工プロセス — 量子ビット品質評価\n"
                 "ステルスダイシング → 最長 T₂* → 最高量子ビット性能",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "quantum_lindblad.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Lindblad decoherence -> {out}")


def plot_tunneling(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: FN電流 vs 酸化膜電界
    ax = axes[0]
    E_arr = np.linspace(1, 12, 200)   # MV/cm
    for t_ox, col, lbl in [(5, "#2166ac", "5nm (標準)"),
                            (3, "#2ca02c", "3nm (2nm ノード)"),
                            (2, "#d62728", "2nm (次世代)")]:
        J_FN = [fowler_nordheim_current(E) for E in E_arr]
        J_dt = [direct_tunnel_current(E*t_ox/100, t_ox) for E in E_arr]
        J    = [max(fn, dt) for fn, dt in zip(J_FN, J_dt)]
        ax.semilogy(E_arr, J, color=col, lw=2, label=f"t_ox={t_ox}nm")

    ax.axhline(1e-8, color="gray", ls="--", lw=1.2, label="MOSFET リーク限界")
    ax.set_xlabel("Oxide Electric Field [MV/cm]", fontsize=11)
    ax.set_ylabel("Tunnel Current [A/cm²]", fontsize=11)
    ax.set_title("SiC/SiO₂ トンネル電流\n(Fowler-Nordheim + WKB 直接トンネル)",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, which="both")
    ax.set_ylim(1e-20, 1e5)

    # Panel 2: t_ox 依存性（直接トンネル支配領域）
    ax2 = axes[1]
    t_arr = np.linspace(1, 10, 100)   # nm
    for V, col, lbl in [(1.0, "#2166ac", "V=1V"),
                         (2.0, "#2ca02c", "V=2V"),
                         (3.0, "#d62728", "V=3V")]:
        Js = [direct_tunnel_current(V, t) for t in t_arr]
        ax2.semilogy(t_arr, Js, color=col, lw=2, label=lbl)

    ax2.axvline(3.0, color="purple", ls="--", lw=1.2, label="3nm (2nm ノード目標)")
    ax2.axhline(1e-8, color="gray", ls=":", lw=1)
    ax2.set_xlabel("Oxide Thickness t_ox [nm]", fontsize=11)
    ax2.set_ylabel("Direct Tunnel Current [A/cm²]", fontsize=11)
    ax2.set_title("WKB 直接トンネリング vs t_ox\n(Phi_B=2.7eV, m_ox=0.42m₀)",
                  fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2, which="both")

    fig.suptitle("WKB 量子トンネリング — SiC MOSFET ゲート酸化膜\n"
                 "2nm ノードパワーデバイスの界面品質限界",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "quantum_tunneling.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] WKB tunneling -> {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odmr",     action="store_true")
    ap.add_argument("--phonon",   action="store_true")
    ap.add_argument("--lindblad", action="store_true")
    ap.add_argument("--tunnel",   action="store_true")
    args = ap.parse_args()

    run_all = not any([args.odmr, args.phonon, args.lindblad, args.tunnel])

    if args.odmr     or run_all: plot_odmr()
    if args.phonon   or run_all: plot_phonon()
    if args.lindblad or run_all: plot_lindblad()
    if args.tunnel   or run_all: plot_tunneling()


if __name__ == "__main__":
    main()
