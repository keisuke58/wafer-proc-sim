"""
Quantum Advanced Model — 深層量子物理拡張
==========================================
quantum_spin_model.py の続き。さらに深い量子記述。

1. タイトバインディング バンド構造 (4H-SiC)
   最近接 sp³ TB モデル + 欠陥準位の挿入
   → Γ-M-K-Γ-A 高対称点バンド図

2. NEGF 量子輸送 (Non-Equilibrium Green's Function)
   Landauer-Büttiker + SiC/SiO₂ 界面散乱
   → WKB より厳密な量子トンネル電流

3. 量子センシング プロトコル
   V_Si を量子磁力計・温度計として使用
   → ODMR 周波数シフト → B・T・歪みの高感度検出
   → 加工プロセスによる感度差 (T₂* 依存)

4. 電子フォノン結合 (第二量子化 / Fröhlich Hamiltonian)
   ポーラロン質量・移動度への寄与
   → SiC の格子歪み → キャリア散乱 → デバイス性能

実行:
    python ml/quantum_advanced.py              # 全図
    python ml/quantum_advanced.py --bands      # バンド構造
    python ml/quantum_advanced.py --negf       # NEGF
    python ml/quantum_advanced.py --sensing    # 量子センシング
    python ml/quantum_advanced.py --polaron    # 電子フォノン

参考文献:
    Park et al. (2000) Phys Rev B — 4H-SiC tight-binding
    Datta (1995) Electronic Transport in Mesoscopic Systems — NEGF
    Degen et al. (2017) Rev Mod Phys — quantum sensing review
    Frohlich (1954) Adv Phys — polaron theory
"""

import argparse
import cmath
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.linalg import inv, eigh

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ── 物理定数 ────────────────────────────────────────────────────────────────
hbar   = 1.0546e-34   # J·s
kB     = 1.381e-23    # J/K
eV     = 1.602e-19    # J
muB    = 9.274e-24    # J/T
m0     = 9.109e-31    # kg
a_SiC  = 3.073e-10   # m  4H-SiC a 格子定数
c_SiC  = 10.053e-10  # m  4H-SiC c 格子定数


# ════════════════════════════════════════════════════════════════════════════
# 1. タイトバインディング バンド構造 — 4H-SiC
# ════════════════════════════════════════════════════════════════════════════
# 簡略化 sp³ モデル (Park et al. 2000 パラメータ)
# オンサイトエネルギーとホッピング積分

# sp³ 軌道のオンサイトエネルギー [eV]
E_s_Si = -7.03;  E_p_Si =  1.71   # Si サイト
E_s_C  = -15.5;  E_p_C  = -4.74   # C サイト

# 最近接ホッピング積分 [eV]  (Harrison's universal model 近似)
V_ss_sigma = -2.07
V_sp_sigma =  2.45
V_pp_sigma =  4.37
V_pp_pi    = -1.13


def _H_4HSiC_k(kx: float, ky: float, kz: float) -> np.ndarray:
    """
    4H-SiC の最近接 sp³ タイトバインディング Hamiltonian (8×8).
    ウルツァイト構造の 4-atom ユニットセル (Si2C2) を 2 ユニット重ねた簡略モデル。

    本来は 16×16 だが教育目的で 8×8 に圧縮 (2-band zone folding)。
    """
    # 最近接ベクトル (ウルツァイト基本)
    d = a_SiC / np.sqrt(3)
    nn = np.array([
        [ d,  0,       c_SiC/8],
        [-d/2, d*np.sqrt(3)/2, c_SiC/8],
        [-d/2,-d*np.sqrt(3)/2, c_SiC/8],
        [ 0,   0,      -3*c_SiC/8],
    ])

    dim = 8
    H   = np.zeros((dim, dim), dtype=complex)

    # オンサイト (sp³ ×2 サイト ×2 スピン なし → 4軌道×2サイト=8)
    H[0, 0] = E_s_Si; H[1, 1] = E_p_Si; H[2, 2] = E_p_Si; H[3, 3] = E_p_Si
    H[4, 4] = E_s_C;  H[5, 5] = E_p_C;  H[6, 6] = E_p_C;  H[7, 7] = E_p_C

    # ホッピング (簡略: s-s, s-p, p-p 結合のみ z 方向)
    for r in nn:
        phase = np.exp(1j * (kx*r[0] + ky*r[1] + kz*r[2]))
        # Si→C  s-s
        H[0, 4] += V_ss_sigma * phase
        H[4, 0] += V_ss_sigma * phase.conjugate()
        # Si→C  s-p
        H[0, 7] += V_sp_sigma * r[2]/np.linalg.norm(r) * phase
        H[7, 0] += V_sp_sigma * r[2]/np.linalg.norm(r) * phase.conjugate()
        # Si→C  p-p sigma (z 成分)
        H[3, 7] += V_pp_sigma * (r[2]/np.linalg.norm(r))**2 * phase
        H[7, 3] += V_pp_sigma * (r[2]/np.linalg.norm(r))**2 * phase.conjugate()
        # pi
        H[1, 5] += V_pp_pi * phase
        H[5, 1] += V_pp_pi * phase.conjugate()
        H[2, 6] += V_pp_pi * phase
        H[6, 2] += V_pp_pi * phase.conjugate()

    return H


def band_structure(n_k: int = 100) -> dict:
    """
    高対称点経路 Γ-M-K-Γ-A に沿ったバンド計算。
    欠陥なし + V_C Z1/2 欠陥準位 (E_c - 0.63 eV) を追加表示。
    """
    # 逆格子ベクトル (六方晶)
    b1 = 2*np.pi/a_SiC * np.array([1, -1/np.sqrt(3), 0])
    b2 = 2*np.pi/a_SiC * np.array([0,  2/np.sqrt(3), 0])
    b3 = 2*np.pi/c_SiC * np.array([0,  0,            1])

    G = np.array([0, 0, 0])
    M = (b1 + b2) / 2
    K = (2*b1 + b2) / 3
    A = b3 / 2

    # k パス
    paths = [
        ("G-M", G, M),
        ("M-K", M, K),
        ("K-G", K, G),
        ("G-A", G, A),
    ]
    k_vals, energies, labels, label_pos = [], [], [], []
    k_cum = 0.0

    for (name, k0, k1) in paths:
        labels.append(name.split("-")[0])
        label_pos.append(k_cum)
        ks = np.linspace(0, 1, n_k)
        for t in ks:
            k = k0 + t * (k1 - k0)
            H = _H_4HSiC_k(*k)
            eig = np.sort(np.linalg.eigvalsh(H))
            energies.append(eig)
            k_vals.append(k_cum + t * np.linalg.norm(k1 - k0))
        k_cum += np.linalg.norm(k1 - k0)

    labels.append("A")
    label_pos.append(k_cum)

    return {
        "k":          np.array(k_vals),
        "E":          np.array(energies),
        "labels":     labels,
        "label_pos":  label_pos,
        "Eg":         3.26,     # 実験値 [eV]
        "defect_levels": {
            "V_C Z1/2":  3.26 - 0.63,   # eV (VBM 基準)
            "V_Si":      3.26 - 0.44,
            "V_Si-V_C":  3.26 - 1.55,
        },
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. NEGF 量子輸送
# ════════════════════════════════════════════════════════════════════════════

def negf_transmission(E_arr: np.ndarray,
                       t_ox_nm: float = 5.0,
                       phi_B_eV: float = 2.7,
                       V_bias: float = 0.0,
                       n_layers: int = 10) -> np.ndarray:
    """
    SiC/SiO₂/Metal 接合の NEGF 透過係数 T(E)。

    離散化した 1D 酸化膜領域を有限差分 Hamiltonian で表現し、
    接触自己エネルギー Σ を通じてリード電極に接続。
    Landauer 公式: J = (2e/h) ∫ T(E) [f_L - f_R] dE

    n_layers: 酸化膜離散化数 (精度と速度のトレードオフ)
    """
    t_ox = t_ox_nm * 1e-9
    dz   = t_ox / n_layers
    m_ox = 0.42 * m0

    # ホッピング積分
    t0   = hbar**2 / (2 * m_ox * dz**2) / eV   # eV

    T_arr = []
    for E in E_arr:
        # オンサイトエネルギー（線形ポテンシャル + バリア）
        on_site = np.array([
            phi_B_eV - V_bias * i/n_layers
            for i in range(n_layers)
        ])

        # ハミルトニアン行列 (tridiagonal)
        H_ox = np.diag(on_site + 2*t0) - np.diag([t0]*(n_layers-1), 1) \
               - np.diag([t0]*(n_layers-1), -1)

        # グリーン関数 G = (E - H - Σ_L - Σ_R)^{-1}
        # 自己エネルギー（半無限鎖の解析式）
        eta  = 1e-5   # 収束因子
        k_L  = np.arccos(np.clip(1 - (E + 1j*eta)/(2*t0), -1, 1))
        k_R  = np.arccos(np.clip(1 - (E - V_bias + 1j*eta)/(2*t0), -1, 1))
        Sig_L = -t0 * np.exp(1j * k_L)
        Sig_R = -t0 * np.exp(1j * k_R)

        Sigma = np.zeros((n_layers, n_layers), dtype=complex)
        Sigma[0, 0]   += Sig_L
        Sigma[-1, -1] += Sig_R

        G = inv((E + 1j*eta) * np.eye(n_layers) - H_ox - Sigma)

        # 結合行列 Γ = i(Σ - Σ†)
        Gam_L = np.zeros((n_layers, n_layers), dtype=complex)
        Gam_R = np.zeros((n_layers, n_layers), dtype=complex)
        Gam_L[0, 0]   = -2 * Sig_L.imag
        Gam_R[-1, -1] = -2 * Sig_R.imag

        # 透過係数 T = Tr[Γ_L G Γ_R G†]
        T = np.real(np.trace(Gam_L @ G @ Gam_R @ G.conj().T))
        T_arr.append(max(T, 0))

    return np.array(T_arr)


def negf_current(V_range: np.ndarray, t_ox_nm: float = 5.0,
                  T_K: float = 300.0) -> np.ndarray:
    """
    J(V) = (2e/h) ∫ T(E,V) [f(E) - f(E-eV)] dE  [A/cm²]
    フェルミ分布込みの Landauer 電流。
    """
    E_arr = np.linspace(-1.0, 4.0, 80)
    J_arr = []
    e_h   = 2 * eV / (2 * np.pi * hbar) * 1e-4   # 2e/h [A/eV · cm⁻²]

    for V in V_range:
        T_E = negf_transmission(E_arr, t_ox_nm=t_ox_nm, V_bias=V)
        # フェルミ分布
        f_L = 1.0 / (1 + np.exp(E_arr / (kB*T_K/eV)))
        f_R = 1.0 / (1 + np.exp((E_arr - V) / (kB*T_K/eV)))
        dE  = E_arr[1] - E_arr[0]
        J   = e_h * np.trapz(T_E * (f_L - f_R), dx=dE)
        J_arr.append(abs(J))

    return np.array(J_arr)


# ════════════════════════════════════════════════════════════════════════════
# 3. 量子センシング プロトコル
# ════════════════════════════════════════════════════════════════════════════

def sensing_sensitivity(T2_us: float, B0_T: float = 0.01,
                         freq_Hz: float = 1e3,
                         N_avg: int = 1000) -> dict:
    """
    V_Si 量子磁力計の磁場感度 δB [T/√Hz]:
        δB = hbar / (g·µB·T2·√(η·τ_meas))

    T2_us  : コヒーレンス時間 [µs]
    B0_T   : 静磁場 [T]
    freq_Hz: 測定周波数帯域 [Hz]
    N_avg  : アンサンブル平均数 (スピンアレイ密度依存)

    Returns: 感度 dict (磁場・温度・歪み)
    """
    T2  = T2_us * 1e-6
    g   = 2.0
    tau = T2                      # Ramsey 最適時間 = T2/2 以上

    # 磁場感度 [T/√Hz]
    delta_B = hbar / (g * muB * T2 * math.sqrt(tau * freq_Hz * N_avg))

    # 温度感度 [K/√Hz] (dD/dT ≈ -74 kHz/K for V_Si)
    dD_dT   = 74e3   # Hz/K
    delta_T = 1.0 / (2 * math.pi * dD_dT * T2 * math.sqrt(tau * freq_Hz * N_avg))

    # 歪み感度 (dD/dε ≈ 1 MHz / 0.1% strain = 1 GHz/strain)
    dD_deps  = 1e9   # Hz/strain
    delta_eps = 1.0 / (2 * math.pi * dD_deps * T2 * math.sqrt(tau * freq_Hz * N_avg))

    return {
        "T2_us":       T2_us,
        "delta_B_nT":  delta_B * 1e9,      # nT/√Hz
        "delta_T_mK":  delta_T * 1e3,      # mK/√Hz
        "delta_eps":   delta_eps,           # strain/√Hz
    }


def odmr_B_sensitivity_map(process_list: list) -> list:
    """各加工プロセスの T2* → 量子センシング感度 テーブル。"""
    # quantum_defect_model の結果を再現
    T2_map = {
        "ブレード (blade)":    5.0,
        "レーザー ns":        12.0,
        "レーザー ps":        20.0,
        "レーザー fs":        45.0,
        "ステルス (stealth)": 100.0,
        "バルク (no process)": 200.0,
    }
    results = []
    for proc in process_list:
        T2 = T2_map.get(proc, 20.0)
        s  = sensing_sensitivity(T2)
        results.append({"process": proc, **s})
    return results


# ════════════════════════════════════════════════════════════════════════════
# 4. 電子フォノン結合 — Fröhlich Hamiltonian / ポーラロン
# ════════════════════════════════════════════════════════════════════════════

# 4H-SiC パラメータ
eps_inf_SiC = 6.52    # 高周波誘電率
eps_0_SiC   = 9.66    # 静的誘電率
omega_LO    = 0.120   # eV  LO フォノンエネルギー
m_c_SiC     = 0.33    # m₀  伝導帯有効質量


def frohlich_coupling(eps_inf: float = eps_inf_SiC,
                       eps_s:   float = eps_0_SiC,
                       omega_LO_eV: float = omega_LO,
                       m_eff: float = m_c_SiC) -> float:
    """
    Fröhlich 結合定数 α (無次元):
        α = e²/(4πε₀) × (1/ε_∞ - 1/ε_s) × (1/ℏω_LO) × √(m*ω_LO/(2ℏ))

    α < 1: 弱結合 (SiC~0.1, GaAs~0.07)
    α > 6: 強結合 (イオン性結晶)
    """
    eps0  = 8.854e-12          # F/m
    omega = omega_LO_eV * eV / hbar   # rad/s
    m     = m_eff * m0
    ke    = 1.0 / (4 * math.pi * eps0)   # N·m²/C²
    alpha = (ke * eV**2
             * (1/eps_inf - 1/eps_s)
             / (hbar * omega)
             * math.sqrt(m * omega / (2 * hbar)))
    return alpha


def polaron_mass(alpha: float) -> float:
    """
    ポーラロン有効質量 (Feynman 摂動):
        m*_pol/m* = 1 + α/6 + 0.025α² + ...  (弱結合)
    """
    return 1 + alpha/6 + 0.025*alpha**2


def polaron_mobility(T_K: float, alpha: float,
                      omega_LO_eV: float = omega_LO,
                      m_eff: float = m_c_SiC) -> float:
    """
    Fröhlich ポーラロン移動度 [cm²/Vs] (Lee-Low-Pines 弱結合):
        µ = (e/m*) × (1/alpha) × (1/ω_LO) × exp(ℏω_LO/kT) - 1)
    簡略版 (Feynman variational)。
    """
    omega   = omega_LO_eV * eV / hbar
    m       = m_eff * m0 * polaron_mass(alpha)
    x       = omega_LO_eV / (kB * T_K / eV)
    tau_inv = alpha * omega * (math.exp(x) - 1) ** (-1) * 2 / 3
    if tau_inv <= 0:
        return float("inf")
    mu = eV / (m * tau_inv) * 1e-4   # cm²/Vs
    return mu


# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def plot_bands(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    bd = band_structure(n_k=60)
    E  = bd["E"]
    k  = bd["k"]

    # VBM を 0 に合わせる（最大占有バンド）
    n_occ = E.shape[1] // 2
    E_VBM = E[:, n_occ-1].max()
    E_shifted = E - E_VBM

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = plt.cm.coolwarm(np.linspace(0, 1, E.shape[1]))

    for i, col in enumerate(colors):
        ax.plot(k, E_shifted[:, i], color=col, lw=1.0, alpha=0.8)

    # バンドギャップ
    ax.axhspan(0, bd["Eg"], alpha=0.06, color="gray", label=f"Gap {bd['Eg']}eV")
    ax.axhline(0,        color="black", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(bd["Eg"], color="black", ls="--", lw=0.8, alpha=0.5)

    # 欠陥準位
    dl_colors = {"V_C Z1/2": "#d62728", "V_Si": "#2166ac", "V_Si-V_C": "#2ca02c"}
    for name, lev in bd["defect_levels"].items():
        ax.axhline(lev - E_VBM, color=dl_colors[name], ls=":", lw=1.8,
                   label=f"{name} ({lev - E_VBM:.2f}eV)")

    # 高対称点ラベル
    sym_labels = {"G": "Γ", "M": "M", "K": "K", "A": "A"}
    for lbl, pos in zip(bd["labels"], bd["label_pos"]):
        ax.axvline(pos, color="gray", ls="-", lw=0.5, alpha=0.4)
        ax.text(pos, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -8,
                sym_labels.get(lbl, lbl), ha="center", fontsize=11,
                fontweight="bold")

    ax.set_ylabel("Energy (eV, VBM=0)", fontsize=11)
    ax.set_title("4H-SiC Band Structure — sp³ Tight-Binding\n"
                 "with Point Defect Levels (V_C, V_Si, V_Si-V_C)", fontsize=10)
    ax.set_ylim(-8, 12)
    ax.set_xlim(k[0], k[-1])
    ax.set_xticks(bd["label_pos"])
    ax.set_xticklabels([sym_labels.get(l, l) for l in bd["labels"]], fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.15, axis="y")

    plt.tight_layout()
    out = os.path.join(out_dir, "quantum_bands_SiC.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Band structure -> {out}")


def plot_negf(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: T(E) スペクトル 厚さ比較
    ax = axes[0]
    E_arr = np.linspace(-1.0, 4.0, 100)
    for t_ox, col in [(8, "#2166ac"), (5, "#2ca02c"), (3, "#d62728")]:
        T_E = negf_transmission(E_arr, t_ox_nm=t_ox, V_bias=0.5)
        ax.semilogy(E_arr, np.clip(T_E, 1e-20, 1),
                    color=col, lw=2, label=f"t_ox={t_ox}nm")
    ax.axvline(2.7, color="gray", ls="--", lw=1, label="Barrier top Φ_B=2.7eV")
    ax.set_xlabel("Energy E [eV]", fontsize=11)
    ax.set_ylabel("Transmission T(E)", fontsize=11)
    ax.set_title("NEGF 透過係数 T(E)\nSiC/SiO₂/Metal, V_bias=0.5V", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, which="both")

    # Panel 2: J(V) 特性 WKB vs NEGF
    ax2 = axes[1]
    V_arr = np.linspace(0.1, 5.0, 40)
    for t_ox, col, ls in [(5, "#2166ac", "-"), (3, "#d62728", "-")]:
        J_negf = negf_current(V_arr, t_ox_nm=t_ox)
        ax2.semilogy(V_arr, np.clip(J_negf, 1e-25, 1e10),
                     color=col, lw=2.5, ls=ls, label=f"NEGF t={t_ox}nm")

    # WKB 比較 (quantum_spin_model の結果と比較)
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from quantum_spin_model import fowler_nordheim_current
        for t_ox, col in [(5, "#2166ac"), (3, "#d62728")]:
            J_fn = [abs(fowler_nordheim_current(V/t_ox*10, t_ox)) for V in V_arr]
            ax2.semilogy(V_arr, np.clip(J_fn, 1e-25, 1e10),
                         color=col, lw=1.5, ls="--",
                         label=f"FN (WKB) t={t_ox}nm")
    except Exception:
        pass

    ax2.axhline(1e-8, color="gray", ls=":", lw=1.2, label="リーク限界")
    ax2.set_xlabel("Gate Voltage [V]", fontsize=11)
    ax2.set_ylabel("Tunnel Current [A/cm²]", fontsize=11)
    ax2.set_title("NEGF vs WKB 電流比較\n(実線=NEGF, 破線=Fowler-Nordheim)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2, which="both")
    ax2.set_ylim(1e-20, 1e5)

    fig.suptitle("NEGF 量子輸送 — SiC/SiO₂ 界面 (Landauer-Büttiker)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(out_dir, "quantum_negf.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] NEGF transport -> {out}")


def plot_sensing(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    procs = ["ブレード (blade)", "レーザー ns", "レーザー ps",
             "レーザー fs", "ステルス (stealth)", "バルク (no process)"]
    results = odmr_B_sensitivity_map(procs)

    T2_vals   = [r["T2_us"]    for r in results]
    dB_vals   = [r["delta_B_nT"] for r in results]
    dT_vals   = [r["delta_T_mK"] for r in results]
    colors    = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(procs)))

    # Panel 1: 磁場感度
    ax = axes[0]
    bars = ax.barh(procs, dB_vals, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_xlabel("Magnetic Sensitivity δB [nT/√Hz]", fontsize=11)
    ax.set_title("V_Si 量子磁力計感度\nvs 加工プロセス (T2* 依存)", fontsize=10)
    ax.axvline(1.0, color="purple", ls="--", lw=1.5, label="1nT 目標")
    for bar, val in zip(bars, dB_vals):
        ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, axis="x")

    # Panel 2: T2* vs 3種の感度
    ax2 = axes[1]
    T2_range = np.logspace(0, 3, 100)
    ax2.loglog(T2_range,
               [sensing_sensitivity(t)["delta_B_nT"]  for t in T2_range],
               "#2166ac", lw=2, label="δB [nT/√Hz]")
    ax2.loglog(T2_range,
               [sensing_sensitivity(t)["delta_T_mK"]  for t in T2_range],
               "#d62728", lw=2, label="δT [mK/√Hz]")
    ax2.loglog(T2_range,
               [sensing_sensitivity(t)["delta_eps"]*1e6 for t in T2_range],
               "#2ca02c", lw=2, label="δε [µε/√Hz]")

    # 各プロセスの T2* をマーク
    T2_proc = {"blade":5, "ps":20, "fs":45, "stealth":100, "bulk":200}
    for name, T2 in T2_proc.items():
        ax2.axvline(T2, color="gray", ls=":", lw=0.8, alpha=0.7)
        ax2.text(T2*1.05, 1e-1, name, fontsize=7, color="gray", rotation=90)

    ax2.set_xlabel("T₂* [µs]", fontsize=11)
    ax2.set_ylabel("Sensitivity (nT or mK or µε / √Hz)", fontsize=11)
    ax2.set_title("量子センサ感度 vs T₂*\n(N=1000 スピンアンサンブル)", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2, which="both")

    fig.suptitle("V_Si 量子センシング — 加工プロセスが量子センサ精度を決定する\n"
                 "ステルスダイシング → 最高感度 (最長 T₂*)", fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "quantum_sensing.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    # テキスト表
    print(f"\n  {'プロセス':<22} {'T2 [µs]':>8}  {'δB [nT]':>10}  {'δT [mK]':>10}")
    print("  " + "-"*55)
    for r in results:
        print(f"  {r['process']:<22} {r['T2_us']:>8.0f}  "
              f"{r['delta_B_nT']:>10.3f}  {r['delta_T_mK']:>10.4f}")
    print(f"[✓] Quantum sensing -> {out}")


def plot_polaron(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    alpha = frohlich_coupling()
    print(f"  Frohlich alpha (4H-SiC) = {alpha:.3f}  "
          f"({'弱結合' if alpha < 1 else '中結合'})")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: 移動度 vs 温度 (材料比較)
    ax = axes[0]
    T_arr = np.linspace(100, 800, 100)
    materials = {
        "4H-SiC":  (eps_inf_SiC, eps_0_SiC, 0.120, 0.33, "#2166ac"),
        "GaN":     (5.35, 8.90, 0.092, 0.22, "#d62728"),
        "GaAs":    (10.9, 12.9, 0.036, 0.063, "#2ca02c"),
    }
    for name, (ei, es, wlo, me, col) in materials.items():
        a    = frohlich_coupling(ei, es, wlo, me)
        mus  = [polaron_mobility(T, a, wlo, me) for T in T_arr]
        ax.semilogy(T_arr, mus, color=col, lw=2.5, label=f"{name} (α={a:.2f})")

    ax.set_xlabel("Temperature [K]", fontsize=11)
    ax.set_ylabel("Polaron Mobility [cm²/Vs]", fontsize=11)
    ax.set_title("Frohlich ポーラロン移動度\n材料比較 (Feynman variational)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, which="both")
    ax.axvline(300, color="gray", ls="--", lw=1, label="RT")

    # Panel 2: 格子歪み → ポーラロン質量変化 → 移動度劣化
    ax2 = axes[1]
    strain_pct = np.linspace(0, 2.0, 60)
    # 歪みによる LO フォノン軟化: ω_LO → ω_LO × (1 - γ × ε) (Gruneisen パラメータ γ≈1)
    gamma = 1.0
    for T, col, lbl in [(300, "#2166ac", "300K"),
                         (500, "#2ca02c", "500K"),
                         (700, "#d62728", "700K")]:
        mus = []
        for eps in strain_pct / 100:
            wlo_eff = omega_LO * (1 - gamma * eps)
            a_eff   = frohlich_coupling(omega_LO_eV=wlo_eff)
            mu      = polaron_mobility(T, a_eff, wlo_eff)
            mus.append(mu)
        ax2.plot(strain_pct, mus, color=col, lw=2, label=lbl)

    ax2.set_xlabel("Lattice Strain [%]", fontsize=11)
    ax2.set_ylabel("Polaron Mobility [cm²/Vs]", fontsize=11)
    ax2.set_title("加工歪み × ポーラロン移動度\n(LO フォノン軟化モデル, 4H-SiC)", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2)
    ax2.axvline(0.1, color="purple", ls="--", lw=1.2, label="典型ダイシング歪み")

    fig.suptitle("Frohlich 電子フォノン結合 (第二量子化)\n"
                 "ポーラロン質量・移動度 — 加工誘起格子歪みの影響",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "quantum_polaron.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Polaron mobility -> {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands",   action="store_true")
    ap.add_argument("--negf",    action="store_true")
    ap.add_argument("--sensing", action="store_true")
    ap.add_argument("--polaron", action="store_true")
    args = ap.parse_args()

    run_all = not any([args.bands, args.negf, args.sensing, args.polaron])

    if args.bands   or run_all: plot_bands()
    if args.negf    or run_all: plot_negf()
    if args.sensing or run_all: plot_sensing()
    if args.polaron or run_all: plot_polaron()


if __name__ == "__main__":
    main()
