"""
Quantum Topology & Many-Body — 最深層量子物理
=============================================
quantum_advanced.py のさらなる拡張。

1. Berry 位相 / Zak 位相 / トポロジカル不変量 (4H-SiC)
   Wilson ループ法による Chern 数 + 分極計算

2. 量子もつれ + 2量子ビットゲート
   V_Si ペアの CNOT / iSWAP ゲート + Bell 状態生成
   エンタングルメント・エントロピー (von Neumann)

3. Keldysh 非平衡グリーン関数
   レーザー照射後のキャリアダイナミクス
   占有数分布の時間発展 (非平衡)

4. 量子誤り訂正 (表面符号)
   V_Si 量子ビットアレイの論理量子ビット
   加工品質 (T2*) → 論理エラー率

実行:
    python ml/quantum_topology.py
    python ml/quantum_topology.py --berry
    python ml/quantum_topology.py --entangle
    python ml/quantum_topology.py --keldysh
    python ml/quantum_topology.py --qec

参考文献:
    King-Smith & Vanderbilt (1993) Phys Rev B — Berry phase polarization
    TKNN (1982) Phys Rev Lett — Chern number
    Gottesman (1997) Stabilizer codes — QEC
    Keldysh (1965) JETP — non-equilibrium GF
    Loss & DiVincenzo (1998) Phys Rev A — spin qubit gates
"""

import argparse
import cmath
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.linalg import expm, logm

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.dirname(__file__))

# 物理定数
hbar = 1.0546e-34
eV   = 1.602e-19
kB   = 1.381e-23


# ════════════════════════════════════════════════════════════════════════════
# 1. Berry 位相 / Zak 位相 / Chern 数
# ════════════════════════════════════════════════════════════════════════════

def _bloch_hamiltonian_2band(kx: float, ky: float,
                              t1: float = 1.0,
                              t2: float = 0.3,
                              delta: float = 1.5) -> np.ndarray:
    """
    2バンド モデル Hamiltonian (SiC ウルツァイト構造の簡略版):
        H(k) = d(k) · σ
        d_x = t1(cos kx + cos ky) + t2 cos(kx+ky)
        d_y = t1(sin kx + sin ky) + t2 sin(kx+ky)
        d_z = Δ (onsite asymmetry, SiC の Si/C 違いを模擬)

    これは Haldane モデルの変形で位相的に非自明な相を示す。
    """
    dx = t1*(math.cos(kx) + math.cos(ky)) + t2*math.cos(kx+ky)
    dy = t1*(math.sin(kx) + math.sin(ky)) + t2*math.sin(kx+ky)
    dz = delta
    # パウリ行列
    sx = np.array([[0,1],[1,0]], dtype=complex)
    sy = np.array([[0,-1j],[1j,0]], dtype=complex)
    sz = np.array([[1,0],[0,-1]], dtype=complex)
    return dx*sx + dy*sy + dz*sz


def zak_phase(kpath_x: np.ndarray, kpath_y: float = 0.0,
              n_band: int = 0) -> float:
    """
    Zak 位相 (1D Berry 位相) の Wilson ループ計算:
        γ = Im ln ∏_k ⟨u_k|u_{k+1}⟩

    0 or π の量子化値 → トポロジカル不変量
    """
    overlaps = []
    states   = []
    for kx in kpath_x:
        H   = _bloch_hamiltonian_2band(kx, kpath_y)
        vals, vecs = np.linalg.eigh(H)
        states.append(vecs[:, n_band])

    # 周期的境界: 最後のベクトルを最初に戻す
    states.append(states[0])

    prod = 1.0 + 0j
    for i in range(len(states)-1):
        u1 = states[i];  u2 = states[i+1]
        prod *= np.vdot(u1, u2)

    zak = -np.angle(prod)   # rad  → 0 or π
    return float(zak)


def chern_number(n_k: int = 40, n_band: int = 0) -> int:
    """
    Chern 数の格子 Berry 曲率計算:
        C = (1/2π) Σ_{k} Ω(k) Δkx Δky
        Ω(k) = Im[∂_x⟨u|∂_y|u⟩ - ∂_y⟨u|∂_x|u⟩]

    Wilson ループ法 (Fukui-Hatsugai-Suzuki 2005):
        F(k) = ln(U_x U_y U_x† U_y†)
        C = (1/2πi) Σ F(k)
    """
    dk  = 2*math.pi / n_k
    ks  = np.linspace(-math.pi, math.pi, n_k, endpoint=False)

    def u(kx, ky):
        H = _bloch_hamiltonian_2band(kx, ky)
        _, vecs = np.linalg.eigh(H)
        return vecs[:, n_band]

    C = 0.0
    for i, kx in enumerate(ks):
        for j, ky in enumerate(ks):
            u00 = u(kx,     ky)
            u10 = u(kx+dk,  ky)
            u01 = u(kx,     ky+dk)
            u11 = u(kx+dk,  ky+dk)
            Ux  = np.vdot(u00, u10) / abs(np.vdot(u00, u10) + 1e-20)
            Uy  = np.vdot(u10, u11) / abs(np.vdot(u10, u11) + 1e-20)
            Ux_ = np.vdot(u01, u11) / abs(np.vdot(u01, u11) + 1e-20)
            Uy_ = np.vdot(u00, u01) / abs(np.vdot(u00, u01) + 1e-20)
            F   = cmath.log(Ux * Uy * Ux_.conjugate() * Uy_.conjugate())
            C  += F.imag
    return int(round(C / (2*math.pi)))


# ════════════════════════════════════════════════════════════════════════════
# 2. 量子もつれ + 2量子ビットゲート
# ════════════════════════════════════════════════════════════════════════════

# パウリ行列 (2量子ビット用)
I2 = np.eye(2, dtype=complex)
sx = np.array([[0,1],[1,0]], dtype=complex)
sy = np.array([[0,-1j],[1j,0]], dtype=complex)
sz = np.array([[1,0],[0,-1]], dtype=complex)


def kron(*ops):
    result = ops[0]
    for o in ops[1:]:
        result = np.kron(result, o)
    return result


def isw_gate(theta: float) -> np.ndarray:
    """
    iSWAP ゲート (交換相互作用 J に由来):
        U_iSWAP(θ) = exp(-iθ (XX + YY)/2)
    V_Si ペアのスピン交換で自然に実現。
    θ=π/2 で完全 iSWAP。
    """
    H_int = (kron(sx,sx) + kron(sy,sy)) / 2
    return expm(-1j * theta * H_int)


def cnot_from_isw() -> np.ndarray:
    """
    iSWAP から CNOT を構成:
        CNOT = Rz(π/2)⊗I × iSWAP(π/2) × I⊗Rx(-π/2) × iSWAP(-π/2) × Rz(-π/2)⊗I
    """
    def Rz(angle): return expm(-1j*angle/2*sz)
    def Rx(angle): return expm(-1j*angle/2*sx)

    U = (kron(Rz(math.pi/2), I2)
         @ isw_gate(math.pi/2)
         @ kron(I2, Rx(-math.pi/2))
         @ isw_gate(-math.pi/2)
         @ kron(Rz(-math.pi/2), I2))
    return U


def von_neumann_entropy(rho: np.ndarray) -> float:
    """
    von Neumann エントロピー: S = -Tr[ρ ln ρ]
    エンタングルメントの定量化。最大 ln(2) ≈ 0.693 (Bell 状態)
    """
    eigs = np.linalg.eigvalsh(rho)
    eigs = np.clip(eigs.real, 1e-15, 1)
    return float(-np.sum(eigs * np.log(eigs)))


def entanglement_entropy_vs_fidelity(T2_us_arr: np.ndarray,
                                      gate_time_ns: float = 100.0) -> dict:
    """
    T2* が異なる各プロセスで、iSWAP ゲート後の Bell 状態エントロピーを計算。
    デコヒーレンスによってエンタングルメントが失われる様子を定量化。
    """
    # Bell 状態 |Φ+⟩ = (|00⟩ + |11⟩)/√2
    bell = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
    rho_bell = np.outer(bell, bell.conj())

    S_vals = []
    fid_vals = []

    for T2 in T2_us_arr:
        # デコヒーレンス率
        gamma = 1.0 / (T2 * 1e-6)   # s⁻¹
        t_gate = gate_time_ns * 1e-9  # s

        # 単純モデル: 脱位相により非対角要素が指数減衰
        decay  = math.exp(-gamma * t_gate)
        rho_d  = rho_bell.copy()
        # オフダイアゴナルを減衰
        for i in range(4):
            for j in range(4):
                if i != j:
                    rho_d[i, j] *= decay

        # 固有値を正規化
        rho_d = rho_d / np.trace(rho_d)

        # 部分系 A (第1量子ビット) のエントロピー
        rho_A = np.array([
            [rho_d[0,0] + rho_d[1,1], rho_d[0,2] + rho_d[1,3]],
            [rho_d[2,0] + rho_d[3,1], rho_d[2,2] + rho_d[3,3]],
        ])
        S_vals.append(von_neumann_entropy(rho_A))

        # 忠実度 F = Tr[ρ_bell ρ_d]
        fid_vals.append(float(np.trace(rho_bell @ rho_d).real))

    return {"T2_us": T2_us_arr, "entropy": np.array(S_vals),
            "fidelity": np.array(fid_vals)}


# ════════════════════════════════════════════════════════════════════════════
# 3. Keldysh 非平衡グリーン関数 — 照射後キャリアダイナミクス
# ════════════════════════════════════════════════════════════════════════════

def keldysh_carrier_dynamics(E_arr: np.ndarray,
                              pump_energy_eV: float = 4.0,
                              gamma_e_eV:    float = 0.05,
                              gamma_ph_eV:   float = 0.10,
                              t_arr_ps:      np.ndarray = None) -> dict:
    """
    Keldysh 形式の非平衡分布関数 f(E, t) の簡略モデル。

    レーザーパルス (E_pump) が価電子帯から伝導帯にキャリアを励起。
    その後:
        - 電子-電子散乱 (γ_e): キャリアを熱化 (Fermi-Dirac へ緩和)
        - フォノン放出 (γ_ph): エネルギーを格子に渡し冷却

    非平衡 GF の lesser 成分:
        G^< (E, t) ∝ f(E, t) A(E)
        A(E) = スペクトル関数 (ローレンツ型)

    Returns: f(E, t) 分布関数の時間発展
    """
    if t_arr_ps is None:
        t_arr_ps = np.linspace(0, 5, 80)

    Eg_SiC = 3.26   # eV
    T_lat  = 300    # K  格子温度

    # スペクトル関数 A(E) (単純ローレンツ)
    gamma = 0.02   # eV
    A = gamma / ((E_arr - Eg_SiC/2)**2 + gamma**2) / math.pi

    f_t = []
    for t_ps in t_arr_ps:
        t = t_ps * 1e-12   # s
        t_e  = 0.2e-12     # 電子-電子散乱時定数 0.2ps
        t_ph = 1.5e-12     # フォノン散乱時定数 1.5ps

        # 初期非平衡分布: ポンプエネルギーにガウスピーク
        f0_ne = np.exp(-(E_arr - pump_energy_eV)**2 / (2 * 0.15**2))
        # Fermi-Dirac (熱化後)
        T_hot  = 3000 * math.exp(-t/t_ph) + T_lat   # ホットキャリア温度
        mu_hot = Eg_SiC + 0.5 * math.exp(-t/t_e)    # 化学ポテンシャル
        kT     = 8.617e-5 * T_hot
        f_fd   = 1.0 / (1 + np.exp((E_arr - mu_hot) / kT))

        # 混合: 非平衡成分が指数減衰しながら Fermi-Dirac に収束
        alpha_e  = math.exp(-t/t_e)
        alpha_ph = math.exp(-t/t_ph)
        f = (alpha_e * f0_ne * alpha_ph
             + (1 - alpha_e) * f_fd * (E_arr > Eg_SiC))
        f = np.clip(f, 0, 1)
        f_t.append(f)

    return {"E": E_arr, "t_ps": t_arr_ps, "f": np.array(f_t)}


# ════════════════════════════════════════════════════════════════════════════
# 4. 量子誤り訂正 (表面符号)
# ════════════════════════════════════════════════════════════════════════════

def surface_code_logical_error(p_phys: float, d: int) -> float:
    """
    距離 d の表面符号の論理エラー率 (threshold 近似):
        p_L ≈ A × (p/p_th)^{⌈d/2+1⌉}

    p_phys: 物理エラー率 (1量子ビット)
    d     : 符号距離
    p_th  = 0.01 (閾値, ~ 1% for surface code)
    """
    p_th = 0.01
    A    = 0.1
    k    = math.ceil(d/2) + 1
    if p_phys >= p_th:
        return 1.0
    return min(1.0, A * (p_phys/p_th)**k)


def physical_error_from_T2(T2_us: float, gate_time_ns: float = 100.0,
                             n_gates: int = 1) -> float:
    """
    T2* → 物理エラー率:
        p = 1 - exp(-n_gates × t_gate / T2*)
    """
    t_gate = gate_time_ns * 1e-9
    T2     = T2_us * 1e-6
    return 1 - math.exp(-n_gates * t_gate / T2)


def qec_analysis(processes: dict) -> list:
    """
    各加工プロセス × 符号距離 d → 論理エラー率テーブル。
    """
    results = []
    for proc, T2 in processes.items():
        p_phys = physical_error_from_T2(T2)
        row = {"process": proc, "T2_us": T2, "p_phys": p_phys}
        for d in [3, 5, 7, 9]:
            p_L = surface_code_logical_error(p_phys, d)
            row[f"p_L_d{d}"] = p_L
        results.append(row)
    return results


# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def plot_berry(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Berry 曲率マップ
    ax = axes[0]
    n_k = 50
    ks  = np.linspace(-math.pi, math.pi, n_k)
    omega = np.zeros((n_k, n_k))
    for i, kx in enumerate(ks):
        for j, ky in enumerate(ks):
            dk = 0.01
            def u(kx_, ky_):
                _, v = np.linalg.eigh(_bloch_hamiltonian_2band(kx_, ky_))
                return v[:, 0]
            ux = u(kx+dk, ky); uy = u(kx, ky+dk)
            u0 = u(kx, ky)
            Ax = np.imag(np.vdot(u0, ux) - np.vdot(u0, u0)) / dk
            Ay = np.imag(np.vdot(u0, uy) - np.vdot(u0, u0)) / dk
            omega[j, i] = Ax - Ay   # simplified

    im = ax.contourf(ks, ks, omega, levels=30, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, label="Berry Curvature Ω(k)")
    ax.set_xlabel("$k_x$ [1/a]", fontsize=11)
    ax.set_ylabel("$k_y$ [1/a]", fontsize=11)
    ax.set_title("Berry 曲率 Ω(k)\n4H-SiC 簡略 2バンドモデル", fontsize=10)

    # Panel 2: Zak 位相 vs Δ (オンサイト非対称)
    ax2 = axes[1]
    deltas = np.linspace(-3, 3, 60)
    zaks   = []
    for d in deltas:
        kpath = np.linspace(-math.pi, math.pi, 100)
        states = []
        for kx in kpath:
            H = _bloch_hamiltonian_2band(kx, 0.0, delta=d)
            _, vecs = np.linalg.eigh(H)
            states.append(vecs[:, 0])
        states.append(states[0])
        prod = 1.0 + 0j
        for i in range(len(states)-1):
            prod *= np.vdot(states[i], states[i+1])
        zaks.append(-np.angle(prod))
    ax2.plot(deltas, np.array(zaks)/math.pi, "#2166ac", lw=2.5)
    ax2.axhline(0, color="gray", ls="--", lw=1)
    ax2.axhline(1, color="red",  ls="--", lw=1, label="π (topological)")
    ax2.axvline(0, color="gray", ls=":", lw=1)
    ax2.set_xlabel("Onsite asymmetry Δ (SiC Si/C 差異)", fontsize=11)
    ax2.set_ylabel("Zak Phase / π", fontsize=11)
    ax2.set_title("Zak 位相 vs 格子非対称性\n0→π: トポロジカル相転移", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    # Panel 3: Chern 数 vs パラメータ相図
    ax3 = axes[2]
    t2_range = np.linspace(0, 1.0, 30)
    d_range  = np.linspace(-3.0, 3.0, 30)
    C_map    = np.zeros((len(d_range), len(t2_range)), dtype=int)
    for i, d in enumerate(d_range):
        for j, t2 in enumerate(t2_range):
            try:
                C_map[i,j] = chern_number(
                    n_k=20,
                    n_band=0,
                )
            except Exception:
                C_map[i,j] = 0

    im3 = ax3.contourf(t2_range, d_range, C_map,
                        levels=[-1.5,-0.5,0.5,1.5],
                        colors=["#d62728","#aec7e8","#2166ac"])
    plt.colorbar(im3, ax=ax3, label="Chern number C", ticks=[-1,0,1])
    ax3.set_xlabel("t₂ (NNN hopping)", fontsize=11)
    ax3.set_ylabel("Δ (SiC asymmetry)", fontsize=11)
    ax3.set_title("Chern 数 相図\nC=±1: トポロジカル絶縁体", fontsize=10)

    fig.suptitle("トポロジカル量子物理 — Berry 位相 / Zak / Chern 数\n"
                 "4H-SiC バンドトポロジー", fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "quantum_topology_berry.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Berry phase / topology -> {out}")


def plot_entangle(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Bell 状態エントロピー vs T2*
    ax = axes[0]
    T2_arr  = np.logspace(0, 3, 80)
    res100  = entanglement_entropy_vs_fidelity(T2_arr, gate_time_ns=100)
    res500  = entanglement_entropy_vs_fidelity(T2_arr, gate_time_ns=500)

    ax.semilogx(T2_arr, res100["entropy"], "#2166ac", lw=2.5,
                label="t_gate=100ns (現状)")
    ax.semilogx(T2_arr, res500["entropy"], "#d62728", lw=2.5,
                label="t_gate=500ns (低速)")
    ax.axhline(math.log(2), color="green", ls="--", lw=1.5,
               label=f"最大値 ln2={math.log(2):.3f}")

    proc_T2 = {"blade":5, "laser_ps":20, "stealth":100}
    cols    = {"blade":"#d62728","laser_ps":"#2ca02c","stealth":"#2166ac"}
    for proc, T2 in proc_T2.items():
        S = entanglement_entropy_vs_fidelity(
            np.array([T2]), gate_time_ns=100)["entropy"][0]
        ax.axvline(T2, color=cols[proc], ls=":", lw=1.0, alpha=0.7)
        ax.text(T2*1.1, 0.05, proc, fontsize=8, color=cols[proc])

    ax.set_xlabel("T₂* [µs]", fontsize=11)
    ax.set_ylabel("Entanglement Entropy S_A [nat]", fontsize=11)
    ax.set_title("Bell 状態エンタングルメント vs T₂*\niSWAP ゲート後 (V_Si ペア)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # Panel 2: 2量子ビットゲート忠実度 vs ゲート時間
    ax2 = axes[1]
    gate_times = np.logspace(1, 4, 80)   # ns
    for T2, col, lbl in [(5, "#d62728", "ブレード T2=5µs"),
                          (20, "#2ca02c", "ps laser T2=20µs"),
                          (100, "#2166ac", "ステルス T2=100µs")]:
        fids = []
        for tg in gate_times:
            res = entanglement_entropy_vs_fidelity(
                np.array([T2]), gate_time_ns=tg)
            fids.append(res["fidelity"][0])
        ax2.semilogx(gate_times, fids, color=col, lw=2, label=lbl)

    ax2.axhline(0.99, color="purple", ls="--", lw=1.5, label="F=99% (量子優位性閾値)")
    ax2.axhline(0.999, color="black", ls=":", lw=1.2, label="F=99.9% (誤り訂正閾値)")
    ax2.set_xlabel("Gate Time [ns]", fontsize=11)
    ax2.set_ylabel("2-Qubit Gate Fidelity", fontsize=11)
    ax2.set_title("iSWAP ゲート忠実度 vs ゲート時間\n(加工プロセス別 T₂*)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    fig.suptitle("量子もつれ × 加工プロセス\nV_Si 2量子ビットゲート性能 (iSWAP)", fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "quantum_entanglement.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Entanglement -> {out}")


def plot_keldysh(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    E_arr   = np.linspace(0, 6.0, 200)
    t_ps    = np.linspace(0, 5, 80)

    # Panel 1: 非平衡分布の時間発展コンター
    ax = axes[0]
    res = keldysh_carrier_dynamics(E_arr, pump_energy_eV=4.2, t_arr_ps=t_ps)
    im  = ax.contourf(t_ps, E_arr, res["f"].T, levels=30, cmap="hot_r")
    plt.colorbar(im, ax=ax, label="Occupation f(E,t)")
    ax.axhline(3.26, color="cyan", ls="--", lw=1.5, label="CBM (3.26eV)")
    ax.axhline(4.2,  color="white", ls=":",  lw=1.2, label="Pump E=4.2eV")
    ax.set_xlabel("Time [ps]", fontsize=11)
    ax.set_ylabel("Energy [eV]", fontsize=11)
    ax.set_title("Keldysh 非平衡分布 f(E,t)\nレーザー照射後キャリアダイナミクス", fontsize=10)
    ax.legend(fontsize=8)

    # Panel 2: 各時刻の分布関数スナップショット
    ax2 = axes[1]
    snap_cols = plt.cm.plasma(np.linspace(0.1, 0.9, 5))
    for idx, (t_snap, col) in enumerate(zip([0, 0.2, 0.5, 1.5, 5.0], snap_cols)):
        i = np.argmin(abs(t_ps - t_snap))
        ax2.plot(E_arr, res["f"][i], color=col, lw=2,
                 label=f"t={t_snap}ps")

    ax2.axvline(3.26, color="gray", ls="--", lw=1, alpha=0.7)
    ax2.set_xlabel("Energy [eV]", fontsize=11)
    ax2.set_ylabel("Occupation f(E)", fontsize=11)
    ax2.set_title("非平衡分布 スナップショット\nt=0: 直後, t=5ps: 熱化完了", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    fig.suptitle("Keldysh 非平衡 GF — ダイシングレーザー照射後の\n"
                 "ホットキャリアダイナミクス (4H-SiC)", fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "quantum_keldysh.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Keldysh dynamics -> {out}")


def plot_qec(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    processes = {
        "ブレード":       5.0,
        "レーザー ns":   12.0,
        "レーザー ps":   20.0,
        "レーザー fs":   45.0,
        "ステルス":      100.0,
        "バルク":        200.0,
    }
    results = qec_analysis(processes)

    # Panel 1: 論理エラー率 vs 物理エラー率 (閾値プロット)
    ax = axes[0]
    p_phys_arr = np.logspace(-4, -0.5, 100)
    for d, col in [(3,"#d62728"),(5,"#ff7f0e"),(7,"#2ca02c"),(9,"#2166ac")]:
        p_L = [surface_code_logical_error(p, d) for p in p_phys_arr]
        ax.loglog(p_phys_arr, p_L, color=col, lw=2, label=f"d={d}")

    ax.loglog(p_phys_arr, p_phys_arr, "k--", lw=1.5, label="p_L=p (no code)")
    ax.axvline(0.01, color="gray", ls=":", lw=1.5, label="閾値 p_th=1%")

    # 各プロセスの物理エラー率をマーク
    proc_colors = ["#d62728","#ff7f0e","#2ca02c","#1f77b4","#9467bd","#8c564b"]
    for r, col in zip(results, proc_colors):
        ax.axvline(r["p_phys"], color=col, ls="-", lw=0.8, alpha=0.5)
        ax.text(r["p_phys"]*1.1, 0.8, r["process"][:4],
                fontsize=7, color=col, rotation=90)

    ax.set_xlabel("Physical Error Rate p", fontsize=11)
    ax.set_ylabel("Logical Error Rate p_L", fontsize=11)
    ax.set_title("表面符号 閾値プロット\n各プロセスの物理エラー率", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.2, which="both")

    # Panel 2: プロセス × d の論理エラー率ヒートマップ
    ax2 = axes[1]
    proc_names = [r["process"] for r in results]
    d_vals     = [3, 5, 7, 9]
    mat        = np.array([[r[f"p_L_d{d}"] for d in d_vals] for r in results])

    im = ax2.imshow(np.log10(mat + 1e-15), aspect="auto",
                    cmap="RdYlGn_r", vmin=-10, vmax=0)
    plt.colorbar(im, ax=ax2, label="log₁₀(p_L)")
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([f"d={d}" for d in d_vals], fontsize=10)
    ax2.set_yticks(range(len(proc_names)))
    ax2.set_yticklabels(proc_names, fontsize=9)
    ax2.set_title("表面符号 論理エラー率\n加工プロセス × 符号距離", fontsize=10)

    for i in range(len(proc_names)):
        for j in range(len(d_vals)):
            val = mat[i, j]
            txt = f"{val:.0e}" if val > 1e-10 else "~0"
            ax2.text(j, i, txt, ha="center", va="center", fontsize=7,
                     color="white" if val > 0.01 else "black")

    fig.suptitle("量子誤り訂正 (表面符号) — 加工品質 → 量子コンピュータ性能\n"
                 "ステルスダイシング → 最低論理エラー率", fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "quantum_qec.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  {'プロセス':<14} {'T2[µs]':>7}  {'p_phys':>8}  "
          f"{'p_L(d=3)':>10}  {'p_L(d=7)':>10}")
    print("  " + "-"*55)
    for r in results:
        print(f"  {r['process']:<14} {r['T2_us']:>7.0f}  "
              f"{r['p_phys']:>8.4f}  {r['p_L_d3']:>10.2e}  {r['p_L_d7']:>10.2e}")
    print(f"[✓] QEC surface code -> {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--berry",    action="store_true")
    ap.add_argument("--entangle", action="store_true")
    ap.add_argument("--keldysh",  action="store_true")
    ap.add_argument("--qec",      action="store_true")
    args = ap.parse_args()

    run_all = not any([args.berry, args.entangle, args.keldysh, args.qec])

    if args.berry    or run_all: plot_berry()
    if args.entangle or run_all: plot_entangle()
    if args.keldysh  or run_all: plot_keldysh()
    if args.qec      or run_all: plot_qec()


if __name__ == "__main__":
    main()
