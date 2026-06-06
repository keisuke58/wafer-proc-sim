"""
深層物理モデル集 — ダイシング・切削・破壊の第一原理
=======================================================

スケール階層:
  原子スケール  (Å):  Peierls-Nabarro 転位モデル / Morse ポテンシャル
  ナノスケール  (nm): Griffith破壊 / J積分 / 応力拡大係数
  マイクロスケール(μm): Hertz接触 / EHD潤滑 / Archard摩耗
  マクロスケール (mm): 熱弾性連成 / Cattaneo-Vernotte熱伝導
  統計力学:          Kramers熱活性化 / Weibull破壊統計

各モデルは wafer-proc-sim の GP サロゲートへの入力として使える
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import gamma as gamma_func

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
BG, PANEL, GRID, WHITE, DIM = "#0d0d0d", "#111111", "#1e1e1e", "#f0f0f0", "#777777"

# ── 物理定数 ────────────────────────────────────────────────────────────────────
k_B   = 1.380649e-23   # Boltzmann定数 J/K
hbar  = 1.055e-34      # 換算Planck定数
N_A   = 6.022e23       # Avogadro数
T_RT  = 300            # 室温 K

# ── SiC 材料定数 ────────────────────────────────────────────────────────────────
class SiC:
    E    = 420e9        # 弾性率 Pa
    nu   = 0.17         # Poisson比
    K_Ic = 3.0e6        # 破壊靱性 Pa√m (3 MPa√m)
    rho  = 3210         # 密度 kg/m³
    cp   = 750          # 比熱 J/(kg·K)
    kth  = 120          # 熱伝導率 W/(m·K)
    alpha_th = 4.5e-6   # 熱膨張係数 1/K
    a0   = 3.08e-10     # 格子定数 m (6H-SiC)
    G12  = E / (2*(1+nu))   # せん断弾性率
    # 弾性異方性 (6H-SiC 六方晶 Cij MPa)
    C11  = 501e9; C12 = 111e9; C13 = 52e9
    C33  = 553e9; C44 = 163e9
    sigma_c = 3.9e9     # 理論引張強度 Pa

# ════════════════════════════════════════════════════════════════════════════
# 1. 原子スケール — Peierls-Nabarro 転位モデル
#    Si-C 結合を切断するのに必要な応力を原子論的に計算
# ════════════════════════════════════════════════════════════════════════════

def peierls_stress(a0=SiC.a0, b=SiC.a0, G=SiC.G12, nu=SiC.nu, d=SiC.a0):
    """
    Peierls応力 (転位が動き始める臨界せん断応力):
      τ_P = (2G/(1-ν)) × exp(-4πd/(b(1-ν)))

    SiC の共有結合性が高いため Peierls谷が深く,
    転位移動に必要な応力が金属より 10~100倍大きい.
    これが SiC の高硬度の原子論的起源.
    """
    tau_P = (2 * G / (1 - nu)) * np.exp(-4 * np.pi * d / (b * (1 - nu)))
    return tau_P

def morse_potential(r, D_e=4.6, alpha=21e9, r_e=1.89e-10):
    """
    Morse ポテンシャル: Si-C 結合エネルギー
      V(r) = D_e × [(1 - exp(-α(r-r_e)))² - 1]
    D_e: 解離エネルギー 4.6 eV (Si-C), α: ポテンシャル幅, r_e: 平衡距離
    """
    return D_e * (1 - np.exp(-alpha * (r - r_e)))**2 - D_e

def bond_force(r, D_e=4.6*1.6e-19, alpha=21e9, r_e=1.89e-10):
    """Morse ポテンシャルから導かれる結合力 F = -dV/dr"""
    return -2 * D_e * alpha * (1 - np.exp(-alpha*(r-r_e))) * np.exp(-alpha*(r-r_e))

def theoretical_strength():
    """
    理論引張強度 (Orowan-Kelly):
      σ_th ≈ √(E γ_s / a0)
    γ_s: 表面エネルギー (SiC ≈ 1.7 J/m²)
    """
    gamma_s = 1.7   # J/m²
    return np.sqrt(SiC.E * gamma_s / SiC.a0)

# ════════════════════════════════════════════════════════════════════════════
# 2. ナノスケール — Griffith 破壊力学
#    亀裂先端の応力場 + エネルギー解放率
# ════════════════════════════════════════════════════════════════════════════

def griffith_critical_stress(a_crack, K_Ic=SiC.K_Ic):
    """
    Griffith 臨界応力:
      σ_c = K_Ic / (Y √(πa))
    a_crack: 亀裂半長 [m], Y: 形状係数 (辺端き裂 Y=1.12)
    """
    Y = 1.12
    return K_Ic / (Y * np.sqrt(np.pi * a_crack))

def stress_field_near_crack(r, theta, K_I, nu=SiC.nu):
    """
    Mode-I 亀裂先端近傍の応力場 (Williams展開 第1項):
      σ_ij = (K_I / √(2πr)) × f_ij(θ)

    r:     亀裂先端からの距離 [m]
    theta: 角度 [rad]
    """
    factor = K_I / np.sqrt(2 * np.pi * r)
    sigma_xx = factor * np.cos(theta/2) * (1 - np.sin(theta/2)*np.sin(3*theta/2))
    sigma_yy = factor * np.cos(theta/2) * (1 + np.sin(theta/2)*np.sin(3*theta/2))
    sigma_xy = factor * np.sin(theta/2) * np.cos(theta/2) * np.cos(3*theta/2)
    return sigma_xx, sigma_yy, sigma_xy

def J_integral_elastic(K_I, K_II=0, E=SiC.E, nu=SiC.nu):
    """
    J積分 (エネルギー解放率) — 平面ひずみ:
      J = (K_I² + K_II²)(1-ν²)/E
    """
    E_prime = E / (1 - nu**2)
    return (K_I**2 + K_II**2) / E_prime

def plastic_zone_radius(K_I, sigma_y, plane="strain", nu=SiC.nu):
    """
    Irwin 塑性域半径:
      r_p = (1/2π)(K_I/σ_y)²  [平面応力]
      r_p = (1/6π)(K_I/σ_y)²  [平面ひずみ: (1-2ν)²係数]
    SiC は脆性材料なので r_p << a → LEFM が有効
    """
    factor = 1/(2*np.pi) if plane=="stress" else (1-2*nu)**2/(6*np.pi)
    return factor * (K_I / sigma_y)**2

# ════════════════════════════════════════════════════════════════════════════
# 3. マイクロスケール — Hertz 接触力学 + EHD 潤滑
#    ブレードの砥粒1粒がウェーハ表面に押し付けられる接触モデル
# ════════════════════════════════════════════════════════════════════════════

def hertz_contact(F_normal, R_grain=10e-6, E_grain=1140e9, E_sic=SiC.E,
                   nu_grain=0.07, nu_sic=SiC.nu):
    """
    Hertz 球接触 (砥粒-ウェーハ):
      E* = [(1-ν1²)/E1 + (1-ν2²)/E2]^{-1}
      a  = (3FR/4E*)^{1/3}  [接触半径]
      p0 = 3F/(2πa²)        [最大接触圧力]
      δ  = a²/R             [押し込み深さ]
    """
    E_star = 1 / ((1-nu_grain**2)/E_grain + (1-nu_sic**2)/E_sic)
    a_c    = (3 * F_normal * R_grain / (4 * E_star))**(1/3)
    p0     = 3 * F_normal / (2 * np.pi * a_c**2)
    delta  = a_c**2 / R_grain
    return {"a_contact_m": a_c, "p0_Pa": p0, "depth_m": delta, "E_star": E_star}

def ehd_film_thickness(eta, U, E_star, R, W):
    """
    弾性流体潤滑 (EHD) 最小油膜厚さ (Hamrock-Dowson 式):
      h_min = 3.63 R (η₀U/E*R)^0.68 (αE*)^0.49 (W/E*R²)^{-0.073}

    η: 動粘度 [Pa·s], U: 速度 [m/s], α: 圧力粘度係数
    クーラント潤滑膜がブレード砥粒を保護するかどうかの判定
    """
    alpha_p = 2e-8   # 圧力粘度係数 Pa⁻¹ (水系クーラント)
    G_ehd   = alpha_p * E_star
    U_ehd   = eta * U / (E_star * R)
    W_ehd   = W / (E_star * R**2)
    h_min   = 3.63 * R * (U_ehd**0.68) * (G_ehd**0.49) * (W_ehd**-0.073)
    lambda_ratio = h_min / (0.1e-6)   # 表面粗さ 0.1μm と比較
    return {"h_min_m": h_min, "lambda": lambda_ratio, "lubricated": lambda_ratio > 3}

def archard_wear(F_normal, sliding_distance, H_material=24e9, K_wear=1e-4):
    """
    Archard 摩耗則:
      V_wear = K × F × L / H
    K: 摩耗係数, H: 硬さ [Pa], L: 摺動距離 [m]
    ブレードの摩耗量を予測 → 交換タイミングの計算
    """
    return K_wear * F_normal * sliding_distance / H_material

# ════════════════════════════════════════════════════════════════════════════
# 4. マクロスケール — 熱弾性連成 (Cattaneo-Vernotte 双曲型熱伝導)
#    フーリエ則の破綻: ナノスケールでは熱が有限速度で伝わる
# ════════════════════════════════════════════════════════════════════════════

def cattaneo_vernotte_solution(x, t, Q0, tau_r=1e-12, kth=SiC.kth,
                                alpha_th_diff=None):
    """
    Cattaneo-Vernotte 方程式の解析解:
      τ_r ∂²T/∂t² + ∂T/∂t = α ∂²T/∂x²

    τ_r: 緩和時間 (フォノン平均自由時間, SiCでは~1ps)
    通常のFourier則: τ_r=0 → 熱が瞬時に伝わる (非物理的)
    CV方程式: 熱波が速度 v_th = √(α/τ_r) で伝播する

    ナノスケールダイシングでは この効果が無視できない
    """
    if alpha_th_diff is None:
        alpha_th_diff = kth / (SiC.rho * SiC.cp)
    v_heat = np.sqrt(alpha_th_diff / tau_r)   # 熱波速度 m/s
    # 簡略解: 熱波のフロント
    T = np.zeros_like(x, dtype=float)
    wave_front = v_heat * t
    mask = np.abs(x) < wave_front
    if np.any(mask):
        xi = x[mask]
        T[mask] = (Q0 / (2*kth)) * np.exp(-t/(2*tau_r)) * np.cosh(
            np.sqrt(t**2 - (xi/v_heat)**2) / (2*tau_r) + 1e-30)
    return T, v_heat

def thermal_stress(delta_T, E=SiC.E, alpha=SiC.alpha_th, nu=SiC.nu):
    """
    熱応力 (拘束熱膨張):
      σ_th = E α ΔT / (1 - 2ν)   [三軸拘束]
      σ_th = E α ΔT / (1 - ν)    [平面ひずみ]
    """
    return E * alpha * delta_T / (1 - nu)

def thermoelastic_coupling(T_field, x_field, E=SiC.E, alpha=SiC.alpha_th, nu=SiC.nu):
    """
    Biot 熱弾性連成 — 温度勾配から熱応力場を計算
    ∇²u = -α(1+ν)/(1-ν) × ∇T  (Navier方程式 熱源項)
    """
    dT_dx = np.gradient(T_field, x_field)
    coupling = E * alpha / (1 - 2*nu)
    sigma_th = -coupling * np.cumsum(dT_dx) * (x_field[1]-x_field[0])
    return sigma_th

# ════════════════════════════════════════════════════════════════════════════
# 5. 統計力学 — Kramers 熱活性化 による亀裂核生成
#    切削中に「なぜ突然チッピングが起きるか」の統計的説明
# ════════════════════════════════════════════════════════════════════════════

def kramers_nucleation_rate(T, delta_G, nu_attempt=1e13):
    """
    Kramers 熱活性化速度:
      Γ = ν₀ × exp(-ΔG / k_B T)

    ν₀: 試行振動数 (フォノン周波数 ~10¹³ Hz)
    ΔG: 活性化障壁 = G_c × V_critical (Griffith 臨界核生成エネルギー)

    解釈: 切削中、熱揺らぎが Griffith 臨界核 を確率的に生成する
    温度が高いほど亀裂が突発的に起きやすい → ドライカットが危険な理由
    """
    return nu_attempt * np.exp(-delta_G / (k_B * T))

def critical_nucleus_energy(sigma_applied, gamma_s=1.7, E=SiC.E, nu=SiC.nu):
    """
    Griffith 臨界核生成エネルギー:
      ΔG* = π γ_s² E / ((1-ν²) σ²)

    σ: 印加応力, γ_s: 表面エネルギー
    この障壁を熱揺らぎが超えると亀裂が自発的に成長する
    """
    return np.pi * gamma_s**2 * E / ((1 - nu**2) * sigma_applied**2)

def weibull_survival(sigma, sigma_0, m_weibull, V, V0=1e-9):
    """
    Weibull 破壊統計 (体積スケーリング込み):
      P_s = exp(-(V/V0) × (σ/σ0)^m)

    体積 V が大きいほど弱い欠陥を含む確率が上がる
    (「最弱リンク理論」— 大きなダイほど割れやすい)
    """
    return np.exp(-(V/V0) * (sigma/sigma_0)**m_weibull)

# ════════════════════════════════════════════════════════════════════════════
# 6. 結晶異方性 — SiC 六方晶の完全弾性テンソル
#    c面/m面/a面でなぜ切りやすさが違うのかの物理
# ════════════════════════════════════════════════════════════════════════════

def sic_elastic_tensor():
    """
    6H-SiC の弾性定数テンソル C_ij (GPa) — 六方晶対称性
    独立成分: C11, C12, C13, C33, C44 (C66 = (C11-C12)/2)
    """
    C = np.zeros((6,6))
    C[0,0]=C[1,1]=SiC.C11/1e9; C[2,2]=SiC.C33/1e9
    C[0,1]=C[1,0]=SiC.C12/1e9
    C[0,2]=C[2,0]=C[1,2]=C[2,1]=SiC.C13/1e9
    C[3,3]=C[4,4]=SiC.C44/1e9
    C[5,5]=(SiC.C11-SiC.C12)/(2e9)
    return C

def directional_modulus(theta_deg, phi_deg=0):
    """
    任意方向の Young率 (六方晶 → 方向依存性):
      1/E(n) = S11 sin⁴θ + S33 cos⁴θ + (2S13+S44)sin²θcos²θ

    θ: c軸(0001)からの極角
    c面(θ=90°)とa面(θ=0°)の剛性差 → ダイシング方向依存性の起源
    """
    C = sic_elastic_tensor()
    # コンプライアンステンソルの近似 (六方晶対称)
    S11 = 1/SiC.C11*1e9; S33 = 1/SiC.C33*1e9
    S13 = -SiC.C13/(SiC.C11*SiC.C33)*1e9
    S44 = 1/SiC.C44*1e9
    theta = np.deg2rad(theta_deg)
    s4 = np.sin(theta)**4; c4 = np.cos(theta)**4
    s2c2 = np.sin(theta)**2 * np.cos(theta)**2
    inv_E = S11*s4 + S33*c4 + (2*S13+S44)*s2c2
    return 1/inv_E   # GPa

# ════════════════════════════════════════════════════════════════════════════
# 7. 流体力学 — クーラントの乱流冷却 (ダイシング特有)
#    Nusselt 数 → 熱伝達係数 → 刃先温度の計算
# ════════════════════════════════════════════════════════════════════════════

def coolant_heat_transfer(v_coolant, d_nozzle=0.5e-3, T_blade=400,
                           rho_c=1000, mu_c=0.001, cp_c=4182, k_c=0.6):
    """
    クーラント噴流の強制対流冷却:
      Re = ρ v d / μ
      Nu = 0.023 Re^0.8 Pr^0.4  (Dittus-Boelter, 乱流)
      h  = Nu k / d
      Q_remove = h × A × (T_blade - T_coolant)

    v_coolant: 噴射速度 [m/s]
    d_nozzle:  ノズル径 [m]
    """
    Pr   = mu_c * cp_c / k_c      # Prandtl数
    Re   = rho_c * v_coolant * d_nozzle / mu_c
    Nu   = 0.023 * Re**0.8 * Pr**0.4 if Re > 2300 else 3.66   # 層流
    h    = Nu * k_c / d_nozzle    # 熱伝達係数 W/(m²·K)
    A    = np.pi * (d_nozzle/2)**2
    Q    = h * A * (T_blade - 25)
    return {"Re": Re, "Nu": Nu, "h": h, "Q_remove_W": Q, "turbulent": Re>2300}

# ════════════════════════════════════════════════════════════════════════════
# 8. 電気化学 — Butler-Volmer (全固体電池界面への転用)
#    TMCMCで同定した界面抵抗パラメータの物理的起源
# ════════════════════════════════════════════════════════════════════════════

def butler_volmer(eta, i0=1e-3, alpha_a=0.5, alpha_c=0.5, T=300):
    """
    Butler-Volmer 方程式 (電極反応速度):
      j = i0 × [exp(α_a F η / RT) - exp(-α_c F η / RT)]

    η: 過電圧 [V], i0: 交換電流密度 [A/m²]
    全固体電池の固体電解質/電極界面に適用

    TMCMC でパラメータ (i0, α_a, α_c) を EIS データから同定する
    """
    F = 96485   # Faraday定数
    R_gas = 8.314
    return i0 * (np.exp(alpha_a * F * eta / (R_gas * T)) -
                 np.exp(-alpha_c * F * eta / (R_gas * T)))

def tafel_slope(alpha, T=300):
    """Tafel 勾配 b = RT/(αF): EIS フィッティングの検証に使う"""
    return 8.314 * T / (alpha * 96485)

# ════════════════════════════════════════════════════════════════════════════
# 9. 高分子物理 — Worm-like Chain (DNA の機械的応答)
#    DNA物理切断モデルの基礎物理
# ════════════════════════════════════════════════════════════════════════════

def wlc_force_extension(x, L_c=16.4e-6, L_p=50e-9, T=300):
    """
    Worm-like Chain モデル (Marko-Siggia 近似):
      F = (k_B T / L_p) × [1/(4(1-x/L_c)²) - 1/4 + x/L_c]

    L_c: 輪郭長 (E.coli ゲノム ~1.6mm, 小DNA ~μm)
    L_p: 持続長 (DNA ≈ 50nm, ssDNA ≈ 1nm)
    x:   伸長量 [m]

    物理切断には F_cut ~ ΔG_bond / Δx が必要
    Si-C 結合 (4.6 eV / 0.34nm) → ~2 nN
    DNAはフォースでは伸び切るだけで切れない → 物理切断の難しさ
    """
    x_safe = np.clip(x, 0, 0.99*L_c)
    F = (k_B * T / L_p) * (0.25/(1-x_safe/L_c)**2 - 0.25 + x_safe/L_c)
    return F

def dna_persistence_length_vs_salt(c_NaCl_mM):
    """
    塩濃度依存の DNA 持続長 (Manning 凝縮理論):
      L_p = L_p0 + (1/c)^0.5 × κ_D

    高塩濃度 → 電荷遮蔽 → DNAが柔らかくなる
    物理切断装置の設計で溶液条件を最適化するときに使う
    """
    L_p0 = 50e-9      # 生理的条件での持続長
    kappa = 3e-9      # Debye長スケール係数
    c_M   = c_NaCl_mM / 1000
    return L_p0 + kappa / np.sqrt(c_M)

# ════════════════════════════════════════════════════════════════════════════
# 可視化
# ════════════════════════════════════════════════════════════════════════════

def sa(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE, labelsize=8)
    ax.xaxis.label.set_color(WHITE); ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax.grid(color=GRID, lw=0.5, alpha=0.6)

def p_morse(ax):
    sa(ax)
    ax.set_title("Morse ポテンシャル — Si-C 結合\n(切断に必要なエネルギーの原子論的起源)", fontsize=9, pad=5)
    r = np.linspace(1.0e-10, 5.0e-10, 500)
    V = morse_potential(r)
    F = bond_force(r)
    ax.plot(r*1e10, V, color="#FF6E40", lw=2.5, label="V(r) [eV]")
    ax2 = ax.twinx()
    ax2.plot(r*1e10, F*1e9, color="#4FC3F7", lw=2, ls="--", label="F(r) [nN]")
    ax2.set_ylabel("結合力 (nN)", color="#4FC3F7"); ax2.tick_params(colors="#4FC3F7")
    ax2.set_facecolor(PANEL)
    for sp in ax2.spines.values(): sp.set_edgecolor("#333")
    ax.axhline(0, color=DIM, lw=0.8, ls="-")
    ax.axvline(1.89, color="#FFD700", lw=1.2, ls=":")
    ax.text(1.92, -3, "平衡距離\n1.89Å", color="#FFD700", fontsize=7.5)
    ax.set_xlabel("原子間距離 (Å)"); ax.set_ylabel("エネルギー (eV)")
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE, loc="upper right")
    ax2.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE, loc="center right")

def p_crack_tip(ax):
    sa(ax)
    ax.set_title("亀裂先端応力場 (Williams展開)\nK_I = K_Ic で亀裂伝播が始まる", fontsize=9, pad=5)
    r_vals = np.linspace(1e-9, 50e-6, 300)
    theta_list = [0, np.pi/4, np.pi/2]
    colors_t = ["#FF5252","#FF9800","#4CAF50"]
    labels_t = ["θ=0° (切断線上)", "θ=45°", "θ=90°"]
    K_I = SiC.K_Ic
    for theta, color, label in zip(theta_list, colors_t, labels_t):
        _, sigma_yy, _ = stress_field_near_crack(r_vals, theta, K_I)
        ax.loglog(r_vals*1e6, sigma_yy/1e6, color=color, lw=2, label=label)
    ax.axhline(SiC.sigma_c/1e6, color="#FFD700", lw=1.5, ls="--")
    ax.text(0.01, SiC.sigma_c/1e6*1.1, f"理論強度 {SiC.sigma_c/1e9:.1f}GPa", color="#FFD700", fontsize=7.5)
    ax.set_xlabel("亀裂先端からの距離 (μm)")
    ax.set_ylabel("σ_yy (MPa)")
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE)

def p_hertz(ax):
    sa(ax)
    ax.set_title("Hertz 接触圧力 — 砥粒1粒の接触物理\n砥粒径 × 荷重 → 接触圧力・押込み深さ", fontsize=9, pad=5)
    F_list   = np.logspace(-3, 1, 100)   # 0.001 ~ 10 N
    for R_um, color, label in [(5,"#FF5252","R=5μm"), (10,"#FF9800","R=10μm"),
                                (20,"#4CAF50","R=20μm"), (50,"#42A5F5","R=50μm")]:
        p0s = [hertz_contact(F, R_grain=R_um*1e-6)["p0_Pa"]/1e9 for F in F_list]
        ax.loglog(F_list, p0s, color=color, lw=2, label=label)
    ax.axhline(SiC.sigma_c/1e9, color="#FFD700", lw=1.5, ls="--")
    ax.text(0.0015, SiC.sigma_c/1e9*1.05, "SiC 理論強度", color="#FFD700", fontsize=7.5)
    ax.set_xlabel("砥粒1粒あたりの荷重 (N)")
    ax.set_ylabel("最大接触圧力 (GPa)")
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE)

def p_kramers(ax):
    sa(ax)
    ax.set_title("Kramers 熱活性化 — チッピング突発発生の統計力学\n「なぜ突然割れるか」の物理的説明", fontsize=9, pad=5)
    T_vals = np.linspace(300, 1200, 300)
    sigma_list = [0.5e9, 1.0e9, 2.0e9, 3.0e9]
    colors_k = ["#42A5F5","#4CAF50","#FF9800","#FF5252"]
    for sigma, color in zip(sigma_list, colors_k):
        dG = critical_nucleus_energy(sigma)
        rates = [kramers_nucleation_rate(T, dG) for T in T_vals]
        ax.semilogy(T_vals, rates, color=color, lw=2,
                    label=f"σ={sigma/1e9:.1f}GPa")
    ax.set_xlabel("温度 (K)"); ax.set_ylabel("亀裂核生成速度 (1/s)")
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE)
    ax.axvline(800, color="#FFD700", lw=1.2, ls=":")
    ax.text(810, 1e5, "ドライカット\n刃先温度域", color="#FFD700", fontsize=7.5)

def p_anisotropy(ax):
    sa(ax)
    ax.set_title("SiC 結晶異方性 — 方向依存 Young率\nダイシング方向で切削力が変わる物理的起源", fontsize=9, pad=5)
    theta_vals = np.linspace(0, 90, 200)
    E_vals = [directional_modulus(t) for t in theta_vals]
    theta_rad = np.deg2rad(theta_vals)
    # 極座標プロット (別途)
    ax.plot(theta_vals, E_vals, color="#E040FB", lw=2.5)
    ax.fill_between(theta_vals, E_vals, alpha=0.2, color="#E040FB")
    ax.axvline(0,  color="#FF5252", lw=1.2, ls="--"); ax.text(2,  200, "c軸\n(0001)", color="#FF5252", fontsize=8)
    ax.axvline(90, color="#4CAF50", lw=1.2, ls="--"); ax.text(82, 200, "a面\n(11-20)", color="#4CAF50", fontsize=8)
    ax.set_xlabel("c軸からの角度 (°)")
    ax.set_ylabel("Young率 (GPa)")
    ax.set_ylim(0, 600)

def p_cattaneo(ax):
    sa(ax)
    ax.set_title("Cattaneo-Vernotte 熱波 vs Fourier則\n超短パルスレーザーダイシングでは熱が「波」として伝わる", fontsize=9, pad=5)
    x = np.linspace(-50e-9, 50e-9, 500)
    t_vals = [1e-12, 5e-12, 10e-12]
    colors_cv = ["#FF5252","#FF9800","#4CAF50"]
    for t, color in zip(t_vals, colors_cv):
        T_cv, v_heat = cattaneo_vernotte_solution(x, t, Q0=1e12)
        ax.plot(x*1e9, T_cv/T_cv.max() if T_cv.max()>0 else T_cv,
                color=color, lw=2, label=f"t={t*1e12:.0f}ps")
    ax.axvline(0, color=DIM, lw=1, ls="--")
    ax.text(1, 0.9, "熱源", color=DIM, fontsize=8)
    ax.set_xlabel("位置 (nm)"); ax.set_ylabel("規格化温度")
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE)
    ax.text(0.01,0.05,"熱波速度: √(α/τ_r) ~6,000 m/s\n(音速の約2倍)",
            transform=ax.transAxes, color=DIM, fontsize=8)

def p_weibull(ax):
    sa(ax)
    ax.set_title("Weibull 破壊統計 — ダイサイズ依存性\n大きなダイほど弱い欠陥を含む確率が高い", fontsize=9, pad=5)
    sigma_vals = np.linspace(0, 5e9, 300)
    sigma_0 = 3.5e9; m = 8
    die_sizes = [1e-9, 4e-9, 16e-9, 64e-9]   # m³
    colors_w = ["#42A5F5","#4CAF50","#FF9800","#FF5252"]
    labels_w = ["1mm²","4mm²","16mm²","64mm²"]
    for V, color, label in zip(die_sizes, colors_w, labels_w):
        Ps = weibull_survival(sigma_vals, sigma_0, m, V)
        ax.plot(sigma_vals/1e9, Ps*100, color=color, lw=2, label=label)
    ax.set_xlabel("印加応力 (GPa)"); ax.set_ylabel("生存確率 (%)")
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE, title="ダイ面積", title_fontsize=7)
    ax.axhline(50, color=DIM, lw=1, ls=":", alpha=0.7)

def p_bv(ax):
    sa(ax)
    ax.set_title("Butler-Volmer — 全固体電池界面反応速度\nTMCMCで i₀,α を EIS データから同定", fontsize=9, pad=5)
    eta = np.linspace(-0.3, 0.3, 300)
    for i0, color, label in [(0.1,"#42A5F5","i₀=0.1 mA/cm²"),
                              (1.0,"#4CAF50","i₀=1 mA/cm²"),
                              (10, "#FF9800","i₀=10 mA/cm²")]:
        j = butler_volmer(eta, i0=i0)
        ax.plot(eta*1000, j, color=color, lw=2, label=label)
    ax.axhline(0, color=DIM, lw=0.8)
    ax.axvline(0, color=DIM, lw=0.8)
    ax.set_xlabel("過電圧 η (mV)"); ax.set_ylabel("電流密度 (mA/cm²)")
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE)
    ax.set_ylim(-50, 50)

def p_wlc(ax):
    sa(ax)
    ax.set_title("Worm-like Chain — DNA の機械的応答\n物理切断に必要な力の下限を与える", fontsize=9, pad=5)
    x_frac = np.linspace(0.01, 0.99, 300)
    for L_p_nm, color, label in [(50, "#B2FF59", "DNA (L_p=50nm)"),
                                   (1,  "#69F0AE", "ssDNA (L_p=1nm)"),
                                   (500,"#00E5FF", "dsDNA 高塩 (L_p=500nm)")]:
        L_c = 1e-6
        F = wlc_force_extension(x_frac * L_c, L_c=L_c, L_p=L_p_nm*1e-9)
        ax.semilogy(x_frac, F*1e12, color=color, lw=2, label=label)
    ax.axhline(2000, color="#FF5252", lw=1.5, ls="--")
    ax.text(0.02, 2500, "Si-C共有結合切断力 ~2nN", color="#FF5252", fontsize=7.5)
    ax.set_xlabel("伸長率 (x/L_c)"); ax.set_ylabel("引張力 (pN)")
    ax.legend(fontsize=8, facecolor="#222", edgecolor="#444", labelcolor=WHITE)
    ax.text(0.02,0.05,"WLCモデル: DNAは伸び切るだけで\n切断には化学/物理的なカットが必要",
            transform=ax.transAxes, color=DIM, fontsize=7.5)

def main():
    fig = plt.figure(figsize=(24, 30), facecolor=BG)
    fig.suptitle(
        "深層物理モデル集 — 原子スケールから統計力学まで\n"
        "Disco 魔改造・全固体電池・骨外科・DNA の第一原理",
        fontsize=16, color=WHITE, fontweight="bold", y=0.988
    )
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.50, wspace=0.36,
                           left=0.06, right=0.97, top=0.960, bottom=0.04)

    p_morse(     fig.add_subplot(gs[0, 0]))
    p_crack_tip( fig.add_subplot(gs[0, 1]))
    p_hertz(     fig.add_subplot(gs[0, 2]))
    p_kramers(   fig.add_subplot(gs[1, 0]))
    p_anisotropy(fig.add_subplot(gs[1, 1]))
    p_cattaneo(  fig.add_subplot(gs[1, 2]))
    p_weibull(   fig.add_subplot(gs[2, 0]))
    p_bv(        fig.add_subplot(gs[2, 1]))
    p_wlc(       fig.add_subplot(gs[2, 2]))

    # スケール階層ラベル
    for row, label, color in [
        (0, "原子/ナノスケール: Morse, K_I場, Hertz接触",    "#FF6E40"),
        (1, "統計力学/結晶物理: Kramers活性化, 異方性, 熱波", "#E040FB"),
        (2, "応用ドメイン: Weibull破壊統計, BV電池, WLC-DNA", "#4FC3F7"),
    ]:
        fig.text(0.01, 0.963 - row*0.333, label, color=color,
                 fontsize=8.5, fontweight="bold", va="top")

    out = os.path.join(OUT_DIR, "deep_physics.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
