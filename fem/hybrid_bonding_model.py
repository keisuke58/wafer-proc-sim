"""
Hybrid Bonding モデル — Cu-Cu 直接接合 (Besi / K&S)
=====================================================
ワイヤーボンドを置き換える次世代接合技術。
AI チップ (HBM / CoWoS / SoIC) の積層に必須。

Besi (BESI.AS) と K&S (KLIC) が市場争奪中。

物理:
    1. Cu-Cu 直接接合強度 — 熱圧着モデル (接合温度・圧力・時間)
    2. 表面活性化 (SAB) — 接合前 CMP + 表面活性化
    3. 接合後アニール — Cu 拡散による強度向上
    4. 電気特性 — 接合抵抗・寄生容量
    5. ワイヤーボンド vs Hybrid bonding 比較

実行:
    python fem/hybrid_bonding_model.py
"""

import math, os, sys
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

kB = 8.617e-5  # eV/K

# ── Cu 材料定数 ──────────────────────────────────────────────────────────────
CU_PROPS = {
    "D0_m2_s":    7.8e-5,     # Cu 粒界拡散前指数因子 [m²/s]
    "Q_diff_eV":  0.80,       # 粒界拡散活性化エネルギー [eV] (bulk=2.04, GB=0.80)
    "E_GPa":      120,
    "H_MPa":      600,        # 硬さ [MPa]  (熱圧着荷重計算)
    "sigma_y_MPa": 120,       # 降伏応力
    "rho_ohm_m":  1.72e-8,    # 抵抗率
}

# ── Besi / K&S 装置スペック ───────────────────────────────────────────────────
BONDERS = {
    "Besi Esec 2100": {
        "maker": "BE Semiconductor",
        "type": "Thermocompression (TCB)",
        "T_bond_C": 200,
        "force_N": 10,
        "time_s": 0.5,
        "accuracy_um": 0.5,
        "UPH": 8000,           # Units Per Hour
        "color": "#d62728",
        "app": "HBM / CoWoS / SoIC",
    },
    "K&S APAMA": {
        "maker": "Kulicke & Soffa",
        "type": "Hybrid bonding",
        "T_bond_C": 250,
        "force_N": 15,
        "time_s": 0.3,
        "accuracy_um": 0.3,
        "UPH": 10000,
        "color": "#2166ac",
        "app": "Advanced packaging",
    },
}


# ════════════════════════════════════════════════════════════════════════════
# 1. Cu-Cu 接合強度モデル
# ════════════════════════════════════════════════════════════════════════════

def cu_diffusion_coeff(T_C: float) -> float:
    """Cu 粒界拡散係数 [m²/s]。"""
    T_K = T_C + 273.15
    return CU_PROPS["D0_m2_s"] * math.exp(-CU_PROPS["Q_diff_eV"] / (kB * T_K))


def bond_strength(T_C: float, P_MPa: float, t_s: float,
                  Ra_nm: float = 0.3) -> dict:
    """
    Cu-Cu 接合強度 [MPa]。
    1. 熱活性化拡散: σ_diff ∝ D(T) * t^0.5
    2. 圧力支配塑性変形: σ_plastic ∝ P/H * σ_y
    3. 表面粗さ補正: 粗さが大きいと実接触面積が低下

    実験値: 200°C/10MPa/1s → ~80 MPa (Kim et al. 2020)
    """
    D = cu_diffusion_coeff(T_C)
    # 拡散による接合
    sigma_diff = 150 * math.sqrt(D * t_s / 1e-18)  # 規格化
    sigma_diff = min(sigma_diff, 200)

    # 圧力塑性
    sigma_plastic = P_MPa / CU_PROPS["H_MPa"] * CU_PROPS["sigma_y_MPa"] * 50
    sigma_plastic = min(sigma_plastic, 100)

    # 粗さ補正 (Ra < 1nm が必要)
    roughness_factor = math.exp(-Ra_nm / 0.5)

    total = (sigma_diff * 0.6 + sigma_plastic * 0.4) * roughness_factor
    return {
        "T_C": T_C, "P_MPa": P_MPa, "t_s": t_s, "Ra_nm": Ra_nm,
        "sigma_diff_MPa":    round(sigma_diff, 1),
        "sigma_plastic_MPa": round(sigma_plastic, 1),
        "bond_strength_MPa": round(total, 1),
        "quality": "good" if total > 60 else "marginal" if total > 30 else "poor",
    }


def anneal_strength_increase(T_anneal_C: float, t_anneal_h: float,
                              sigma0_MPa: float = 60) -> float:
    """
    接合後アニールによる強度向上。Cu 拡散が進んで界面が消える。
    σ_final = σ0 * (1 + α * sqrt(D(T) * t))
    """
    D = cu_diffusion_coeff(T_anneal_C)
    t_s = t_anneal_h * 3600
    alpha = 7e3  # 経験定数 (300°C/1h で σ0→2×σ0 になるようキャリブレーション)
    return sigma0_MPa * (1 + alpha * math.sqrt(D * t_s))


# ════════════════════════════════════════════════════════════════════════════
# 2. 電気特性
# ════════════════════════════════════════════════════════════════════════════

def contact_resistance(bond_area_um2: float, Ra_nm: float = 0.3) -> float:
    """接合抵抗 [mΩ]。実接触面積 = 全面積 × (1 - Ra補正)"""
    A_m2 = bond_area_um2 * 1e-12
    A_contact = A_m2 * math.exp(-Ra_nm / 0.3)
    rho_c = CU_PROPS["rho_ohm_m"] * 1e-9  # 界面抵抗率 [Ω·m²]
    R = rho_c / max(A_contact, 1e-18)
    return R * 1000  # mΩ


def parasitic_capacitance(bump_pitch_um: float, dielectric_k: float = 3.9,
                           dielectric_t_um: float = 0.5) -> float:
    """
    バンプ間寄生容量 [fF]。
    Hybrid bonding のバンプ間隔は Wire bond より桁違いに小さい → 注意。
    """
    eps0 = 8.854e-12
    A_m2 = (bump_pitch_um * 1e-6) ** 2
    C = dielectric_k * eps0 * A_m2 / (dielectric_t_um * 1e-6)
    return C * 1e15  # fF


# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Panel 1: 接合強度 vs 温度・圧力
    ax = axes[0, 0]
    T_arr = np.linspace(150, 400, 100)
    for P, col, lbl in [(5,"#2166ac","P=5MPa"),(10,"#ff7f0e","P=10MPa"),(20,"#d62728","P=20MPa")]:
        strengths = [bond_strength(T, P, 1.0)["bond_strength_MPa"] for T in T_arr]
        ax.plot(T_arr, strengths, lw=2.5, color=col, label=lbl)
    ax.axhline(60, color="purple", ls="--", lw=1.5, label="Good bond >60MPa")
    ax.axvspan(180, 300, alpha=0.1, color="green", label="典型プロセス窓")
    ax.set_xlabel("Bonding temperature [°C]", fontsize=10)
    ax.set_ylabel("Bond strength [MPa]", fontsize=10)
    ax.set_title("Cu-Cu Bond Strength\nvs Temperature & Pressure", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    # Panel 2: アニール効果
    ax2 = axes[0, 1]
    t_arr = np.linspace(0, 4, 100)
    for T_a, col, lbl in [(150,"#2166ac","Anneal 150°C"),(200,"#ff7f0e","200°C"),(300,"#d62728","300°C")]:
        strengths = [anneal_strength_increase(T_a, t) for t in t_arr]
        strengths = [min(s, 250) for s in strengths]
        ax2.plot(t_arr, strengths, lw=2.5, color=col, label=lbl)
    ax2.set_xlabel("Anneal time [h]", fontsize=10)
    ax2.set_ylabel("Bond strength [MPa]", fontsize=10)
    ax2.set_title("Post-Bond Anneal\nCu Diffusion Strengthening", fontsize=10)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.2)

    # Panel 3: 表面粗さ vs 接合強度
    ax3 = axes[0, 2]
    Ra_arr = np.linspace(0.05, 3.0, 100)
    for T, col, lbl in [(200,"#2166ac","200°C"),(250,"#ff7f0e","250°C"),(300,"#d62728","300°C")]:
        strengths = [bond_strength(T, 10, 1.0, Ra)["bond_strength_MPa"] for Ra in Ra_arr]
        ax3.plot(Ra_arr, strengths, lw=2.5, color=col, label=lbl)
    ax3.axvline(0.3, color="green", ls="--", lw=1.5, label="CMP目標 Ra<0.3nm")
    ax3.axvline(1.0, color="red", ls=":",   lw=1.5, label="上限 Ra=1nm")
    ax3.set_xlabel("Surface roughness Ra [nm]", fontsize=10)
    ax3.set_ylabel("Bond strength [MPa]", fontsize=10)
    ax3.set_title("Surface Roughness vs Bond Quality\n(CMP Ra < 0.3nm 必須)", fontsize=10)
    ax3.legend(fontsize=8); ax3.grid(alpha=0.2)

    # Panel 4: Hybrid bonding vs Wire bond 比較
    ax4 = axes[1, 0]
    ax4.axis("off")
    rows = [
        ["特性", "Wire Bond (Au)", "Hybrid Bonding"],
        ["バンプピッチ", ">50 µm", "<10 µm"],
        ["寄生インダクタンス", "~1 nH", "~0.01 nH (100×↓)"],
        ["寄生容量", "~0.1 pF", "~0.01 pF"],
        ["帯域幅", "~10 GHz", ">100 GHz"],
        ["接合強度", "~7g pull", ">60 MPa"],
        ["接合温度", "175°C", "200-300°C"],
        ["適用", "パワー/RF", "AI/HBM/CoWoS"],
        ["主要装置", "K&S/Shinkawa", "Besi/K&S APAMA"],
    ]
    tbl = ax4.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    tbl.scale(1.35, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0: cell.set_facecolor("#333"); cell.set_text_props(color="w")
        elif c == 2 and r > 0: cell.set_facecolor("#d4edda")  # Hybrid bonding 強調
        elif r % 2 == 0: cell.set_facecolor("#f5f5f5")
    ax4.set_title("Wire Bond vs Hybrid Bonding\n電気特性比較", fontsize=10)

    # Panel 5: バンプピッチ vs 寄生容量・帯域幅
    ax5 = axes[1, 1]
    pitches = np.linspace(2, 50, 100)
    caps = [parasitic_capacitance(p) for p in pitches]
    # 帯域幅 ∝ 1/(L*C)^0.5, L ∝ pitch (概算)
    L_nH = pitches / 100  # nH (rough)
    C_fF = np.array(caps)
    BW_GHz = 1 / (2 * math.pi * np.sqrt(L_nH * 1e-9 * C_fF * 1e-15)) / 1e9
    ax5_twin = ax5.twinx()
    ax5.plot(pitches, caps, "b-", lw=2.5, label="Capacitance [fF]")
    ax5_twin.semilogy(pitches, BW_GHz, "r--", lw=2.5, label="Bandwidth [GHz]")
    ax5.axvline(10, color="purple", ls="--", lw=1.5, label="Hybrid bonding ~10µm")
    ax5.axvline(50, color="gray",   ls=":",  lw=1.5, label="Wire bond ~50µm")
    ax5.set_xlabel("Bump pitch [µm]", fontsize=10)
    ax5.set_ylabel("Capacitance [fF]", color="blue", fontsize=10)
    ax5_twin.set_ylabel("Bandwidth [GHz]", color="red", fontsize=10)
    ax5.set_title("Bump Pitch → Parasitics → Bandwidth\nHybrid bonding の電気優位性", fontsize=10)
    lines1, labs1 = ax5.get_legend_handles_labels()
    lines2, labs2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1+lines2, labs1+labs2, fontsize=8)
    ax5.grid(alpha=0.2)

    # Panel 6: Besi vs K&S 市場
    ax6 = axes[1, 2]
    years = np.arange(2022, 2031)
    hb_market = np.array([0.3, 0.5, 0.9, 1.5, 2.3, 3.2, 4.2, 5.3, 6.5])  # B USD
    besi_share = np.array([0.35,0.38,0.42,0.45,0.46,0.46,0.45,0.44,0.43])
    ks_share   = np.array([0.20,0.22,0.25,0.28,0.30,0.31,0.32,0.33,0.34])
    ax6.fill_between(years, 0, hb_market*besi_share, alpha=0.7, color="#d62728", label="Besi")
    ax6.fill_between(years, hb_market*besi_share, hb_market*(besi_share+ks_share),
                     alpha=0.7, color="#2166ac", label="K&S")
    ax6.fill_between(years, hb_market*(besi_share+ks_share), hb_market,
                     alpha=0.4, color="gray", label="Others")
    ax6.plot(years, hb_market, "ko-", lw=2, ms=5, label="Total market")
    ax6.axvline(2024, color="gray", ls="--", lw=1)
    ax6.set_xlabel("Year", fontsize=10)
    ax6.set_ylabel("Market [B USD]", fontsize=10)
    ax6.set_title("Hybrid Bonding Equipment Market\nBesi vs K&S シェア争い", fontsize=10)
    ax6.legend(fontsize=8); ax6.grid(alpha=0.2)

    fig.suptitle("Hybrid Bonding モデル — Cu-Cu 直接接合 (Besi / K&S)\n"
                 "接合強度 / アニール / 電気特性 / ワイヤーボンド比較",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(out_dir, "hybrid_bonding.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print("\n  Cu-Cu 接合 (200°C, 10MPa, 1s, Ra=0.3nm):")
    print(f"  {bond_strength(200, 10, 1.0, 0.3)}")
    print(f"  アニール後 (300°C, 1h): {anneal_strength_increase(300, 1):.0f} MPa")
    print(f"  10µm ピッチ 容量: {parasitic_capacitance(10):.2f} fF")
    print(f"\n[OK] Hybrid bonding -> {out}")


if __name__ == "__main__":
    plot_all()
