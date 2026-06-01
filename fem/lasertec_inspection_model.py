"""
Lasertec 検査モデル — SiC ウェーハ + EUV マスク検査
=====================================================
Lasertec (6920.T) の2大製品ラインを物理モデル化:
  1. SICA シリーズ   — SiC ウェーハ欠陥検査 (Disco と連携)
  2. ABICS シリーズ  — EUV フォトマスク検査 (ASML と連携)

検査原理:
  SiC: レーザー散乱 + 蛍光検出 → ピット / マイクロパイプ / BPD 欠陥
  EUV: EUV 光源 (13.5nm) で反射率マップ → 吸収体欠陥 < 20nm 検出

実行:
    python fem/lasertec_inspection_model.py
"""

import math, os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ════════════════════════════════════════════════════════════════════════════
# 1. SiC ウェーハ欠陥検査 (Lasertec SICA)
# ════════════════════════════════════════════════════════════════════════════

def laser_scatter_signal(defect_size_nm: float, wavelength_nm: float = 355,
                          defect_type: str = "pit") -> float:
    """
    レーザー散乱強度 [arb. units]。
    Mie 散乱近似: I ∝ (d/λ)^4 for d << λ (Rayleigh),  ∝ (d/λ)^2 for d~λ
    """
    ratio = defect_size_nm / wavelength_nm
    if ratio < 0.1:   # Rayleigh 域
        I = ratio ** 4
    elif ratio < 1.0: # Mie 遷移域
        I = ratio ** 2.5
    else:             # 幾何光学域
        I = ratio ** 2

    # 欠陥種別による散乱効率補正
    type_factor = {"pit": 1.0, "micropipe": 2.5, "BPD": 0.6, "hill": 0.8}
    return I * type_factor.get(defect_type, 1.0)


def detection_limit(wavelength_nm: float, NA: float = 0.9,
                     SNR_required: float = 3.0) -> float:
    """
    検出限界欠陥サイズ [nm]。
    半波長分解能 + SNR 要求から算出。
    """
    # アッベ分解能
    abbe_nm = wavelength_nm / (2 * NA)
    # SNR による補正 (小さな欠陥は SNR が低い)
    return abbe_nm / math.sqrt(SNR_required) * 1.5


def wafer_inspection_time(D_mm: float, pixel_size_nm: float,
                           scan_speed_mm_s: float = 200) -> float:
    """
    ウェーハ全面検査時間 [min]。
    """
    A_wafer = math.pi * (D_mm / 2) ** 2   # mm²
    pixels_total = A_wafer * 1e6 / (pixel_size_nm / 1000) ** 2
    scan_lines = D_mm / (pixel_size_nm / 1e6)  # ライン数
    time_s = A_wafer / scan_speed_mm_s / (pixel_size_nm / 1e6) * 0.001
    return time_s / 60


def defect_map_simulate(n_defects: int = 50, wafer_D_mm: float = 150,
                         rng=None) -> np.ndarray:
    """
    SiC ウェーハ欠陥マップをシミュレーション。
    BPD: 面内均一分布, マイクロパイプ: 中心集中
    """
    rng = rng or np.random.default_rng(42)
    R = wafer_D_mm / 2
    # BPD: ランダム分布
    n_bpd = int(n_defects * 0.7)
    theta = rng.uniform(0, 2*math.pi, n_bpd)
    r_bpd = R * np.sqrt(rng.uniform(0, 1, n_bpd))
    x_bpd = r_bpd * np.cos(theta)
    y_bpd = r_bpd * np.sin(theta)
    # マイクロパイプ: 中心付近集中
    n_mp = n_defects - n_bpd
    x_mp = rng.normal(0, R*0.2, n_mp)
    y_mp = rng.normal(0, R*0.2, n_mp)
    # [x, y, type(0=BPD,1=MP)]
    bpd = np.column_stack([x_bpd, y_bpd, np.zeros(n_bpd)])
    mp  = np.column_stack([x_mp,  y_mp,  np.ones(n_mp)])
    return np.vstack([bpd, mp])


# ════════════════════════════════════════════════════════════════════════════
# 2. EUV マスク検査 (Lasertec ABICS)
# ════════════════════════════════════════════════════════════════════════════

EUV_PARAMS = {
    "wavelength_nm": 13.5,     # EUV 波長
    "NA": 0.08,                # 検査光学系 NA (低め)
    "Mo_Si_reflectance": 0.70, # Mo/Si 多層膜反射率
    "defect_contrast_min": 0.01,  # 検出可能最小コントラスト
}


def euv_mask_reflectance_map(mask_size_mm: float = 152,
                              n_defects: int = 5, rng=None) -> tuple:
    """
    EUV マスク反射率マップをシミュレーション。
    欠陥位置で反射率が低下する。
    """
    rng = rng or np.random.default_rng(42)
    N = 256
    R_map = np.ones((N, N)) * EUV_PARAMS["Mo_Si_reflectance"]
    # ノイズ追加
    R_map += rng.normal(0, 0.003, (N, N))

    # 欠陥: 局所的な反射率低下
    defect_coords = []
    for _ in range(n_defects):
        ix = rng.integers(10, N-10)
        iy = rng.integers(10, N-10)
        size_px = rng.integers(1, 4)
        defect_type = rng.choice(["absorber", "phase", "particle"])
        dR = {"absorber": -0.15, "phase": -0.05, "particle": -0.08}[defect_type]
        R_map[ix-size_px:ix+size_px, iy-size_px:iy+size_px] += dR
        defect_coords.append((ix, iy, defect_type, abs(dR)))

    R_map = np.clip(R_map, 0, 1)
    return R_map, defect_coords


def euv_defect_detectability(defect_size_nm: float,
                               defect_type: str = "absorber") -> dict:
    """
    EUV マスク欠陥の検出可能性。
    SNR = contrast * sqrt(N_photons) / noise_floor
    """
    p = EUV_PARAMS
    contrast = {"absorber": 0.20, "phase": 0.08, "particle": 0.12}.get(defect_type, 0.10)

    # 検出限界: ABICS の実仕様 ~20nm (公開情報)
    detect_limit_nm = p["wavelength_nm"] / (2 * p["NA"]) * 0.5  # 実効分解能
    # SNR ∝ (size/limit)^2
    SNR = (defect_size_nm / detect_limit_nm) ** 2 * contrast / p["defect_contrast_min"]

    return {
        "defect_size_nm":   defect_size_nm,
        "defect_type":      defect_type,
        "SNR":              round(SNR, 2),
        "detectable":       SNR >= 3.0,
        "detect_limit_nm":  round(detect_limit_nm, 1),
    }


# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    rng = np.random.default_rng(42)

    # Panel 1: レーザー散乱強度 vs 欠陥サイズ
    ax = axes[0, 0]
    sizes = np.logspace(1, 4, 200)  # 10nm - 10µm
    for wl, col, lbl in [(266,"#d62728","266nm (DUV)"),(355,"#2166ac","355nm (UV)"),(532,"#2ca02c","532nm (Green)")]:
        sig = [laser_scatter_signal(s, wl) for s in sizes]
        ax.loglog(sizes, sig, lw=2.5, color=col, label=lbl)
        dl = detection_limit(wl)
        ax.axvline(dl, color=col, ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("Defect size [nm]", fontsize=10)
    ax.set_ylabel("Scatter signal [arb.]", fontsize=10)
    ax.set_title("SiC Defect Detection\nLaser Scatter vs Defect Size", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.2, which="both")
    ax.text(15, 1e-6, "← 検出限界", fontsize=8, color="gray")

    # Panel 2: SiC ウェーハ欠陥マップ
    ax2 = axes[0, 1]
    defects = defect_map_simulate(80, 150, rng)
    bpd = defects[defects[:,2]==0]
    mp  = defects[defects[:,2]==1]
    circle = plt.Circle((0,0), 75, fill=False, color="gray", lw=1.5)
    ax2.add_patch(circle)
    ax2.scatter(bpd[:,0], bpd[:,1], c="#ff7f0e", s=15, alpha=0.7, label=f"BPD ({len(bpd)})")
    ax2.scatter(mp[:,0],  mp[:,1],  c="#d62728", s=40, marker="x", lw=1.5, label=f"Micropipe ({len(mp)})")
    ax2.set_xlim(-85, 85); ax2.set_ylim(-85, 85)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x [mm]"); ax2.set_ylabel("y [mm]")
    ax2.set_title("SiC Wafer Defect Map (150mm)\nLasertec SICA simulation", fontsize=10)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.2)

    # Panel 3: 検査時間 vs 解像度
    ax3 = axes[0, 2]
    pixel_sizes = np.linspace(50, 500, 100)
    for D, col, lbl in [(150,"#2166ac","150mm (6in)"),(200,"#d62728","200mm (8in)"),(300,"#ff7f0e","300mm (12in)")]:
        times = [wafer_inspection_time(D, ps) for ps in pixel_sizes]
        ax3.semilogy(pixel_sizes, times, lw=2.5, color=col, label=lbl)
    ax3.axvline(100, color="purple", ls="--", lw=1.5, label="SiC 標準 100nm pixel")
    ax3.axhline(5, color="green", ls=":", lw=1.5, label="目標 5 min/wafer")
    ax3.set_xlabel("Pixel size [nm]", fontsize=10)
    ax3.set_ylabel("Inspection time [min]", fontsize=10)
    ax3.set_title("Inspection Time vs Resolution\nThroughput-Quality Tradeoff", fontsize=10)
    ax3.legend(fontsize=8); ax3.grid(alpha=0.2, which="both")

    # Panel 4: EUV マスク反射率マップ
    ax4 = axes[1, 0]
    R_map, defect_coords = euv_mask_reflectance_map(rng=rng)
    im = ax4.imshow(R_map, cmap="RdYlGn", vmin=0.5, vmax=0.8, aspect="equal")
    plt.colorbar(im, ax=ax4, label="Reflectance")
    for (ix, iy, dtype, dR) in defect_coords:
        marker = {"absorber":"x","phase":"o","particle":"^"}[dtype]
        ax4.plot(iy, ix, marker, ms=10, color="blue", markeredgewidth=2)
    ax4.set_title("EUV Mask Reflectance Map\n(Lasertec ABICS simulation)", fontsize=10)
    ax4.set_xlabel("x [pixels]"); ax4.set_ylabel("y [pixels]")

    # Panel 5: EUV 欠陥検出可能性 vs サイズ
    ax5 = axes[1, 1]
    sizes5 = np.linspace(5, 100, 100)
    for dtype, col, lbl in [("absorber","#d62728","Absorber defect"),
                              ("phase","#2166ac","Phase defect"),
                              ("particle","#ff7f0e","Particle")]:
        snrs = [euv_defect_detectability(s, dtype)["SNR"] for s in sizes5]
        ax5.plot(sizes5, snrs, lw=2.5, color=col, label=lbl)
    ax5.axhline(3.0, color="purple", ls="--", lw=1.5, label="SNR=3 (検出閾値)")
    dl = euv_defect_detectability(20)["detect_limit_nm"]
    ax5.axvline(dl, color="gray", ls=":", lw=1.5, label=f"検出限界 ~{dl:.0f}nm")
    ax5.set_xlabel("Defect size [nm]", fontsize=10)
    ax5.set_ylabel("SNR", fontsize=10)
    ax5.set_title("EUV Mask Defect Detectability\n(ABICS, λ=13.5nm)", fontsize=10)
    ax5.legend(fontsize=8); ax5.grid(alpha=0.2); ax5.set_ylim(0, 30)

    # Panel 6: Lasertec 製品・市場サマリー
    ax6 = axes[1, 2]
    ax6.axis("off")
    rows = [
        ["製品", "用途", "分解能", "スループット", "対象顧客"],
        ["SICA88",   "SiC ウェーハ",    "< 50nm",  "150mm: 5min", "フェローテック\nWolfspeed"],
        ["SICA89",   "SiC 200mm対応",   "< 40nm",  "200mm: 8min", "フェローテック\nST Micro"],
        ["ABICS11",  "EUV マスク",       "< 20nm",  "30min/mask",  "TSMC / Intel\nSamsung"],
        ["ABICS12",  "EUV HV (高電圧)", "< 16nm",  "45min/mask",  "TSMC 3nm+\nASML顧客"],
    ]
    tbl = ax6.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    tbl.scale(1.3, 2.2)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0: cell.set_facecolor("#1a3a5c"); cell.set_text_props(color="w")
        elif r <= 2: cell.set_facecolor("#fff3cd")  # SiC 系
        else: cell.set_facecolor("#d1ecf1")          # EUV 系
    ax6.set_title("Lasertec 製品ラインアップ\nSiC ÷ EUV 二本柱", fontsize=10)

    fig.suptitle("Lasertec (6920) 検査モデル — SiC ウェーハ × EUV マスク\n"
                 "SICA (SiC欠陥) / ABICS (EUVマスク) 物理シミュレーション",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(out_dir, "lasertec_inspection.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  SiC 検出限界 (355nm): {detection_limit(355):.0f} nm")
    print(f"  EUV 欠陥検出 (20nm absorber): {euv_defect_detectability(20, 'absorber')}")
    print(f"\n[OK] Lasertec inspection -> {out}")


if __name__ == "__main__":
    plot_all()
