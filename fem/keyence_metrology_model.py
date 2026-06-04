"""
Keyence VK-X 共焦点レーザー顕微鏡 — SiC ダイシング計測モデル
=============================================================
Keyence VK-X3100 (405 nm 紫レーザー) による 4H-SiC ブレードダイシング後の
チッピング幅計測を物理モデル化する。

計測原理:
  共焦点光学系によりピンホール面での光強度が z 依存のガウシアン応答を示す。
  チッピングエッジの 3D 形状を nm スケールで再構成し、チッピング幅 c_chip を
  サブミクロン精度で算出する。

モジュールの役割:
  1. VK-X 計測シミュレーション (横・深さ分解能、SiC 反射率、ノイズ)
  2. 計測手法別 σ_meas モデル (VK-X / 光学顕微鏡 / 目測)
  3. データ品質グレード (A/B/C/D) の計測ツール起点での自動格付け
  4. ROI 計算: VK-X 導入コスト vs 歩留まり改善利益
  5. GP サロゲート精度への伝播解析 (wafer-proc-sim と統合)

実行:
    python fem/keyence_metrology_model.py
    python fem/keyence_metrology_model.py --roi
    python fem/keyence_metrology_model.py --sigma-map

参考文献:
    Keyence VK-X3100 仕様 (公開情報, 2024)
    Born & Wolf (2013) Principles of Optics — confocal theory
    Nishioka K. et al. (2025) Frontiers — heteroscedastic GP for SiC dicing
    Wang Y. et al. (2026) Micromachines 17(2):187 — SiC dicing dataset
"""

import argparse
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import List, Optional

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ════════════════════════════════════════════════════════════════════════════
# 0. 機器仕様 (VK-X3100 公開スペック)
# ════════════════════════════════════════════════════════════════════════════

VKX_SPEC = {
    "wavelength_nm":     405.0,   # 紫半導体レーザー
    "NA_100x":             0.95,  # 対物レンズ 100×: 最高 NA
    "NA_50x":              0.80,  # 50× 対物レンズ
    "NA_20x":              0.45,  # 20× 対物レンズ
    "lateral_res_nm":    260.0,   # 横分解能 (Rayleigh: 0.61λ/NA @ 100×)
    "depth_res_nm":      225.0,   # 深さ分解能 δz = nλ/(2NA²), n=1
    "height_noise_nm":     1.0,   # 標準高さノイズ [nm] (公称値)
    "stage_repeat_nm":    10.0,   # ステージ位置再現性 [nm]
    "sic_reflectance":   0.204,   # 4H-SiC @405 nm: ((n-1)/(n+1))², n≈2.65
    "throughput_wph":     12.0,   # ダイシング後計測スループット [wafers/hr]
    "price_jpy_M":        25.0,   # 参考機器価格 [百万円] (高倍対物込み)
}

# 計測手法別 σ_meas [µm] (チッピング幅 1σ 不確かさ, SiC 典型運用条件)
TOOL_SIGMA = {
    "vkx_100x": 0.28,   # VK-X 100× → Grade A
    "vkx_50x":  0.45,   # VK-X 50×  → Grade A
    "sem":      0.15,   # SEM        → Grade A (最高精度、低スループット)
    "opt_100x": 1.20,   # 光学顕微鏡 100× → Grade B
    "opt_50x":  1.80,   # 光学顕微鏡 50×  → Grade C
    "visual":   4.50,   # 目測 / 粗い画像推定 → Grade D
}

# Grade → σ_meas [µm] (GP ヘテロセダスティックノイズ入力用)
GRADE_TO_SIGMA = {"A": 0.30, "B": 0.80, "C": 1.50, "D": 3.50}
GRADE_TO_TOOL  = {"A": "vkx_100x / SEM", "B": "opt_100x",
                  "C": "opt_50x",         "D": "visual / estimated"}


# ════════════════════════════════════════════════════════════════════════════
# 1. SiC 光学物性
# ════════════════════════════════════════════════════════════════════════════

def sic_refractive_index(wavelength_nm: float = 405.0) -> float:
    """4H-SiC 実部屈折率 (Sellmeier 近似, Shaffer 1971 常光線)."""
    lam = wavelength_nm
    n2 = 1.0 + 6.72 * lam**2 / (lam**2 - 282.0**2)
    return math.sqrt(max(n2, 1.01))


def sic_reflectance(wavelength_nm: float = 405.0) -> float:
    """4H-SiC 垂直入射反射率."""
    n = sic_refractive_index(wavelength_nm)
    return ((n - 1.0) / (n + 1.0))**2


def chipping_scatter_factor(roughness_rms_nm: float,
                            wavelength_nm: float = 405.0) -> float:
    """
    チッピング面の有効反射率補正係数 (Beckmann–Spizzichino 近似).
    Strehl ≈ exp(-(4π σ_Ra / λ)²)
    """
    strehl = math.exp(-(4.0 * math.pi * roughness_rms_nm / wavelength_nm)**2)
    return max(strehl, 0.01)


# ════════════════════════════════════════════════════════════════════════════
# 2. 共焦点 PSF・深さ応答
# ════════════════════════════════════════════════════════════════════════════

def confocal_depth_response(z_offset_nm: np.ndarray,
                            NA: float = 0.95,
                            wavelength_nm: float = 405.0) -> np.ndarray:
    """
    共焦点軸方向強度応答 I(z) — Stokseth 近似.
    I(z) = sinc²(u/2π),  u = (8π/λ) NA² z
    """
    u = (8.0 * math.pi / wavelength_nm) * NA**2 * z_offset_nm
    return np.sinc(u / (2.0 * math.pi))**2


def confocal_lateral_psf(r_nm: np.ndarray,
                         NA: float = 0.95,
                         wavelength_nm: float = 405.0) -> np.ndarray:
    """共焦点横方向 PSF (Airy 関数の Gaussian 近似)."""
    sigma_nm = 0.61 * wavelength_nm / NA / 2.355
    return np.exp(-0.5 * (r_nm / sigma_nm)**2)


# ════════════════════════════════════════════════════════════════════════════
# 3. チッピングエッジ計測シミュレーション
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ChipProfile:
    tool:          str
    true_width_um: float
    meas_width_um: float
    sigma_meas_um: float
    snr:           float
    grade:         str


def simulate_chip_measurement(
    true_width_um: float,
    tool:          str = "vkx_100x",
    Ra_nm:         float = 150.0,
    rng:           Optional[np.random.Generator] = None,
) -> ChipProfile:
    """
    チッピング幅の単回計測をシミュレート。
    真値 c_chip に σ_meas のガウシアンノイズを加える。
    SNR は共焦点 PSF × SiC 反射率 × 粗さ散乱から物理推定。
    """
    if rng is None:
        rng = np.random.default_rng()

    sigma_base = TOOL_SIGMA.get(tool, 4.5)

    if tool.startswith("vkx"):
        NA = VKX_SPEC["NA_100x"] if "100x" in tool else VKX_SPEC["NA_50x"]
        base_R  = sic_reflectance(VKX_SPEC["wavelength_nm"])
        scatter = chipping_scatter_factor(Ra_nm, VKX_SPEC["wavelength_nm"])
        eff_R   = base_R * scatter
        snr = 40.0 * math.sqrt(eff_R / 0.20)   # 基準: 20% 反射率で SNR=40
    elif tool == "sem":
        snr = 120.0
    elif tool.startswith("opt"):
        snr = 8.0
    else:
        snr = 2.5

    roughness_penalty = 1.0 + max(0.0, (Ra_nm - 100.0) / 500.0)
    sigma_eff = sigma_base * roughness_penalty

    meas = max(0.1, true_width_um + rng.normal(0.0, sigma_eff))

    if sigma_eff < 0.50:
        grade = "A"
    elif sigma_eff < 1.00:
        grade = "B"
    elif sigma_eff < 2.50:
        grade = "C"
    else:
        grade = "D"

    return ChipProfile(tool=tool, true_width_um=true_width_um,
                       meas_width_um=meas, sigma_meas_um=sigma_eff,
                       snr=snr, grade=grade)


def batch_measurement(true_widths: np.ndarray, tool: str,
                      Ra_nm: float = 150.0, seed: int = 42) -> List[ChipProfile]:
    rng = np.random.default_rng(seed)
    return [simulate_chip_measurement(w, tool, Ra_nm, rng) for w in true_widths]


# ════════════════════════════════════════════════════════════════════════════
# 4. σ_meas vs 表面粗さ
# ════════════════════════════════════════════════════════════════════════════

def sigma_vs_roughness(Ra_range_nm: np.ndarray,
                       tools: List[str]) -> dict:
    results = {}
    for tool in tools:
        sigma_base = TOOL_SIGMA.get(tool, 4.5)
        if tool.startswith("vkx"):
            scatter = np.array([chipping_scatter_factor(ra) for ra in Ra_range_nm])
            penalty = 1.0 + np.maximum(0.0, (Ra_range_nm - 100.0) / 500.0)
            sigma_arr = sigma_base * penalty / np.sqrt(np.maximum(scatter / 0.20, 0.01))
        else:
            penalty = 1.0 + np.maximum(0.0, (Ra_range_nm - 80.0) / 200.0)
            sigma_arr = sigma_base * penalty
        results[tool] = sigma_arr
    return results


# ════════════════════════════════════════════════════════════════════════════
# 5. ROI 分析
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ROIScenario:
    name:           str
    color:          str
    gp_rmse_um:     float
    yield_loss_pct: float
    wafers_per_yr:  int
    die_price_usd:  float


ROI_SCENARIOS = [
    ROIScenario("VK-X (Grade A)",        "#1f77b4", 1.62, 2.1, 50_000, 12.0),
    ROIScenario("光学顕微鏡 (Grade C)",  "#ff7f0e", 2.50, 3.8, 50_000, 12.0),
    ROIScenario("目測 (Grade D)",        "#d62728", 3.12, 5.5, 50_000, 12.0),
]

DIES_PER_WAFER = 400   # 4H-SiC 150 mm ウェーハ典型値


def compute_roi(scenario: ROIScenario,
                vkx_price_jpy_M: float = 25.0,
                usd_jpy: float = 150.0,
                years: int = 5) -> dict:
    """VK-X 導入 ROI (Grade D baseline との歩留まり損失差分で算出)."""
    baseline = ROI_SCENARIOS[-1]
    yield_delta_pct = baseline.yield_loss_pct - scenario.yield_loss_pct
    extra_dies_yr   = yield_delta_pct / 100.0 * scenario.wafers_per_yr * DIES_PER_WAFER
    revenue_yr      = extra_dies_yr * scenario.die_price_usd
    revenue_5yr     = revenue_yr * years
    capex           = vkx_price_jpy_M * 1e6 / usd_jpy
    net_5yr         = revenue_5yr - capex
    payback         = capex / revenue_yr if revenue_yr > 0 else float("inf")
    return dict(yield_delta_pct=yield_delta_pct, extra_dies_yr=extra_dies_yr,
                revenue_yr=revenue_yr, revenue_5yr=revenue_5yr,
                capex=capex, net_5yr=net_5yr, payback_yr=payback)


# ════════════════════════════════════════════════════════════════════════════
# 6. 可視化
# ════════════════════════════════════════════════════════════════════════════

def plot_main(save_dir: str = OUT_DIR) -> None:
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(15, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    # (A) 共焦点深さ応答
    ax = axes[0]
    z_nm = np.linspace(-2000, 2000, 500)
    for NA, label, ls in [(0.95, "VK-X 100× NA=0.95", "-"),
                          (0.80, "VK-X 50×  NA=0.80",  "--"),
                          (0.45, "20×      NA=0.45",   ":")]:
        ax.plot(z_nm, confocal_depth_response(z_nm, NA), ls=ls, label=label, lw=1.8)
    ax.set_xlabel("Defocus z [nm]")
    ax.set_ylabel("Normalised intensity")
    ax.set_title("(A) Confocal depth response")
    ax.legend(fontsize=7.5)

    # (B) チッピングエッジプロファイル
    ax = axes[1]
    x_um = np.linspace(-3, 12, 300)
    true_edge = 5.0
    chip_h    = 4.0
    tool_styles = [("vkx_100x", "C0", 1.0, 0.3), ("opt_50x", "C1", 0.8, 1.5),
                   ("visual", "C3", 0.6, 4.0)]
    for tool, color, alpha, sigma in tool_styles:
        base = np.where(x_um < true_edge,
                        chip_h * np.exp(-0.5 * ((x_um - true_edge) / sigma)**2), 0.0)
        noise = rng.normal(0, sigma * 0.3, len(x_um)) if tool == "vkx_100x" \
                else rng.normal(0, sigma * 0.5, len(x_um))
        ax.plot(x_um, base + noise, color=color, alpha=alpha, lw=1.5,
                label=f"{tool}  σ={sigma:.2f}µm")
    ax.axvline(true_edge, color="k", lw=0.8, ls="--", label="True edge")
    ax.set_xlabel("Position [µm]")
    ax.set_ylabel("Height [µm]")
    ax.set_title("(B) Chipping edge profile (simulated)")
    ax.legend(fontsize=7)

    # (C) σ_meas vs 表面粗さ
    ax = axes[2]
    Ra_arr = np.linspace(50, 600, 200)
    tools_c  = ["vkx_100x", "opt_100x", "opt_50x", "visual"]
    colors_c = ["C0", "C2", "C1", "C3"]
    labels_c = ["VK-X 100× (Grade A)", "Opt. 100× (Grade B)",
                "Opt. 50× (Grade C)", "Visual (Grade D)"]
    smap = sigma_vs_roughness(Ra_arr, tools_c)
    for t, col, lab in zip(tools_c, colors_c, labels_c):
        ax.plot(Ra_arr, smap[t], color=col, label=lab, lw=1.8)
    ax.axhline(0.5, color="gray", ls=":",  lw=1.0, label="A/B boundary")
    ax.axhline(2.5, color="gray", ls="--", lw=1.0, label="C/D boundary")
    ax.set_xlabel("Surface roughness Ra [nm]")
    ax.set_ylabel("σ_meas [µm]")
    ax.set_title("(C) σ_meas vs roughness")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 8)

    # (D) データ点数 vs GP LOO-RMSE
    ax = axes[3]
    n_arr  = np.arange(5, 26)
    rmse_a = 3.5 / np.sqrt(n_arr / 5) * 0.52
    rmse_c = 3.5 / np.sqrt(n_arr / 5) * 0.80
    rmse_d = 3.5 / np.sqrt(n_arr / 5)
    ax.plot(n_arr, rmse_a, "C0-o", ms=4, label="VK-X Grade A")
    ax.plot(n_arr, rmse_c, "C1-s", ms=4, label="Opt. Grade C")
    ax.plot(n_arr, rmse_d, "C3-^", ms=4, label="Mixed / Grade D")
    ax.axhline(1.62, color="C0", ls=":", lw=1.0, label="Paper: Het-GP A n=11 (1.62µm)")
    ax.axhline(3.12, color="C3", ls=":", lw=1.0, label="Paper: Homo-GP all (3.12µm)")
    ax.set_xlabel("Number of data points n")
    ax.set_ylabel("LOO-RMSE [µm]")
    ax.set_title("(D) Data quality vs GP accuracy")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 4.5)

    # (E) 歩留まり損失
    ax = axes[4]
    lab_e = [s.name.split("(")[0].strip() for s in ROI_SCENARIOS]
    col_e = [s.color for s in ROI_SCENARIOS]
    yl_e  = [s.yield_loss_pct for s in ROI_SCENARIOS]
    bars = ax.bar(lab_e, yl_e, color=col_e, alpha=0.85, edgecolor="k", lw=0.7)
    for b, v in zip(bars, yl_e):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Chipping yield loss [%]")
    ax.set_title("(E) Yield loss by measurement tool")
    ax.set_ylim(0, 7)

    # (F) 5年 ROI 累積
    ax = axes[5]
    roi = compute_roi(ROI_SCENARIOS[0])
    years_arr = np.arange(0, 6)
    cumul = roi["revenue_yr"] * years_arr - roi["capex"]
    ax.plot(years_arr, cumul / 1e6, "C0-o", ms=5, lw=2)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.fill_between(years_arr, cumul / 1e6, 0,
                    where=cumul >= 0, alpha=0.15, color="C0", label="Net profit")
    ax.fill_between(years_arr, cumul / 1e6, 0,
                    where=cumul <  0, alpha=0.15, color="C3", label="Net loss")
    ax.axvline(roi["payback_yr"], color="C2", ls=":", lw=1.5,
               label=f"Payback {roi['payback_yr']:.1f} yr")
    ax.set_xlabel("Years after VK-X acquisition")
    ax.set_ylabel("Cumulative net benefit [M USD]")
    ax.set_title(f"(F) VK-X ROI (¥{VKX_SPEC['price_jpy_M']:.0f}M, "
                 f"{ROI_SCENARIOS[0].wafers_per_yr//1000}k wfr/yr)")
    ax.legend(fontsize=7.5)

    fig.suptitle(
        "Keyence VK-X3100 Confocal Profilometry — SiC Blade Dicing Metrology",
        fontsize=12, fontweight="bold")
    path = os.path.join(save_dir, "keyence_metrology.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[keyence_metrology] saved → {path}")


def print_roi_table() -> None:
    print("\n" + "=" * 65)
    print(f"{'Keyence VK-X ROI Analysis':^65}")
    print("=" * 65)
    for s in ROI_SCENARIOS:
        roi = compute_roi(s)
        print(f"\n  {s.name}")
        print(f"    GP RMSE              : {s.gp_rmse_um:.2f} µm")
        print(f"    Yield loss           : {s.yield_loss_pct:.1f}%")
        print(f"    Yield improvement    : +{roi['yield_delta_pct']:.1f}% vs Grade D")
        print(f"    Extra dies / yr      : {roi['extra_dies_yr']:,.0f}")
        print(f"    Revenue gain 5yr     : ${roi['revenue_5yr']:,.0f}")
        print(f"    VK-X capex           : ${roi['capex']:,.0f}")
        print(f"    Net benefit 5yr      : ${roi['net_5yr']:,.0f}")
        if roi["payback_yr"] < 50:
            print(f"    Payback period       : {roi['payback_yr']:.2f} yr")


# ════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Keyence VK-X metrology model")
    ap.add_argument("--roi",       action="store_true", help="Print ROI table only")
    ap.add_argument("--sigma-map", action="store_true", help="Print σ_meas table only")
    args = ap.parse_args()

    if args.roi:
        print_roi_table()
        return

    if args.sigma_map:
        print(f"\n  {'Tool':<14}  σ_meas [µm]  Grade")
        for t, s in TOOL_SIGMA.items():
            g = "A" if s < 0.50 else "B" if s < 1.00 else "C" if s < 2.50 else "D"
            print(f"  {t:<14}  {s:.2f}         {g}")
        return

    print("[keyence_metrology] Running full analysis...")
    print_roi_table()
    plot_main()
    print("[keyence_metrology] Done.")


if __name__ == "__main__":
    main()
