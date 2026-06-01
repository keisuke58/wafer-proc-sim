"""
KLA Corporation — Wafer Inspection & Process Control Model
==========================================================
世界シェア1位の検査・計測装置メーカー。
Puma (暗視野ウェーハ検査), Surfscan SP7 (表面欠陥), eDR-7xxx (e-beam),
Archer (オーバーレイ計測), CIRCL (プロセス制御) をカバー。

Key physics:
  - 暗視野スキャタリング: Mie / Rayleigh 散乱で欠陥サイズ → 散乱強度
  - SNR モデル: 光子統計 + 電気ノイズ → 最小検出欠陥サイズ
  - e-beam 分解能: de Broglie 波長 + 収差 (Scherzer criterion)
  - オーバーレイ計測誤差: ショット再現性 × スキャナ収差
  - プロセス制御 ROI: 欠陥コスト削減 vs 検査コスト
  - CD-SEM 3σ 精度: 電子ビーム幅 + ショットノイズ

実行:
    python fem/kla_model.py
    python fem/kla_model.py --inspection --tool Puma
    python fem/kla_model.py --overlay --node 2

参考文献:
    Mie (1908) Ann. Phys. — 粒子散乱理論
    Neureuther & Zach (1975) J. Opt. Soc. Am. — 暗視野検査 SNR
    Scherzer (1949) J. Appl. Phys. — e-beam 球面収差限界
    ITRS Metrology Roadmap (2023)
    KLA Annual Report / product briefs (2023)
"""

import argparse
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ════════════════════════════════════════════════════════════════════════════
# 1. 検査ツール パラメータ テーブル
# ════════════════════════════════════════════════════════════════════════════

INSPECTION_TOOLS = {
    "Puma_9900": {
        "type": "optical_darkfield",
        "wavelength_nm": 266.0,        # UV 266nm
        "NA": 0.9,
        "pixel_nm": 30.0,
        "scan_speed_mm2_s": 1200.0,
        "throughput_wph": 120,
        "sensitivity_nm": 16.0,        # min detectable defect [nm]
        "price_MUSD": 8.5,
        "color": "#2166ac",
        "ref": "KLA Puma 9900 product brief (2023)",
    },
    "Surfscan_SP7": {
        "type": "optical_brightfield",
        "wavelength_nm": 355.0,
        "NA": 0.7,
        "pixel_nm": 50.0,
        "scan_speed_mm2_s": 2000.0,
        "throughput_wph": 200,
        "sensitivity_nm": 26.0,
        "price_MUSD": 5.0,
        "color": "#d62728",
        "ref": "KLA Surfscan SP7 product brief (2023)",
    },
    "eDR_7380": {
        "type": "ebeam",
        "wavelength_nm": 0.0,          # e-beam: λ from energy
        "NA": 0.0,
        "pixel_nm": 2.0,
        "scan_speed_mm2_s": 0.5,
        "throughput_wph": 5,
        "sensitivity_nm": 2.5,
        "price_MUSD": 12.0,
        "color": "#2ca02c",
        "ref": "KLA eDR-7380 product brief (2023)",
    },
    "CIRCL": {
        "type": "broadband_plasma",
        "wavelength_nm": 200.0,        # broadband UV-VIS
        "NA": 0.85,
        "pixel_nm": 15.0,
        "scan_speed_mm2_s": 5000.0,
        "throughput_wph": 350,
        "sensitivity_nm": 20.0,
        "price_MUSD": 6.0,
        "color": "#9467bd",
        "ref": "KLA CIRCL product brief (2024)",
    },
}

OVERLAY_TOOLS = {
    "Archer_750": {
        "TMU_nm": 0.15,               # Total Measurement Uncertainty
        "throughput_wph": 280,
        "price_MUSD": 4.5,
        "target_type": "SCOL",
        "color": "#2166ac",
        "ref": "KLA Archer 750 product brief (2023)",
    },
    "Archer_1000": {
        "TMU_nm": 0.10,
        "throughput_wph": 320,
        "price_MUSD": 6.0,
        "target_type": "SCOL+DBO",
        "color": "#d62728",
        "ref": "KLA Archer 1000 product brief (2024)",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 2. Core Physics Functions
# ════════════════════════════════════════════════════════════════════════════

def scatter_signal(defect_d_nm: float, wavelength_nm: float,
                   NA: float = 0.9) -> dict:
    """
    暗視野散乱強度: Mie/Rayleigh レジーム判定 + 強度推定。

    Rayleigh (d << λ): I ∝ (d/λ)⁴  [Rayleigh 散乱]
    Mie 遷移 (d ~ λ): I ∝ (d/λ)²
    幾何光学 (d >> λ): I ∝ (d/λ)⁰ (飽和)

    集光効率 η ∝ NA²; 検出信号 S = η × I × I0

    参考: Mie (1908); Neureuther (1975)
    """
    ratio = defect_d_nm / wavelength_nm
    if ratio < 0.1:
        I_norm = ratio ** 4          # Rayleigh
        regime = "Rayleigh"
    elif ratio < 1.0:
        I_norm = ratio ** 2          # Mie transition
        regime = "Mie"
    else:
        I_norm = 1.0                 # geometric
        regime = "geometric"
    eta_collection = NA ** 2
    signal = I_norm * eta_collection * 1e6   # arbitrary units
    return {
        "defect_d_nm": defect_d_nm,
        "wavelength_nm": wavelength_nm,
        "regime": regime,
        "I_norm": round(I_norm, 8),
        "signal_au": round(signal, 4),
    }


def detection_snr(defect_d_nm: float, tool: str = "Puma_9900",
                  background_roughness_nm: float = 0.3) -> dict:
    """
    SNR = S_defect / σ_noise
    σ_noise = sqrt(σ_shot² + σ_roughness²)
    σ_shot ∝ 1/sqrt(N_photons) (ショットノイズ)
    σ_roughness ∝ background_roughness_nm / λ (表面粗さ散乱)

    SNR > 3 → 検出可能
    """
    p = INSPECTION_TOOLS.get(tool, INSPECTION_TOOLS["Puma_9900"])
    wl = p["wavelength_nm"]
    NA = p["NA"]
    sc = scatter_signal(defect_d_nm, wl, NA)
    # Shot noise: N_photons ∝ 1/pixel_area, σ_shot ∝ 1/sqrt(N)
    N_photons = 1e6 / (p["pixel_nm"] ** 2) * (wl / 266.0) ** 2
    sigma_shot = 1.0 / math.sqrt(max(N_photons, 1.0))
    sigma_rough = (background_roughness_nm / wl) ** 2
    sigma_total = math.sqrt(sigma_shot ** 2 + sigma_rough ** 2)
    snr = sc["I_norm"] / max(sigma_total, 1e-15)
    detectable = snr > 3.0

    return {
        "tool": tool,
        "defect_d_nm": defect_d_nm,
        "SNR": round(snr, 2),
        "detectable": detectable,
        "min_detectable_nm": p["sensitivity_nm"],
        "regime": sc["regime"],
    }


def ebeam_resolution(landing_energy_eV: float = 2000.0,
                     probe_current_pA: float = 50.0) -> dict:
    """
    e-beam SEM 分解能: de Broglie 波長 + Scherzer 球面収差限界。

    λ_dB = h / sqrt(2 * m * e * V) [nm]
    d_Scherzer = 0.43 * Cs^(1/4) * λ^(3/4)  [Cs ≈ 1 mm]
    d_diffraction = 0.61 * λ / NA_ebeam
    d_total = sqrt(d_Scherzer² + d_diffraction² + d_Coulomb²)

    d_Coulomb ∝ I^(1/3) / V^(4/3)  (空間電荷)

    参考: Scherzer (1949) J. Appl. Phys. 20, 20
    """
    h = 6.626e-34; m = 9.109e-31; e = 1.602e-19
    lam_m = h / math.sqrt(2 * m * e * landing_energy_eV)
    lam_nm = lam_m * 1e9
    Cs_mm = 1.0   # spherical aberration coefficient [mm]
    Cs_m = Cs_mm * 1e-3
    d_Scherzer_m = 0.43 * (Cs_m ** 0.25) * (lam_m ** 0.75)
    d_diff_m     = 0.61 * lam_m / 0.1   # NA_ebeam ≈ 0.1 rad
    d_Coulomb_m  = 2e-9 * (probe_current_pA / 50.0) ** (1.0/3.0) * (2000.0 / landing_energy_eV) ** (4.0/3.0)
    d_total_nm = math.sqrt(d_Scherzer_m**2 + d_diff_m**2 + d_Coulomb_m**2) * 1e9
    throughput_factor = math.sqrt(50.0 / probe_current_pA) * (landing_energy_eV / 2000.0)

    return {
        "landing_energy_eV": landing_energy_eV,
        "probe_current_pA": probe_current_pA,
        "lambda_dB_pm": round(lam_nm * 1000, 3),
        "d_Scherzer_nm": round(d_Scherzer_m * 1e9, 3),
        "d_total_nm": round(d_total_nm, 3),
        "throughput_relative": round(throughput_factor, 3),
    }


def overlay_error(node_nm: float, tool: str = "Archer_750",
                  n_fields: int = 100) -> dict:
    """
    オーバーレイ総誤差:
    3σ_overlay = sqrt(TMU² + scanner_residual² + process_induced²)

    scanner_residual ∝ node_nm / 10 (スケーリング)
    spec: overlay < node_nm / 5

    参考: ITRS Metrology Roadmap 2023
    """
    p = OVERLAY_TOOLS.get(tool, OVERLAY_TOOLS["Archer_750"])
    TMU = p["TMU_nm"]
    scanner_residual = node_nm / 12.0
    process_induced  = node_nm / 15.0
    overlay_3sigma = math.sqrt(TMU**2 + scanner_residual**2 + process_induced**2)
    spec_nm = node_nm / 5.0
    # Measurement uncertainty improves with sqrt(n_fields)
    TMU_effective = TMU / math.sqrt(max(n_fields, 1) / 25.0)

    return {
        "tool": tool,
        "node_nm": node_nm,
        "TMU_nm": TMU,
        "scanner_residual_nm": round(scanner_residual, 3),
        "overlay_3sigma_nm": round(overlay_3sigma, 3),
        "spec_nm": round(spec_nm, 2),
        "within_spec": overlay_3sigma < spec_nm,
        "n_fields_for_spec": int((overlay_3sigma / spec_nm) ** 2 * 25 + 1),
    }


def inspection_roi(tool: str = "Puma_9900", defect_density_per_cm2: float = 0.1,
                   die_value_usd: float = 5000.0) -> dict:
    """
    検査 ROI: 欠陥による損失コスト削減 vs 検査コスト

    Risk_reduction = defect_density * die_value * capture_rate * wafers_per_year
    Inspection_cost = (price_MUSD * 0.15 + opex) / wafers_per_year

    capture_rate: 検査ツールの欠陥捕捉率 (sensitivity 依存)
    """
    p = INSPECTION_TOOLS.get(tool, INSPECTION_TOOLS["Puma_9900"])
    wafers_per_year = p["throughput_wph"] * 8000    # 8000 h/yr utilization 85%
    inspection_cost_per_wafer = (p["price_MUSD"] * 1e6 * 0.15 + 500000) / max(wafers_per_year, 1)
    # Capture rate: smaller sensitivity → higher capture
    capture_rate = min(0.99, 0.70 + 0.30 * (50.0 / max(p["sensitivity_nm"], 1.0)))
    risk_per_wafer = defect_density_per_cm2 * 70 * die_value_usd * capture_rate  # ~70 cm² per wafer
    roi = risk_per_wafer / max(inspection_cost_per_wafer, 0.01)

    return {
        "tool": tool,
        "inspection_cost_usd_wafer": round(inspection_cost_per_wafer, 2),
        "risk_savings_usd_wafer": round(risk_per_wafer, 2),
        "capture_rate_pct": round(capture_rate * 100, 1),
        "ROI": round(roi, 1),
        "wafers_per_year": wafers_per_year,
        "payback_months": round(p["price_MUSD"] * 1e6 / max(risk_per_wafer * wafers_per_year, 1) * 12, 1),
    }


def full_pipeline(node_nm: float = 2.0, tool: str = "Puma_9900") -> dict:
    """KLA検査パイプライン: スキャン → SNR → オーバーレイ → ROI"""
    snr = detection_snr(node_nm * 5.0, tool)   # 欠陥サイズ = 5×node
    ebm = ebeam_resolution(2000.0)
    ovl = overlay_error(node_nm, "Archer_750")
    roi = inspection_roi(tool, defect_density_per_cm2=0.15)
    return {
        "node_nm": node_nm,
        "tool": tool,
        "SNR_at_5xnode": snr["SNR"],
        "detectable": snr["detectable"],
        "overlay_3sigma_nm": ovl["overlay_3sigma_nm"],
        "overlay_spec_nm": ovl["spec_nm"],
        "overlay_ok": ovl["within_spec"],
        "ebeam_resolution_nm": ebm["d_total_nm"],
        "inspection_ROI": roi["ROI"],
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. Visualization
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(17, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    ax = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    d_arr = np.linspace(5, 200, 300)

    # ── Panel 0: 散乱強度 vs 欠陥サイズ ──────────────────────────────────
    for wl, col, lbl in [(193, "#d62728", "ArF 193nm"), (266, "#2166ac", "UV 266nm"),
                          (355, "#ff7f0e", "UV 355nm"), (13.5, "#2ca02c", "EUV 13.5nm")]:
        sig = [scatter_signal(d, wl)["I_norm"] for d in d_arr]
        ax[0].loglog(d_arr, np.maximum(sig, 1e-10), lw=2.2, color=col, label=f"λ={wl}nm")
    ax[0].axvline(16, color="#2166ac", ls=":", lw=1.2, label="Puma limit 16nm")
    ax[0].set_xlabel("欠陥サイズ [nm]")
    ax[0].set_ylabel("正規化散乱強度 [a.u.]")
    ax[0].set_title("暗視野散乱強度 vs 欠陥サイズ\n(Mie/Rayleigh モデル)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.25, which="both")

    # ── Panel 1: SNR vs 欠陥サイズ (ツール比較) ──────────────────────────
    for tool, col in [("Puma_9900", "#2166ac"), ("Surfscan_SP7", "#d62728"),
                       ("CIRCL", "#9467bd")]:
        snr_arr = [detection_snr(d, tool)["SNR"] for d in d_arr]
        ax[1].semilogy(d_arr, np.maximum(snr_arr, 0.01), lw=2.2, color=col,
                       label=tool.replace("_", " "))
    ax[1].axhline(3.0, color="k", ls="--", lw=1.5, label="SNR=3 検出限界")
    ax[1].set_xlabel("欠陥サイズ [nm]")
    ax[1].set_ylabel("SNR")
    ax[1].set_title("検出 SNR vs 欠陥サイズ\n(ツール別 感度比較)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.25, which="both")
    ax[1].set_xlim(5, 200)

    # ── Panel 2: e-beam 分解能 vs 着地エネルギー ─────────────────────────
    E_arr = np.linspace(500, 5000, 100)
    for I_pA, col, lbl in [(10, "#2166ac", "10 pA"), (50, "#d62728", "50 pA"),
                            (200, "#2ca02c", "200 pA")]:
        res = [ebeam_resolution(E, I_pA)["d_total_nm"] for E in E_arr]
        ax[2].plot(E_arr, res, lw=2.2, color=col, label=lbl)
    ax[2].set_xlabel("着地エネルギー [eV]")
    ax[2].set_ylabel("総合分解能 [nm]")
    ax[2].set_title("e-beam CD-SEM 分解能\n(Scherzer + 収差 + 空間電荷)")
    ax[2].legend(fontsize=9)
    ax[2].grid(alpha=0.25)

    # ── Panel 3: オーバーレイ誤差 vs ノード ──────────────────────────────
    nodes = np.linspace(1, 20, 100)
    for otool, col in [("Archer_750", "#2166ac"), ("Archer_1000", "#d62728")]:
        ov3 = [overlay_error(n, otool)["overlay_3sigma_nm"] for n in nodes]
        sp  = [overlay_error(n, otool)["spec_nm"] for n in nodes]
        ax[3].plot(nodes, ov3, lw=2.2, color=col, label=f"{otool.replace('_',' ')} 3σ")
    ax[3].plot(nodes, [n/5 for n in nodes], "k--", lw=1.5, label="Spec=node/5")
    ax[3].set_xlabel("ノード [nm]")
    ax[3].set_ylabel("オーバーレイ誤差 [nm]")
    ax[3].set_title("オーバーレイ 3σ vs プロセスノード\n(KLA Archer 750 / 1000)")
    ax[3].legend(fontsize=8)
    ax[3].grid(alpha=0.25)

    # ── Panel 4: 検査 ROI vs 欠陥密度 ───────────────────────────────────
    D0_arr = np.linspace(0.01, 1.0, 100)
    for tool, col in [("Puma_9900", "#2166ac"), ("Surfscan_SP7", "#d62728"),
                       ("eDR_7380", "#2ca02c"), ("CIRCL", "#9467bd")]:
        roi_arr = [inspection_roi(tool, D0)["ROI"] for D0 in D0_arr]
        ax[4].plot(D0_arr, roi_arr, lw=2.2, color=col, label=tool.replace("_", " "))
    ax[4].axhline(1.0, color="k", ls="--", lw=1.5, label="ROI=1 (損益分岐)")
    ax[4].set_xlabel("欠陥密度 D₀ [/cm²]")
    ax[4].set_ylabel("検査 ROI")
    ax[4].set_title("検査 ROI vs 欠陥密度\n(ダイ価値=$5,000、年間稼働率85%)")
    ax[4].legend(fontsize=8)
    ax[4].grid(alpha=0.25)
    ax[4].set_ylim(0, 50)

    # ── Panel 5: KLA ツール比較サマリー ─────────────────────────────────
    ax[5].axis("off")
    col_labels = ["ツール", "方式", "感度\n[nm]", "WPH", "価格\n[$M]", "ROI\n(D₀=0.1)"]
    tdata = []
    for tname, tp in INSPECTION_TOOLS.items():
        r = inspection_roi(tname, 0.1)
        tdata.append([tname.replace("_", " "), tp["type"].replace("_", " "),
                      f"{tp['sensitivity_nm']:.0f}", f"{tp['throughput_wph']}",
                      f"${tp['price_MUSD']:.1f}M", f"{r['ROI']:.0f}×"])
    tbl = ax[5].table(cellText=tdata, colLabels=col_labels,
                      cellLoc="center", loc="center", bbox=[0.0, 0.10, 1.0, 0.85])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1b4a8a"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#eef3ff")
    ax[5].set_title("KLA ツール比較サマリー\nPuma · Surfscan · eDR · CIRCL")

    fig.suptitle(
        "KLA Corporation — ウェーハ検査・計測物理モデル\n"
        "Puma 9900 · Surfscan SP7 · eDR-7380 · Archer オーバーレイ",
        fontsize=12, fontweight="bold")
    out = os.path.join(out_dir, "kla_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] KLA model → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="KLA inspection/metrology model")
    ap.add_argument("--inspection", action="store_true")
    ap.add_argument("--overlay",    action="store_true")
    ap.add_argument("--tool",  default="Puma_9900", choices=list(INSPECTION_TOOLS.keys()))
    ap.add_argument("--node",  type=float, default=2.0)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    if args.inspection:
        for d in [10, 16, 20, 30, 50, 100]:
            r = detection_snr(d, args.tool)
            print(f"  d={d:3d}nm  SNR={r['SNR']:8.2f}  detect={r['detectable']}  {r['regime']}")
    if args.overlay:
        r = overlay_error(args.node, "Archer_1000")
        print(f"  node={args.node}nm  overlay_3σ={r['overlay_3sigma_nm']:.3f}nm  "
              f"spec={r['spec_nm']:.2f}nm  ok={r['within_spec']}")
    if not args.no_plot:
        plot_all()

if __name__ == "__main__":
    main()
