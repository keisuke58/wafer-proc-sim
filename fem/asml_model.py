"""
ASML EUV リソグラフィー モデル
================================
TWINSCAN NXE:3600D / EXE:5000 (High-NA) 相当の露光シミュレーション。

Lasertec ACTIS (マスク検査) → ASML EUV 露光 → TEL 現像 という
フロントエンドの核心を物理モデル化。

1. 空中像 (Aerial Image) — Abbe 結像理論の簡略版
   EUV 光源 (13.5nm) + 反射型光学系
   → 空中像コントラスト / 解像限界

2. レジスト露光 (PRISM モデル)
   露光量 → CD (Critical Dimension) 変換
   EUV レジスト感度 / LWR (Line Width Roughness)

3. プロセスウィンドウ
   Focus-Exposure Matrix → DOF × EL
   2nm ノード (k1=0.33) での余裕度

4. スループット × コスト
   WPH (Wafers Per Hour) vs NA × 光源パワー
   High-NA EUV の ROI

5. ASML → Lasertec → ダイシングへの接続
   CD ばらつき → V_th ばらつき → ソシオネクスト タイミング yield

実行:
    python fem/asml_model.py
    python fem/asml_model.py --aerial
    python fem/asml_model.py --process-window
    python fem/asml_model.py --throughput
"""

import argparse
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0  # Bessel function

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ── ASML 機種パラメータ ─────────────────────────────────────────────────────
ASML_SYSTEMS = {
    "NXE:3600D": {
        "NA": 0.33, "wavelength_nm": 13.5, "sigma": 0.87,
        "source_W": 250, "WPH_max": 170, "price_M_usd": 180,
        "node": "5-3nm", "color": "#2166ac",
        "ref": "ASML 2023 Annual Report",
    },
    "EXE:5000 (High-NA)": {
        "NA": 0.55, "wavelength_nm": 13.5, "sigma": 0.90,
        "source_W": 500, "WPH_max": 200, "price_M_usd": 380,
        "node": "2nm以下", "color": "#d62728",
        "ref": "ASML EXE:5000 spec",
    },
    "NXT:2050i (DUV)": {
        "NA": 1.35, "wavelength_nm": 193, "sigma": 0.85,
        "source_W": 90, "WPH_max": 295, "price_M_usd": 80,
        "node": "7-10nm", "color": "#2ca02c",
        "ref": "ASML NXT spec",
    },
}


# ════════════════════════════════════════════════════════════════════════════
# 1. 空中像 (Abbe 結像理論 — 簡略版)
# ════════════════════════════════════════════════════════════════════════════

def aerial_image(pitch_nm: float, CD_nm: float,
                 NA: float = 0.55, wavelength_nm: float = 13.5,
                 sigma: float = 0.87) -> dict:
    """
    1D 周期パターンの空中像強度分布 I(x)。

    Abbe 法: I(x) = Σ_m Σ_n T_m T_n* × s_coherence(m,n) × cos(2π(m-n)x/p)
    簡略化: コヒーレントTCC + partial coherence 補正。

    k1 = 0.25 × pitch / (λ/NA):  小さいほど解像が難しい
    NILS (Normalized Image Log Slope): コントラストの指標
    """
    k1     = 0.5 * CD_nm / (wavelength_nm / NA)
    # 解像限界チェック: k1 < 0.25 は物理的に不可能
    resolvable = k1 >= 0.25

    # 簡略正弦波モデル: I(x) = 0.5(1 + V·cos(2πx/p))
    xs = np.linspace(-pitch_nm/2, pitch_nm/2, 500)
    # Abbe コントラスト V (部分コヒーレント近似)
    u_norm = pitch_nm / (wavelength_nm / NA)   # 正規化空間周波数
    # Hopkins TCC ≈ sinc型
    V = max(0, float(np.sinc(u_norm * (1 - sigma))) - 0.1 * sigma)
    V = min(V, 1.0)

    I = 0.5 * (1 + V * np.cos(2 * np.pi * xs / pitch_nm))

    # NILS = (dln I/dx)|_threshold × CD/2
    idx_threshold = np.argmin(np.abs(I - 0.5))
    if idx_threshold > 0 and idx_threshold < len(I)-1:
        dI_dx = (I[idx_threshold+1] - I[idx_threshold-1]) / (xs[1]-xs[0])
        nils  = abs(dI_dx / max(I[idx_threshold], 1e-10)) * (CD_nm/2)
    else:
        nils  = 0.0

    return {
        "system":       f"NA={NA}, λ={wavelength_nm}nm",
        "CD_nm":        CD_nm,
        "pitch_nm":     pitch_nm,
        "k1":           round(k1, 3),
        "contrast_V":   round(V, 3),
        "NILS":         round(nils, 3),
        "resolvable":   resolvable,
        "xs_nm":        xs,
        "I":            I,
        "ILS_good":     nils > 2.0,   # NILS > 2: 十分なコントラスト
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. レジスト露光 (CD vs Dose)
# ════════════════════════════════════════════════════════════════════════════

def resist_cd(dose_mJ_cm2: float, dose_target: float = 30.0,
               EL_pct: float = 10.0, CD_target_nm: float = 13.0,
               LWR_nm: float = 2.5) -> dict:
    """
    EUV レジスト CD モデル:
        CD(D) = CD_target × (1 + (1/γ) × ln(D/D_target))
        γ = Dill B parameter (contrast)

    EL (Exposure Latitude): CD が ±10% 以内になる dose 範囲
    LWR (Line Width Roughness): EUV レジストの本質的ノイズ
    """
    gamma = 100 / EL_pct   # EL 10% → γ = 10
    CD    = CD_target * (1 + (1/gamma) * math.log(max(dose_mJ_cm2, 0.1) / dose_target))
    CD    = max(CD, 1.0)

    # EUV 固有: ショットノイズによる LWR
    # σ_LWR ∝ 1/√(dose × CD)
    photons_per_nm2 = dose_mJ_cm2 * 1e-3 / (13.5e-9 * 6.63e-34 * 3e8 / 13.5e-9) * 1e18
    sigma_shot = LWR_nm * math.sqrt(30.0 / max(dose_mJ_cm2, 1))
    CD_3sigma  = CD + 3 * sigma_shot

    return {
        "dose_mJ_cm2":   dose_mJ_cm2,
        "CD_nm":         round(CD, 2),
        "CD_3sigma_nm":  round(CD_3sigma, 2),
        "LWR_3sigma_nm": round(3 * sigma_shot, 2),
        "within_spec":   abs(CD - CD_target) < CD_target * 0.1,
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. プロセスウィンドウ (Focus-Exposure Matrix)
# ════════════════════════════════════════════════════════════════════════════

def process_window(system: str = "EXE:5000 (High-NA)",
                   CD_target_nm: float = 13.0,
                   CD_tol_pct: float = 10.0) -> dict:
    """
    Focus-Exposure Matrix から DOF × EL を計算。
    2nm ノード (CD=13nm half-pitch) でのプロセスウィンドウ面積。
    """
    s = ASML_SYSTEMS[system]
    NA = s["NA"]
    lam = s["wavelength_nm"]

    # DOF (Depth of Focus): 焦点深度
    # DOF = k2 × λ / NA²  (k2 ≈ 0.5-0.8)
    k2  = 0.7
    DOF_nm = k2 * lam / NA**2 * 1e3   # nm (lam in nm)

    # EL (Exposure Latitude) ≈ 10% for EUV
    EL_pct = 12.0 if "High-NA" in system else 9.0

    # PW 面積 = DOF × EL
    pw_area = DOF_nm * EL_pct

    # CD ばらつきに変換
    sigma_CD_nm = CD_target_nm * (CD_tol_pct/100) / 3   # 3σ = tol

    return {
        "system":       system,
        "NA":           NA,
        "DOF_nm":       round(DOF_nm, 1),
        "EL_pct":       EL_pct,
        "PW_area":      round(pw_area, 1),
        "sigma_CD_nm":  round(sigma_CD_nm, 2),
        "node_capable": CD_target_nm > lam / (2 * NA) * 0.6,
    }


# ════════════════════════════════════════════════════════════════════════════
# 4. スループット × コスト
# ════════════════════════════════════════════════════════════════════════════

def throughput_cost(system: str, utilization: float = 0.85,
                    wafer_cost_usd: float = 15000) -> dict:
    """
    WPH × 稼働率 → 有効スループット
    Cost of Ownership (CoO) per wafer layer
    """
    s = ASML_SYSTEMS[system]
    wph_eff   = s["WPH_max"] * utilization
    # 1年の稼動時間 = 8760 × 稼働率
    wafers_yr = wph_eff * 8760 * 0.90   # 90% uptime
    # 装置償却 (10年) + 保守 (15%/年)
    capex_yr  = s["price_M_usd"] * 1e6 / 10
    opex_yr   = s["price_M_usd"] * 1e6 * 0.15
    cost_per_w_layer = (capex_yr + opex_yr) / max(wafers_yr, 1)

    return {
        "system":            system,
        "WPH_effective":     round(wph_eff, 1),
        "wafers_per_year":   int(wafers_yr),
        "cost_per_layer_usd":round(cost_per_w_layer, 2),
        "price_M_usd":       s["price_M_usd"],
        "node":              s["node"],
    }


# ════════════════════════════════════════════════════════════════════════════
# 5. ASML → CD → V_th → ソシオネクスト タイミング
# ════════════════════════════════════════════════════════════════════════════

def cd_to_vth_sigma(sigma_CD_nm: float, node_nm: float = 2.0) -> float:
    """
    CD ばらつき → V_th ばらつき の変換:
    ΔV_th/V_th ≈ ΔL/L  (L = gate length ≈ CD/2)
    σ_Vth [mV] = V_th_ref × σ_CD / (node_nm / 2)
    """
    V_th_ref = 200   # mV (2nm ノード典型値)
    L_nm     = node_nm / 2
    return V_th_ref * sigma_CD_nm / L_nm


# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def plot_aerial(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: 空中像 (3システム比較)
    ax = axes[0]
    CD = 13.0; pitch = 26.0
    for name, s in ASML_SYSTEMS.items():
        r = aerial_image(pitch, CD, s["NA"], s["wavelength_nm"], s["sigma"])
        ax.plot(r["xs_nm"], r["I"], color=s["color"], lw=2,
                label=f"{name}\nk1={r['k1']:.2f}, V={r['contrast_V']:.2f}")
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="threshold")
    ax.axhline(0.3, color="red",  ls=":",  lw=1, label="min contrast")
    ax.set_xlabel("Position [nm]", fontsize=10)
    ax.set_ylabel("Aerial Image Intensity", fontsize=10)
    ax.set_title(f"空中像 (CD={CD}nm, pitch={pitch}nm)\nAbbe 結像理論", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.2)

    # Panel 2: CD vs Dose (EUV レジスト)
    ax2 = axes[1]
    doses = np.linspace(5, 80, 100)
    for system_name, (NA, col) in [("EXE:5000", (0.55, "#d62728")),
                                    ("NXE:3600D",(0.33, "#2166ac"))]:
        EL = 12 if "EXE" in system_name else 9
        cds = [resist_cd(d, EL_pct=EL)["CD_nm"] for d in doses]
        cd3s= [resist_cd(d, EL_pct=EL)["CD_3sigma_nm"] for d in doses]
        ax2.plot(doses, cds,  color=col, lw=2,   label=f"{system_name} CD")
        ax2.fill_between(doses, cds, cd3s, alpha=0.12, color=col)
    ax2.axhline(13.0, color="purple", ls="--", lw=1.5, label="CD target 13nm")
    ax2.axhspan(11.7, 14.3, alpha=0.07, color="green", label="±10% spec")
    ax2.set_xlabel("Dose [mJ/cm²]", fontsize=10)
    ax2.set_ylabel("CD [nm]", fontsize=10)
    ax2.set_title("EUV レジスト CD vs 露光量\n(帯域: CD+3σ_shot)", fontsize=10)
    ax2.legend(fontsize=7); ax2.grid(alpha=0.2)
    ax2.set_ylim(0, 40)

    # Panel 3: k1 vs ノード × システム比較
    ax3 = axes[2]
    nodes = np.linspace(1, 20, 100)
    for name, s in ASML_SYSTEMS.items():
        k1s = [0.5 * n / (s["wavelength_nm"] / s["NA"]) for n in nodes]
        ax3.plot(nodes, k1s, color=s["color"], lw=2, label=name)
    ax3.axhline(0.25, color="black", ls="-",  lw=1.5, label="k1=0.25 (物理限界)")
    ax3.axhline(0.33, color="red",   ls="--", lw=1.2, label="k1=0.33 (2nm 実用)")
    ax3.axvline(2,    color="purple", ls=":",  lw=1.2)
    ax3.set_xlabel("Half-Pitch [nm]", fontsize=10)
    ax3.set_ylabel("k1 Factor", fontsize=10)
    ax3.set_title("k1 vs ノード\nHigh-NA で 2nm 以下が可能に", fontsize=10)
    ax3.legend(fontsize=7); ax3.grid(alpha=0.2)
    ax3.set_xlim(1, 20); ax3.set_ylim(0, 1)
    ax3.invert_xaxis()

    fig.suptitle("ASML EUV リソグラフィー — 空中像 / CD / k1 解像度解析\n"
                 "2nm ノードは High-NA EUV (EXE:5000) でのみ到達可能",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "asml_aerial.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Aerial image -> {out}")


def plot_process_window(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: DOF × EL × ノード
    ax = axes[0]
    nodes = np.linspace(1.5, 20, 80)
    for name, s in ASML_SYSTEMS.items():
        dofs, els, pws = [], [], []
        for n in nodes:
            pw = process_window(name, CD_target_nm=n)
            dofs.append(pw["DOF_nm"])
            els.append(pw["EL_pct"])
            pws.append(pw["PW_area"])
        ax.plot(nodes, dofs, color=s["color"], lw=2, label=f"{name} DOF")
        ax.plot(nodes, pws,  color=s["color"], lw=1.5, ls="--")
    ax.axvline(2, color="purple", ls=":", lw=1.5, label="2nm")
    ax.axhline(30, color="gray", ls="--", lw=1, label="DOF 30nm (実用下限)")
    ax.set_xlabel("Half-Pitch [nm]", fontsize=10)
    ax.set_ylabel("DOF [nm] / PW Area [nm·%]", fontsize=10)
    ax.set_title("プロセスウィンドウ DOF × EL\n実線=DOF, 破線=PW面積", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.2); ax.invert_xaxis()

    # Panel 2: CoO 比較テーブル
    ax2 = axes[1]
    ax2.axis("off")
    rows = [["システム", "NA", "WPH", "$/wafer-layer", "ノード", "価格"]]
    for name, s in ASML_SYSTEMS.items():
        tc = throughput_cost(name)
        rows.append([name.replace(" (High-NA)","*").replace(" (DUV)","†"),
                     f"{s['NA']}", f"{tc['WPH_effective']:.0f}",
                     f"${tc['cost_per_layer_usd']:.0f}",
                     s["node"], f"${s['price_M_usd']}M"])
    tbl = ax2.table(cellText=rows[1:], colLabels=rows[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.2, 2.2)
    colors_row = [ASML_SYSTEMS[n]["color"] for n in ASML_SYSTEMS]
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#333"); cell.set_text_props(color="white")
        elif r <= 3:
            cell.set_facecolor(colors_row[r-1] + "33")
    ax2.set_title("ASML システム比較 — スループット × CoO", fontsize=10)

    fig.suptitle("ASML プロセスウィンドウ & コスト分析\n"
                 "EXE:5000 (High-NA): $380M/台 — 世界で最も高価な製造装置",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "asml_process_window.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Process window -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aerial",         action="store_true")
    ap.add_argument("--process-window", action="store_true")
    args = ap.parse_args()
    run_all = not any([args.aerial, args.process_window])
    if args.aerial         or run_all: plot_aerial()
    if args.process_window or run_all: plot_process_window()


if __name__ == "__main__":
    main()
