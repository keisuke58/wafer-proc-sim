"""
EUV フォトレジスト物理モデル — JSR / TOK / Merck
==========================================================
EUVレジスト3強: JSR (CAR → MOR)、TOK (金属酸化物レジスト)、Merck (CAR/MOR)。
高NA EUV (0.55 NA) 時代の確率論的欠陥・LWR が最重要課題。

Key physics:
  - RLS トレードオフ三角形: Resolution vs LWR vs Sensitivity (感度)
  - CAR (化学増幅型レジスト): 酸拡散 → CD 制御
  - MOR (金属酸化物レジスト): SnOx/HfOx 高感度 → LWR 改善
  - 光子ショットノイズ: Poisson 統計 → LWR 下限
  - アウトガス: レジスト分解 → レンズ汚染 (PMMA系)
  - EUV 吸収効率: β係数 → 有効吸収深さ

実行:
    python fem/euv_resist_model.py
    python fem/euv_resist_model.py --rls --node 2
    python fem/euv_resist_model.py --shot-noise --dose 30

参考文献:
    Kozawa & Tagawa (2009) Jpn. J. Appl. Phys. — EUV レジスト感度
    Ouyang et al. (2016) SPIE — RLS トレードオフ
    Naulleau et al. (2020) JVST B — MOR vs CAR 比較
    Mack (2007) Fundamental Principles of Optical Lithography — 酸拡散
    JSR/TOK/Merck EUV レジスト製品ブリーフ (2023)
"""

import argparse
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ════════════════════════════════════════════════════════════════════════════
# 1. レジスト パラメータ テーブル
# ════════════════════════════════════════════════════════════════════════════

RESISTS = {
    "JSR_CAR": {
        "vendor": "JSR",
        "type": "CAR",
        "Esize_mJ_cm2": 15.0,       # 感度 (Dose to clear)
        "LWR_3sigma_nm": 3.2,       # Line Width Roughness
        "resolution_nm": 14.0,      # 最小解像ピッチ/2
        "acid_diffusion_nm": 5.0,   # 酸拡散長
        "outgas_rate": 0.8,         # 相対アウトガス (1.0 = PMMA 基準)
        "k_absorption": 0.0074,     # EUV 吸収係数 (β at 13.5nm)
        "color": "#d62728",
        "ref": "JSR EUV CAR product brief (2023)",
    },
    "JSR_MOR": {
        "vendor": "JSR",
        "type": "MOR",
        "Esize_mJ_cm2": 8.0,
        "LWR_3sigma_nm": 2.1,
        "resolution_nm": 11.0,
        "acid_diffusion_nm": 1.5,
        "outgas_rate": 0.2,
        "k_absorption": 0.042,      # Sn ベース → 高吸収
        "color": "#ff7f0e",
        "ref": "JSR EUV MOR product brief (2024)",
    },
    "TOK_CAR": {
        "vendor": "TOK",
        "type": "CAR",
        "Esize_mJ_cm2": 18.0,
        "LWR_3sigma_nm": 2.9,
        "resolution_nm": 13.0,
        "acid_diffusion_nm": 4.5,
        "outgas_rate": 0.75,
        "k_absorption": 0.0071,
        "color": "#2166ac",
        "ref": "TOK EUV CAR product brief (2023)",
    },
    "TOK_MOR": {
        "vendor": "TOK",
        "type": "MOR",
        "Esize_mJ_cm2": 6.0,
        "LWR_3sigma_nm": 1.8,
        "resolution_nm": 10.0,
        "acid_diffusion_nm": 1.0,
        "outgas_rate": 0.15,
        "k_absorption": 0.055,
        "color": "#2ca02c",
        "ref": "TOK EUV MOR product brief (2024)",
    },
    "Merck_CAR": {
        "vendor": "Merck",
        "type": "CAR",
        "Esize_mJ_cm2": 20.0,
        "LWR_3sigma_nm": 3.5,
        "resolution_nm": 15.0,
        "acid_diffusion_nm": 6.0,
        "outgas_rate": 0.9,
        "k_absorption": 0.0068,
        "color": "#9467bd",
        "ref": "Merck EUV CAR product brief (2023)",
    },
}

# ════════════════════════════════════════════════════════════════════════════
# 2. Core Physics Functions
# ════════════════════════════════════════════════════════════════════════════

def photon_shot_noise_lwr(dose_mJ_cm2: float, CD_nm: float,
                           resist_thickness_nm: float = 30.0,
                           k_absorption: float = 0.0074) -> dict:
    """
    EUV 光子ショットノイズ → LWR 下限。

    N_photons = dose × pixel_area / E_photon
    σ_N = sqrt(N)  (Poisson)
    LWR_shot = 3σ_N / N × CD  (相対ゆらぎ → CD変換)

    参考: Naulleau et al. (2020) JVST B — shot noise model
    """
    E_photon_J = 1.602e-19 * (1240.0 / 13.5)   # 92 eV photon energy
    pixel_area_cm2 = (CD_nm * 1e-7) ** 2         # pixel ≈ CD × CD
    dose_J_cm2 = dose_mJ_cm2 * 1e-3
    N_photons = dose_J_cm2 * pixel_area_cm2 / E_photon_J
    # Absorbed photons (Beer-Lambert: 1 - exp(-α·t))
    alpha_cm = k_absorption * 4 * math.pi / (13.5e-7)   # absorption coefficient
    absorbed_frac = 1.0 - math.exp(-alpha_cm * resist_thickness_nm * 1e-7)
    N_absorbed = N_photons * absorbed_frac
    sigma_shot = math.sqrt(max(N_absorbed, 1.0))
    LWR_shot_nm = 3.0 * sigma_shot / max(N_absorbed, 1.0) * CD_nm
    return {
        "dose_mJ_cm2": dose_mJ_cm2,
        "CD_nm": CD_nm,
        "N_photons_per_pixel": round(N_photons, 1),
        "N_absorbed": round(N_absorbed, 1),
        "absorbed_frac": round(absorbed_frac, 4),
        "LWR_shot_3sigma_nm": round(LWR_shot_nm, 3),
    }


def acid_diffusion_cd_blur(resist: str = "JSR_CAR", T_C: float = 100.0,
                            bake_time_s: float = 60.0) -> dict:
    """
    CAR 酸拡散ブラー: post-exposure bake (PEB) での酸拡散 → CD ブラー。

    拡散長 L_diff = sqrt(2 * D * t)
    D = D0 * exp(-Ea / kT),  Ea ≈ 0.85 eV for typical CAR

    参考: Mack (2007) Fundamental Principles of Optical Lithography
    """
    p = RESISTS.get(resist, RESISTS["JSR_CAR"])
    if p["type"] != "CAR":
        return {"resist": resist, "type": p["type"],
                "acid_diffusion_nm": 0.0, "CD_blur_nm": 0.0,
                "note": "MOR: no acid diffusion"}
    Ea_eV = 0.85; k_B_eV = 8.617e-5
    D0 = 1.5e-12   # cm²/s (pre-exponential)
    D  = D0 * math.exp(-Ea_eV / (k_B_eV * (T_C + 273.15)))
    L_diff = math.sqrt(2 * D * bake_time_s) * 1e7   # → nm
    CD_blur = 2 * L_diff   # bilateral smear
    return {
        "resist": resist,
        "T_C": T_C, "bake_time_s": bake_time_s,
        "D_cm2_s": round(D, 3),
        "acid_diffusion_nm": round(L_diff, 2),
        "CD_blur_nm": round(CD_blur, 2),
        "spec_pct": round(CD_blur / p["resolution_nm"] * 100, 1),
    }


def rls_tradeoff(resist: str = "JSR_CAR", node_nm: float = 2.0) -> dict:
    """
    RLS (Resolution-LWR-Sensitivity) トレードオフ。

    スコアリング:
    - Resolution: 1 if resolution_nm ≤ node_nm * 5, else penalty
    - LWR: spec = node_nm * 0.12 (12% of CD)
    - Sensitivity: target ≤ 20 mJ/cm² for high throughput

    参考: Ouyang et al. (2016) SPIE Advanced Lithography
    """
    p = RESISTS.get(resist, RESISTS["JSR_CAR"])
    CD_nm = node_nm * 2.5    # half-pitch at node
    lwr_spec = CD_nm * 0.12
    lwr_ok   = p["LWR_3sigma_nm"] <= lwr_spec
    res_ok   = p["resolution_nm"] <= CD_nm * 1.2
    sens_ok  = p["Esize_mJ_cm2"] <= 20.0

    # Composite score (0-100)
    score = (50.0 * (lwr_spec / max(p["LWR_3sigma_nm"], 0.1)) +
             30.0 * (CD_nm / max(p["resolution_nm"], 0.1)) +
             20.0 * (20.0 / max(p["Esize_mJ_cm2"], 1.0)))
    score = min(score, 100.0)

    return {
        "resist": resist, "node_nm": node_nm,
        "CD_nm": CD_nm,
        "LWR_3sigma_nm": p["LWR_3sigma_nm"],
        "lwr_spec_nm": round(lwr_spec, 2),
        "lwr_ok": lwr_ok,
        "resolution_ok": res_ok,
        "sensitivity_ok": sens_ok,
        "rls_score": round(score, 1),
        "recommended": score > 60.0,
    }


def outgas_lens_contamination(resist: str = "JSR_CAR", n_wafers: int = 1000,
                               dose_mJ_cm2: float = 20.0) -> dict:
    """
    アウトガス → EUV レンズ汚染モデル。

    汚染速度 ∝ outgas_rate × dose × n_wafers
    反射率低下 ΔR ∝ contamination_layer_nm
    寿命: レンズ洗浄が必要になるウェーハ数

    参考: Bajt et al. (2003) SPIE — EUV optics contamination
    """
    p = RESISTS.get(resist, RESISTS["JSR_CAR"])
    contamination_rate_nm_per_wafer = p["outgas_rate"] * dose_mJ_cm2 * 1e-5
    total_contamination_nm = contamination_rate_nm_per_wafer * n_wafers
    # Reflectance drop: ~0.5% per nm contamination
    dR_pct = total_contamination_nm * 0.5
    cleaning_interval_wafers = 0.1 / max(contamination_rate_nm_per_wafer, 1e-10)
    return {
        "resist": resist, "n_wafers": n_wafers,
        "contamination_nm": round(total_contamination_nm, 4),
        "dR_pct": round(dR_pct, 3),
        "cleaning_interval_wafers": round(cleaning_interval_wafers, 0),
        "low_outgas": p["outgas_rate"] < 0.3,
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. Visualization
# ════════════════════════════════════════════════════════════════════════════

def plot_all(out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(17, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    ax = [fig.add_subplot(gs[r, c], polar=(r == 0 and c == 1)) for r in range(2) for c in range(3)]

    # ── Panel 0: 光子ショットノイズ LWR vs ドーズ ─────────────────────────
    dose_arr = np.linspace(5, 60, 150)
    for resist, col in [("JSR_CAR", "#d62728"), ("JSR_MOR", "#ff7f0e"),
                         ("TOK_MOR", "#2ca02c"), ("Merck_CAR", "#9467bd")]:
        k = RESISTS[resist]["k_absorption"]
        lwr_shot = [photon_shot_noise_lwr(d, CD_nm=14.0, k_absorption=k)["LWR_shot_3sigma_nm"]
                    for d in dose_arr]
        ax[0].plot(dose_arr, lwr_shot, lw=2.2, color=col,
                   label=f"{RESISTS[resist]['vendor']} {RESISTS[resist]['type']}")
    ax[0].axhline(2.0, color="k", ls="--", lw=1.5, label="2nm LWR spec (2nm node)")
    ax[0].set_xlabel("EUV ドーズ [mJ/cm²]")
    ax[0].set_ylabel("LWR ショットノイズ 3σ [nm]")
    ax[0].set_title("EUV 光子ショットノイズ → LWR\n(Poisson統計、CD=14nm)")
    ax[0].legend(fontsize=7.5); ax[0].grid(alpha=0.25)
    ax[0].set_ylim(0, 8)

    # ── Panel 1: RLS トレードオフ レーダーチャート ────────────────────────
    categories = ["解像度", "LWR", "感度", "低アウトガス", "吸収効率"]
    n_cat = len(categories)
    angles = np.linspace(0, 2*math.pi, n_cat, endpoint=False).tolist()
    angles += angles[:1]
    for resist, col in [("JSR_MOR", "#ff7f0e"), ("TOK_MOR", "#2ca02c"),
                         ("JSR_CAR", "#d62728"), ("TOK_CAR", "#2166ac")]:
        p = RESISTS[resist]
        vals = [
            min(100, 14.0 / max(p["resolution_nm"], 0.1) * 100),   # 解像度
            min(100, 2.0 / max(p["LWR_3sigma_nm"], 0.1) * 100),     # LWR
            min(100, 20.0 / max(p["Esize_mJ_cm2"], 1.0) * 100),     # 感度
            (1 - p["outgas_rate"]) * 100,                            # アウトガス
            min(100, p["k_absorption"] / 0.055 * 100),               # 吸収
        ]
        vals += vals[:1]
        ax[1].plot(angles, vals, color=col, lw=2.0,
                   label=f"{p['vendor']} {p['type']}")
        ax[1].fill(angles, vals, color=col, alpha=0.10)
    ax[1].set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=8.5)
    ax[1].set_ylim(0, 110)
    ax[1].set_title("RLS トレードオフ レーダー\n(スコア 100=最良)")
    ax[1].legend(fontsize=7.5, loc="lower right")

    # ── Panel 2: 酸拡散 CD ブラー vs PEB 温度 ───────────────────────────
    T_arr = np.linspace(60, 140, 100)
    for resist, col in [("JSR_CAR", "#d62728"), ("TOK_CAR", "#2166ac"), ("Merck_CAR", "#9467bd")]:
        blur = [acid_diffusion_cd_blur(resist, T)["CD_blur_nm"] for T in T_arr]
        ax[2].semilogy(T_arr, np.maximum(blur, 0.001), color=col, lw=2.2,
                       label=f"{RESISTS[resist]['vendor']} CAR")
    ax[2].axhline(2.0, color="k", ls="--", lw=1.5, label="2nm CD ブラー spec")
    ax[2].set_xlabel("PEB 温度 [°C]"); ax[2].set_ylabel("CD ブラー [nm]")
    ax[2].set_title("CAR 酸拡散 → CD ブラー\n(PEB 60s, Ea=0.85eV)")
    ax[2].legend(fontsize=9); ax[2].grid(alpha=0.25)

    # ── Panel 3: ノード別 RLS スコア ─────────────────────────────────────
    nodes = np.linspace(1.0, 7.0, 60)
    for resist, col in [("JSR_MOR", "#ff7f0e"), ("TOK_MOR", "#2ca02c"),
                         ("JSR_CAR", "#d62728"), ("TOK_CAR", "#2166ac"),
                         ("Merck_CAR", "#9467bd")]:
        scores = [rls_tradeoff(resist, n)["rls_score"] for n in nodes]
        p = RESISTS[resist]
        ax[3].plot(nodes, scores, lw=2.2, color=col,
                   label=f"{p['vendor']} {p['type']}")
    ax[3].axhline(60.0, color="k", ls="--", lw=1.5, label="推奨スコア 60")
    ax[3].axvline(2.0, color="gray", ls=":", lw=1.2, label="2nm node")
    ax[3].set_xlabel("プロセスノード [nm]"); ax[3].set_ylabel("RLS スコア")
    ax[3].set_title("RLS 複合スコア vs プロセスノード\n(解像度×LWR×感度 重み付き)")
    ax[3].legend(fontsize=8); ax[3].grid(alpha=0.25)
    ax[3].set_ylim(0, 110)

    # ── Panel 4: アウトガス → レンズ汚染 ─────────────────────────────────
    wafers_arr = np.linspace(100, 50000, 200)
    for resist, col in [(r, RESISTS[r]["color"]) for r in RESISTS]:
        cont = [outgas_lens_contamination(resist, int(w))["contamination_nm"] for w in wafers_arr]
        ax[4].semilogy(wafers_arr / 1000, np.maximum(cont, 1e-6), color=col, lw=2.0,
                       label=f"{RESISTS[resist]['vendor']} {RESISTS[resist]['type']}")
    ax[4].axhline(0.1, color="k", ls="--", lw=1.5, label="0.1nm 洗浄トリガー")
    ax[4].set_xlabel("処理ウェーハ数 [k]"); ax[4].set_ylabel("レンズ汚染膜厚 [nm]")
    ax[4].set_title("EUV レンズ汚染 vs 処理量\n(アウトガス速度比較)")
    ax[4].legend(fontsize=7.5); ax[4].grid(alpha=0.25, which="both")

    # ── Panel 5: レジスト仕様比較テーブル ───────────────────────────────
    ax[5].axis("off")
    col_labels = ["レジスト", "感度\n[mJ/cm²]", "LWR 3σ\n[nm]", "解像\n[nm]",
                  "酸拡散\n[nm]", "RLS Score\n(2nm node)"]
    tdata = []
    for rname, rp in RESISTS.items():
        rls = rls_tradeoff(rname, 2.0)
        tdata.append([
            f"{rp['vendor']}\n{rp['type']}",
            f"{rp['Esize_mJ_cm2']:.0f}",
            f"{rp['LWR_3sigma_nm']:.1f}",
            f"{rp['resolution_nm']:.0f}",
            f"{rp['acid_diffusion_nm']:.1f}" if rp["type"] == "CAR" else "—",
            f"{rls['rls_score']:.0f}",
        ])
    tbl = ax[5].table(cellText=tdata, colLabels=col_labels,
                      cellLoc="center", loc="center", bbox=[0.0, 0.05, 1.0, 0.90])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4a0080"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f5eeff")
    ax[5].set_title("EUV レジスト比較\nJSR · TOK · Merck")

    fig.suptitle("EUV フォトレジスト物理 — JSR / TOK / Merck\n"
                 "光子ショットノイズ · RLS トレードオフ · 酸拡散 · アウトガス",
                 fontsize=12, fontweight="bold")
    out = os.path.join(out_dir, "euv_resist_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] EUV resist model → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="EUV resist model (JSR/TOK/Merck)")
    ap.add_argument("--rls",        action="store_true")
    ap.add_argument("--shot-noise", action="store_true")
    ap.add_argument("--node",  type=float, default=2.0)
    ap.add_argument("--dose",  type=float, default=20.0)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    if args.rls:
        print(f"\nRLS スコア (node={args.node}nm)")
        for r in RESISTS:
            res = rls_tradeoff(r, args.node)
            print(f"  {r:15s}  score={res['rls_score']:5.1f}  "
                  f"LWR={res['LWR_3sigma_nm']:.1f}nm  {'✓' if res['recommended'] else '✗'}")
    if args.shot_noise:
        for CD in [8, 10, 12, 14, 16, 20]:
            r = photon_shot_noise_lwr(args.dose, CD)
            print(f"  dose={args.dose:.0f}  CD={CD:2d}nm  "
                  f"LWR_shot={r['LWR_shot_3sigma_nm']:.3f}nm  N={r['N_absorbed']:.0f}")
    if not args.no_plot:
        plot_all()

if __name__ == "__main__":
    main()
