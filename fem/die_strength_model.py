"""
Die Strength Model — Chipping → Fracture Probability
======================================================
Connects the dicing chipping model to device-level fracture reliability.

Pipeline:
    GP surrogate (chipping_um)
        ↓ Griffith / Weibull notch model
    Die fracture strength [MPa]
        ↓ Weibull failure probability
    P(fracture) at assembly / thermal cycling stress

Physics:
    1. Griffith notch: σ_f = K_Ic / (Y √(π a))
       where a = chipping size [m], Y = 1.12 (edge crack)
    2. Weibull distribution over die population:
       P(σ > σ_f) = 1 − exp(−(σ/σ_0)^m)
    3. Thermal stress at assembly: σ_assembly = E·α·ΔT / (1−ν)

Crystal anisotropy link:
    K_Ic(θ) from crystal_anisotropy.py → direction-dependent σ_f

Usage:
    python fem/die_strength_model.py
    python fem/die_strength_model.py --material GaN

References:
    Griffith (1921) Phil. Trans. R. Soc. — fracture mechanics
    Weibull (1951) J. Appl. Mech. — statistical failure model
    Lawn (1993) Fracture of Brittle Solids, Cambridge Univ. Press
    Jaccodine (1963) J. Electrochem. Soc. — Si fracture strength
"""

import argparse, math, os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.materials.material_properties import SiC, Si, GaN

# ── Weibull 形状パラメータ (m) — 材料別 ─────────────────────────────────────
# 高い m → 信頼性高い（ばらつき小）
WEIBULL_M = {"4H-SiC": 8.0, "Si": 5.5, "GaN": 4.5}

MATERIALS = {
    "4H-SiC": {"mat": SiC,  "name": "4H-SiC", "color": "#2166ac"},
    "Si":      {"mat": Si,   "name": "Si",      "color": "#d62728"},
    "GaN":     {"mat": GaN,  "name": "GaN",     "color": "#ff7f0e"},
}


# ════════════════════════════════════════════════════════════════════════════
# 1. Griffith 破壊強度
# ════════════════════════════════════════════════════════════════════════════

def fracture_strength(chipping_um: float, material: str = "4H-SiC",
                       theta_deg: float = 0) -> float:
    """
    チッピングサイズ → 破壊強度 [MPa]。
    σ_f = K_Ic(θ) / (Y · √(π · a))

    theta_deg: ダイシング方向角（結晶異方性補正）
    """
    from fem.crystal_anisotropy import kic_vs_angle
    mat = MATERIALS[material]["mat"]
    KIc_Pa_m = mat["K_Ic"]          # Pa√m
    KIc_MPa  = KIc_Pa_m / 1e6       # MPa√m

    # 結晶方向補正: a面方向はK_Icが低い
    if material == "4H-SiC":
        KIc_MPa = kic_vs_angle(theta_deg)

    Y = 1.12   # 端部亀裂形状係数
    a_m = max(chipping_um, 0.01) * 1e-6   # µm → m

    return KIc_MPa / (Y * math.sqrt(math.pi * a_m))   # MPa


def strength_distribution(chipping_um_arr: np.ndarray,
                           material: str = "4H-SiC",
                           theta_deg: float = 0) -> np.ndarray:
    """チッピング分布 → 破壊強度分布 [MPa]。"""
    return np.array([fracture_strength(c, material, theta_deg)
                     for c in chipping_um_arr])


CHIP_COEFF = 0.95   # calibrated to 11 Grade-A points (Micro2026+Mat2022): MAE 0.87 µm


def empirical_chipping_um(depth_um: float, feed_mm_s: float,
                          spindle_rpm: float = 30000.0) -> float:
    """
    ブレードダイシングの front-side 最大チッピング [µm] — 軽量冪則プロキシ。

        chipping = 0.95 · depth^0.6 · feed^0.4 / (spindle_krpm)^0.3

    指数は Si/SiC ダイシングレビューの傾向（チッピングは送り↑で増・主軸回転↑で減）、
    係数 0.95 は ml.surrogate_transformer の 11 点実験データ
    (Micro2026 + Mat2022, depth 40–120 µm / feed 1.5–2.0 mm/s / 26–34 krpm) に
    最小二乗フィット（MAE 0.87 µm, MAPE 18.5 %）。
    完全な脆性破壊チッピングFEM = fem/dicing_blade_2d.py (ABAQUS/Explicit)。
    結晶方位補正は apply_crystal_correction を別途適用。
    """
    spindle_krpm = max(spindle_rpm, 1.0) / 1000.0
    return CHIP_COEFF * depth_um ** 0.6 * feed_mm_s ** 0.4 / spindle_krpm ** 0.3


# ════════════════════════════════════════════════════════════════════════════
# 2. Weibull 破壊確率
# ════════════════════════════════════════════════════════════════════════════

def weibull_failure_prob(sigma_applied: float, sigma_0: float,
                          m: float) -> float:
    """
    Weibull 破壊確率。
    P_f = 1 − exp(−(σ / σ_0)^m)
    sigma_0: 特性強度 (63.2% 破壊強度)
    m: Weibull 形状パラメータ
    """
    return 1.0 - math.exp(-(sigma_applied / max(sigma_0, 1e-6)) ** m)


def assembly_failure_rate(chipping_um: float, material: str = "4H-SiC",
                           theta_deg: float = 0,
                           sigma_assembly_MPa: float = 80.0) -> dict:
    """
    ダイシング後の実装時破壊率。
    sigma_assembly: ダイアタッチ / ワイヤーボンド工程の熱応力 [MPa]
    """
    sigma_f   = fracture_strength(chipping_um, material, theta_deg)
    m_val     = WEIBULL_M.get(material, 6.0)
    # σ_0 は σ_f の分布中央値として扱う
    P_failure = weibull_failure_prob(sigma_assembly_MPa, sigma_f * 0.8, m_val)

    return {
        "chipping_um":        round(chipping_um, 2),
        "fracture_str_MPa":   round(sigma_f, 1),
        "assembly_stress_MPa": sigma_assembly_MPa,
        "P_failure_ppm":      round(P_failure * 1e6, 1),
        "safety_margin":      round(sigma_f / max(sigma_assembly_MPa, 1), 2),
        "pass":               sigma_f > sigma_assembly_MPa * 1.5,
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. 結晶方向 × チッピング → 破壊率 連成
# ════════════════════════════════════════════════════════════════════════════

def direction_reliability_map(depths: np.ndarray, feeds: np.ndarray,
                               material: str = "4H-SiC") -> dict:
    """
    (depth, feed, θ) の3次元パラメータ空間で破壊率マップを生成。
    GP 予測チッピング → Griffith → Weibull の完全パイプライン。
    """
    from fem.crystal_anisotropy import apply_crystal_correction
    results = {}
    for theta in [0, 30]:
        ppm_map = np.zeros((len(feeds), len(depths)))
        for i, feed in enumerate(feeds):
            for j, depth in enumerate(depths):
                base_chip = 0.52 * depth ** 0.65 * (feed / 1.0) ** 0.4
                chip = apply_crystal_correction(base_chip, theta)
                r = assembly_failure_rate(chip, material, theta)
                ppm_map[i, j] = r["P_failure_ppm"]
        results[f"theta_{theta}"] = ppm_map
    results["improvement_ppm"] = results["theta_30"] - results["theta_0"]
    return results


# ════════════════════════════════════════════════════════════════════════════
# プロット
# ════════════════════════════════════════════════════════════════════════════

def plot_all(material: str = "4H-SiC", out_dir: str = OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    m_val = WEIBULL_M[material]

    # ── Panel 1: σ_f vs chipping (方向別) ──────────────────────────────
    ax = axes[0, 0]
    chip_arr = np.linspace(0.5, 30, 200)
    for theta, col, lbl in [(0, "#2166ac", "m-face θ=0°"),
                             (15, "#ff7f0e", "θ=15°"),
                             (30, "#d62728", "a-face θ=30°")]:
        sf = [fracture_strength(c, material, theta) for c in chip_arr]
        ax.plot(chip_arr, sf, lw=2.5, color=col, label=lbl)
    ax.axhline(80, color="purple", ls="--", lw=1.5, label="Assembly stress 80MPa")
    ax.set_xlabel("Chipping size [µm]", fontsize=10)
    ax.set_ylabel("Fracture strength [MPa]", fontsize=10)
    ax.set_title("Griffith Fracture Strength\nvs Chipping & Crystal Direction", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    # ── Panel 2: Weibull 破壊確率 vs 応力 ──────────────────────────────
    ax2 = axes[0, 1]
    sigma_arr = np.linspace(10, 400, 200)
    for chip, col, lbl in [(2, "#2166ac", "chip=2µm (good)"),
                            (5, "#ff7f0e", "chip=5µm"),
                            (15, "#d62728", "chip=15µm (poor)")]:
        sf = fracture_strength(chip, material, 0)
        pf = [weibull_failure_prob(s, sf * 0.8, m_val) * 1e6 for s in sigma_arr]
        ax2.semilogy(sigma_arr, np.maximum(pf, 0.1), lw=2.5, color=col, label=lbl)
    ax2.axvline(80, color="gray", ls="--", lw=1.5, label="Assembly 80MPa")
    ax2.set_xlabel("Applied stress [MPa]", fontsize=10)
    ax2.set_ylabel("Failure rate [PPM]", fontsize=10)
    ax2.set_title(f"Weibull Failure Rate (m={m_val})\nvs Applied Stress", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.2, which="both")

    # ── Panel 3: 材料比較 ───────────────────────────────────────────────
    ax3 = axes[0, 2]
    chips_cmp = np.linspace(1, 20, 100)
    for mat_key, mat_d in MATERIALS.items():
        sf = [fracture_strength(c, mat_key, 0) for c in chips_cmp]
        ax3.plot(chips_cmp, sf, lw=2.5, color=mat_d["color"], label=mat_key)
    ax3.axhline(80, color="purple", ls="--", lw=1.5, label="Assembly stress")
    ax3.set_xlabel("Chipping size [µm]", fontsize=10)
    ax3.set_ylabel("Fracture strength [MPa]", fontsize=10)
    ax3.set_title("Material Comparison\nSiC vs Si vs GaN fracture strength", fontsize=10)
    ax3.legend(fontsize=8); ax3.grid(alpha=0.2)

    # ── Panel 4: 方向最適化 × 破壊率マップ ─────────────────────────────
    ax4 = axes[1, 0]
    depths = np.linspace(30, 300, 40)
    feeds  = np.linspace(0.5, 3.5, 40)
    D, F   = np.meshgrid(depths, feeds)
    res    = direction_reliability_map(depths, feeds, material)
    cp = ax4.contourf(D, F, np.log10(np.maximum(res["theta_0"], 0.01)),
                      levels=15, cmap="YlOrRd")
    plt.colorbar(cp, ax=ax4, label="log₁₀ Failure rate [PPM]")
    ax4.set_xlabel("Cut depth [µm]", fontsize=10)
    ax4.set_ylabel("Feed [mm/s]", fontsize=10)
    ax4.set_title("Failure Rate Map (m-face θ=0°)\n(GP chipping → Griffith → Weibull)", fontsize=10)

    # ── Panel 5: m面 vs a面 破壊率改善量 ───────────────────────────────
    ax5 = axes[1, 1]
    improvement = res["improvement_ppm"]
    cp2 = ax5.contourf(D, F, improvement, levels=15, cmap="RdYlGn_r")
    plt.colorbar(cp2, ax=ax5, label="Failure rate increase (PPM): a-face vs m-face")
    ax5.set_xlabel("Cut depth [µm]", fontsize=10)
    ax5.set_ylabel("Feed [mm/s]", fontsize=10)
    ax5.set_title("Reliability Gain: m-face vs a-face\n(choosing θ=0° reduces failure rate)", fontsize=10)

    # ── Panel 6: パイプラインサマリー表 ────────────────────────────────
    ax6 = axes[1, 2]
    ax6.axis("off")
    test_cases = [
        ("Optimal (m-face, 2µm chip)",    0,  2,  80),
        ("Typical (m-face, 5µm chip)",    0,  5,  80),
        ("Suboptimal (a-face, 5µm chip)", 30, 5,  80),
        ("Poor (a-face, 15µm chip)",      30, 15, 80),
        ("Deep cut (m-face, 10µm chip)",  0,  10, 120),
    ]
    rows = [["Case", "σ_f [MPa]", "PPM", "Safety", "PASS"]]
    for label, theta, chip, stress in test_cases:
        r = assembly_failure_rate(chip, material, theta, stress)
        rows.append([
            label,
            f"{r['fracture_str_MPa']:.0f}",
            f"{r['P_failure_ppm']:.1f}",
            f"×{r['safety_margin']:.1f}",
            "✓" if r["pass"] else "✗",
        ])
    tbl = ax6.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    tbl.scale(1.2, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0: cell.set_facecolor("#333"); cell.set_text_props(color="w")
        elif c == 4 and r > 0:
            cell.set_facecolor("#c8e6c9" if rows[r][4] == "✓" else "#ffcdd2")
        elif r % 2 == 0: cell.set_facecolor("#f5f5f5")
    ax6.set_title(f"Die Reliability Pipeline — {material}\nGP chipping → Griffith → Weibull", fontsize=10)

    fig.suptitle(f"Die Strength Model — {material}\n"
                 "Chipping → Fracture Strength (Griffith) → Failure Rate (Weibull)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(out_dir, "die_strength_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Die Strength Pipeline ({material}):")
    for label, theta, chip, stress in [
        ("m-face 2µm",  0,  2,  80),
        ("m-face 5µm",  0,  5,  80),
        ("a-face 5µm",  30, 5,  80),
        ("a-face 15µm", 30, 15, 80),
    ]:
        r = assembly_failure_rate(chip, material, theta, stress)
        print(f"  {label:16s} σ_f={r['fracture_str_MPa']:5.0f}MPa  "
              f"PPM={r['P_failure_ppm']:8.1f}  ×{r['safety_margin']:.1f}")
    print(f"\n[OK] die strength model -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="4H-SiC",
                    choices=["4H-SiC", "Si", "GaN"])
    args = ap.parse_args()
    plot_all(args.material)


if __name__ == "__main__":
    main()
