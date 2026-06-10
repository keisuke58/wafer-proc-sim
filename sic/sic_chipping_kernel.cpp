// sic_chipping_kernel.cpp — 4H-SiC crystal anisotropy + chipping prediction
//
// AP試験 / DISCO engineer concepts:
//   - Griffith 破壊基準: K_I = σ√(πa) ≥ K_IC → チッピング条件
//   - Lawn-Evans インデンテーション破壊力学:
//       h_chip ∝ (E/H)^(1/2) × (F_N)^(1/2) / K_IC
//   - 4H-SiC 結晶異方性: E, K_IC がダイシング方向 θ で変化
//     a軸[11-20]: E≈448GPa, K_IC≈2.0 MPa√m  (オフ軸研磨方向)
//     c軸[0001]:  E≈476GPa, K_IC≈2.8 MPa√m
//   - 最適ダイシング方向: K_IC が高い → チッピング深さ最小
//
// DISCO context:
//   - SiC は EV 向けパワーデバイスの主力材料 (2026年需要急拡大)
//   - 目標チッピング < 2 µm (現状 2-10 µm)
//   - ブレードダイシング vs ステルスダイシングの選択基準にも使う
//   - 4° オフアングル基板が標準 → 結晶方向考慮必須

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace py = pybind11;

static constexpr double PI = M_PI;

// ── 4H-SiC anisotropic elastic constants ─────────────────────────────────────
// S_ij (compliance tensor) for 4H-SiC (hexagonal symmetry, Class 6mm)
// Literature: Li et al. (2019), Levinshtein (2001)
//   C_11 = 501 GPa,  C_12 = 111 GPa,  C_13 = 52 GPa
//   C_33 = 553 GPa,  C_44 = 163 GPa,  C_66 = (C_11-C_12)/2 = 195 GPa
static constexpr double C11 = 501.0, C12 = 111.0, C13 = 52.0;
static constexpr double C33 = 553.0, C44 = 163.0;

// Young's modulus for a direction at angle θ from c-axis [0001] toward a-axis [11-20]
// Using Voigt notation rotation: E(θ) from compliance S(θ)
// Simplified anisotropy model (valid for in-plane rotation in basal plane):
//   E(θ) = 1 / (S₁₁ sin⁴θ + S₃₃ cos⁴θ + (2S₁₃+S₄₄)sin²θ cos²θ)
// where S₁₁ = 1/C_eff_a,  S₃₃ = 1/C_eff_c
static double sic_young_modulus(double theta_deg) {
    double th = theta_deg * PI / 180.0;
    double s2 = std::sin(th) * std::sin(th);
    double c2 = std::cos(th) * std::cos(th);

    // Compliance (GPa^-1) from stiffness
    double S11 = 1.0 / 448.0;   // ≈ 1/E_a
    double S33 = 1.0 / 476.0;   // ≈ 1/E_c
    // Cross-compliance S13 ≈ -ν_ac / E_a  (ν_ac ≈ 0.10)
    double S44 = 1.0 / C44;     // shear
    double S13 = -0.10 / 448.0;

    double inv_E = S11 * s2 * s2
                 + S33 * c2 * c2
                 + (2.0 * S13 + S44) * s2 * c2;
    return 1.0 / inv_E;   // GPa
}

// Fracture toughness K_IC(θ) for 4H-SiC
// Basal plane (0001): K_IC ≈ 1.0 MPa√m  (weakest cleavage plane)
// Prismatic plane (10-10): K_IC ≈ 2.0 MPa√m
// Non-basal / c-axis: K_IC ≈ 2.8 MPa√m
// Model: K_IC(θ) = K_IC_c - (K_IC_c - K_IC_a) × cos²θ
//   where θ = angle from c-axis
static double sic_K_IC(double theta_deg) {
    double th = theta_deg * PI / 180.0;
    double c2 = std::cos(th) * std::cos(th);
    double K_IC_a = 2.0;   // a-axis (prismatic): MPa√m
    double K_IC_c = 2.8;   // c-axis (non-basal): MPa√m
    // Also account for basal cleavage weakening near θ=0
    double K_base = K_IC_a + (K_IC_c - K_IC_a) * (1.0 - c2);
    // Weak basal plane: subtract contribution when θ≈0
    double K_basal_dip = 1.0;   // minimum at exact c-axis perp cut
    double basal_factor = std::exp(-std::pow(theta_deg / 15.0, 2));
    return K_base * (1.0 - 0.3 * basal_factor) + K_basal_dip * 0.3 * basal_factor;
}

// Vickers hardness (GPa) — nearly isotropic for 4H-SiC, ~28 GPa
// Small orientation dependence (~5%)
static double sic_hardness(double theta_deg) {
    double th = theta_deg * PI / 180.0;
    return 28.0 - 1.4 * std::pow(std::cos(th), 2);   // ±5% variation
}

// ── Lawn–Evans chipping depth model ─────────────────────────────────────────
// For brittle material under localized load (blade edge = Vickers-like indenter):
//   h_chip ≈ α × (E/H)^(1/2) × F_N^(1/2) / K_IC
// where α = geometry constant ≈ 0.015 (empirical, units: µm / (N^0.5 MPa√m))
// F_N in Newtons, E in GPa, H in GPa, K_IC in MPa√m → h_chip in µm

static double lawn_evans_chipping(
    double F_N_newtons,
    double E_GPa,
    double H_GPa,
    double K_IC_MPa_sqrtm,
    double alpha = 0.015
) {
    // Convert to consistent units
    double E_MPa = E_GPa * 1000.0;
    double H_MPa = H_GPa * 1000.0;
    double ratio = std::sqrt(E_MPa / H_MPa);
    double h_chip = alpha * ratio * std::sqrt(F_N_newtons) / K_IC_MPa_sqrtm;
    return h_chip;   // µm
}

// Normal force from dicing parameters (reused from blade_wear_kernel logic)
static double estimate_force(double depth_um, double feed_um_rev,
                              double spindle_krpm, double H_GPa) {
    // F_N ≈ k_c × depth × feed × (H/H_ref)^0.6
    double k_c = 0.08;  // N/(µm·µm/rev)
    double feed_mm_s = feed_um_rev * 1e-3 * spindle_krpm * 1000.0 / 60.0;
    return k_c * depth_um * feed_mm_s * std::pow(H_GPa / 13.0, 0.6);
}

// ── Crystal properties at given orientation ───────────────────────────────────
py::dict sic_crystal_properties(double orientation_deg) {
    double E   = sic_young_modulus(orientation_deg);
    double K   = sic_K_IC(orientation_deg);
    double H   = sic_hardness(orientation_deg);

    // Griffith critical crack half-length (µm) for σ_c = H/10 (fracture stress approx)
    double sigma_c_MPa = H * 1000.0 / 10.0;
    double a_c_um = 1e6 * (1.0 / PI) * std::pow(K / sigma_c_MPa, 2) * 1e6;
    // (K in MPa√m, σ in MPa → a in m → convert to µm)
    double a_c = (1.0 / PI) * std::pow(K / sigma_c_MPa, 2) * 1e6;  // µm

    py::dict r;
    r["orientation_deg"]        = orientation_deg;
    r["E_GPa"]                  = E;
    r["K_IC_MPa_sqrtm"]         = K;
    r["H_GPa"]                  = H;
    r["critical_crack_um"]      = a_c;
    r["anisotropy_note"] = std::string(
        orientation_deg < 10.0  ? "near c-axis: weakest cleavage (basal)" :
        orientation_deg < 30.0  ? "transitional: mixed a/c character" :
        orientation_deg < 70.0  ? "intermediate: prismatic planes active" :
                                   "near a-axis: toughest against chipping");
    return r;
}

// ── Chipping prediction for given dicing parameters ───────────────────────────
py::dict predict_chipping(
    double orientation_deg,
    double blade_depth_um,     // blade cut depth [µm]
    double feed_um_rev,        // feed per revolution [µm/rev]
    double spindle_krpm
) {
    double E = sic_young_modulus(orientation_deg);
    double K = sic_K_IC(orientation_deg);
    double H = sic_hardness(orientation_deg);

    double F_N = estimate_force(blade_depth_um, feed_um_rev, spindle_krpm, H);
    double h_chip = lawn_evans_chipping(F_N, E, H, K);

    // Standard deviation estimate: ~20% of mean (process variability)
    double h_std = h_chip * 0.20;

    // Chipping quality classification
    const char* quality;
    if (h_chip < 2.0)       quality = "excellent (<2µm target met)";
    else if (h_chip < 5.0)  quality = "acceptable (2-5µm)";
    else if (h_chip < 10.0) quality = "marginal (5-10µm)";
    else                     quality = "reject (>10µm)";

    py::dict r;
    r["orientation_deg"]    = orientation_deg;
    r["blade_depth_um"]     = blade_depth_um;
    r["feed_um_rev"]        = feed_um_rev;
    r["spindle_krpm"]       = spindle_krpm;
    r["F_N_newtons"]        = F_N;
    r["E_GPa"]              = E;
    r["K_IC_MPa_sqrtm"]     = K;
    r["H_GPa"]              = H;
    r["chipping_mean_um"]   = h_chip;
    r["chipping_std_um"]    = h_std;
    r["quality"]            = std::string(quality);
    return r;
}

// ── Optimize dicing direction to minimize chipping ────────────────────────────
py::dict optimize_dicing_direction(
    double blade_depth_um,
    double feed_um_rev,
    double spindle_krpm,
    double target_chipping_um
) {
    double best_angle = 0.0;
    double best_chip  = 1e9;
    std::vector<py::dict> sweep;

    for (int theta = 0; theta <= 90; theta += 5) {
        auto r = predict_chipping(static_cast<double>(theta),
                                   blade_depth_um, feed_um_rev, spindle_krpm);
        double chip = py::cast<double>(r["chipping_mean_um"]);
        sweep.push_back(r);
        if (chip < best_chip) { best_chip = chip; best_angle = theta; }
    }

    bool target_met = best_chip <= target_chipping_um;

    py::dict r;
    r["best_orientation_deg"]   = best_angle;
    r["best_chipping_um"]       = best_chip;
    r["target_chipping_um"]     = target_chipping_um;
    r["target_met"]             = target_met;
    r["recommendation"] = std::string(target_met ?
        "Dicing direction optimized: target chipping depth achievable." :
        "Target not met with blade dicing. Consider stealth dicing for SiC <2µm.");
    r["sweep"] = sweep;
    return r;
}

// ── Demo: Si vs SiC chipping comparison ──────────────────────────────────────
py::dict chipping_demo() {
    // Compare Si and SiC at standard DFL7160 parameters
    struct CutParams {
        std::string material;
        double depth_um, feed_um_rev, spindle_krpm;
        // For Si: use isotropic model; for SiC: use orientation-dependent
    };

    // Si reference values (isotropic)
    double Si_E = 130.0, Si_K = 0.70, Si_H = 13.0;
    double Si_F = estimate_force(50.0, 1.5, 30.0, Si_H);
    double Si_chip = lawn_evans_chipping(Si_F, Si_E, Si_H, Si_K);

    // SiC at optimal and worst orientation
    auto sic_opt  = predict_chipping(60.0, 30.0, 1.0, 25.0);   // near a-axis
    auto sic_worst= predict_chipping(5.0,  30.0, 1.0, 25.0);   // near basal plane

    py::dict r;
    r["Si_chipping_um"]            = Si_chip;
    r["Si_params"]                 = std::string("depth=50µm, feed=10mm/s, 30krpm");
    r["SiC_optimal_chipping_um"]   = py::cast<double>(sic_opt["chipping_mean_um"]);
    r["SiC_optimal_orientation"]   = py::cast<double>(sic_opt["orientation_deg"]);
    r["SiC_worst_chipping_um"]     = py::cast<double>(sic_worst["chipping_mean_um"]);
    r["SiC_worst_orientation"]     = py::cast<double>(sic_worst["orientation_deg"]);
    r["chipping_ratio_worst_best"] = py::cast<double>(sic_worst["chipping_mean_um"])
                                     / std::max(py::cast<double>(sic_opt["chipping_mean_um"]), 0.01);
    r["insight"] = std::string(
        "SiC anisotropy: chipping varies 2-4x with crystal orientation. "
        "Dicing parallel to a-axis [11-20] minimizes chipping via higher K_IC.");
    return r;
}

// ── pybind11 module ──────────────────────────────────────────────────────────
PYBIND11_MODULE(_sic_chipping_kernel, m) {
    m.doc() =
        "4H-SiC crystal anisotropy and Lawn-Evans chipping prediction.\n"
        "\n"
        "AP試験 / DISCO engineer topics:\n"
        "  - Griffith破壊基準: K_I = σ√(πa) ≥ K_IC\n"
        "  - Lawn-Evans: h_chip ∝ (E/H)^(1/2) × F_N^(1/2) / K_IC\n"
        "  - 4H-SiC六方晶弾性定数: C_11=501, C_33=553, C_44=163 (GPa)\n"
        "  - 結晶方向別 K_IC: basal面(0001)最弱 1.0 → c軸非直交 2.8 MPa√m\n"
        "\n"
        "DISCO context:\n"
        "  - SiC EV向けパワーデバイス: 目標チッピング < 2µm\n"
        "  - 切断方向最適化: [11-20]a軸方向でチッピング最小化\n"
        "  - ステルスダイシングへの切替基準: ブレードで<2µm達成困難なら推奨";

    m.def("sic_crystal_properties", &sic_crystal_properties,
          py::arg("orientation_deg") = 45.0,
          "Return 4H-SiC crystal properties at given orientation from c-axis.\n"
          "Returns: E_GPa, K_IC_MPa_sqrtm, H_GPa, critical_crack_um.");

    m.def("predict_chipping", &predict_chipping,
          py::arg("orientation_deg") = 45.0,
          py::arg("blade_depth_um")  = 30.0,
          py::arg("feed_um_rev")     = 1.0,
          py::arg("spindle_krpm")    = 25.0,
          "Predict chipping depth (µm) using Lawn-Evans fracture mechanics.\n"
          "Returns: chipping_mean_um, chipping_std_um, quality, F_N, E, K_IC, H.");

    m.def("optimize_dicing_direction", &optimize_dicing_direction,
          py::arg("blade_depth_um")     = 30.0,
          py::arg("feed_um_rev")        = 1.0,
          py::arg("spindle_krpm")       = 25.0,
          py::arg("target_chipping_um") = 2.0,
          "Sweep orientation 0-90° to find minimum-chipping dicing direction.\n"
          "Returns: best_orientation_deg, best_chipping_um, target_met, sweep.");

    m.def("chipping_demo", &chipping_demo,
          "Compare Si vs SiC chipping: optimal vs worst orientation, anisotropy ratio.");
}
