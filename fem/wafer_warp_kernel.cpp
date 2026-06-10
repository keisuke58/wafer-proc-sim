// wafer_warp_kernel.cpp — Grinding-induced wafer warp (TAIKO® context)
//
// AP試験 / DISCO engineer concepts:
//   - 研削残留応力 → ウェーハ反り（Warp / BOW / TTV）
//   - Stoney 方程式: σ_f = (E_s · h_s²) / (6 · (1-ν_s) · h_f · R)
//     σ_f: 薄膜応力 [Pa], h_s: 基板厚, h_f: 研削層厚, R: 曲率半径
//   - Timoshenko バイメタル梁: ΔT → 反り量 δ = α·L²/h (熱応力アナログ)
//   - TAIKO® エッジリブ効果: エッジ幅 W_rim でリブが反りを抑制
//     δ_TAIKO / δ_full = 1 − tanh(W_rim / λ)  (λ: 特性長 ≈ 0.12 × wafer_r)
//   - 研削砥石番手（grit size）↑ → 表面粗さ↓・残留応力 ↓
//     σ_residual ≈ σ₀ × (grit_nominal / grit)^0.4
//   - HBM4 目標: 25µm 厚ウェーハ、反り < 1 mm
//
// DISCO context:
//   - DFG8560 グラインダー: 300mm ウェーハ 25µm 対応 (HBM4 向け)
//   - TAIKO® = エッジリブ残し薄ウェーハ → 本モジュールはその反り最小化設計

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace py = pybind11;

static constexpr double PI = M_PI;

// ── Material database ─────────────────────────────────────────────────────────
struct WaferMat {
    std::string name;
    double E_GPa;      // Young's modulus
    double nu;         // Poisson's ratio
    double alpha_ppm;  // CTE [ppm/K]
};

static const WaferMat WAFER_MATS[] = {
    {"Si",       130.0, 0.28, 2.6},
    {"SiC",      450.0, 0.17, 4.2},
    {"GaN",      295.0, 0.23, 3.2},
    {"Sapphire", 400.0, 0.25, 5.3},
};

static const WaferMat* find_mat(const std::string& name) {
    for (auto& m : WAFER_MATS)
        if (m.name == name) return &m;
    return &WAFER_MATS[0];
}

// ── Stoney equation: warp from residual stress ────────────────────────────────
// R_curv [m] = (E_s · h_s²) / (6 · (1-ν_s) · h_f · σ_f)
// δ_center [µm] = diameter² / (8 · R_curv) × 1e6
static py::dict stoney_warp(
    const std::string& material,
    double wafer_diam_mm,
    double target_thickness_um,   // thinned wafer thickness
    double ground_layer_um,       // thickness of damaged/stressed layer
    double grit_number,           // #320, #2000, #8000 ... (higher = finer)
    double delta_T_C              // thermal gradient during grinding [K]
) {
    const WaferMat* m = find_mat(material);
    double h_s = target_thickness_um * 1e-6;   // m
    double h_f = ground_layer_um * 1e-6;       // m
    double R   = wafer_diam_mm * 1e-3 / 2.0;  // wafer radius [m]
    double L   = wafer_diam_mm * 1e-3;         // diameter

    // Residual stress model: σ₀ depends on material hardness, grit
    double sigma0_MPa = 350.0 * (m->E_GPa / 130.0) * 0.5;  // baseline at #320
    double sigma_f_MPa = sigma0_MPa * std::pow(320.0 / grit_number, 0.4);

    // Thermal stress contribution (Timoshenko bimetal analog)
    double sigma_thermal_MPa = m->E_GPa * 1e3 * m->alpha_ppm * 1e-6 * delta_T_C;
    double sigma_total_MPa = std::hypot(sigma_f_MPa, sigma_thermal_MPa);

    double sigma_Pa = sigma_total_MPa * 1e6;
    double E_s = m->E_GPa * 1e9;
    double nu_s = m->nu;

    // Stoney: R_curv = E_s·h_s² / (6·(1-ν_s)·h_f·σ_f)
    double R_curv_m = (E_s * h_s * h_s) / (6.0 * (1.0 - nu_s) * h_f * sigma_Pa);
    double bow_mm   = (L * L) / (8.0 * std::abs(R_curv_m)) * 1e3;  // mm

    // Warp (peak-to-valley across diameter)
    double warp_um  = bow_mm * 1e3;  // µm

    // TTV estimate (Total Thickness Variation): ~10% of warp for grinding
    double ttv_um = warp_um * 0.10;

    // Handling risk: > 5mm warp → fragile at 25µm
    std::string risk = (warp_um > 5000) ? "CRITICAL" :
                       (warp_um > 1000) ? "HIGH" :
                       (warp_um > 200)  ? "MODERATE" : "LOW";

    py::dict r;
    r["material"]          = material;
    r["target_thickness_um"] = target_thickness_um;
    r["grit_number"]       = grit_number;
    r["sigma_residual_MPa"]= sigma_f_MPa;
    r["sigma_thermal_MPa"] = sigma_thermal_MPa;
    r["sigma_total_MPa"]   = sigma_total_MPa;
    r["R_curvature_m"]     = R_curv_m;
    r["bow_mm"]            = bow_mm;
    r["warp_um"]           = warp_um;
    r["ttv_um"]            = ttv_um;
    r["handling_risk"]     = risk;
    return r;
}

// ── TAIKO® rib suppression model ──────────────────────────────────────────────
// エッジリブが残っている場合の反り抑制率
// δ_TAIKO / δ_full = 1 − tanh(W_rim / λ)
// λ = 0.12 × R_wafer
static py::dict taiko_warp_reduction(
    const std::string& material,
    double wafer_diam_mm,
    double target_thickness_um,
    double rim_width_mm,           // TAIKO® rim width (typically 3–5 mm)
    double rim_thickness_um,       // rim thickness (full wafer thickness)
    double grit_number,
    double delta_T_C
) {
    auto full = stoney_warp(material, wafer_diam_mm,
                             target_thickness_um, 5.0, grit_number, delta_T_C);
    double warp_full_um = full["warp_um"].cast<double>();

    double R_wafer = wafer_diam_mm / 2.0;
    double lambda  = 0.12 * R_wafer;
    double suppression = std::tanh(rim_width_mm / lambda);
    double warp_taiko_um = warp_full_um * (1.0 - suppression);

    // Effective stiffness increase from rim
    double h_full = rim_thickness_um * 1e-6;
    double h_thin = target_thickness_um * 1e-6;
    double stiffness_ratio = std::pow(h_full / h_thin, 3.0);  // I ∝ h³

    std::string feasible = (warp_taiko_um < 1000.0) ? "YES" : "NO";
    std::string vs_hbm4  = (warp_taiko_um < 1000.0) ? "PASS (< 1 mm)" : "FAIL";

    py::dict r;
    r["material"]            = material;
    r["target_thickness_um"] = target_thickness_um;
    r["rim_width_mm"]        = rim_width_mm;
    r["rim_thickness_um"]    = rim_thickness_um;
    r["warp_full_grind_um"]  = warp_full_um;
    r["suppression_fraction"]= suppression;
    r["warp_taiko_um"]       = warp_taiko_um;
    r["stiffness_ratio"]     = stiffness_ratio;
    r["feasible_25um"]       = feasible;
    r["hbm4_spec_1mm"]       = vs_hbm4;
    return r;
}

// ── Grit sweep: find minimum grit for warp < target ──────────────────────────
static py::dict grit_sweep(
    const std::string& material,
    double wafer_diam_mm,
    double target_thickness_um,
    double warp_limit_um,
    bool   use_taiko,
    double rim_width_mm
) {
    std::vector<double> grits = {
        60, 120, 220, 320, 400, 600, 1200, 2000, 4000, 8000
    };
    py::list sweep;
    int best_grit = -1;
    double best_warp = 1e18;

    for (double g : grits) {
        double warp;
        if (use_taiko) {
            auto r = taiko_warp_reduction(material, wafer_diam_mm,
                                           target_thickness_um, rim_width_mm,
                                           775.0, g, 5.0);
            warp = r["warp_taiko_um"].cast<double>();
        } else {
            auto r = stoney_warp(material, wafer_diam_mm,
                                  target_thickness_um, 5.0, g, 5.0);
            warp = r["warp_um"].cast<double>();
        }
        py::dict entry;
        entry["grit"]    = g;
        entry["warp_um"] = warp;
        entry["pass"]    = (warp <= warp_limit_um);
        sweep.append(entry);
        if (warp <= warp_limit_um && best_grit < 0) {
            best_grit = static_cast<int>(g);
            best_warp = warp;
        }
    }

    py::dict r;
    r["material"]        = material;
    r["warp_limit_um"]   = warp_limit_um;
    r["use_taiko"]       = use_taiko;
    r["sweep"]           = sweep;
    r["min_grit_needed"] = best_grit;
    r["best_warp_um"]    = (best_grit > 0) ? best_warp : -1.0;
    r["achievable"]      = (best_grit > 0);
    return r;
}

// ── Demo: Si vs SiC, with and without TAIKO® ─────────────────────────────────
static py::list wafer_warp_demo() {
    py::list results;
    std::vector<std::string> mats = {"Si", "SiC"};
    for (auto& mat : mats) {
        auto no_taiko = stoney_warp(mat, 300.0, 25.0, 5.0, 2000.0, 5.0);
        auto with_taiko = taiko_warp_reduction(mat, 300.0, 25.0, 3.5, 775.0, 2000.0, 5.0);
        py::dict r;
        r["material"]             = mat;
        r["warp_no_taiko_um"]     = no_taiko["warp_um"];
        r["warp_with_taiko_um"]   = with_taiko["warp_taiko_um"];
        r["taiko_reduction_pct"]  = (1.0 - with_taiko["warp_taiko_um"].cast<double>()
                                         / no_taiko["warp_um"].cast<double>()) * 100.0;
        r["hbm4_spec"]            = with_taiko["hbm4_spec_1mm"];
        results.append(r);
    }
    return results;
}

PYBIND11_MODULE(_wafer_warp_kernel, m) {
    m.doc() = "Wafer warp from grinding residual stress — TAIKO® rim suppression model";

    m.def("stoney_warp", &stoney_warp,
          py::arg("material")            = "Si",
          py::arg("wafer_diam_mm")       = 300.0,
          py::arg("target_thickness_um") = 25.0,
          py::arg("ground_layer_um")     = 5.0,
          py::arg("grit_number")         = 2000.0,
          py::arg("delta_T_C")           = 5.0,
          "Stoney equation: residual stress + thermal gradient -> warp.\n"
          "Returns: sigma_residual/thermal/total_MPa, R_curvature_m, bow_mm, warp_um, TTV.");

    m.def("taiko_warp_reduction", &taiko_warp_reduction,
          py::arg("material")            = "Si",
          py::arg("wafer_diam_mm")       = 300.0,
          py::arg("target_thickness_um") = 25.0,
          py::arg("rim_width_mm")        = 3.5,
          py::arg("rim_thickness_um")    = 775.0,
          py::arg("grit_number")         = 2000.0,
          py::arg("delta_T_C")           = 5.0,
          "TAIKO(R) rim suppression: delta_TAIKO/delta_full = 1-tanh(W/lambda).\n"
          "Returns: warp_full/taiko_um, suppression_fraction, stiffness_ratio, HBM4 pass/fail.");

    m.def("grit_sweep", &grit_sweep,
          py::arg("material")            = "Si",
          py::arg("wafer_diam_mm")       = 300.0,
          py::arg("target_thickness_um") = 25.0,
          py::arg("warp_limit_um")       = 1000.0,
          py::arg("use_taiko")           = true,
          py::arg("rim_width_mm")        = 3.5,
          "Sweep grit #60 to #8000: find minimum grit meeting warp spec.\n"
          "Returns: sweep list, min_grit_needed, achievable flag.");

    m.def("wafer_warp_demo", &wafer_warp_demo,
          "Si vs SiC at 25µm: compare warp with/without TAIKO(R), HBM4 feasibility.");
}
