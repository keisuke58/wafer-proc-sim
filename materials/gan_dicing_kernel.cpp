// gan_dicing_kernel.cpp — GaN substrate dicing & epitaxy-induced bow
//
// AP試験 / DISCO engineer concepts:
//   - GaN RF市場: GaN-on-SiC (高熱伝導・高価)  → 基地局パワーアンプ
//     GaN電力市場: GaN-on-Si (低コスト・大口径) → EV充電・インバータ
//   - エピ誘起反り: GaN(5.6ppm/K) vs Si(2.6ppm/K) → ΔαΔT → Stoney式反り
//     300mm GaN-on-Si は反り >1mm になり扱いが難しい → DFG8560 TAIKO対応
//   - ダイシング方法選択:
//     ① ブレードダイシング: GaN硬度 ~12GPa → SiC並に難しい (K_IC≈0.8)
//     ② ステルスダイシング: GaN-on-Si に有効。内部改質で応力割断
//        波長: 1064nm (Si透過), GaN層を局所改質
//     ③ レーザーアブレーション: GaN-on-SiC 薄層 RF チップ
//   - ストリート幅: GaN-on-SiC ~40µm (高価→小ストリート),
//                   GaN-on-Si ~80µm (Si基板のブレード対応)
//
// DISCO context:
//   - DFL7560 (レーザー), DFD6362 (ブレード) がGaN市場に投入済み
//   - GaN-on-Si の大口径化 (200mm→300mm) に伴い反り対応が新課題

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <string>
#include <vector>

namespace py = pybind11;

struct GaNSubstrate {
    std::string name;
    double E_substrate_GPa;
    double nu_substrate;
    double alpha_sub_ppm;  // CTE of substrate
    double alpha_gan_ppm;  // CTE of GaN epilayer = 5.6
    double K_IC_composite; // effective K_IC [MPa√m]
    double H_GPa;          // hardness
    std::string primary_use;
};

static const GaNSubstrate GAN_SUBS[] = {
    {"GaN-on-SiC", 450.0, 0.17, 4.2,  5.6, 0.9, 12.0, "RF/5G power amplifier"},
    {"GaN-on-Si",  130.0, 0.28, 2.6,  5.6, 0.7,  9.5, "EV inverter/charger"},
    {"Freestanding", 295.0, 0.23, 5.6, 5.6, 0.8, 12.0, "R&D / high-power LED"},
};

static const GaNSubstrate* find_sub(const std::string& name) {
    for (auto& s : GAN_SUBS)
        if (s.name == name) return &s;
    return &GAN_SUBS[1]; // default GaN-on-Si
}

// ── Substrate properties + dicing recommendation ──────────────────────────────
static py::dict gan_substrate_properties(const std::string& substrate_type) {
    const GaNSubstrate* s = find_sub(substrate_type);

    // Dicing method scoring (0–10): higher = more suitable
    double blade_score  = 10.0 - s->K_IC_composite * 3.0
                        - (s->H_GPa > 11.0 ? 2.0 : 0.0);
    double stealth_score = (substrate_type == "GaN-on-Si")  ? 8.5 :
                           (substrate_type == "GaN-on-SiC") ? 5.0 : 6.0;
    double laser_score  = (substrate_type == "GaN-on-SiC")  ? 8.0 :
                          (substrate_type == "GaN-on-Si")   ? 5.5 : 7.0;

    std::string best;
    if (stealth_score >= blade_score && stealth_score >= laser_score)
        best = "Stealth dicing";
    else if (laser_score >= blade_score)
        best = "Laser ablation";
    else
        best = "Blade dicing";

    py::dict r;
    r["substrate"]            = substrate_type;
    r["E_substrate_GPa"]      = s->E_substrate_GPa;
    r["nu_substrate"]         = s->nu_substrate;
    r["alpha_substrate_ppm"]  = s->alpha_sub_ppm;
    r["alpha_GaN_ppm"]        = s->alpha_gan_ppm;
    r["delta_alpha_ppm"]      = std::abs(s->alpha_gan_ppm - s->alpha_sub_ppm);
    r["K_IC_MPa_sqrtm"]       = s->K_IC_composite;
    r["H_GPa"]                = s->H_GPa;
    r["primary_use"]          = s->primary_use;
    r["blade_score"]          = blade_score;
    r["stealth_score"]        = stealth_score;
    r["laser_score"]          = laser_score;
    r["recommended_method"]   = best;
    return r;
}

// ── Epitaxy-induced bow (Stoney analog) ───────────────────────────────────────
// Stress from CTE mismatch during cooldown from growth temperature (~1050°C)
// σ_epi = E_sub / (1 - ν_sub) * Δα * ΔT
// R_curv = E_sub * h_sub² / (6 * (1-ν_sub) * h_epi * σ_epi)
// bow_mm = diam² / (8 * R_curv) * 1e3
static py::dict bow_from_epitaxy(
    const std::string& substrate_type,
    double wafer_diam_mm,
    double gan_thickness_um,   // epi layer [µm]
    double growth_temp_C,      // MOCVD temperature (typically 1050 °C)
    double room_temp_C         // room temperature (25 °C)
) {
    const GaNSubstrate* s = find_sub(substrate_type);
    double delta_T = growth_temp_C - room_temp_C;
    double delta_alpha = s->alpha_gan_ppm * 1e-6 - s->alpha_sub_ppm * 1e-6;

    double E_s  = s->E_substrate_GPa * 1e9;
    double nu_s = s->nu_substrate;
    double h_sub_m = wafer_diam_mm * 0.5 * 1e-3;  // rough: h_sub ~ radius/300
    // More realistic: typical substrate thickness
    double h_sub_typical_um = (wafer_diam_mm >= 200.0) ? 675.0 :
                              (wafer_diam_mm >= 150.0) ? 525.0 : 350.0;
    double h_s = h_sub_typical_um * 1e-6;
    double h_f = gan_thickness_um  * 1e-6;
    double L   = wafer_diam_mm * 1e-3;

    // Biaxial stress in epi layer
    double sigma_epi_Pa = E_s / (1.0 - nu_s) * delta_alpha * delta_T;

    // Sign: GaN-on-Si → delta_alpha > 0 → sigma tensile → concave up
    //       GaN-on-SiC → delta_alpha < 0 → sigma compressive → convex
    double sigma_abs = std::abs(sigma_epi_Pa);
    double R_curv_m  = (E_s * h_s * h_s) / (6.0 * (1.0 - nu_s) * h_f * sigma_abs);
    double bow_mm    = (L * L) / (8.0 * R_curv_m) * 1e3;

    std::string shape = (delta_alpha > 0) ? "concave (GaN in tension)"
                                          : "convex (GaN in compression)";
    std::string handling = (bow_mm > 5.0) ? "CRITICAL — TAIKO required" :
                           (bow_mm > 1.0) ? "HIGH — edge handling only" :
                           (bow_mm > 0.3) ? "MODERATE" : "LOW";

    py::dict r;
    r["substrate"]           = substrate_type;
    r["wafer_diam_mm"]       = wafer_diam_mm;
    r["gan_thickness_um"]    = gan_thickness_um;
    r["delta_T_C"]           = delta_T;
    r["delta_alpha_ppm"]     = std::abs(delta_alpha) * 1e6;
    r["sigma_epi_MPa"]       = sigma_abs * 1e-6;
    r["R_curvature_m"]       = R_curv_m;
    r["bow_mm"]              = bow_mm;
    r["bow_shape"]           = shape;
    r["handling_risk"]       = handling;
    return r;
}

// ── Dicing method comparison ──────────────────────────────────────────────────
// Returns ranked method list with scores for a specific scenario
static py::dict dicing_method_comparison(
    const std::string& substrate_type,
    double chip_size_mm,
    double street_width_um,
    double wafer_thickness_um
) {
    const GaNSubstrate* s = find_sub(substrate_type);

    bool is_thin = (wafer_thickness_um < 100.0);
    bool narrow  = (street_width_um < 50.0);

    double blade_chip_um  = 5.0 * (s->H_GPa / 10.0) * (12.0 / (s->K_IC_composite + 1.0));
    double blade_kerf_um  = street_width_um * 0.6;
    double blade_yield    = 1.0 - blade_chip_um / (chip_size_mm * 1e3 * 0.5) * 0.2;
    blade_yield = std::min(std::max(blade_yield, 0.5), 1.0);

    double stealth_chip_um = 1.5;
    double stealth_kerf_um = 3.0;
    double stealth_yield   = (substrate_type == "GaN-on-Si") ? 0.96 :
                             (substrate_type == "GaN-on-SiC") ? 0.82 : 0.90;
    double stealth_available = (substrate_type == "GaN-on-Si") ? 1.0 :
                               (substrate_type == "GaN-on-SiC") ? 0.5 : 0.7;

    double laser_chip_um  = 3.0;
    double laser_kerf_um  = 10.0;
    double laser_haz_um   = 15.0;
    double laser_yield    = narrow ? 0.95 : 0.93;

    py::dict blade, stealth, laser;
    blade["method"]      = std::string("Blade dicing");
    blade["chip_um"]     = blade_chip_um;
    blade["kerf_um"]     = blade_kerf_um;
    blade["yield_est"]   = blade_yield;
    blade["throughput"]  = std::string("high");
    blade["note"]        = std::string("Standard; GaN hardness needs SD/CBN blade");

    stealth["method"]      = std::string("Stealth dicing");
    stealth["chip_um"]     = stealth_chip_um;
    stealth["kerf_um"]     = stealth_kerf_um;
    stealth["yield_est"]   = stealth_yield;
    stealth["availability"]= stealth_available;
    stealth["throughput"]  = std::string("medium");
    stealth["note"]        = substrate_type == "GaN-on-Si"
                           ? std::string("Best for GaN-on-Si: 1064nm Si-transparent")
                           : std::string("Limited: SiC opaque at 1064nm");

    laser["method"]      = std::string("Laser ablation");
    laser["chip_um"]     = laser_chip_um;
    laser["kerf_um"]     = laser_kerf_um;
    laser["haz_um"]      = laser_haz_um;
    laser["yield_est"]   = laser_yield;
    laser["throughput"]  = std::string("medium");
    laser["note"]        = std::string("Best for GaN-on-SiC RF chips; DISCO DFL7560");

    py::dict r;
    r["substrate"]     = substrate_type;
    r["chip_size_mm"]  = chip_size_mm;
    r["street_um"]     = street_width_um;
    r["blade"]         = blade;
    r["stealth"]       = stealth;
    r["laser"]         = laser;
    return r;
}

// ── Market context ─────────────────────────────────────────────────────────────
static py::dict gan_market_context() {
    py::dict r;
    r["rf_market"]         = std::string("GaN-on-SiC: 5G base station PA, defense radar");
    r["power_market"]      = std::string("GaN-on-Si: EV OBC, solar inverter, data-center PSU");
    r["rf_wafer_size"]     = std::string("4-inch dominant; moving to 6-inch 2025+");
    r["power_wafer_size"]  = std::string("150mm/200mm; 300mm GaN-on-Si pilot by 2026");
    r["disco_tools"]       = std::string("DFD6362 (blade), DFL7560 (laser), DFG8560 (grind)");
    r["growth_driver"]     = std::string("5G capex + EV adoption; 25% CAGR 2024-2030");
    r["main_challenge"]    = std::string("GaN-on-Si 300mm bow > 1mm → TAIKO grinding needed");
    r["vs_sic_power"]      = std::string("GaN: higher freq (MHz) | SiC: higher voltage (1200V+)");
    return r;
}

PYBIND11_MODULE(_gan_dicing_kernel, m) {
    m.doc() = "GaN substrate dicing: epitaxy bow, method selection (blade/stealth/laser), market context";

    m.def("gan_substrate_properties", &gan_substrate_properties,
          py::arg("substrate_type") = "GaN-on-Si",
          "Properties + dicing method scores for GaN-on-SiC, GaN-on-Si, Freestanding.\n"
          "Returns: E, K_IC, H, CTE mismatch, blade/stealth/laser scores, recommendation.");

    m.def("bow_from_epitaxy", &bow_from_epitaxy,
          py::arg("substrate_type")  = "GaN-on-Si",
          py::arg("wafer_diam_mm")   = 200.0,
          py::arg("gan_thickness_um")= 5.0,
          py::arg("growth_temp_C")   = 1050.0,
          py::arg("room_temp_C")     = 25.0,
          "Stoney-model bow from GaN/substrate CTE mismatch during MOCVD cooldown.\n"
          "Returns: sigma_epi, R_curv, bow_mm, handling risk.");

    m.def("dicing_method_comparison", &dicing_method_comparison,
          py::arg("substrate_type")     = "GaN-on-Si",
          py::arg("chip_size_mm")       = 2.0,
          py::arg("street_width_um")    = 80.0,
          py::arg("wafer_thickness_um") = 150.0,
          "Compare blade / stealth / laser dicing for given GaN substrate and geometry.\n"
          "Returns: chip size, kerf, yield, throughput, notes per method.");

    m.def("gan_market_context", &gan_market_context,
          "GaN market summary: RF vs Power, wafer sizes, DISCO tools, growth drivers.");
}
