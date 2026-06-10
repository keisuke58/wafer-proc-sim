// inp_dicing_kernel.cpp — InP photonic chip dicing & IOWN context
//
// AP試験 / DISCO engineer concepts:
//   - InP（インジウムリン）: 亜鉛閃亜鉛鉱型構造 (zinc-blende, F-43m)
//     バンドギャップ: 1.35eV (直接遷移) → 光デバイスに最適
//     劈開面: {110} → 4方向に完全劈開 → ウェーハが非常に割れやすい
//     K_IC ≈ 0.3 MPa√m (SiCの1/7 !! 非常に脆い)
//
//   - 用途:
//     ① DFBレーザー (1.3µm / 1.55µm) → 光ファイバー通信の光源
//     ② EML (電界吸収変調器付きレーザー) → 高速光通信、IOWNのO-RAN fronthaul
//     ③ フォトニック集積回路 (PIC) → Si-Phとの比較・補完
//
//   - IOWNとの接続:
//     NTT IOWN構想の「光電融合チップ」では、レーザー光源としてInP-DFBが必須
//     → Si-PhotonicsチップにInPチップを3D積層 (die bonding)
//     → InPの精密ダイシングがSi-Phチップ歩留りを左右する
//
//   - ダイシングの難しさ:
//     K_IC=0.3 → ブレードダイシング時のチッピングが深刻
//     通常: レーザーダイシング（熱影響少なく精密）か
//           ブレード極細 (#4000+) 極低送り (<2 mm/s)
//     ウェーハサイズ: 2〜4インチ (小径 → 枚数多く扱いが面倒)
//
// DISCO context:
//   - DFL7560: InP薄ウェーハレーザーダイシングに対応
//   - IOWNロードマップ上でInP需要は 2025→2030 急増見通し
//   - 「光電融合パッケージング」向けの高精度ダイシングはDISCOの差別化領域

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <string>
#include <vector>

namespace py = pybind11;

static constexpr double PI = M_PI;

// ── InP photonic chip properties ──────────────────────────────────────────────
static py::dict inp_photonic_chip_properties() {
    py::dict r;
    // Mechanical
    r["crystal_system"]      = std::string("Zinc-blende (F-43m)");
    r["E_GPa"]               = 61.0;
    r["nu"]                  = 0.36;
    r["H_GPa"]               = 4.5;
    r["K_IC_MPa_sqrtm"]      = 0.30;   // among the lowest of all semiconductors
    r["cleavage_planes"]     = std::string("{110} — 4 equivalent cleavage directions");
    r["CTE_ppm_K"]           = 4.6;
    // Electronic / Optical
    r["bandgap_eV"]          = 1.35;
    r["transition_type"]     = std::string("Direct gap — efficient light emission");
    r["refractive_index"]    = 3.17;
    r["wavelength_range_nm"] = std::string("1260-1620 nm (O/C/L band)");
    // Wafer specs
    r["wafer_sizes_inch"]    = std::string("2, 3, 4 inch (no 6-inch yet)");
    r["typical_thickness_um"]= 350.0;
    r["thinned_for_dicing_um"]= 100.0;
    // DISCO relevance
    r["fragility_vs_Si"]     = std::string("K_IC = 0.30 vs Si 0.80 — 2.7x more fragile");
    r["fragility_vs_SiC"]    = std::string("K_IC = 0.30 vs SiC 2.0 — 6.7x more fragile");
    r["recommended_method"]  = std::string("Laser dicing (DFL7560) or ultra-fine blade #4000+ @<2mm/s");
    return r;
}

// ── Laser dicing HAZ model ────────────────────────────────────────────────────
// Heat-affected zone (HAZ): HAZ ∝ sqrt(P / (v * κ))
// where P = laser power, v = feed speed, κ = thermal diffusivity of InP
// InP: κ ≈ 2.7e-6 m²/s (low thermal diffusivity)
// Si:  κ ≈ 9.0e-5 m²/s → InP HAZ spreads much less → laser favored
static py::dict laser_haz_model(
    double laser_power_mW,
    double feed_mm_s,
    double wafer_thickness_um,
    double pulse_duration_ns  // 0 = CW; >0 = pulsed
) {
    // Thermal diffusivity [m²/s]
    double kappa_InP = 2.7e-6;
    double kappa_Si  = 9.0e-5;

    // HAZ radius ≈ 2 * sqrt(kappa * t_interaction)
    // t_interaction = beam_diameter / feed_speed
    // Beam diameter ~ 3 µm (focus), feed speed in mm/s
    double beam_um = 3.0;
    double feed_m_s = feed_mm_s * 1e-3;
    double t_interact = (beam_um * 1e-6) / std::max(feed_m_s, 1e-9);

    double haz_radius_inp_um = 2.0 * std::sqrt(kappa_InP * t_interact) * 1e6;
    double haz_radius_si_um  = 2.0 * std::sqrt(kappa_Si  * t_interact) * 1e6;

    // Add pulse effect: pulsed laser reduces HAZ (shorter interaction time)
    double pulse_factor = (pulse_duration_ns > 0)
                        ? std::sqrt(pulse_duration_ns * 1e-9 / t_interact)
                        : 1.0;
    pulse_factor = std::min(pulse_factor, 1.0);
    double haz_pulsed_um = haz_radius_inp_um * pulse_factor;

    // Kerf width = beam_diameter + 2 * HAZ
    double kerf_um = beam_um + 2.0 * haz_pulsed_um;

    // Active chip area: (chip_size - kerf) / chip_size → yield proxy
    std::string quality = (haz_pulsed_um < 3.0) ? "EXCELLENT (< 3 µm HAZ)" :
                          (haz_pulsed_um < 8.0) ? "GOOD" : "MODERATE — pulsed preferred";

    py::dict r;
    r["laser_power_mW"]     = laser_power_mW;
    r["feed_mm_s"]          = feed_mm_s;
    r["wafer_thickness_um"] = wafer_thickness_um;
    r["pulse_duration_ns"]  = pulse_duration_ns;
    r["haz_cw_um"]          = haz_radius_inp_um;
    r["haz_si_equiv_um"]    = haz_radius_si_um;
    r["haz_pulsed_um"]      = haz_pulsed_um;
    r["kerf_width_um"]      = kerf_um;
    r["quality_grade"]      = quality;
    r["note"] = std::string(
        "InP low thermal diffusivity (2.7e-6 m2/s) reduces HAZ vs Si (9.0e-5) "
        "— laser is preferred over blade for InP photonic chips");
    return r;
}

// ── Blade vs laser yield model ────────────────────────────────────────────────
// Chip yield model: P_yield = exp(- n_edges * P_chip_per_edge)
// P_chip_per_edge ∝ (chip_um / min_spacing_um)^2
static py::dict blade_vs_laser_yield(
    double chip_size_mm,
    double street_width_um,
    const std::string& dicing_method,  // "blade" or "laser"
    double wafer_thickness_um
) {
    double K_IC   = 0.30;
    double E_GPa  = 61.0;
    double H_GPa  = 4.5;

    // Blade: chipping ∝ Lawn-Evans
    double F_N_norm = 0.002;  // relative force at standard conditions
    double chip_blade_um = 3.0 * std::sqrt(E_GPa / H_GPa) * std::sqrt(F_N_norm) / K_IC;

    // Laser: HAZ-based "soft chip"
    double chip_laser_um = 2.5;  // typical for UV pulsed laser on InP

    double chip_um = (dicing_method == "blade") ? chip_blade_um : chip_laser_um;

    // Number of edges per chip (4 sides), each with chip probability
    int n_edges = 4;
    double p_chip_per_edge = std::min(chip_um / (chip_size_mm * 500.0), 0.5);
    double p_yield_per_chip = std::pow(1.0 - p_chip_per_edge, n_edges);

    // Kerf efficiency
    double kerf_um = (dicing_method == "blade") ? street_width_um * 0.5
                                                 : 10.0;  // laser: ~10µm kerf
    double chip_area = chip_size_mm * chip_size_mm;
    double pitch_mm  = chip_size_mm + (street_width_um * 1e-3);
    double silicon_eff = chip_area / (pitch_mm * pitch_mm);

    py::dict r;
    r["dicing_method"]      = dicing_method;
    r["chip_size_mm"]       = chip_size_mm;
    r["street_width_um"]    = street_width_um;
    r["chip_um_estimated"]  = chip_um;
    r["kerf_um"]            = kerf_um;
    r["p_yield_per_chip"]   = p_yield_per_chip;
    r["silicon_utilization"]= silicon_eff;
    r["recommendation"] = (p_yield_per_chip > 0.90)
        ? std::string("Acceptable yield")
        : std::string("Yield risk — switch to laser or reduce feed speed");
    return r;
}

// ── IOWN market context ───────────────────────────────────────────────────────
static py::dict iown_market_context() {
    py::dict r;
    r["iown_full_name"]     = std::string("Innovative Optical and Wireless Network (NTT)");
    r["target_year"]        = std::string("2030 full deployment");
    r["inp_role"]           = std::string("DFB/EML laser source in Si-Ph co-packaged optics");
    r["chip_type"]          = std::string("EML (Electro-absorption Modulated Laser)");
    r["bitrate_per_chip"]   = std::string("100G-400G per lane (coherent/direct detect)");
    r["wafer_to_chip"]      = std::string("InP 2-4 inch -> die bond onto Si-Ph interposer");
    r["disco_role"]         = std::string(
        "Precision laser dicing of InP EML chips; "
        "street width ~30um, K_IC=0.3 requires DFL7560 UV laser");
    r["demand_driver"]      = std::string(
        "AI datacenter traffic 60%+ CAGR -> transceiver demand -> InP EML supply");
    r["market_size_2030"]   = std::string("InP epitaxy+fab market: ~$3B (2030 estimate)");
    r["competitor"]         = std::string("II-VI (Coherent), Lumentum, Sumitomo Electric");
    r["interview_point"]    = std::string(
        "IOWN x DISCO: InP dicing accuracy directly sets EML chip yield -> "
        "optical transceiver cost -> data-center deployment speed");
    return r;
}

PYBIND11_MODULE(_inp_dicing_kernel, m) {
    m.doc() = "InP photonic chip dicing: crystal properties, laser HAZ, yield model, IOWN market context";

    m.def("inp_photonic_chip_properties", &inp_photonic_chip_properties,
          "InP zinc-blende properties: E=61GPa, K_IC=0.3, H=4.5, {110} cleavage.\n"
          "Returns: mechanical props, optical params, wafer specs, DISCO relevance.");

    m.def("laser_haz_model", &laser_haz_model,
          py::arg("laser_power_mW")    = 200.0,
          py::arg("feed_mm_s")         = 5.0,
          py::arg("wafer_thickness_um")= 100.0,
          py::arg("pulse_duration_ns") = 10.0,
          "Heat-affected zone for laser dicing InP.\n"
          "Compares CW vs pulsed; InP vs Si thermal diffusivity.\n"
          "Returns: HAZ radius, kerf width, quality grade.");

    m.def("blade_vs_laser_yield", &blade_vs_laser_yield,
          py::arg("chip_size_mm")       = 1.0,
          py::arg("street_width_um")    = 40.0,
          py::arg("dicing_method")      = "laser",
          py::arg("wafer_thickness_um") = 100.0,
          "Chip yield model for InP: blade (Lawn-Evans chipping) vs laser.\n"
          "Returns: chip_um, kerf, p_yield, silicon utilization.");

    m.def("iown_market_context", &iown_market_context,
          "IOWN x InP x DISCO connection: EML chips, Si-Ph co-packaging, demand drivers.\n"
          "Returns: market context dict for interview preparation.");
}
