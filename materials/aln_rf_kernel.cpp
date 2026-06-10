// aln_rf_kernel.cpp — AlN substrate: RF power modules, mmWave, thermal management
//
// AP試験 / DISCO engineer concepts:
//   - AlN（窒化アルミニウム）: ウルツ鉱型構造 (hexagonal, P6₃mc) = GaNと同じ系
//     バンドギャップ 6.2 eV (最広クラスの実用半導体)
//     熱伝導率 285 W/mK (AlNセラミクスは 170-200)
//     CTE 4.6 ppm/K ≒ GaN(5.6) → GaN-on-AlNは格子整合が良い
//
//   - 主要用途:
//     ① AlN基板 (セラミクス): パワーモジュールの放熱基板
//        Si や SiC パワーデバイスの下に置いてDBC (Direct Bonded Copper) 基板
//     ② AlN/GaN HEMT: RF増幅器の高電子移動度構造
//        GaN-on-AlN ≒ GaN-on-SiC の代替 (格子整合改善)
//     ③ AlN bulk ウェーハ: 深紫外LED(DUV-LED) の基板
//        → 殺菌・医療・半導体露光
//
//   - 市場:
//     宇宙・軍事レーダー: 高温耐性が必要 → AlN基板が必須
//     5G mmWave基地局: 28GHz/39GHz帯PAにGaN-on-SiC/AlN
//     DUV LED: UVC (222nm・254nm) → コロナ後に市場急拡大
//
//   - ダイシング特性:
//     AlNセラミクス基板: 非常に硬くて脆い (K_IC ≈ 3.0)
//     → ダイヤモンド砥石ブレード (#600-#1200) でダイシング可能
//     → ただし粗さ管理が必要: 医療・光学用途は Ra < 0.5µm 要求
//     AlN ウェーハ (バルク): 同様にレーザーが有利
//
// DISCO context:
//   - DFD6361ブレードダイサー: AlNセラミクス基板の切断実績多数
//   - 精密切断でアルミナ(Al₂O₃)・AlN混在基板の単品化に使用
//   - GaN-on-AlN RFチップ: DFL7560レーザーダイシング

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <string>
#include <vector>

namespace py = pybind11;

// ── AlN substrate properties ──────────────────────────────────────────────────
static py::dict aln_substrate_properties(const std::string& substrate_type) {
    // substrate_type: "AlN_ceramic", "AlN_single_crystal", "AlN_GaN_HEMT"
    py::dict r;
    r["substrate_type"] = substrate_type;

    if (substrate_type == "AlN_ceramic") {
        r["E_GPa"]                     = 310.0;
        r["nu"]                        = 0.24;
        r["H_GPa"]                     = 11.0;
        r["K_IC_MPa_sqrtm"]            = 3.0;
        r["thermal_conductivity_W_mK"] = 170.0;  // ceramic grade
        r["CTE_ppm_K"]                 = 4.5;
        r["bandgap_eV"]                = std::numeric_limits<double>::quiet_NaN();
        r["application"]               = std::string("Power module heat spreader (DBC substrate)");
        r["wafer_size"]                = std::string("Custom tiles 50-200mm");
        r["dicing_method"]             = std::string("Diamond blade #600-#1200");
    } else if (substrate_type == "AlN_single_crystal") {
        r["E_GPa"]                     = 344.0;
        r["nu"]                        = 0.23;
        r["H_GPa"]                     = 13.0;
        r["K_IC_MPa_sqrtm"]            = 3.5;
        r["thermal_conductivity_W_mK"] = 285.0;
        r["CTE_ppm_K"]                 = 4.6;
        r["bandgap_eV"]                = 6.2;
        r["application"]               = std::string("DUV-LED substrate, GaN HEMT, RF power");
        r["wafer_size"]                = std::string("2-inch dominant; 4-inch R&D (2024)");
        r["dicing_method"]             = std::string("Laser scribing preferred; fine blade possible");
    } else {  // AlN_GaN_HEMT
        r["E_GPa"]                     = 325.0;  // composite
        r["nu"]                        = 0.22;
        r["H_GPa"]                     = 12.5;
        r["K_IC_MPa_sqrtm"]            = 3.2;
        r["thermal_conductivity_W_mK"] = 250.0;  // GaN/AlN composite
        r["CTE_ppm_K"]                 = 4.8;
        r["bandgap_eV"]                = 3.4;  // GaN channel
        r["application"]               = std::string("mmWave RF HEMT (28GHz/39GHz 5G, Ka-band satellite)");
        r["wafer_size"]                = std::string("4-inch (SiC substrate); transitioning from SiC");
        r["dicing_method"]             = std::string("Laser ablation for narrow streets (<40µm)");
    }

    // Common: AlN hardness requires special blade selection
    r["blade_material"]   = std::string("Metal-bonded or resin-bonded diamond (not standard SD)");
    r["coolant_required"] = std::string("Yes — AlN reacts with water slowly; use deionized water");
    return r;
}

// ── RF thermal model: GaN-on-AlN vs alternatives ─────────────────────────────
// Channel temperature model for GaN HEMT:
// T_j = T_ambient + P_diss * (R_th_channel + R_th_buffer + R_th_substrate + R_th_package)
static py::dict rf_thermal_model(
    double P_diss_W,           // dissipated power per chip
    double gate_width_mm,      // total gate periphery
    const std::string& substrate
) {
    struct SubThermal { std::string name; double k; double thickness_um; };
    SubThermal subs[] = {
        {"GaN-on-SiC",   490.0,  100.0},
        {"GaN-on-AlN",   285.0,  100.0},
        {"GaN-on-Si",     50.0,  100.0},
        {"GaN-on-Diamond",2000.0, 50.0},
    };

    // GaN channel layer: k=230, thickness=2µm
    double k_GaN   = 230.0;
    double h_GaN   = 2.0e-6;
    double area_m2 = gate_width_mm * 1e-3 * 0.1e-3;  // gate_width × 100µm cell

    double R_GaN = h_GaN / (k_GaN * area_m2);

    py::list results;
    for (auto& s : subs) {
        double h_sub = s.thickness_um * 1e-6;
        double R_sub = h_sub / (s.k * area_m2);
        double R_total = R_GaN + R_sub;
        double T_rise = P_diss_W * R_total;
        py::dict d;
        d["substrate"]     = s.name;
        d["k_W_mK"]        = s.k;
        d["R_th_total_K_W"] = R_total;
        d["T_rise_K"]       = T_rise;
        d["T_junction_C"]   = 25.0 + T_rise;
        d["feasible_150C"]  = (25.0 + T_rise < 150.0);
        results.append(d);
    }

    py::dict r;
    r["P_diss_W"]       = P_diss_W;
    r["gate_width_mm"]  = gate_width_mm;
    r["results"]        = results;
    r["insight"] = std::string(
        "GaN-on-Diamond offers 4x lower T_rise vs SiC; "
        "GaN-on-AlN intermediate — better lattice match than SiC, lower cost than Diamond");
    return r;
}

// ── AlN blade dicing model ────────────────────────────────────────────────────
// Predicts surface roughness and chipping for AlN ceramic dicing
// Roughness: Ra ∝ (1/grit)^0.5 × feed^0.3 / (spindle_speed^0.2)
static py::dict aln_dicing_model(
    double blade_grit,       // diamond grit #320 – #2000
    double feed_mm_s,
    double spindle_krpm,
    double substrate_thickness_mm
) {
    // Ra model for hard ceramics (empirical fit)
    double Ra_um = 2.5 * std::pow(1000.0 / blade_grit, 0.5)
                 * std::pow(feed_mm_s / 5.0, 0.3)
                 / std::pow(spindle_krpm / 20.0, 0.2);

    // Chipping: Lawn-Evans with AlN properties
    double E = 310.0, H = 11.0, K_IC = 3.0;
    double F_norm = 10.0 / blade_grit;
    double chip_um = 2.5 * std::sqrt(E / H) * std::sqrt(F_norm) / K_IC;

    // Throughput estimate: singulation rate [chips/hr]
    double kerf_passes = std::ceil(substrate_thickness_mm / (blade_grit * 1e-3 * 0.5));
    double chips_per_hr = (spindle_krpm > 0) ? 3600.0 / (kerf_passes * 0.1 / feed_mm_s) : 0;

    std::string surface_grade = (Ra_um < 0.5) ? "Fine (optical/medical grade)" :
                                (Ra_um < 1.0) ? "Standard (power module grade)" : "Rough";

    py::dict r;
    r["blade_grit"]        = blade_grit;
    r["feed_mm_s"]         = feed_mm_s;
    r["spindle_krpm"]      = spindle_krpm;
    r["Ra_um"]             = Ra_um;
    r["chip_um"]           = chip_um;
    r["surface_grade"]     = surface_grade;
    r["est_chips_per_hr"]  = chips_per_hr;
    r["recommendation"] = (Ra_um < 0.5 && chip_um < 5.0)
        ? std::string("Parameters suitable for precision AlN dicing")
        : std::string("Reduce feed or increase grit for better surface quality");
    return r;
}

// ── mmWave + space market context ─────────────────────────────────────────────
static py::dict aln_market_context() {
    py::dict r;
    r["duv_led_market"]        = std::string("UVC 222/254nm: water purification, medical sterilization; 30% CAGR 2022-2028");
    r["mmwave_5g"]             = std::string("28/39GHz base station PA: GaN-on-SiC dominant, AlN growing");
    r["space_radar"]           = std::string("Ka-band (26.5-40GHz) SAR satellites, AESA radar");
    r["power_module_substrate"]= std::string("EV inverter DCB: AlN replaces Al2O3 for > 10kW modules");
    r["wafer_supply"]          = std::string("Tokuyama, Hexatech (US), HexaTech (Germany)");
    r["disco_tools"]           = std::string("DFD6361 blade (ceramic tiles); DFL7560 laser (wafers)");
    r["cte_advantage"]         = std::string("AlN CTE=4.6 vs GaN=5.6: 7x less thermal stress than Si(2.6)");
    r["vs_sic_economics"]      = std::string("AlN substrate ~3x cheaper than SiC per cm2; gaining share in RF");
    return r;
}

PYBIND11_MODULE(_aln_rf_kernel, m) {
    m.doc() = "AlN substrate: properties (ceramic/single-crystal/GaN-HEMT), RF thermal model, dicing, market";

    m.def("aln_substrate_properties", &aln_substrate_properties,
          py::arg("substrate_type") = "AlN_single_crystal",
          "AlN substrate properties for: AlN_ceramic, AlN_single_crystal, AlN_GaN_HEMT.\n"
          "Returns: E, K_IC, H, thermal conductivity, application, dicing method.");

    m.def("rf_thermal_model", &rf_thermal_model,
          py::arg("P_diss_W")      = 5.0,
          py::arg("gate_width_mm") = 1.0,
          py::arg("substrate")     = "GaN-on-SiC",
          "Channel temperature rise for GaN HEMT across substrate options.\n"
          "Compares GaN-on-SiC / AlN / Si / Diamond. Returns T_junction per substrate.");

    m.def("aln_dicing_model", &aln_dicing_model,
          py::arg("blade_grit")              = 800.0,
          py::arg("feed_mm_s")               = 5.0,
          py::arg("spindle_krpm")            = 20.0,
          py::arg("substrate_thickness_mm")  = 0.635,
          "Surface roughness + chipping model for AlN ceramic blade dicing.\n"
          "Returns: Ra, chip size, throughput estimate, recommendation.");

    m.def("aln_market_context", &aln_market_context,
          "AlN market: DUV-LED, mmWave 5G, space radar, power module substrate.\n"
          "Returns market context dict for interview preparation.");
}
