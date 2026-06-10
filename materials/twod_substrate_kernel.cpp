// twod_substrate_kernel.cpp — 2D materials (MoS2 / graphene / hBN) + sapphire substrate dicing
//
// AP試験 / DISCO engineer concepts:
//   - 2D材料の本質:
//     原子1層（~0.3–0.7nm）で動作するデバイス
//     シリコンの微細化限界（~1nm）を超える唯一の候補
//     ただしウェーハ規模の均一成膜がまだ難しい
//
//   - 主要2D材料:
//     ① グラフェン (C): バンドギャップなし → 超高速トランジスタ候補・放熱材料
//        電子移動度: ~200,000 cm²/Vs (室温、suspendedで)
//     ② MoS₂ (二硫化モリブデン): 直接遷移バンドギャップ 1.8eV (単層)
//        → MOSFET・光センサー・フレキシブル半導体
//     ③ hBN (六方晶窒化ホウ素): バンドギャップ ~6eV → グラフェンの理想ゲート絶縁体
//     ④ WS₂, WSe₂: MoS₂系列、より高い移動度
//
//   - DISCO文脈での2D材料:
//     ★ 2D材料層自体をダイシングすることはない（薄すぎる）
//     ★ 2D材料が成膜されている「基板」をダイシングする
//        基板の候補: サファイア(Al₂O₃)・SiC・Si・hBN自立膜
//
//   - サファイア (Al₂O₃) ダイシング:
//     硬度: モース9 (ダイヤモンドに次ぐ第2位)
//     K_IC ≈ 2.2 MPa√m
//     H ≈ 17-20 GPa (ビッカース)
//     E ≈ 345-400 GPa
//     劈開面: {1-102} (r面), {0001} (c面)
//     LED工場: GaN-on-Sapphire → サファイア基板ダイシングはDISCO主力品
//     → DFD6361 ブレード / DFL7560 レーザー 両対応
//
//   - 2D材料デバイス工場の流れ:
//     CVDでMoS₂/グラフェンをサファイア基板に成膜
//     → リソグラフィでデバイス形成
//     → サファイア基板ごとダイシング (DISCOの出番)
//     → チップをフレキシブル基板等に転写

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <string>
#include <vector>

namespace py = pybind11;

// ── 2D material properties ────────────────────────────────────────────────────
static py::dict twod_material_properties(const std::string& material) {
    py::dict r;
    r["material"] = material;

    if (material == "graphene") {
        r["layers"]                    = 1;
        r["thickness_nm"]              = 0.34;
        r["crystal_structure"]         = std::string("Hexagonal (honeycomb), sp2");
        r["bandgap_eV"]                = 0.0;  // zero bandgap
        r["bandgap_note"]              = std::string("Zero gap — needs substrate engineering or BLG for gap");
        r["electron_mobility_cm2_Vs"]  = 200000.0;  // theoretical, suspended
        r["practical_mobility_on_SiO2"]= 10000.0;
        r["thermal_conductivity_W_mK"] = 5000.0;  // suspended monolayer
        r["application"]               = std::string("RF transistor, transparent electrode, heat spreader, sensor");
        r["substrate"]                 = std::string("SiO2/Si, SiC, hBN (ideal), sapphire");
        r["growth_method"]             = std::string("CVD on Cu foil, then transfer; epitaxial on SiC");
        r["maturity"]                  = std::string("R&D + niche applications (2024)");
        r["disco_relevance"]           = std::string("SiC substrate dicing for epitaxial graphene");

    } else if (material == "MoS2") {
        r["layers"]                    = 1;
        r["thickness_nm"]              = 0.65;
        r["crystal_structure"]         = std::string("Hexagonal (trigonal prismatic), D6h");
        r["bandgap_eV"]                = 1.8;  // monolayer direct gap
        r["bandgap_bulk_eV"]           = 1.2;  // bulk indirect
        r["bandgap_note"]              = std::string("Monolayer: direct 1.8eV. Bulk: indirect 1.2eV");
        r["electron_mobility_cm2_Vs"]  = 200.0;  // typical on SiO2 at RT
        r["intrinsic_mobility_cm2_Vs"] = 400.0;
        r["application"]               = std::string("Thin-film FET, photodetector, flexible electronics");
        r["substrate"]                 = std::string("SiO2/Si, sapphire, flexible PET");
        r["growth_method"]             = std::string("CVD (sulfurization of MoO3), mechanical exfoliation");
        r["maturity"]                  = std::string("Research / early prototype");
        r["disco_relevance"]           = std::string("Sapphire substrate dicing for MoS2 device arrays");

    } else if (material == "hBN") {
        r["layers"]                    = 1;
        r["thickness_nm"]              = 0.33;
        r["crystal_structure"]         = std::string("Hexagonal, sp2 (isostructural with graphene)");
        r["bandgap_eV"]                = 6.0;
        r["bandgap_note"]              = std::string("Widest 2D bandgap — ideal dielectric for graphene/MoS2");
        r["electron_mobility_cm2_Vs"]  = 0.0;  // insulator
        r["application"]               = std::string("Gate dielectric, encapsulant for graphene, neutron detector");
        r["substrate"]                 = std::string("Si, sapphire, freestanding bulk crystal");
        r["growth_method"]             = std::string("CVD; bulk crystal growth (Watanabe-Taniguchi method)");
        r["maturity"]                  = std::string("Research; bulk crystal supply limited");
        r["disco_relevance"]           = std::string("Bulk hBN crystal cleaving for substrate preparation");

    } else {  // WS2 default
        r["layers"]                    = 1;
        r["thickness_nm"]              = 0.62;
        r["crystal_structure"]         = std::string("Hexagonal D6h (same as MoS2)");
        r["bandgap_eV"]                = 2.0;  // monolayer
        r["electron_mobility_cm2_Vs"]  = 234.0;
        r["hole_mobility_cm2_Vs"]      = 250.0;  // ambipolar
        r["application"]               = std::string("Ambipolar transistor, LED/laser, valleytronics");
        r["substrate"]                 = std::string("SiO2/Si, sapphire");
        r["growth_method"]             = std::string("CVD (WO3 sulfurization)");
        r["maturity"]                  = std::string("Research");
        r["disco_relevance"]           = std::string("Sapphire substrate dicing");
    }
    return r;
}

// ── Sapphire substrate properties ─────────────────────────────────────────────
static py::dict sapphire_properties() {
    py::dict r;
    r["formula"]                   = std::string("Al2O3 (corundum)");
    r["crystal_system"]            = std::string("Rhombohedral (R-3c), hexagonal axes");
    r["E_c_GPa"]                   = 400.0;  // c-axis (0001)
    r["E_a_GPa"]                   = 345.0;  // a-plane
    r["nu"]                        = 0.25;
    r["H_GPa"]                     = 18.0;   // Vickers ~1800 kg/mm2
    r["mohs_hardness"]             = 9.0;    // second hardest natural material
    r["K_IC_MPa_sqrtm"]            = 2.2;    // moderate toughness
    r["cleavage_planes"]           = std::string("{1-102} r-plane (preferred), {0001} c-plane");
    r["CTE_ppm_K"]                 = 5.4;
    r["thermal_conductivity_W_mK"] = 35.0;
    r["bandgap_eV"]                = 9.9;    // transparent UV to IR
    r["transmission_range"]        = std::string("0.17–5.5 µm (UV to mid-IR)");
    r["common_wafer_sizes"]        = std::string("2,4,6,8 inch — LED industry standard");
    r["LED_use"]                   = std::string("GaN-on-Sapphire: 80%+ of white LED production");
    r["dicing_challenge"]          = std::string("H=18GPa requires fine diamond blade; hard on blade life");
    r["recommended_methods"]       = std::string("Laser scribing (DFL7560) + breaking, or fine blade #2000+");
    return r;
}

// ── Sapphire dicing: blade vs laser ──────────────────────────────────────────
static py::dict sapphire_dicing(
    double wafer_diam_mm,
    double thickness_um,
    const std::string& method,  // "blade" or "laser"
    double blade_grit,
    double feed_mm_s
) {
    // Sapphire properties
    double E = 400.0, H = 18.0, K_IC = 2.2;

    if (method == "blade") {
        double F_norm    = 10.0 / blade_grit;
        double chip_um   = 2.5 * std::sqrt(E / H) * std::sqrt(F_norm) / K_IC;
        double Ra_um     = 1.8 * std::pow(1000.0 / blade_grit, 0.5)
                         * std::pow(feed_mm_s / 5.0, 0.3);
        double kerf_um   = blade_grit < 600 ? 80.0 : 50.0;
        double blade_life = 1000.0 * (K_IC / E) * (blade_grit / 1000.0);  // relative life

        py::dict r;
        r["method"]       = method;
        r["blade_grit"]   = blade_grit;
        r["chip_um"]      = chip_um;
        r["Ra_um"]        = Ra_um;
        r["kerf_um"]      = kerf_um;
        r["blade_life_rel"]= blade_life;
        r["note"] = std::string(
            "Sapphire hard on blades — use resin bond diamond. "
            "Blade life ~30-50% of SiC. #2000+ for LED-grade surface.");
        return r;
    } else {
        // Laser: scribing along r-plane {1-102} then breaking
        double scribe_depth_um = thickness_um * 0.3;  // 30% scribing depth
        double breaking_force_N = K_IC * std::sqrt(M_PI * scribe_depth_um * 1e-6) * 1e6 * 1e-3;
        double kerf_um = 5.0;  // laser scribing kerf
        double t_per_chip_ms = (wafer_diam_mm / 200.0) * 2.0;  // relative time estimate

        py::dict r;
        r["method"]             = method;
        r["kerf_um"]            = kerf_um;
        r["scribe_depth_um"]    = scribe_depth_um;
        r["breaking_force_N"]   = breaking_force_N;
        r["surface_quality"]    = std::string("Better than blade; cleavage-controlled edge");
        r["note"] = std::string(
            "Laser scribing along r-plane (1-102) is the dominant method for LED sapphire. "
            "DISCO DFL7560 UV laser; throughput ~5000 chips/hr for 4-inch");
        return r;
    }
}

// ── Post-silicon roadmap ──────────────────────────────────────────────────────
static py::dict future_semiconductor_roadmap() {
    py::dict r;

    py::dict era1, era2, era3, era4, era5;
    era1["era"]       = std::string("2000-2020: Si CMOS scaling");
    era1["key_material"] = std::string("Si (bulk, SOI)");
    era1["node"]      = std::string("65nm → 7nm → 3nm");
    era1["disco_role"]= std::string("Si wafer dicing, thinning (HBM grinding)");

    era2["era"]       = std::string("2015-2030: WBG power");
    era2["key_material"] = std::string("SiC + GaN");
    era2["driver"]    = std::string("EV + 5G base station");
    era2["disco_role"]= std::string("SiC/GaN dicing — core growth market NOW");

    era3["era"]       = std::string("2025-2035: UWBG + photonics");
    era3["key_material"] = std::string("Ga2O3 + InP + AlN");
    era3["driver"]    = std::string("Next-gen EV inverter + IOWN + mmWave radar");
    era3["disco_role"]= std::string("Ga2O3 cleavage dicing, InP laser, AlN ceramic");

    era4["era"]       = std::string("2030-2040: 2D + Diamond");
    era4["key_material"] = std::string("MoS2/graphene (substrate) + CVD diamond (thermal)");
    era4["driver"]    = std::string("Post-Si transistors + AI chip heat management");
    era4["disco_role"]= std::string("Sapphire/SiC substrate dicing + diamond laser scribe");

    era5["era"]       = std::string("2040+: Quantum + Neuromorphic");
    era5["key_material"] = std::string("Diamond (spin qubit) + phase-change materials");
    era5["driver"]    = std::string("Quantum computing, brain-inspired chips");
    era5["disco_role"]= std::string("Ultra-precision femtosecond laser dicing (speculative)");

    r["era1_si_cmos"]     = era1;
    r["era2_wbg_power"]   = era2;
    r["era3_uwbg_photon"] = era3;
    r["era4_2d_diamond"]  = era4;
    r["era5_quantum"]     = era5;
    r["summary"] = std::string(
        "Each era opens new dicing challenges. DISCO's 50+ year track record = "
        "accumulated know-how for each new material. 'Cut anything precisely' is the moat.");
    return r;
}

PYBIND11_MODULE(_twod_substrate_kernel, m) {
    m.doc() = "2D materials (graphene/MoS2/hBN) properties + sapphire dicing + semiconductor roadmap";

    m.def("twod_material_properties", &twod_material_properties,
          py::arg("material") = "MoS2",
          "2D material properties: graphene, MoS2, hBN, WS2.\n"
          "Returns: thickness, bandgap, mobility, application, substrate, DISCO relevance.");

    m.def("sapphire_properties", &sapphire_properties,
          "Sapphire (Al2O3) substrate: E=400GPa, H=18GPa, K_IC=2.2, Mohs=9.\n"
          "Primary substrate for GaN-on-Sapphire LED and 2D material growth.");

    m.def("sapphire_dicing", &sapphire_dicing,
          py::arg("wafer_diam_mm") = 150.0,
          py::arg("thickness_um")  = 430.0,
          py::arg("method")        = "laser",
          py::arg("blade_grit")    = 2000.0,
          py::arg("feed_mm_s")     = 5.0,
          "Blade vs laser dicing model for sapphire substrate.\n"
          "Laser: r-plane scribing + breaking (DISCO DFL7560). Blade: chip/Ra/life.");

    m.def("future_semiconductor_roadmap", &future_semiconductor_roadmap,
          "Post-silicon material roadmap: Si → WBG → UWBG → 2D/Diamond → Quantum.\n"
          "Each era's key material, driver, and DISCO role.");
}
