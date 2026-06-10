// diamond_kernel.cpp — Diamond semiconductor: properties, CVD limits, laser scribing
//
// AP試験 / DISCO engineer concepts:
//   - ダイヤモンド = 炭素(C)のsp3同素体
//     熱伝導率 2000 W/mK = SiCの4倍・Siの13倍
//     バンドギャップ 5.5 eV = 究極の電力半導体ポテンシャル
//     硬度: モース10 (全物質中最硬) → ブレードで切れない
//
//   - なぜ普及しないか:
//     CVD (化学気相成長) で作れるが大口径単結晶が困難
//     2024年時点: 直径10mm以下が限界 (Siの300mmと比較にならない)
//     HPHT (高温高圧法): さらに小さい (<5mm, 宝石用)
//
//   - ダイシング: ブレード不可 → レーザーのみ
//     ① UV ピコ秒レーザー: 波長355nm (アブレーション)
//     ② IR フェムト秒レーザー: 内部改質 → ステルスライク
//     ③ {111} 劈開面に沿ったスクライブ + ブレーキング
//
//   - DISCOとの関係:
//     現状: 極小市場 (試験・研究用)
//     将来: ① AIチップの放熱基板としてのCVDダイヤモンド薄板
//           ② ダイヤモンドSBD (ショットキーダイオード) 量産 2030年代?
//           → レーザーダイシング技術 (DFL7560系) が唯一の選択肢
//
// DISCO context:
//   - ブレードダイシングは原理的に不可（ブレード自体がダイヤモンド砥石だが
//     ウェーハ切断には使えない — 砥粒と素材が同じ硬度で研磨不能）
//   - DFLシリーズのUVレーザーが唯一の現実解

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <string>
#include <vector>

namespace py = pybind11;

// ── Diamond semiconductor properties ─────────────────────────────────────────
static py::dict diamond_properties() {
    py::dict r;
    // Crystal
    r["crystal_structure"]         = std::string("Diamond cubic (Fd-3m)");
    r["bond_type"]                 = std::string("sp3 covalent");
    // Mechanical
    r["E_GPa"]                     = 1050.0;  // [100] direction
    r["E_isotropic_GPa"]           = 1050.0;
    r["nu"]                        = 0.07;    // nearly incompressible
    r["H_GPa"]                     = 98.0;    // Vickers ~10000 kg/mm2
    r["K_IC_MPa_sqrtm"]            = 6.5;     // moderate — not as brittle as InP
    r["cleavage_planes"]           = std::string("{111} — 4 perfect cleavage planes");
    r["mohs_hardness"]             = 10.0;
    // Thermal
    r["thermal_conductivity_W_mK"] = 2000.0;
    r["CTE_ppm_K"]                 = 1.0;
    // Electronic
    r["bandgap_eV"]                = 5.5;
    r["bandgap_type"]              = std::string("Indirect (direct at ~7.3 eV)");
    r["breakdown_field_MV_cm"]     = 10.0;
    r["electron_mobility_cm2_Vs"]  = 4500.0;
    r["hole_mobility_cm2_Vs"]      = 3800.0;  // unusually high holes
    r["dielectric_constant"]       = 5.7;
    // Manufacturing
    r["max_wafer_mm_2024"]         = 10.0;    // CVD polycrystalline up to 150mm; single crystal ~10mm
    r["growth_method"]             = std::string("CVD (polycrystal), HPHT (single crystal <5mm)");
    r["single_crystal_wafer_mm"]   = 10.0;
    r["cost_estimate"]             = std::string("~$10,000/cm2 for single crystal — 1000x SiC");
    // Dicing
    r["blade_dicing"]              = std::string("IMPOSSIBLE — same hardness as blade abrasive");
    r["recommended_method"]        = std::string("UV picosecond laser (355nm) or IR femtosecond stealth");
    r["disco_tool"]                = std::string("DFL7560 UV laser; future: femtosecond system");
    return r;
}

// ── CVD wafer growth model ────────────────────────────────────────────────────
// Rough cost/time model for CVD diamond growth
// Growth rate ~1 µm/h for single crystal, ~10 µm/h for polycrystalline
static py::dict cvd_wafer_limits(double target_diameter_mm, double target_thickness_um) {
    bool is_single_crystal = (target_diameter_mm <= 15.0);
    double growth_rate_um_h = is_single_crystal ? 1.0 : 8.0;
    double growth_time_h    = target_thickness_um / growth_rate_um_h;

    // Cost model: base cost + area factor
    double area_cm2   = M_PI * (target_diameter_mm / 20.0) * (target_diameter_mm / 20.0);
    double cost_usd   = is_single_crystal
                      ? area_cm2 * 10000.0       // $10k/cm² single crystal
                      : area_cm2 * 200.0 + 500.0; // $200/cm² polycrystalline

    std::string feasibility = (target_diameter_mm <= 10.0 && is_single_crystal) ? "Feasible (2024 frontier)" :
                              (target_diameter_mm <= 50.0 && !is_single_crystal) ? "Feasible (polycrystalline only)" :
                              (target_diameter_mm <= 150.0 && !is_single_crystal) ? "Limited — R&D stage" :
                              "Not feasible currently";

    py::dict r;
    r["target_diameter_mm"]   = target_diameter_mm;
    r["target_thickness_um"]  = target_thickness_um;
    r["crystal_type"]         = is_single_crystal ? std::string("Single crystal") : std::string("Polycrystalline");
    r["growth_rate_um_h"]     = growth_rate_um_h;
    r["growth_time_h"]        = growth_time_h;
    r["cost_usd_estimate"]    = cost_usd;
    r["feasibility_2024"]     = feasibility;
    r["note"] = std::string(
        "Single crystal limited to ~10mm — wafer-size bottleneck. "
        "Polycrystalline 150mm possible but not suitable for active devices.");
    return r;
}

// ── Laser scribing model for diamond ─────────────────────────────────────────
// Diamond is transparent at lambda > 225 nm → UV laser (355 nm) needed for ablation
// OR IR femtosecond laser for internal modification (stealth-like)
static py::dict laser_scribe_diamond(
    double laser_power_mW,
    double wavelength_nm,   // 355 = UV, 1064 = IR femtosecond
    double pulse_width_ps,  // picosecond (0.1–10)
    double feed_mm_s,
    double wafer_thickness_um
) {
    bool is_uv = (wavelength_nm < 400.0);
    bool is_femto = (wavelength_nm > 900.0 && pulse_width_ps < 0.01);

    // Ablation threshold for diamond: ~0.5 J/cm² for UV, ~1.5 J/cm² for IR
    double threshold_J_cm2 = is_uv ? 0.5 : 1.5;

    // Beam area (focus ~2µm UV, ~3µm IR)
    double beam_um  = is_uv ? 2.0 : 3.0;
    double beam_cm2 = M_PI * (beam_um * 1e-4 / 2.0) * (beam_um * 1e-4 / 2.0);

    // Pulse energy at repetition rate ~ feed_speed / beam_diameter
    // Material removal per pulse
    double fluence_J_cm2 = (laser_power_mW * 1e-3 * pulse_width_ps * 1e-12)
                         / beam_cm2;
    fluence_J_cm2 = std::max(fluence_J_cm2, laser_power_mW * 1e-3 / (feed_mm_s * 1e-3 * beam_cm2 * 1e4));

    bool ablates = (fluence_J_cm2 > threshold_J_cm2);
    double kerf_um = ablates ? (is_uv ? 5.0 : 8.0) : 0.0;
    double passes  = ablates ? std::ceil(wafer_thickness_um / (is_uv ? 2.0 : 5.0)) : -1;

    std::string quality = is_uv && ablates ? "GOOD (UV ablation)" :
                          is_femto        ? "EXCELLENT (internal modification, clean edge)" :
                          ablates         ? "MODERATE" : "INSUFFICIENT — increase power";

    py::dict r;
    r["wavelength_nm"]        = wavelength_nm;
    r["is_uv"]                = is_uv;
    r["pulse_width_ps"]       = pulse_width_ps;
    r["fluence_J_cm2"]        = fluence_J_cm2;
    r["threshold_J_cm2"]      = threshold_J_cm2;
    r["ablates"]              = ablates;
    r["kerf_um"]              = kerf_um;
    r["passes_required"]      = passes;
    r["quality"]              = quality;
    r["disco_relevance"] = std::string(
        ablates ? "DFL7560 UV system applicable; multi-pass needed for thick wafers"
                : "Increase power or switch to UV");
    return r;
}

// ── Thermal resistance comparison ─────────────────────────────────────────────
// θ_chip = thickness / (k * area)  [K/W]
// Compare substrate options for the same GaN power chip
static py::dict thermal_comparison(double power_W, double chip_area_mm2, double substrate_thickness_um) {
    struct Sub { std::string name; double k; };  // thermal conductivity [W/mK]
    Sub subs[] = {
        {"Diamond (CVD)", 1500.0},
        {"SiC",            490.0},
        {"AlN",            285.0},
        {"GaN-on-Si",       50.0},  // effective (Si dominated)
        {"Cu heat slug",   400.0},
    };

    double area_m2 = chip_area_mm2 * 1e-6;
    double h_m     = substrate_thickness_um * 1e-6;

    py::list results;
    for (auto& s : subs) {
        double theta = h_m / (s.k * area_m2);  // K/W
        double delta_T = theta * power_W;
        py::dict d;
        d["substrate"]        = s.name;
        d["k_W_mK"]           = s.k;
        d["theta_K_W"]        = theta;
        d["junction_dT_K"]    = delta_T;
        d["feasible_100W"]    = (delta_T < 50.0);
        results.append(d);
    }

    py::dict r;
    r["power_W"]            = power_W;
    r["chip_area_mm2"]      = chip_area_mm2;
    r["substrate_thickness_um"] = substrate_thickness_um;
    r["results"]            = results;
    r["conclusion"] = std::string(
        "Diamond 4x better than SiC; enables GaN power chips at 100W+ without liquid cooling");
    return r;
}

PYBIND11_MODULE(_diamond_kernel, m) {
    m.doc() = "Diamond semiconductor: properties, CVD wafer limits, laser scribing, thermal comparison";

    m.def("diamond_properties", &diamond_properties,
          "Full diamond semiconductor properties (mechanical, thermal, electronic, dicing).");

    m.def("cvd_wafer_limits", &cvd_wafer_limits,
          py::arg("target_diameter_mm")  = 10.0,
          py::arg("target_thickness_um") = 500.0,
          "CVD diamond growth model: time, cost, feasibility by diameter.\n"
          "Single crystal <10mm; polycrystalline up to 150mm (not for active devices).");

    m.def("laser_scribe_diamond", &laser_scribe_diamond,
          py::arg("laser_power_mW")   = 500.0,
          py::arg("wavelength_nm")    = 355.0,
          py::arg("pulse_width_ps")   = 10.0,
          py::arg("feed_mm_s")        = 1.0,
          py::arg("wafer_thickness_um") = 300.0,
          "UV (355nm) or IR femtosecond laser scribing for diamond.\n"
          "Returns: fluence vs threshold, kerf, passes, quality, DISCO relevance.");

    m.def("thermal_comparison", &thermal_comparison,
          py::arg("power_W")              = 50.0,
          py::arg("chip_area_mm2")        = 1.0,
          py::arg("substrate_thickness_um") = 300.0,
          "Thermal resistance comparison: Diamond / SiC / AlN / GaN-on-Si / Cu.\n"
          "Returns junction temperature rise per substrate for given power.");
}
