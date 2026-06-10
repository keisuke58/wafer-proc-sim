// ga2o3_kernel.cpp — beta-Ga2O3 dicing & material properties
//
// AP試験 / DISCO engineer concepts:
//   - β-Ga₂O₃（酸化ガリウム）: バンドギャップ 4.8eV > SiC(3.3eV) > Si(1.1eV)
//     → 理論的耐圧: SiCの4倍以上
//     → 製造コスト: 融液法（フローティングゾーン）でコスト < SiC (昇華法)
//
//   - 結晶構造: 単斜晶系 (monoclinic, β角 103.8°)
//     → 完全へき開面: (100)・(001) → ダイシング方向が制限される
//     → ストリートを劈開面に平行に設定するとウェーハが自然に割れる
//     → DISCOの解決策: ① ストリートを劈開面に45°で設定
//                       ② レーザーでスクライブ後にブレーキング
//                       ③ ステルスダイシング（改質線＋劈開利用）
//
//   - 機械特性:
//     E  ≈ 240 GPa (平均; c軸方向は 370 GPa)
//     K_IC ≈ 0.5 MPa√m on (100) [SiCの1/3!]
//     H  ≈ 11.5 GPa (モース硬度 ~6.5)
//     → 硬い × 脆い × 劈開性 という三重苦
//
//   - 市場: 電力変換・EV。ただし信頼性データ蓄積待ち → 量産初期 (2024-2028)
//
// DISCO context:
//   - 現状: 2〜4インチウェーハのみ → まだ市場小さい
//   - 将来: Ga₂O₃ウェーハが6インチ→8インチへ成長すると本格参入
//   - 劈開性の問題はDISCOの「精密レーザーダイシング技術」が強みを発揮する局面

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <string>
#include <vector>

namespace py = pybind11;

// ── β-Ga₂O₃ crystal properties ───────────────────────────────────────────────
static py::dict ga2o3_crystal_properties() {
    py::dict r;
    // Mechanical
    r["crystal_system"]      = std::string("Monoclinic (beta phase)");
    r["space_group"]         = std::string("C2/m (No.12)");
    r["beta_angle_deg"]      = 103.8;
    r["E_avg_GPa"]           = 240.0;
    r["E_c_axis_GPa"]        = 370.0;  // stiffer along c-axis
    r["nu"]                  = 0.26;
    r["H_GPa"]               = 11.5;
    r["K_IC_100_MPa_sqrtm"]  = 0.50;   // on (100) cleavage plane — weakest
    r["K_IC_avg_MPa_sqrtm"]  = 0.78;   // averaged over directions
    r["cleavage_planes"]     = std::string("(100) primary, (001) secondary");
    // Electronic
    r["bandgap_eV"]          = 4.8;
    r["breakdown_field_MV_cm"] = 8.0;  // theoretical; Si=0.3, SiC=3.5
    r["electron_mobility_cm2_Vs"] = 300.0;  // vs SiC ~1000 (lower!)
    r["thermal_conductivity_W_mK"] = 11.0;  // vs SiC 490 (much lower!)
    // Wafer status
    r["max_wafer_inch_2024"] = 4.0;
    r["growth_method"]       = std::string("Floating zone / EFG (edge-fed growth)");
    r["cost_vs_sic"]         = std::string("Lower (melt growth vs. sublimation)");
    // DISCO relevance
    r["dicing_challenge"]    = std::string("Perfect cleavage on (100)/(001) — must control cut direction");
    r["recommended_method"]  = std::string("Laser scribing + breaking, or stealth dicing along cleavage");
    return r;
}

// ── Cleavage risk model ───────────────────────────────────────────────────────
// Street angle relative to (100) cleavage plane:
//   angle = 0°  → cut parallel to cleavage → maximum spontaneous fracture risk
//   angle = 90° → cut perpendicular → controlled fracture possible
// Risk: R(θ) = cos²(θ) × (K_IC_100 / K_IC_avg)² × exp(-blade_speed / 5.0)
static py::dict cleavage_risk(
    double street_angle_deg,  // angle of dicing street w.r.t. (100) plane
    double blade_depth_um,    // cut depth [µm]
    double feed_mm_s          // blade feed speed [mm/s]
) {
    double theta = street_angle_deg * M_PI / 180.0;
    double K_ratio = 0.50 / 0.78;  // K_IC(100) / K_IC(avg)

    // Risk = geometric alignment × fracture toughness ratio × speed factor
    double alignment_factor = std::cos(theta) * std::cos(theta);
    double speed_factor     = std::exp(-feed_mm_s / 5.0);  // slower = more risk
    double depth_factor     = (blade_depth_um > 200.0) ? 1.3 : 1.0;

    double risk = alignment_factor * K_ratio * K_ratio * speed_factor * depth_factor;
    // Normalize to 0-1
    risk = std::min(risk, 1.0);

    // Optimal angle: ~45° gives balanced risk
    double optimal_angle  = 45.0;
    double risk_at_45     = std::cos(45.0 * M_PI / 180.0) * std::cos(45.0 * M_PI / 180.0)
                          * K_ratio * K_ratio;

    std::string grade = (risk > 0.4) ? "HIGH — cleavage fracture likely" :
                        (risk > 0.2) ? "MODERATE — precision required" : "LOW";

    py::dict r;
    r["street_angle_deg"]    = street_angle_deg;
    r["cleavage_risk_score"] = risk;
    r["risk_grade"]          = grade;
    r["alignment_factor"]    = alignment_factor;
    r["optimal_angle_deg"]   = optimal_angle;
    r["risk_at_optimal"]     = risk_at_45;
    r["recommendation"] = (risk > 0.35)
        ? std::string("Rotate street angle toward 45 deg; use laser scribing")
        : std::string("Proceed; monitor for edge chipping");
    return r;
}

// ── Ga₂O₃ vs SiC dicing performance ─────────────────────────────────────────
// Compare expected chip size and kerf quality under the same blade conditions
static py::dict ga2o3_vs_sic_dicing(
    double blade_grit,       // e.g. 2000
    double feed_mm_s,        // e.g. 5.0
    double street_angle_deg  // relative to Ga₂O₃ cleavage plane
) {
    // Lawn-Evans chipping: h_chip ∝ (E/H)^0.5 * F^0.5 / K_IC
    // Normalize to Si=1.0 baseline
    struct MatChip {
        double E, H, K_IC;
        std::string name;
    };
    MatChip mats[3] = {
        {130.0, 10.0, 0.8, "Si"},
        {450.0, 26.0, 2.0, "SiC"},
        {240.0, 11.5, 0.5, "Ga2O3"},
    };

    // F_N ∝ hardness × grit_area; grit_area ∝ 1/grit_number
    double F_norm = 10.0 / blade_grit;  // relative cutting force

    py::list comparison;
    double si_chip = 0.0;
    for (int i = 0; i < 3; ++i) {
        auto& m = mats[i];
        double chip_um = 2.5 * std::sqrt(m.E / m.H) * std::sqrt(F_norm) / m.K_IC;
        if (m.name == "Si") si_chip = chip_um;
        py::dict d;
        d["material"]  = m.name;
        d["E_GPa"]     = m.E;
        d["H_GPa"]     = m.H;
        d["K_IC"]      = m.K_IC;
        d["chip_um_estimated"] = chip_um;
        d["chip_ratio_vs_Si"]  = (si_chip > 0) ? chip_um / si_chip : 1.0;
        comparison.append(d);
    }

    // Ga₂O₃ extra penalty from cleavage if angle is bad
    double cleavage_penalty = 1.0;
    double theta = street_angle_deg * M_PI / 180.0;
    cleavage_penalty = 1.0 + 0.5 * std::cos(theta) * std::cos(theta);

    py::dict r;
    r["comparison"]         = comparison;
    r["blade_grit"]         = blade_grit;
    r["feed_mm_s"]          = feed_mm_s;
    r["street_angle_deg"]   = street_angle_deg;
    r["ga2o3_cleavage_penalty"] = cleavage_penalty;
    r["key_insight"] = std::string(
        "Ga2O3 has LOWER K_IC than SiC (0.5 vs 2.0) — easier to chip, "
        "but cleavage anisotropy requires angle-controlled dicing");
    return r;
}

// ── Material benchmark: power semiconductor comparison ───────────────────────
static py::dict material_benchmark() {
    py::dict r;

    // Baliga figure of merit: BFM ∝ μ * ε * E_breakdown³
    // (relative to Si = 1)
    py::dict si, sic, gan, ga2o3, diamond;

    si["bandgap_eV"]       = 1.1;
    si["breakdown_MV_cm"]  = 0.3;
    si["mobility_cm2_Vs"]  = 1400.0;
    si["thermal_W_mK"]     = 150.0;
    si["K_IC"]             = 0.8;
    si["BFM_vs_Si"]        = 1.0;
    si["wafer_inch"]       = 12.0;
    si["maturity"]         = std::string("Production");

    sic["bandgap_eV"]      = 3.3;
    sic["breakdown_MV_cm"] = 3.5;
    sic["mobility_cm2_Vs"] = 1000.0;
    sic["thermal_W_mK"]    = 490.0;
    sic["K_IC"]            = 2.0;
    sic["BFM_vs_Si"]       = 400.0;
    sic["wafer_inch"]      = 8.0;
    sic["maturity"]        = std::string("Mass production");

    gan["bandgap_eV"]      = 3.4;
    gan["breakdown_MV_cm"] = 3.3;
    gan["mobility_cm2_Vs"] = 1500.0;
    gan["thermal_W_mK"]    = 230.0;
    gan["K_IC"]            = 0.9;
    gan["BFM_vs_Si"]       = 650.0;
    gan["wafer_inch"]      = 8.0;
    gan["maturity"]        = std::string("Mass production (on-Si) / ramp");

    ga2o3["bandgap_eV"]      = 4.8;
    ga2o3["breakdown_MV_cm"] = 8.0;
    ga2o3["mobility_cm2_Vs"] = 300.0;
    ga2o3["thermal_W_mK"]    = 11.0;
    ga2o3["K_IC"]            = 0.5;
    ga2o3["BFM_vs_Si"]       = 3000.0;  // theoretical upper bound
    ga2o3["wafer_inch"]      = 4.0;
    ga2o3["maturity"]        = std::string("Early production 2024");

    diamond["bandgap_eV"]      = 5.5;
    diamond["breakdown_MV_cm"] = 10.0;
    diamond["mobility_cm2_Vs"] = 4500.0;
    diamond["thermal_W_mK"]    = 2000.0;
    diamond["K_IC"]            = 7.0;
    diamond["BFM_vs_Si"]       = 100000.0;  // theoretical
    diamond["wafer_inch"]      = 0.3;  // ~8mm today
    diamond["maturity"]        = std::string("Research — wafer size bottleneck");

    r["Si"]      = si;
    r["SiC"]     = sic;
    r["GaN"]     = gan;
    r["Ga2O3"]   = ga2o3;
    r["Diamond"] = diamond;
    r["disco_priority"] = std::string(
        "1. SiC (NOW) 2. GaN (NOW) 3. Ga2O3 (2026+) 4. Diamond (2030+)");
    r["ga2o3_weakness"]  = std::string(
        "Low thermal conductivity (11 W/mK) and K_IC=0.5 — heat management critical");
    return r;
}

PYBIND11_MODULE(_ga2o3_kernel, m) {
    m.doc() = "beta-Ga2O3: crystal properties, cleavage risk, dicing vs SiC, power semiconductor benchmark";

    m.def("ga2o3_crystal_properties", &ga2o3_crystal_properties,
          "beta-Ga2O3 mechanical + electronic properties.\n"
          "Returns: E, K_IC, H, band gap, cleavage planes, growth method, DISCO context.");

    m.def("cleavage_risk", &cleavage_risk,
          py::arg("street_angle_deg") = 0.0,
          py::arg("blade_depth_um")   = 100.0,
          py::arg("feed_mm_s")        = 5.0,
          "Cleavage fracture risk for given street angle w.r.t. (100) plane.\n"
          "Returns: risk score 0-1, grade, optimal angle, recommendation.");

    m.def("ga2o3_vs_sic_dicing", &ga2o3_vs_sic_dicing,
          py::arg("blade_grit")        = 2000.0,
          py::arg("feed_mm_s")         = 5.0,
          py::arg("street_angle_deg")  = 45.0,
          "Chipping comparison: Si / SiC / Ga2O3 under same blade conditions.\n"
          "Lawn-Evans model. Returns per-material chip size + cleavage penalty.");

    m.def("material_benchmark", &material_benchmark,
          "Power semiconductor benchmark: Si, SiC, GaN, Ga2O3, Diamond.\n"
          "Returns: band gap, breakdown field, BFM vs Si, K_IC, wafer size, maturity.");
}
