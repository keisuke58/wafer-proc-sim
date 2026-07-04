// housing.cpp — material selection + rib stiffness-to-weight.
#include "optomech/housing.hpp"

#include <algorithm>
#include <cmath>

namespace optm::housing {
namespace {
constexpr double kPi = 3.14159265358979323846;
}

std::vector<Material> default_materials() {
    return {
        {"Al 6061", 68.9, 2700.0, 276.0},
        {"Mg AZ31", 45.0, 1770.0, 200.0},
        {"Ti-6Al-4V", 114.0, 4430.0, 880.0},
        {"CFRP (quasi-iso)", 70.0, 1600.0, 600.0},
        {"Polycarbonate", 2.4, 1200.0, 62.0},
        {"Glass-filled PC", 7.0, 1430.0, 90.0},
        {"Stainless 304", 193.0, 8000.0, 215.0},
    };
}

SectionProps section_props(const Section& s) {
    SectionProps p;
    const double Do = s.outer_dia_mm, Di = s.outer_dia_mm - 2.0 * s.wall_mm;
    if (Do <= 0.0 || Di < 0.0 || Di >= Do) return p;
    p.area_mm2 = kPi / 4.0 * (Do * Do - Di * Di);
    p.inertia_mm4 = kPi / 64.0 * (std::pow(Do, 4) - std::pow(Di, 4));
    // Axial ribs on the inner wall: rectangular, contributing via parallel axis.
    if (s.n_ribs > 0 && s.rib_h_mm > 0.0 && s.rib_w_mm > 0.0) {
        const double arm = Di / 2.0 - s.rib_h_mm / 2.0;  // rib centroid radius
        const double i_self = s.rib_w_mm * std::pow(s.rib_h_mm, 3) / 12.0;
        const double a_rib = s.rib_w_mm * s.rib_h_mm;
        // Average orientation factor 1/2 (ribs around the ring see bending on
        // both axes); use the mean contribution to the section inertia.
        const double i_rib = i_self + a_rib * arm * arm;
        p.area_mm2 += s.n_ribs * a_rib;
        p.inertia_mm4 += 0.5 * s.n_ribs * i_rib;
    }
    p.stiffness_per_mass = p.area_mm2 > 0.0 ? p.inertia_mm4 / p.area_mm2 : 0.0;
    return p;
}

Result analyze(const std::vector<Material>& materials, const Section& solid,
               const Section& ribbed) {
    Result r;
    for (const Material& m : materials) {
        if (m.density <= 0.0) continue;
        MatRank mr;
        mr.name = m.name;
        mr.ashby_index = std::sqrt(m.youngs_GPa * 1e9) / m.density;
        mr.specific_stiffness = m.youngs_GPa * 1e9 / m.density;
        mr.specific_strength = m.yield_MPa * 1e6 / m.density;
        r.ranking.push_back(mr);
    }
    std::sort(r.ranking.begin(), r.ranking.end(),
              [](const MatRank& a, const MatRank& b) { return a.ashby_index > b.ashby_index; });
    if (!r.ranking.empty()) r.best_material = r.ranking.front().name;

    r.solid = section_props(solid);
    r.ribbed = section_props(ribbed);
    r.stiffness_to_weight_gain_pct =
        r.solid.stiffness_per_mass > 0.0
            ? 100.0 * (r.ribbed.stiffness_per_mass - r.solid.stiffness_per_mass) /
                  r.solid.stiffness_per_mass
            : 0.0;
    return r;
}

}  // namespace optm::housing
