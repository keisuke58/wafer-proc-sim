// housing.hpp — barrel/housing stiffness-to-weight + material selection.
//
// The lens housing must be stiff and strong yet light and slim ("strength and
// design in one"). Two design levers are modelled:
//
//   * Material selection — for a light, stiff structure the figure of merit is
//     the Ashby beam index E^(1/2)/rho (maximize); this ranks candidate barrel
//     materials.
//   * Rib topology — a thin sleek wall braced by axial ribs beats a thick solid
//     wall in stiffness-per-gram; the module compares I/mass for a solid vs a
//     ribbed section.
//
// Pure C++17, no external dependencies.
#pragma once

#include <string>
#include <vector>

namespace optm::housing {

struct Material {
    std::string name;
    double youngs_GPa = 0.0;
    double density = 0.0;      // kg/m^3
    double yield_MPa = 0.0;
};

// Ranked material by the light-stiff-beam Ashby index.
struct MatRank {
    std::string name;
    double ashby_index = 0.0;      // E^(1/2)/rho  (higher = lighter for stiffness)
    double specific_stiffness = 0.0;// E/rho
    double specific_strength = 0.0; // yield/rho
};

// A tube cross-section, optionally braced by n_ribs axial ribs on the inside.
struct Section {
    double outer_dia_mm = 30.0;
    double wall_mm = 2.0;
    int n_ribs = 0;
    double rib_h_mm = 0.0;     // rib height (radially inward)
    double rib_w_mm = 0.0;     // rib width
};

struct SectionProps {
    double area_mm2 = 0.0;
    double inertia_mm4 = 0.0;      // second moment of area
    double stiffness_per_mass = 0.0;  // I / area (∝ bending stiffness per gram)
};

struct Result {
    std::vector<MatRank> ranking;  // best material first
    std::string best_material;
    SectionProps solid;
    SectionProps ribbed;
    double stiffness_to_weight_gain_pct = 0.0;  // ribbed vs solid
};

// Rank `materials` and compare a solid vs a `ribbed` section (same outer dia).
Result analyze(const std::vector<Material>& materials,
               const Section& solid, const Section& ribbed);

// The default candidate set (Al, Mg, Ti, CFRP, PC, glass-filled PC, steel).
std::vector<Material> default_materials();

// Cross-section properties of a (possibly ribbed) tube.
SectionProps section_props(const Section& s);

}  // namespace optm::housing
