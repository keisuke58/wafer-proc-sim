// test_housing.cpp — material selection + rib stiffness-to-weight.
#include <cmath>
#include <iostream>

#include "optomech/housing.hpp"

using namespace optm;

static int g_total = 0, g_failed = 0;
#define CHECK(cond)                                                     \
    do { ++g_total; if (!(cond)) { ++g_failed;                          \
        std::cerr << "FAIL [" << __LINE__ << "]: " #cond "\n"; } } while (0)

int main() {
    housing::Section solid; solid.outer_dia_mm = 30; solid.wall_mm = 2.5;
    housing::Section ribbed; ribbed.outer_dia_mm = 30; ribbed.wall_mm = 1.4;
    ribbed.n_ribs = 6; ribbed.rib_h_mm = 2.5; ribbed.rib_w_mm = 1.5;

    housing::Result r = housing::analyze(housing::default_materials(), solid, ribbed);
    std::cout << "best material=" << r.best_material << " ashby="
              << r.ranking.front().ashby_index << "\n";
    std::cout << "solid I/A=" << r.solid.stiffness_per_mass << " ribbed I/A="
              << r.ribbed.stiffness_per_mass << " gain=" << r.stiffness_to_weight_gain_pct << "%\n";
    CHECK(!r.ranking.empty());
    // Ranking sorted by Ashby index descending; CFRP/Mg beat Al beat steel.
    for (std::size_t i = 1; i < r.ranking.size(); ++i)
        CHECK(r.ranking[i].ashby_index <= r.ranking[i - 1].ashby_index + 1e-12);
    CHECK(r.best_material == "CFRP (quasi-iso)");   // highest E^(1/2)/rho
    // Steel is stiff but heavy -> it should NOT top the light-stiff index.
    CHECK(r.ranking.back().name == "Polycarbonate" || r.ranking.back().name == "Stainless 304");
    // Ribs raise stiffness-per-gram vs a solid thicker wall.
    CHECK(r.ribbed.stiffness_per_mass > r.solid.stiffness_per_mass);
    CHECK(r.stiffness_to_weight_gain_pct > 0.0);
    // Section properties are physical.
    CHECK(r.solid.area_mm2 > 0.0 && r.solid.inertia_mm4 > 0.0);

    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) { std::cerr << g_failed << " FAILED\n"; return 1; }
    std::cout << "HOUSING OK\n";
    return 0;
}
