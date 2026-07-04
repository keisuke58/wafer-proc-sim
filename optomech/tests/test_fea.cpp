// test_fea.cpp — barrel thermal / modal / drop-shock analysis.
#include <cmath>
#include <iostream>

#include "optomech/fea.hpp"

using namespace optm;

static int g_total = 0, g_failed = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        ++g_total;                                                      \
        if (!(cond)) {                                                  \
            ++g_failed;                                                 \
            std::cerr << "FAIL [" << __LINE__ << "]: " #cond "\n";      \
        }                                                               \
    } while (0)

static void test_baseline() {
    fea::Barrel b;  // Al, 30 mm OD, 2 mm wall, 40 mm, 8 g lens
    fea::Result r = fea::analyze(b, /*drop_m=*/1.0, /*shock_ms=*/0.5);
    std::cout << "thermal=" << r.thermal_focus_drift_um_per_C << "µm/°C f1="
              << r.first_mode_hz << "Hz shock=" << r.shock_g_peak << "g DAF="
              << r.daf << " stress=" << r.drop_shock_stress_MPa << "MPa SF="
              << r.safety_factor << "\n";
    // CTE * L: 23.6e-6 * 40 mm = 0.944 µm/°C.
    CHECK(std::fabs(r.thermal_focus_drift_um_per_C - 0.944) < 0.01);
    CHECK(r.first_mode_hz > 1000.0);          // short stiff tube
    CHECK(r.axial_stiffness_N_per_um > 0.0);
    CHECK(r.daf > 0.3 && r.daf < 2.0);        // physical SRS range
    CHECK(r.drop_shock_stress_MPa > 0.0);
    CHECK(r.safety_factor > 2.0 && r.ok);     // survives a 1 m drop
}

static void test_monotonic() {
    fea::Barrel stiff; stiff.wall_mm = 3.0;
    fea::Barrel thin;  thin.wall_mm = 1.0;
    fea::Result rs = fea::analyze(stiff);
    fea::Result rt = fea::analyze(thin);
    std::cout << "wall 3mm f1=" << rs.first_mode_hz << " SF=" << rs.safety_factor
              << " | wall 1mm f1=" << rt.first_mode_hz << " SF=" << rt.safety_factor << "\n";
    // A thinner wall => lower first mode and lower safety factor.
    CHECK(rt.first_mode_hz < rs.first_mode_hz);
    CHECK(rt.safety_factor < rs.safety_factor);

    // Hotter environment / larger CTE => more focus drift (polycarbonate mount).
    fea::Barrel poly; poly.mat.cte_per_C = 70e-6;
    CHECK(fea::analyze(poly).thermal_focus_drift_um_per_C >
          fea::analyze(fea::Barrel{}).thermal_focus_drift_um_per_C);
}

int main() {
    test_baseline();
    test_monotonic();
    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) { std::cerr << g_failed << " FAILED\n"; return 1; }
    std::cout << "FEA OK\n";
    return 0;
}
