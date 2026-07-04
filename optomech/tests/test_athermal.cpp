// test_athermal.cpp — athermalization + random-vibration (Miles).
#include <cmath>
#include <iostream>

#include "optomech/athermal.hpp"
#include "optomech/fea.hpp"

using namespace optm;

static int g_total = 0, g_failed = 0;
#define CHECK(cond)                                                     \
    do { ++g_total; if (!(cond)) { ++g_failed;                          \
        std::cerr << "FAIL [" << __LINE__ << "]: " #cond "\n"; } } while (0)

int main() {
    fea::Barrel b;  // Al, 30/2/40 mm, 8 g lens
    athermal::Result r = athermal::analyze(b);
    std::cout << "uncomp=" << r.uncomp_drift_um_per_C << " comp_len=" << r.comp_len_mm
              << "mm comp_drift=" << r.comp_drift_um_per_C << " f1=" << r.first_mode_hz
              << " Grms=" << r.grms_response << " 3sig=" << r.accel_3sigma_g
              << "g stress=" << r.rand_stress_MPa << "MPa margin=" << r.fatigue_margin << "\n";
    // Uncompensated drift = CTE * L (0.944 µm/°C for Al, 40 mm).
    CHECK(std::fabs(r.uncomp_drift_um_per_C - 0.944) < 0.01);
    // Athermalization drives the net drift near zero, at a realizable length.
    CHECK(r.comp_len_mm > 0.0);
    CHECK(r.comp_drift_um_per_C < 0.1);
    CHECK(r.comp_drift_um_per_C < r.uncomp_drift_um_per_C);   // big improvement
    // Random vibration: positive response and a fatigue margin.
    CHECK(r.first_mode_hz > 1000.0);
    CHECK(r.grms_response > 0.0 && r.accel_3sigma_g > r.grms_response);
    CHECK(r.rand_stress_MPa > 0.0);
    CHECK(r.fatigue_margin > 0.0);
    CHECK(r.ok);   // athermalized + adequate fatigue margin

    // A stiffer (thicker) barrel raises the mode and changes the vib response.
    fea::Barrel stiff = b; stiff.wall_mm = 3.5;
    CHECK(athermal::analyze(stiff).first_mode_hz > r.first_mode_hz);

    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) { std::cerr << g_failed << " FAILED\n"; return 1; }
    std::cout << "ATHERMAL OK\n";
    return 0;
}
