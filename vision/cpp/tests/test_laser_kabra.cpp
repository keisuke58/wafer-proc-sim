// test_laser_kabra.cpp — SiC ingot slice yield, laser-vs-saw economics, layer QC.
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "wproc/laser_kabra.hpp"

using namespace wproc;

static int g_total = 0, g_failed = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        ++g_total;                                                      \
        if (!(cond)) {                                                  \
            ++g_failed;                                                 \
            std::cerr << "FAIL [" << __LINE__ << "]: " #cond "\n";      \
        }                                                               \
    } while (0)

static void test_slice_yield() {
    // 30 mm ingot, 350 um wafers, 100 um laser loss ⇒ pitch 450 um ⇒ 66 wafers.
    laser::SliceInputs in;
    in.ingot_length_mm = 30.0;
    in.wafer_thickness_um = 350.0;
    in.loss_per_slice_um = 100.0;
    laser::SliceYield y = laser::slice_yield(in);
    std::cout << "laser wafers=" << y.wafers << " util=" << y.utilization_pct << "%\n";
    CHECK(y.wafers == 66);            // floor(30000 / 450)
    CHECK(y.pitch_um == 450.0);
    CHECK(y.utilization_pct > 76.0 && y.utilization_pct < 78.0);  // 66*350/30000
}

static void test_comparison_and_econ() {
    // Laser 100 um loss vs diamond-wire 250 um kerf, same 350 um wafers.
    laser::SliceComparison cmp = laser::compare_slice(30.0, 350.0, 100.0, 250.0);
    std::cout << "laser=" << cmp.laser.wafers << " saw=" << cmp.saw.wafers
              << " extra=" << cmp.extra_wafers << " (" << cmp.extra_wafers_pct
              << "%)\n";
    CHECK(cmp.laser.wafers == 66);
    CHECK(cmp.saw.wafers == 50);      // floor(30000 / 600)
    CHECK(cmp.extra_wafers == 16);
    CHECK(cmp.extra_wafers_pct > 30.0);      // 16/50 = 32%
    CHECK(cmp.material_saved_pct > 8.0);     // utilisation gain in points

    // Ingot $5000; laser processing pricier per wafer but far more wafers.
    laser::SliceEcon e = laser::slice_economics(cmp, 5000.0, 8.0, 5.0);
    std::cout << "cost/wafer laser=$" << e.cost_per_wafer_laser << " saw=$"
              << e.cost_per_wafer_saw << " save=" << e.savings_pct << "%\n";
    // laser 5000/66 + 8 = 83.8 ; saw 5000/50 + 5 = 105 ; laser cheaper.
    CHECK(e.cost_per_wafer_laser < e.cost_per_wafer_saw);
    CHECK(e.savings_per_wafer > 15.0);
    CHECK(e.savings_pct > 15.0);
}

static void test_separation_layer() {
    // Uniform, on-target separation depth ⇒ clean cleave (ok).
    std::mt19937 gen(4);
    std::normal_distribution<double> noise(0.0, 0.5);
    std::vector<double> good;
    for (int i = 0; i < 200; ++i) good.push_back(120.0 + noise(gen));
    laser::SepLayerStats gs = laser::analyze_separation(good, 120.0, /*tol=*/3.0);
    std::cout << "sep good: mean=" << gs.mean_depth_um << " sigma=" << gs.sigma_um
              << " unif=" << gs.uniformity_pct << "% ok=" << gs.ok << "\n";
    CHECK(gs.ok);
    CHECK(gs.uniformity_pct > 98.0);
    CHECK(gs.max_dev_um < 3.0);

    // A wavy / off-target layer ⇒ risk of a bad break (not ok).
    std::vector<double> bad;
    for (int i = 0; i < 200; ++i)
        bad.push_back(120.0 + 6.0 * std::sin(i * 0.1) + noise(gen));
    laser::SepLayerStats bs = laser::analyze_separation(bad, 120.0, 3.0);
    std::cout << "sep bad: max_dev=" << bs.max_dev_um << " ok=" << bs.ok << "\n";
    CHECK(!bs.ok);
    CHECK(bs.max_dev_um > 3.0);
}

int main() {
    test_slice_yield();
    test_comparison_and_econ();
    test_separation_layer();
    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "LASER KABRA OK\n";
    return 0;
}
