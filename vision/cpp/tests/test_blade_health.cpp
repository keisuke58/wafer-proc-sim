// test_blade_health.cpp — SPC run rules + blade-wear trend / RUL prediction.
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "wproc/blade_health.hpp"
#include "wproc/spc.hpp"

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
#define CHECK_NEAR(a, b, tol)                                                  \
    do {                                                                       \
        ++g_total;                                                             \
        double _d = std::fabs(double(a) - double(b));                          \
        if (_d > (tol)) {                                                      \
            ++g_failed;                                                        \
            std::cerr << "FAIL [" << __LINE__ << "]: |" << (a) << " - " << (b) \
                      << "| = " << _d << " > " << (tol) << "\n";               \
        }                                                                      \
    } while (0)

static void test_spc_rules() {
    spc::Chart chart({/*mean=*/40.0, /*sigma=*/0.5});

    // In-control points near the mean: no violations.
    auto p = chart.add(40.1);
    CHECK(p.in_control);
    p = chart.add(39.9);
    CHECK(p.in_control);

    // A single point beyond 3-sigma (>41.5) trips WE1.
    p = chart.add(42.0);
    bool we1 = false;
    for (auto r : p.violations) we1 |= (r == spc::Rule::Beyond3Sigma);
    CHECK(we1);
    CHECK(!p.in_control);

    // Two of three beyond 2-sigma (>41.0), same side, trips WE2.
    spc::Chart c2({40.0, 0.5});
    c2.add(40.0);           // z = 0
    c2.add(41.1);           // z = 2.2
    spc::Point w2 = c2.add(41.2);  // z = 2.4 -> 2 of last 3 beyond 2 sigma
    bool we2 = false;
    for (auto r : w2.violations) we2 |= (r == spc::Rule::TwoOfThree2Sigma);
    CHECK(we2);

    // Four of five beyond 1-sigma (>40.5), same side, trips WE3.
    spc::Chart c3({40.0, 0.5});
    spc::Point w3;
    for (int i = 0; i < 5; ++i) w3 = c3.add(40.6);  // z = 1.2 each
    bool we3 = false;
    for (auto r : w3.violations) we3 |= (r == spc::Rule::FourOfFive1Sigma);
    CHECK(we3);

    // Eight consecutive above the mean trips WE4.
    spc::Chart c4({40.0, 0.5});
    spc::Point q;
    for (int i = 0; i < 8; ++i) q = c4.add(40.2);  // all slightly high
    bool we4 = false;
    for (auto r : q.violations) we4 |= (r == spc::Rule::EightOneSide);
    CHECK(we4);

    // A degenerate sigma must not silently pass everything (NaN-guard).
    spc::Chart degenerate({40.0, 0.0});
    CHECK(degenerate.limits().sigma > 0.0);
}

static void test_wear_prediction() {
    // Baseline: fresh blade, width ~40 um, tight spread.
    std::mt19937 gen(3);
    std::normal_distribution<double> noise(0.0, 0.15);
    std::vector<double> baseline;
    for (int i = 0; i < 30; ++i) baseline.push_back(40.0 + noise(gen));

    blade::Spec spec;  // nominal 40, USL 45, chip limit 5
    blade::Monitor mon(spec, baseline);

    // Degradation: width drifts +0.01 um/cut, chipping +0.004 um/cut.
    const double wslope = 0.01, cslope = 0.004;
    bool flagged = false;
    long flag_cut = -1;
    for (long cut = 1; cut <= 300; ++cut) {
        double w = 40.0 + wslope * cut + noise(gen);
        double c = 1.0 + cslope * cut + std::fabs(noise(gen));
        blade::Sample s = mon.add(cut, w, c);
        if (!s.in_control && !flagged) {
            flagged = true;
            flag_cut = cut;
        }
    }
    CHECK(flagged);  // SPC eventually catches the drift
    std::cout << "SPC first flagged at cut " << flag_cut << "\n";

    blade::Health h = mon.health();
    CHECK(h.width_slope_um_per_cut > 0.0);
    CHECK_NEAR(h.width_slope_um_per_cut, wslope, 0.003);  // recovered slope
    CHECK(h.wear_index > 0.0);
    // Width reaches USL(45) at cut ~500; from cut 300 that's ~200 cuts out.
    CHECK(h.rul_cuts > 100.0 && h.rul_cuts < 320.0);
    std::cout << "wear_index=" << h.wear_index << " RUL=" << h.rul_cuts
              << " cuts, slope=" << h.width_slope_um_per_cut << " um/cut\n";

    // A stable blade (no drift) should not raise a persistent alarm or a
    // near-term replacement.
    blade::Monitor stable(spec, baseline);
    for (long cut = 1; cut <= 200; ++cut)
        stable.add(cut, 40.0 + noise(gen), 1.0 + std::fabs(noise(gen)));
    blade::Health hs = stable.health();
    CHECK(!hs.replace_now);
    CHECK(hs.rul_cuts > 1000.0);
    std::cout << "stable RUL=" << hs.rul_cuts << "\n";
}

int main() {
    test_spc_rules();
    test_wear_prediction();
    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "BLADE HEALTH OK\n";
    return 0;
}
