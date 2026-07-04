// test_tolerance.cpp — paraxial optics + tolerance stack-up.
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "optomech/optics.hpp"
#include "optomech/tolerance.hpp"

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

static void test_optics() {
    // Single thin lens f=50: EFL=50, BFD=50.
    LensSystem s1; s1.elements = {{50.0, 50.0, 10.0}};
    FirstOrder f1 = first_order(s1);
    std::cout << "single lens efl=" << f1.efl_mm << " bfd=" << f1.bfd_mm << "\n";
    CHECK(std::fabs(f1.efl_mm - 50.0) < 1e-6);
    CHECK(std::fabs(f1.bfd_mm - 50.0) < 1e-6);

    // Two f=50 lenses 10 mm apart: 1/f = 1/50 + 1/50 - 10/2500 => f = 27.78 mm.
    LensSystem s2; s2.elements = {{50.0, 10.0, 10.0}, {50.0, 22.22, 10.0}};
    FirstOrder f2 = first_order(s2);
    std::cout << "doublet efl=" << f2.efl_mm << "\n";
    CHECK(std::fabs(f2.efl_mm - 27.778) < 0.05);
}

// A plausible 4-element camera lens (EFL ~ 35 mm).
static LensSystem make_lens() {
    LensSystem s;
    s.elements = {
        {60.0, 6.0, 12.0},    // front positive
        {-40.0, 4.0, 10.0},   // negative
        {35.0, 8.0, 9.0},     // positive
        {80.0, 18.0, 8.0},    // rear + back distance to sensor
    };
    return s;
}

static void test_stackup() {
    const LensSystem sys = make_lens();
    std::vector<tol::Tolerance> tols = {
        {5.0, 1.0, 10.0}, {8.0, 1.5, 10.0}, {5.0, 1.0, 8.0}, {4.0, 0.8, 8.0}};
    // Budget = one 3.9 µm sensor pixel.
    tol::Result r = tol::analyze(sys, tols, /*budget_um=*/3.9, /*n=*/40000);
    std::cout << "lateral RSS=" << r.lateral_rss_um << "µm MC RMS="
              << r.lateral_mc_rms_um << "µm P99=" << r.lateral_mc_p99_um
              << "µm focus RSS=" << r.focus_rss_um << "µm ok=" << r.ok << "\n";
    std::cout << "critical: elem " << r.ranking[0].element << " " << r.ranking[0].source
              << " (" << r.ranking[0].variance_pct << "% var)\n";
    CHECK(r.efl_mm > 20.0 && r.efl_mm < 60.0);
    CHECK(r.lateral_rss_um > 0.0);
    // Analytic RSS and Monte Carlo RMS of a sum of Gaussians should agree
    // closely (the MC RMS of the 2-D magnitude ~ RSS within ~5%).
    CHECK(std::fabs(r.lateral_mc_rms_um - r.lateral_rss_um) / r.lateral_rss_um < 0.08);
    // Ranking is sorted by sigma and its variance shares sum to ~100%.
    for (std::size_t i = 1; i < r.ranking.size(); ++i)
        CHECK(r.ranking[i].sigma_um <= r.ranking[i - 1].sigma_um + 1e-9);
    const double sum_pct = std::accumulate(r.ranking.begin(), r.ranking.end(), 0.0,
        [](double a, const tol::Contribution& c) { return a + c.variance_pct; });
    CHECK(std::fabs(sum_pct - 100.0) < 1.0);
    CHECK(r.mc_lateral_um.size() == 40000);

    // Tightening the dominant tolerance to zero reduces the budget.
    const auto crit = r.ranking[0];
    std::vector<tol::Tolerance> tightened = tols;
    if (crit.source == "decenter") tightened[crit.element].decenter_um = 0.0;
    else if (crit.source == "tilt") tightened[crit.element].tilt_mrad = 0.0;
    tol::Result r2 = tol::analyze(sys, tightened, 3.9, 40000);
    CHECK(r2.lateral_rss_um < r.lateral_rss_um);

    // n_samples <= 0: analytic-only path, no Monte Carlo, no crash.
    tol::Result r0 = tol::analyze(sys, tols, 20.0, /*n=*/0);
    CHECK(r0.mc_lateral_um.empty());
    CHECK(r0.lateral_rss_um > 0.0);
    CHECK(r0.ok == (r0.lateral_rss_um <= 20.0));
}

int main() {
    test_optics();
    test_stackup();
    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) { std::cerr << g_failed << " FAILED\n"; return 1; }
    std::cout << "TOLERANCE OK\n";
    return 0;
}
