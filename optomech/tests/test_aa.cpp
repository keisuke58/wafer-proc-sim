// test_aa.cpp — active alignment via Nelder-Mead.
#include <cmath>
#include <iostream>

#include "optomech/aa.hpp"
#include "optomech/optics.hpp"

using namespace optm;

static int g_total = 0, g_failed = 0;
#define CHECK(cond)                                                     \
    do { ++g_total; if (!(cond)) { ++g_failed;                          \
        std::cerr << "FAIL [" << __LINE__ << "]: " #cond "\n"; } } while (0)

static LensSystem make_lens() {
    LensSystem s;
    s.elements = {{60.0, 6.0, 12.0}, {-40.0, 4.0, 10.0}, {35.0, 8.0, 9.0}, {80.0, 18.0, 8.0}};
    return s;
}

int main() {
    const LensSystem sys = make_lens();
    aa::Assembly err;
    err.element = 2;
    err.decenter_x_um = 30.0; err.decenter_y_um = -20.0;
    err.tilt_mrad = 0.05; err.focus_um = 15.0;

    aa::Result r = aa::align(sys, err);
    std::cout << "AA merit " << r.merit_before_um << " -> " << r.merit_after_um
              << " µm in " << r.iterations << " iters; adj=("
              << r.adj_x_um << "," << r.adj_y_um << "," << r.adj_focus_um << ")\n";
    CHECK(r.converged);
    CHECK(r.merit_after_um < r.merit_before_um);
    // Active alignment recovers most of the error (>5x reduction here).
    CHECK(r.merit_after_um < r.merit_before_um / 5.0);
    // The applied decenter cancels the built-in decenter (x ~ -30, y ~ +20).
    CHECK(std::fabs(r.adj_x_um + 30.0) < 3.0);
    CHECK(std::fabs(r.adj_y_um - 20.0) < 3.0);
    // Merit history is monotone non-increasing.
    for (std::size_t i = 1; i < r.merit_history.size(); ++i)
        CHECK(r.merit_history[i] <= r.merit_history[i - 1] + 1e-9);
    // A perfectly-assembled element needs ~no correction.
    aa::Assembly clean; clean.element = 2;
    aa::Result rc = aa::align(sys, clean);
    CHECK(rc.merit_after_um < 1.0);

    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) { std::cerr << g_failed << " FAILED\n"; return 1; }
    std::cout << "AA OK\n";
    return 0;
}
