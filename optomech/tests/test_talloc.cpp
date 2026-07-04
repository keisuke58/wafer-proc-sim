// test_talloc.cpp — cost-optimal tolerance allocation.
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "optomech/optics.hpp"
#include "optomech/talloc.hpp"

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
    const double budget = 20.0;
    talloc::Result r = talloc::allocate(sys, budget);
    std::cout << "opt cost=" << r.opt_cost << " equal cost=" << r.equal_cost
              << " savings=" << r.savings_pct << "% tightest: e"
              << r.allocation[0].element << " " << r.allocation[0].kind
              << " sigma=" << r.allocation[0].sigma << "\n";
    CHECK(r.ok);
    // Optimized allocation is never costlier than equal tolerances.
    CHECK(r.opt_cost <= r.equal_cost + 1e-9);
    CHECK(r.savings_pct > 0.0);         // and strictly cheaper here
    // Allocation is sorted tightest-first.
    for (std::size_t i = 1; i < r.allocation.size(); ++i)
        CHECK(r.allocation[i].sigma >= r.allocation[i - 1].sigma - 1e-9);
    // The allocation exactly meets the image budget: 2*sum (S*sigma)^2 = B^2.
    const double rss2 = std::accumulate(r.allocation.begin(), r.allocation.end(), 0.0,
        [](double acc, const talloc::Alloc& a) { return acc + std::pow(a.sensitivity * a.sigma, 2); });
    const double image_rss = std::sqrt(2.0 * rss2);
    std::cout << "image RSS at allocation = " << image_rss << " µm\n";
    CHECK(std::fabs(image_rss - budget) < 0.05);
    // The most sensitive source gets the tightest tolerance.
    CHECK(r.allocation.front().sensitivity >= r.allocation.back().sensitivity);

    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) { std::cerr << g_failed << " FAILED\n"; return 1; }
    std::cout << "TALLOC OK\n";
    return 0;
}
