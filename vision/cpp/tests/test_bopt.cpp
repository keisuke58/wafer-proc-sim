// test_bopt.cpp — GP prediction sanity + Bayesian-optimization convergence.
#include <cmath>
#include <iostream>
#include <vector>

#include "wproc/bopt.hpp"

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

// The GP interpolates its training data and is uncertain away from it.
static void test_gp_interpolates() {
    bopt::Kernel k; k.lengthscale = 1.0; k.signal_var = 1.0; k.noise = 1e-8;
    bopt::GP gp(k);
    std::vector<bopt::Point> X = {{0.0}, {1.0}, {2.0}, {3.0}};
    std::vector<double> Y = {0.0, 0.84, 0.91, 0.14};   // ~sin
    CHECK(gp.fit(X, Y));
    for (std::size_t i = 0; i < X.size(); ++i) {
        const bopt::Posterior p = gp.predict(X[i]);
        CHECK(std::fabs(p.mean - Y[i]) < 1e-3);   // near-exact at training pts
        CHECK(p.variance < 1e-3);                 // confident at training pts
    }
    const bopt::Posterior far = gp.predict({10.0});
    CHECK(far.variance > 0.5);                    // uncertain far away
}

// EI is positive where improvement is likely, zero for a certain no-improvement.
static void test_acquisition() {
    bopt::Posterior promising; promising.mean = -1.0; promising.stddev = 1.0;
    promising.variance = 1.0;
    CHECK(bopt::expected_improvement(promising, 0.0) > 0.0);

    bopt::Posterior certain_bad; certain_bad.mean = 5.0; certain_bad.stddev = 0.0;
    CHECK(bopt::expected_improvement(certain_bad, 0.0) == 0.0);

    // LCB prefers low mean / high uncertainty.
    bopt::Posterior a; a.mean = 1.0; a.stddev = 0.1;
    bopt::Posterior b; b.mean = 1.0; b.stddev = 2.0;
    CHECK(bopt::lcb_acquisition(b) > bopt::lcb_acquisition(a));
}

// A 2-D "recipe" objective (chipping vs feed & load): a smooth bowl with the
// minimum at (0.3, 0.7). BO should find it in far fewer evals than a grid.
static double recipe(const bopt::Point& x) {
    const double a = x[0] - 0.3, b = x[1] - 0.7;
    return 2.0 * a * a + 3.0 * b * b + 0.03 * std::sin(6.0 * x[0]);
}

static void test_bo_converges() {
    bopt::Bounds bnd; bnd.lo = {0.0, 0.0}; bnd.hi = {1.0, 1.0};
    bopt::Config cfg;
    cfg.n_init = 6; cfg.n_iter = 30; cfg.candidates = 400;
    cfg.kernel.lengthscale = 0.25; cfg.kernel.signal_var = 1.0; cfg.kernel.noise = 1e-6;
    const bopt::Result r = bopt::minimize(recipe, bnd, cfg);
    std::cout << "BO best=(" << r.best_x[0] << "," << r.best_x[1] << ") y="
              << r.best_y << " evals=" << r.evaluations << "\n";
    CHECK(r.ok);
    CHECK(std::fabs(r.best_x[0] - 0.3) < 0.12);
    CHECK(std::fabs(r.best_x[1] - 0.7) < 0.12);
    CHECK(r.best_y < 0.05);
    // best_so_far is monotone non-increasing.
    for (std::size_t i = 1; i < r.history.size(); ++i)
        CHECK(r.history[i].best_so_far <= r.history[i - 1].best_so_far + 1e-12);
}

int main() {
    test_gp_interpolates();
    test_acquisition();
    test_bo_converges();
    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "BOPT OK\n";
    return 0;
}
