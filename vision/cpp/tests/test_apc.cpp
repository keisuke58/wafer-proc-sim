// test_apc.cpp — closed-loop kerf regulation rejects blade-wear drift.
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "wproc/apc.hpp"

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

// RMS deviation of a series from target.
static double rms_err(const std::vector<double>& w, double target) {
    double s = std::accumulate(
        w.begin(), w.end(), 0.0,
        [target](double a, double x) { return a + (x - target) * (x - target); });
    return std::sqrt(s / w.size());
}

// The controller must reject a degenerate model up front.
static bool throws_invalid(const apc::R2RParams& p) {
    try {
        apc::R2RController c(p);
        (void)c;
        return false;
    } catch (const std::invalid_argument&) {
        return true;
    }
}

static void test_validation() {
    apc::R2RParams bad_gain;
    bad_gain.gain = 0.0;
    CHECK(throws_invalid(bad_gain));

    apc::R2RParams bad_lambda_hi;
    bad_lambda_hi.lambda = 1.5;
    CHECK(throws_invalid(bad_lambda_hi));

    apc::R2RParams bad_lambda_lo;
    bad_lambda_lo.lambda = 0.0;
    CHECK(throws_invalid(bad_lambda_lo));
}

int main() {
    test_validation();

    apc::DicingPlant plant;
    plant.offset0_um = 32.5;      // width = 32.5 + 5*feed + drift*k
    plant.gain = 5.0;             // feed 1.5 -> 40 um at k=0
    plant.drift_um_per_cut = 0.02;  // wear widens the kerf over time
    plant.noise_sigma = 0.1;

    const double target = 40.0;
    const double nominal_feed = 1.5;
    const int N = 250;
    std::mt19937 gen(11);
    std::normal_distribution<double> noise(0.0, plant.noise_sigma);

    // Open loop: fixed nominal feed. Drift carries the width off target.
    std::vector<double> open_w;
    for (long k = 1; k <= N; ++k)
        open_w.push_back(plant.width(nominal_feed, k, noise(gen)));
    double open_rms = rms_err(open_w, target);

    // Closed loop: R2R EWMA controller adjusts feed each cut. Compensating the
    // 5 um total drift lowers feed from 1.5 to ~0.5 — within the actuator range.
    apc::R2RParams p;
    p.target_um = target;
    p.gain = 5.0;
    p.lambda = 0.3;
    p.feed0 = nominal_feed;
    apc::R2RController ctrl(p);

    std::vector<double> cl_w;
    double feed = ctrl.current_feed();
    for (long k = 1; k <= N; ++k) {
        double w = plant.width(feed, k, noise(gen));
        cl_w.push_back(w);
        feed = ctrl.update(w);
    }
    double cl_rms = rms_err(cl_w, target);

    std::cout << "open-loop RMS=" << open_rms << " um, closed-loop RMS="
              << cl_rms << " um, final feed=" << ctrl.current_feed() << "\n";

    CHECK(open_rms > 2.0);           // uncontrolled drift is large
    CHECK(cl_rms < 0.4);             // controller holds target tightly
    CHECK(cl_rms < 0.2 * open_rms);  // big improvement
    // Steady-state: last 50 cuts hug the target.
    double tail = rms_err({cl_w.end() - 50, cl_w.end()}, target);
    CHECK(tail < 0.3);
    // Feed was pushed down to compensate the widening drift (no saturation).
    CHECK(ctrl.current_feed() < p.feed0);
    CHECK(ctrl.current_feed() > p.feed_min);

    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "APC OK\n";
    return 0;
}
