// test_grind_control.cpp — infeed APC holds thickness; wheel RUL; economics.
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "wproc/grind_control.hpp"

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

static double rms_err(const std::vector<double>& v, double target) {
    double s = std::accumulate(
        v.begin(), v.end(), 0.0,
        [target](double a, double x) { return a + (x - target) * (x - target); });
    return std::sqrt(s / v.size());
}

static bool throws_invalid(const grind::GrindParams& p) {
    try {
        grind::InfeedController c(p);
        (void)c;
        return false;
    } catch (const std::invalid_argument&) {
        return true;
    }
}

static void test_validation() {
    grind::GrindParams bad_gain;
    bad_gain.gain = 0.0;
    CHECK(throws_invalid(bad_gain));

    grind::GrindParams bad_lambda_hi;
    bad_lambda_hi.lambda = 1.5;
    CHECK(throws_invalid(bad_lambda_hi));

    grind::GrindParams bad_lambda_lo;
    bad_lambda_lo.lambda = 0.0;
    CHECK(throws_invalid(bad_lambda_lo));

    grind::GrindParams bad_bounds;
    bad_bounds.infeed_min = 5.0;
    bad_bounds.infeed_max = 1.0;
    CHECK(throws_invalid(bad_bounds));
}

static void test_closed_loop() {
    grind::GrindPlant plant;
    plant.offset0_um = 780.0;
    plant.gain_um_per_infeed = 100.0;
    plant.dull_um_per_wafer = 0.15;  // wheel dulls: same infeed removes less
    plant.noise_sigma = 0.4;

    const double target = 100.0;      // target final thickness
    const double nominal = (plant.offset0_um - target) / plant.gain_um_per_infeed;  // 6.8
    const int N = 300;
    std::mt19937 gen(7);
    std::normal_distribution<double> noise(0.0, plant.noise_sigma);

    // Open loop: fixed nominal infeed. Dulling drives thickness off target.
    std::vector<double> open_t;
    for (long k = 1; k <= N; ++k)
        open_t.push_back(plant.final_thickness(nominal, k, noise(gen)));
    const double open_rms = rms_err(open_t, target);

    // Closed loop: R2R infeed controller compensates the dulling drift.
    grind::GrindParams p;
    p.target_um = target;
    p.gain = 100.0;
    p.lambda = 0.3;
    p.infeed0 = nominal;
    grind::InfeedController ctrl(p);

    std::vector<double> cl_t;
    double infeed = ctrl.current_infeed();
    for (long k = 1; k <= N; ++k) {
        const double t = plant.final_thickness(infeed, k, noise(gen));
        cl_t.push_back(t);
        infeed = ctrl.update(t);
    }
    const double cl_rms = rms_err(cl_t, target);

    std::cout << "grind open-loop RMS=" << open_rms << " um, closed-loop RMS="
              << cl_rms << " um, final infeed=" << ctrl.current_infeed() << "\n";

    CHECK(open_rms > 5.0);            // uncontrolled drift is large
    CHECK(cl_rms < 1.0);             // controller holds target tightly
    CHECK(cl_rms < 0.2 * open_rms);  // big improvement
    // Steady-state tail hugs target.
    const double tail = rms_err({cl_t.end() - 50, cl_t.end()}, target);
    CHECK(tail < 0.8);
    // Compensating dulling raises infeed above the fresh-wheel nominal.
    CHECK(ctrl.current_infeed() > p.infeed0);
    CHECK(ctrl.current_infeed() < p.infeed_max);
}

static void test_wheel_health() {
    grind::GrindPlant plant;
    plant.force_gain = 2.0;
    plant.force_dull_per_wafer = 0.05;  // force climbs as the wheel dulls

    grind::WheelMonitor mon;
    const double infeed = 6.8;
    for (long k = 1; k <= 200; ++k) {
        const double removed = plant.gain_um_per_infeed * infeed;  // ~680 um
        mon.add(k, removed, plant.force(infeed, k));
    }
    const double dress_limit = 30.0;  // force at 200: 2*6.8 + 0.05*200 = 23.6
    grind::WheelHealth h = mon.health(dress_limit);
    std::cout << "wheel force=" << h.current_force << " slope="
              << h.force_slope_per_wafer << " RUL=" << h.rul_wafers
              << " G-ratio=" << h.g_ratio << "\n";

    CHECK(h.n == 200);
    CHECK(std::fabs(h.force_slope_per_wafer - 0.05) < 0.01);  // recovered slope
    CHECK(!h.dress_now);
    // Force at limit 30 from ~23.6 at slope 0.05 => ~128 wafers out.
    CHECK(h.rul_wafers > 60.0 && h.rul_wafers < 200.0);
    CHECK(h.g_ratio > 0.0);
    CHECK(h.total_removed_um > 0.0);

    // A blade already over the limit must flag dress-now with zero RUL.
    grind::WheelHealth over = mon.health(10.0);
    CHECK(over.dress_now);
    CHECK(over.rul_wafers == 0.0);
}

static void test_economics() {
    grind::EconInputs in;
    in.wafers_per_hour = 20.0;
    in.dies_per_wafer = 1000.0;
    in.yield_pct = 95.0;
    in.machine_cost_per_hour = 120.0;
    in.wheel_cost = 800.0;
    in.wafers_per_wheel = 2000.0;
    in.consumables_per_wafer = 1.5;

    grind::EconSummary s = grind::economics(in);
    std::cout << "cost/wafer=$" << s.cost_per_wafer << " cost/good-die=$"
              << s.cost_per_good_die << " good dies/hr=" << s.good_dies_per_hour
              << "\n";

    // machine 120/20=6, wheel 800/2000=0.4, consumables 1.5 => 7.9 /wafer.
    CHECK(std::fabs(s.cost_per_wafer - 7.9) < 1e-6);
    // good dies/wafer = 1000*0.95 = 950 => cost/die = 7.9/950.
    CHECK(std::fabs(s.cost_per_good_die - 7.9 / 950.0) < 1e-9);
    CHECK(std::fabs(s.good_dies_per_hour - 20.0 * 950.0) < 1e-6);
}

int main() {
    test_validation();
    test_closed_loop();
    test_wheel_health();
    test_economics();
    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "GRIND CONTROL OK\n";
    return 0;
}
