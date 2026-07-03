// test_laser_control.cpp — pulse-energy APC holds ablation depth; optics RUL.
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "wproc/laser_control.hpp"

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

static bool throws_invalid(const laser::PulseParams& p) {
    try {
        laser::PulseEnergyController c(p);
        (void)c;
        return false;
    } catch (const std::invalid_argument&) {
        return true;
    }
}

static void test_validation() {
    laser::PulseParams bad_gain;
    bad_gain.gain = 0.0;
    CHECK(throws_invalid(bad_gain));

    laser::PulseParams bad_lambda;
    bad_lambda.lambda = 1.5;
    CHECK(throws_invalid(bad_lambda));

    laser::PulseParams bad_lambda_lo;
    bad_lambda_lo.lambda = 0.0;
    CHECK(throws_invalid(bad_lambda_lo));

    laser::PulseParams bad_bounds;
    bad_bounds.energy_min = 30.0;
    bad_bounds.energy_max = 5.0;
    CHECK(throws_invalid(bad_bounds));
}

static void test_closed_loop() {
    laser::AblationPlant plant;
    plant.depth_gain = 1.0;      // depth = energy*passes/speed - drift*k
    plant.passes = 1.0;
    plant.speed = 1.0;
    plant.drift_um_per_shot = 0.02;  // optics dull: same energy removes less
    plant.noise_sigma = 0.15;

    const double target = 20.0;
    const double nominal = target;  // gain=1, passes/speed=1 -> energy≈depth at k=0
    const int N = 300;
    std::mt19937 gen(9);
    std::normal_distribution<double> noise(0.0, plant.noise_sigma);

    // Open loop: fixed energy. Optics drift drives depth off target.
    std::vector<double> open_d;
    for (long k = 1; k <= N; ++k)
        open_d.push_back(plant.depth(nominal, k, noise(gen)));
    const double open_rms = rms_err(open_d, target);

    // Closed loop: R2R pulse-energy controller compensates the drift.
    laser::PulseParams p;
    p.target_um = target;
    p.gain = 1.0;
    p.lambda = 0.3;
    p.energy0 = nominal;
    laser::PulseEnergyController ctrl(p);

    std::vector<double> cl_d;
    double energy = ctrl.current_energy();
    for (long k = 1; k <= N; ++k) {
        const double d = plant.depth(energy, k, noise(gen));
        cl_d.push_back(d);
        energy = ctrl.update(d);
    }
    const double cl_rms = rms_err(cl_d, target);

    std::cout << "laser open-loop RMS=" << open_rms << " um, closed-loop RMS="
              << cl_rms << " um, final energy=" << ctrl.current_energy() << "\n";

    CHECK(open_rms > 2.0);
    CHECK(cl_rms < 0.4);
    CHECK(cl_rms < 0.2 * open_rms);
    const double tail = rms_err({cl_d.end() - 50, cl_d.end()}, target);
    CHECK(tail < 0.35);
    // Compensating the drift pushes energy above the fresh-optics nominal.
    CHECK(ctrl.current_energy() > p.energy0);
    CHECK(ctrl.current_energy() < p.energy_max);
}

static void test_optics_health() {
    laser::AblationPlant plant;
    plant.power_gain = 1.0;
    plant.power_drop_per_shot = 0.01;  // transmitted power decays with shots

    laser::OpticsMonitor mon;
    const double energy = 20.0;  // power at k=0 ≈ 20 W
    for (long k = 1; k <= 400; ++k) mon.add(k, plant.power(energy, k));

    const double floor = 15.0;  // power at 400: 20 - 0.01*400 = 16 W (above floor)
    laser::OpticsHealth h = mon.health(floor);
    std::cout << "optics power=" << h.current_power << " slope="
              << h.power_slope_per_shot << " RUL=" << h.rul_shots << "\n";

    CHECK(h.n == 400);
    CHECK(std::fabs(h.power_slope_per_shot + 0.01) < 0.002);  // ~ -0.01
    CHECK(!h.service_now);
    // From ~16 W at slope -0.01, floor 15 is ~100 shots out.
    CHECK(h.rul_shots > 50.0 && h.rul_shots < 200.0);

    // Already below the floor ⇒ service now, zero RUL.
    laser::OpticsHealth over = mon.health(18.0);
    CHECK(over.service_now);
    CHECK(over.rul_shots == 0.0);
}

int main() {
    test_validation();
    test_closed_loop();
    test_optics_health();
    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "LASER CONTROL OK\n";
    return 0;
}
