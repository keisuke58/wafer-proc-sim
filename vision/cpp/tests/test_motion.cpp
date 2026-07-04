// test_motion.cpp — S-curve limits/endpoints + servo feedforward tracking.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "wproc/motion.hpp"

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

static double maxabs(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0,
        [](double m, double x) { return std::max(m, std::fabs(x)); });
}

static void check_profile(const motion::Trajectory& t, double D,
                          const motion::Limits& lim) {
    CHECK(!t.pos.empty());
    CHECK(std::fabs(t.pos.back() - D) < 1e-6);          // reaches target
    CHECK(std::fabs(t.vel.back()) < 1e-6);              // ends at rest
    CHECK(maxabs(t.vel) <= lim.vmax * 1.01 + 1e-9);     // velocity limit
    CHECK(maxabs(t.acc) <= lim.amax * 1.01 + 1e-9);     // accel limit
    // Position monotonically non-decreasing (no overshoot in the reference).
    bool mono = true;
    for (std::size_t i = 1; i < t.pos.size(); ++i)
        if (t.pos[i] + 1e-9 < t.pos[i - 1]) mono = false;
    CHECK(mono);
    // Jerk limit: bounded second difference of velocity.
    double jmax_obs = 0.0;
    for (std::size_t i = 2; i < t.vel.size(); ++i) {
        const double j = (t.vel[i] - 2 * t.vel[i - 1] + t.vel[i - 2]) / (t.dt * t.dt);
        jmax_obs = std::max(jmax_obs, std::fabs(j));
    }
    CHECK(jmax_obs <= lim.jmax * 1.5 + 1e-6);           // jerk roughly bounded
}

static void test_scurve() {
    motion::Limits lim; lim.vmax = 50.0; lim.amax = 500.0; lim.jmax = 5000.0;
    const double dt = 1e-4;

    // Long move: cruise phase reached.
    motion::Trajectory a = motion::scurve(20.0, lim, dt);
    std::cout << "long move: T=" << a.total_time << " vmax_obs=" << maxabs(a.vel) << "\n";
    check_profile(a, 20.0, lim);
    CHECK(maxabs(a.vel) > 0.9 * lim.vmax);   // actually cruises near vmax

    // Short move: peak velocity reduced below vmax.
    motion::Trajectory b = motion::scurve(0.5, lim, dt);
    std::cout << "short move: T=" << b.total_time << " vmax_obs=" << maxabs(b.vel) << "\n";
    check_profile(b, 0.5, lim);
    CHECK(maxabs(b.vel) < lim.vmax);         // never reaches vmax
}

static void test_servo_feedforward() {
    motion::Limits lim; lim.vmax = 40.0; lim.amax = 400.0; lim.jmax = 4000.0;
    motion::Trajectory ref = motion::scurve(10.0, lim, 1e-4);
    // Dwell at the target so the servo can settle after the move ends.
    for (int i = 0; i < 3000; ++i) {
        ref.pos.push_back(10.0); ref.vel.push_back(0.0); ref.acc.push_back(0.0);
    }

    motion::StagePlant plant; plant.mass = 1.0; plant.damping = 5.0;
    motion::ServoGains g; g.kp = 400.0; g.kd = 40.0; g.ki = 0.0;

    g.feedforward = true;
    motion::TrackResult with_ff = motion::track(ref, plant, g);
    g.feedforward = false;
    motion::TrackResult no_ff = motion::track(ref, plant, g);

    std::cout << "servo: FF max err=" << with_ff.max_error
              << " vs no-FF max err=" << no_ff.max_error << "\n";
    CHECK(with_ff.max_error < no_ff.max_error * 0.2);   // FF collapses lag error
    CHECK(with_ff.max_error < 1e-2);                    // tight tracking
    CHECK(with_ff.final_error < 1e-3);                  // settles on target
}

int main() {
    test_scurve();
    test_servo_feedforward();
    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "MOTION OK\n";
    return 0;
}
