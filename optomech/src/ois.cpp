// ois.cpp — optical image stabilization simulation.
#include "optomech/ois.hpp"

#include <cmath>
#include <numeric>
#include <random>

namespace optm::ois {
namespace {
constexpr double kTwoPi = 6.283185307179586;

double rms(const std::vector<double>& v, std::size_t skip) {
    if (skip >= v.size()) return 0.0;
    const double ss = std::accumulate(v.begin() + skip, v.end(), 0.0,
        [](double a, double x) { return a + x * x; });
    return std::sqrt(ss / (v.size() - skip));
}
}  // namespace

Result simulate(double focal_mm, const Plant& plant, unsigned seed) {
    Result r;
    r.focal_mm = focal_mm;
    if (plant.dt_s <= 0.0 || plant.duration_s <= 0.0) return r;
    const double dt = plant.dt_s;
    const int steps = static_cast<int>(plant.duration_s / dt) + 1;

    // Multi-tone hand tremor as an angular-rate spectrum (rad/s); amplitudes fall
    // with frequency. Total angular excursion ~0.2-0.3 deg, typical of handheld.
    struct Tone { double f_hz, amp_rad; double phase; };
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> ph(0.0, kTwoPi);
    const double freqs[5] = {1.2, 3.0, 6.0, 9.0, 13.0};
    const double amps_deg[5] = {0.15, 0.10, 0.06, 0.035, 0.02};  // per-tone angle
    Tone tones[5];
    for (int i = 0; i < 5; ++i)
        tones[i] = {freqs[i], amps_deg[i] * kTwoPi / 360.0, ph(gen)};

    std::normal_distribution<double> gnoise(0.0, plant.gyro_noise_urad * 1e-6);

    // Second-order actuator: x'' + 2 z wn x' + wn^2 x = wn^2 * cmd.
    const double wn = kTwoPi * plant.act_freq_hz;
    const double zt = plant.damping;
    double x = 0.0, xd = 0.0;

    r.t_s.reserve(static_cast<std::size_t>(steps));
    r.demand_um.reserve(static_cast<std::size_t>(steps));
    r.residual_um.reserve(static_cast<std::size_t>(steps));

    for (int i = 0; i < steps; ++i) {
        const double t = i * dt;
        // Camera angle theta(t) = sum of per-tone angular displacements.
        const double theta = std::accumulate(tones, tones + 5, 0.0,
            [t](double s, const Tone& tn) {
                return s + tn.amp_rad * std::sin(kTwoPi * tn.f_hz * t + tn.phase);
            });
        // Image demand at the sensor (small angle): d = f * theta.
        const double demand_mm = focal_mm * theta;

        // Gyro-measured angle (with noise) drives the cancellation command.
        const double theta_meas = theta + gnoise(gen);
        const double cmd_mm = -focal_mm * theta_meas;   // desired actuator shift

        // Integrate the actuator toward the command.
        const double xdd = wn * wn * (cmd_mm - x) - 2.0 * zt * wn * xd;
        xd += xdd * dt;
        x += xd * dt;

        const double residual_mm = demand_mm + x;  // x cancels demand when tracking
        r.t_s.push_back(t);
        r.demand_um.push_back(demand_mm * 1e3);
        r.residual_um.push_back(residual_mm * 1e3);
    }

    // Skip the first 10% (actuator start-up transient) for the RMS statistics.
    const std::size_t skip = r.t_s.size() / 10;
    r.off_rms_um = rms(r.demand_um, skip);
    r.on_rms_um = rms(r.residual_um, skip);
    r.ratio = r.on_rms_um > 1e-9 ? r.off_rms_um / r.on_rms_um : 0.0;
    r.stops = r.ratio > 0.0 ? std::log2(r.ratio) : 0.0;
    return r;
}

}  // namespace optm::ois
