// cam.cpp — cam-barrel kinematics + VCM autofocus control.
#include "optomech/cam.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace optm::cam {
namespace {
constexpr double kTwoPi = 6.283185307179586;
}

std::vector<CamSample> cam_curve(const CamProfile& p, int n) {
    std::vector<CamSample> out;
    if (n < 2 || p.sweep_deg <= 0.0) return out;
    out.reserve(static_cast<std::size_t>(n));
    const double S = p.stroke_mm, sw = p.sweep_deg;
    for (int i = 0; i < n; ++i) {
        const double phi = static_cast<double>(i) / (n - 1);  // 0..1
        CamSample s;
        s.theta_deg = phi * sw;
        // Cycloidal motion law: smooth (zero end velocity/acceleration).
        s.z_mm = S * (phi - std::sin(kTwoPi * phi) / kTwoPi);
        s.vel = (S / sw) * (1.0 - std::cos(kTwoPi * phi));
        s.accel = (S / (sw * sw)) * kTwoPi * std::sin(kTwoPi * phi);
        out.push_back(s);
    }
    return out;
}

double peak_accel(const CamProfile& p, int n) {
    const auto curve = cam_curve(p, n);
    return std::accumulate(curve.begin(), curve.end(), 0.0,
        [](double m, const CamSample& s) { return std::max(m, std::fabs(s.accel)); });
}

StepResult step_response(const Vcm& v, const Pid& pid, double target_um,
                         double t_end_ms, double dt_ms) {
    StepResult r;
    const double m = v.mass_g * 1e-3;         // kg
    const double k = v.flexure_n_per_mm * 1e3;// N/m
    const double c = v.damping * 1e3;         // N/(m/s)
    const double dt = dt_ms * 1e-3;           // s
    const double tgt_m = target_um * 1e-6;    // m

    double x = 0.0, xd = 0.0, integ = 0.0, prev_e = target_um;
    const int steps = static_cast<int>(t_end_ms / dt_ms) + 1;
    r.t_ms.reserve(static_cast<std::size_t>(steps));
    r.pos_um.reserve(static_cast<std::size_t>(steps));
    r.target_um.reserve(static_cast<std::size_t>(steps));

    double peak = 0.0;
    double last_out_of_band_ms = 0.0;
    const double band = 0.02 * std::fabs(target_um);  // 2% settling band
    const double ff_cmd = pid.feedforward ? (k * tgt_m) / v.force_max_n : 0.0;

    for (int i = 0; i < steps; ++i) {
        const double t_ms = i * dt_ms;
        const double e_um = (tgt_m - x) * 1e6;
        const double de = (e_um - prev_e) / dt;
        prev_e = e_um;
        // PID command, plus a static feedforward bias (as a fraction of full
        // force) that cancels the spring load at the target.
        double u = pid.kp * e_um + pid.ki * integ + pid.kd * de + ff_cmd;
        // Coil force saturates; only integrate while unsaturated (anti-windup).
        double u_sat = std::clamp(u, -1.0, 1.0);
        if (std::fabs(u) <= 1.0) integ += e_um * dt;
        const double force = u_sat * v.force_max_n;
        const double xdd = (force - k * x - c * xd) / m;
        xd += xdd * dt;
        x += xd * dt;

        const double pos_um = x * 1e6;
        r.t_ms.push_back(t_ms);
        r.pos_um.push_back(pos_um);
        r.target_um.push_back(target_um);
        peak = std::max(peak, pos_um);
        if (std::fabs(pos_um - target_um) > band) last_out_of_band_ms = t_ms;
    }

    r.settling_ms = last_out_of_band_ms;
    r.overshoot_pct = target_um > 0.0 && peak > target_um
                          ? 100.0 * (peak - target_um) / target_um : 0.0;
    r.ss_error_um = std::fabs(r.pos_um.back() - target_um);
    return r;
}

}  // namespace optm::cam
