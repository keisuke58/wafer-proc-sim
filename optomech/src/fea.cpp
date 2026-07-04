// fea.cpp — barrel structural analysis (thermal, modal, drop-shock).
#include "optomech/fea.hpp"

#include <algorithm>
#include <cmath>

namespace optm::fea {
namespace {
constexpr double kG = 9.80665;      // m/s^2
constexpr double kPi = 3.14159265358979323846;

// Maximax dynamic amplification factor of an undamped SDOF (natural freq f_hz)
// to a half-sine base-acceleration pulse of duration tau: integrate
// z'' + w^2 z = -a_base(t); the absolute response acceleration is -w^2 z, so
// DAF = w^2 * max|z| / a_peak.
double half_sine_daf(double f_hz, double tau_s) {
    if (f_hz <= 0.0 || tau_s <= 0.0) return 1.0;
    const double w = 2.0 * kPi * f_hz;
    const double dt = std::min(tau_s, 1.0 / f_hz) / 400.0;
    const double t_end = tau_s + 2.0 / f_hz;   // include free-vibration tail
    double z = 0.0, zd = 0.0, maxz = 0.0;
    for (double t = 0.0; t <= t_end; t += dt) {
        const double a_base = t < tau_s ? std::sin(kPi * t / tau_s) : 0.0;  // unit amp
        const double zdd = -w * w * z - a_base;
        zd += zdd * dt;
        z += zd * dt;
        maxz = std::max(maxz, std::fabs(z));
    }
    return w * w * maxz;  // a_peak is unit amplitude, so this is the DAF
}

}  // namespace

Result analyze(const Barrel& b, double drop_height_m, double shock_ms) {
    Result r;
    const double E = b.mat.youngs_GPa * 1e9;      // Pa
    const double Do = b.outer_dia_mm * 1e-3;      // m
    const double Di = (b.outer_dia_mm - 2.0 * b.wall_mm) * 1e-3;
    const double L = b.length_mm * 1e-3;
    if (Do <= 0.0 || Di < 0.0 || Di >= Do || L <= 0.0) return r;

    const double area = kPi / 4.0 * (Do * Do - Di * Di);           // m^2
    const double I = kPi / 64.0 * (std::pow(Do, 4) - std::pow(Di, 4)); // m^4
    const double c = Do / 2.0;                                     // extreme fibre

    // Thermal focus drift: axial expansion of the barrel length moves the image.
    r.thermal_focus_drift_um_per_C = b.mat.cte_per_C * b.length_mm * 1e3;  // µm/°C

    // Axial stiffness EA/L.
    const double k_axial = E * area / L;                 // N/m
    r.axial_stiffness_N_per_um = k_axial * 1e-6;

    // First bending mode: cantilever tube (tip lens mass + effective tube mass).
    const double k_bend = 3.0 * E * I / (L * L * L);     // N/m
    const double m_lens = b.lens_mass_g * 1e-3;          // kg
    const double m_tube = b.mat.density * area * L;      // kg
    const double m_eff = m_lens + 0.24 * m_tube;         // Rayleigh cantilever
    r.first_mode_hz = m_eff > 0.0 ? std::sqrt(k_bend / m_eff) / (2.0 * kPi) : 0.0;

    // Drop shock: impact velocity -> half-sine pulse -> peak accel -> SRS.
    const double v = std::sqrt(2.0 * kG * std::max(0.0, drop_height_m));
    const double tau = shock_ms * 1e-3;
    const double a_peak = tau > 0.0 ? kPi * v / (2.0 * tau) : 0.0;  // m/s^2
    r.shock_g_peak = a_peak / kG;
    r.daf = half_sine_daf(r.first_mode_hz, tau);

    // Worst case: lateral shock bends the cantilever (tip force = m*a*DAF).
    const double force = m_lens * a_peak * r.daf;        // N
    const double moment = force * L;                     // N·m
    const double stress = I > 0.0 ? moment * c / I : 0.0;// Pa
    r.drop_shock_stress_MPa = stress * 1e-6;
    r.safety_factor = r.drop_shock_stress_MPa > 0.0
                          ? b.mat.yield_MPa / r.drop_shock_stress_MPa : 0.0;
    r.ok = r.safety_factor >= 2.0;
    return r;
}

}  // namespace optm::fea
