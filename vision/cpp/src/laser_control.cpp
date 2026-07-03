// laser_control.cpp — laser ablation APC + optics health.
#include "wproc/laser_control.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace wproc::laser {

// ── PulseEnergyController ────────────────────────────────────────────────────

PulseEnergyController::PulseEnergyController(PulseParams p)
    : p_(p), energy_(p.energy0) {
    if (p_.gain == 0.0)
        throw std::invalid_argument("PulseEnergyController: gain must be non-zero");
    if (!(p_.lambda > 0.0 && p_.lambda <= 1.0))
        throw std::invalid_argument("PulseEnergyController: lambda in (0,1]");
    if (p_.energy_min > p_.energy_max)
        throw std::invalid_argument("PulseEnergyController: energy_min > energy_max");
    // Offset consistent with the initial energy hitting target:
    // depth = offset + gain*energy  =>  offset = target - gain*energy0.
    offset_hat_ = p_.target_um - p_.gain * p_.energy0;
}

double PulseEnergyController::update(double measured_depth_um) {
    // Invert the model to estimate the current offset, then EWMA-smooth it.
    const double offset_meas = measured_depth_um - p_.gain * energy_;
    offset_hat_ = p_.lambda * offset_meas + (1.0 - p_.lambda) * offset_hat_;

    // Set the next energy so the predicted depth lands on target.
    double next = (p_.target_um - offset_hat_) / p_.gain;
    next = std::clamp(next, p_.energy_min, p_.energy_max);
    energy_ = next;
    return energy_;
}

// ── OpticsMonitor ────────────────────────────────────────────────────────────

void OpticsMonitor::add(long shot, double power_w) {
    k_.push_back(static_cast<double>(shot));
    power_.push_back(power_w);
}

OpticsHealth OpticsMonitor::health(double power_floor) const {
    OpticsHealth h;
    h.n = static_cast<int>(power_.size());
    if (h.n == 0) return h;

    h.current_power = power_.back();

    // Least-squares slope of power vs shot index (expected negative).
    const double n = h.n;
    const double sk = std::accumulate(k_.begin(), k_.end(), 0.0);
    const double sp = std::accumulate(power_.begin(), power_.end(), 0.0);
    const double skk = std::inner_product(k_.begin(), k_.end(), k_.begin(), 0.0);
    const double skp = std::inner_product(k_.begin(), k_.end(), power_.begin(), 0.0);
    const double denom = n * skk - sk * sk;
    const double slope = std::fabs(denom) > 1e-12 ? (n * skp - sk * sp) / denom : 0.0;
    h.power_slope_per_shot = slope;

    h.service_now = h.current_power <= power_floor;
    if (h.service_now) {
        h.rul_shots = 0.0;
    } else if (slope < -1e-12) {
        // power falling: shots until it reaches the floor.
        h.rul_shots = (power_floor - h.current_power) / slope;
    } else {
        h.rul_shots = 1e9;  // stable/rising: effectively unlimited
    }
    return h;
}

}  // namespace wproc::laser
