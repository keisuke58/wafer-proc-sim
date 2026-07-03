// grind_control.cpp — grinding APC, wheel health, economics.
#include "wproc/grind_control.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace wproc::grind {

// ── InfeedController ─────────────────────────────────────────────────────────

InfeedController::InfeedController(GrindParams p) : p_(p), infeed_(p.infeed0) {
    if (p_.gain == 0.0)
        throw std::invalid_argument("InfeedController: gain must be non-zero");
    if (!(p_.lambda > 0.0 && p_.lambda <= 1.0))
        throw std::invalid_argument("InfeedController: lambda must be in (0,1]");
    if (p_.infeed_min > p_.infeed_max)
        throw std::invalid_argument("InfeedController: infeed_min > infeed_max");
    // Offset consistent with the initial infeed hitting target:
    // thickness = offset - gain*infeed  =>  offset = target + gain*infeed0.
    offset_hat_ = p_.target_um + p_.gain * p_.infeed0;
}

double InfeedController::update(double measured_thickness_um) {
    // Invert the model to estimate the current offset, then EWMA-smooth it.
    const double offset_meas = measured_thickness_um + p_.gain * infeed_;
    offset_hat_ = p_.lambda * offset_meas + (1.0 - p_.lambda) * offset_hat_;

    // Set the next infeed so the predicted thickness lands on target.
    double next = (offset_hat_ - p_.target_um) / p_.gain;
    next = std::clamp(next, p_.infeed_min, p_.infeed_max);
    infeed_ = next;
    return infeed_;
}

// ── WheelMonitor ─────────────────────────────────────────────────────────────

void WheelMonitor::add(long wafer, double removed_um, double force_n) {
    k_.push_back(static_cast<double>(wafer));
    removed_.push_back(removed_um);
    force_.push_back(force_n);
}

WheelHealth WheelMonitor::health(double force_dress_limit) const {
    WheelHealth h;
    h.n = static_cast<int>(force_.size());
    if (h.n == 0) return h;

    h.current_force = force_.back();
    h.total_removed_um = std::accumulate(removed_.begin(), removed_.end(), 0.0);

    // Least-squares slope of force vs wafer index.
    const double n = h.n;
    const double sk = std::accumulate(k_.begin(), k_.end(), 0.0);
    const double sf = std::accumulate(force_.begin(), force_.end(), 0.0);
    const double skk = std::inner_product(k_.begin(), k_.end(), k_.begin(), 0.0);
    const double skf = std::inner_product(k_.begin(), k_.end(), force_.begin(), 0.0);
    const double denom = n * skk - sk * sk;
    const double slope = std::fabs(denom) > 1e-12 ? (n * skf - sk * sf) / denom : 0.0;
    const double intercept = (sf - slope * sk) / n;
    h.force_slope_per_wafer = slope;

    // G-ratio proxy: material removed per unit of force rise above the fresh
    // (intercept) force — a stand-in for removed-volume / wheel-wear-volume.
    const double force_rise = std::max(1e-6, h.current_force - intercept);
    h.g_ratio = h.total_removed_um / force_rise;

    h.dress_now = h.current_force >= force_dress_limit;
    if (h.dress_now) {
        h.rul_wafers = 0.0;
    } else if (slope > 1e-9) {
        h.rul_wafers = (force_dress_limit - h.current_force) / slope;
    } else {
        h.rul_wafers = 1e9;  // no rising trend: effectively unlimited
    }
    return h;
}

// ── Economics ────────────────────────────────────────────────────────────────

EconSummary economics(const EconInputs& in) {
    EconSummary s;
    s.throughput_wph = in.wafers_per_hour;
    const double good_frac = std::clamp(in.yield_pct / 100.0, 0.0, 1.0);
    s.good_dies_per_hour = in.wafers_per_hour * in.dies_per_wafer * good_frac;

    const double machine_per_wafer =
        in.wafers_per_hour > 0 ? in.machine_cost_per_hour / in.wafers_per_hour : 0.0;
    const double wheel_per_wafer =
        in.wafers_per_wheel > 0 ? in.wheel_cost / in.wafers_per_wheel : 0.0;
    s.cost_per_wafer = machine_per_wafer + wheel_per_wafer + in.consumables_per_wafer;

    const double good_dies_per_wafer = in.dies_per_wafer * good_frac;
    s.cost_per_good_die =
        good_dies_per_wafer > 0 ? s.cost_per_wafer / good_dies_per_wafer : 0.0;
    return s;
}

}  // namespace wproc::grind
