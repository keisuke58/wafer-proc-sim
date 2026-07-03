// grind_control.hpp — grinding physics, infeed APC, wheel health, economics.
//
// The back-grind analog of the dicing APC stack. Four cooperating pieces:
//
//   GrindPlant      first-order thinning "plant": final thickness for a given
//                   infeed at wafer k, with wheel-dulling drift, plus the
//                   grinding force that infeed/dulling produce.
//   InfeedController run-to-run EWMA controller that trims the infeed each wafer
//                   so the achieved final thickness holds target against dulling.
//   WheelMonitor    tracks grinding force over wafers, fits its trend, and
//                   predicts wafers-until-dress (RUL) with a G-ratio proxy.
//   economics()     rolls throughput / yield / consumables into cost per wafer
//                   and cost per good die.
//
// Pure C++17, header + one TU. Mirrors the style of apc.hpp / blade_health.hpp.
#pragma once

#include <vector>

namespace wproc::grind {

// ── Physics plant ────────────────────────────────────────────────────────────

// final_thickness = offset0 + dull*k - gain*infeed + noise
//   more infeed removes more material (thinner wafer); as the wheel dulls the
//   same infeed removes less, so the effective offset drifts up with k.
struct GrindPlant {
    double offset0_um = 780.0;        // final thickness at zero infeed, fresh wheel
    double gain_um_per_infeed = 100.0;// material removed per unit infeed
    double dull_um_per_wafer = 0.0;   // offset drift from wheel dulling
    double force_gain = 1.0;          // grinding force per unit infeed (N)
    double force_dull_per_wafer = 0.0;// extra force per wafer of dulling (N)
    double noise_sigma = 0.0;

    // Achieved final thickness for `infeed` at wafer k, with supplied noise n
    // (caller provides noise so simulations stay reproducible).
    double final_thickness(double infeed, long k, double n) const {
        return offset0_um + dull_um_per_wafer * k - gain_um_per_infeed * infeed + n;
    }
    // Grinding force (arbitrary N) for `infeed` at wafer k.
    double force(double infeed, long k) const {
        return force_gain * infeed + force_dull_per_wafer * k;
    }
};

// ── Run-to-run infeed controller ─────────────────────────────────────────────

struct GrindParams {
    double target_um = 100.0;   // desired final thickness
    double gain = 100.0;        // model d(removed)/d(infeed) (gain != 0)
    double lambda = 0.3;        // EWMA weight in (0, 1]
    double infeed0 = 6.0;       // initial infeed
    double infeed_min = 0.0;    // actuator bounds
    double infeed_max = 12.0;
};

class InfeedController {
public:
    explicit InfeedController(GrindParams p);

    // Feed the measured final thickness of the wafer ground at the last
    // commanded infeed; returns the infeed to command for the next wafer.
    double update(double measured_thickness_um);

    double current_infeed() const { return infeed_; }
    double offset_estimate() const { return offset_hat_; }

private:
    GrindParams p_;
    double infeed_;      // last commanded infeed
    double offset_hat_;  // EWMA estimate of offset (thickness = offset - gain*infeed)
};

// ── Wheel health / dress prediction ──────────────────────────────────────────

struct WheelHealth {
    int n = 0;
    double current_force = 0.0;
    double force_slope_per_wafer = 0.0;  // fitted force trend
    double total_removed_um = 0.0;
    double g_ratio = 0.0;                // removed / accumulated wear proxy
    double rul_wafers = 0.0;             // wafers until force crosses dress limit
    bool dress_now = false;              // force already at/over limit
};

// Online monitor of grinding force vs wafer index.
class WheelMonitor {
public:
    // Record one wafer's outcome: material removed (um) and grinding force (N).
    void add(long wafer, double removed_um, double force_n);

    // Health estimate; `force_dress_limit` is the force at which the wheel must
    // be dressed.
    WheelHealth health(double force_dress_limit) const;

private:
    std::vector<double> k_, removed_, force_;
};

// ── Economics roll-up ────────────────────────────────────────────────────────

struct EconInputs {
    double wafers_per_hour = 20.0;
    double dies_per_wafer = 1000.0;
    double yield_pct = 95.0;
    double machine_cost_per_hour = 120.0;  // depreciation + labor + overhead
    double wheel_cost = 800.0;             // per grinding wheel
    double wafers_per_wheel = 2000.0;      // wheel life
    double consumables_per_wafer = 1.5;    // DI water, slurry, tape, etc.
};

struct EconSummary {
    double throughput_wph = 0.0;
    double good_dies_per_hour = 0.0;
    double cost_per_wafer = 0.0;
    double cost_per_good_die = 0.0;
};

// Roll inputs into per-wafer and per-good-die cost.
EconSummary economics(const EconInputs& in);

}  // namespace wproc::grind
