// laser_control.hpp — laser ablation physics, pulse-energy APC, optics health.
//
// The laser-processing analog of the dicing/grinding APC stacks. Three pieces:
//
//   AblationPlant        first-order ablation "plant": groove depth for a given
//                        pulse energy / scan speed / pass count, with a slow
//                        optics-power degradation drift and shot noise.
//   PulseEnergyController run-to-run EWMA controller that trims the pulse energy
//                        each wafer so the ablated depth holds target as the
//                        beam delivery slowly loses efficiency.
//   OpticsMonitor        tracks delivered laser power over shots, fits the
//                        degradation trend, and predicts shots-until-service
//                        (RUL) when transmitted power crosses a floor.
//
// Depth model (per wafer/site): a fixed-fluence ablation law linearised about
// the operating point,
//   depth = k_e * energy * passes / speed  -  drift * shot  +  noise
// so more energy / more passes / slower scan removes more; as the optics dull
// the same command removes less (the drift term).
//
// Pure C++17, header + one TU. Mirrors apc.hpp / grind_control.hpp.
#pragma once

#include <vector>

namespace wproc::laser {

// ── Physics plant ────────────────────────────────────────────────────────────

struct AblationPlant {
    double depth_gain = 1.0;        // um removed per unit (energy*passes/speed)
    double passes = 1.0;            // number of scan passes
    double speed = 1.0;             // scan speed (normalised)
    double drift_um_per_shot = 0.0; // optics dulling: depth loss per shot
    double power_gain = 1.0;        // delivered power per unit pulse energy
    double power_drop_per_shot = 0.0;  // transmitted-power loss per shot
    double noise_sigma = 0.0;

    // Ablated depth for `energy` at shot k, with supplied noise n.
    double depth(double energy, long k, double n) const {
        return depth_gain * energy * passes / speed - drift_um_per_shot * k + n;
    }
    // Delivered/transmitted laser power (arbitrary W) for `energy` at shot k.
    double power(double energy, long k) const {
        return power_gain * energy - power_drop_per_shot * k;
    }
};

// ── Run-to-run pulse-energy controller ───────────────────────────────────────

struct PulseParams {
    double target_um = 20.0;    // desired ablation depth
    double gain = 1.0;          // model d(depth)/d(energy) (gain != 0)
    double lambda = 0.3;        // EWMA weight in (0, 1]
    double energy0 = 20.0;      // initial pulse energy
    double energy_min = 1.0;    // actuator bounds
    double energy_max = 60.0;
};

class PulseEnergyController {
public:
    explicit PulseEnergyController(PulseParams p);

    // Feed the measured depth ablated at the last commanded energy; returns the
    // energy to command for the next wafer/site.
    double update(double measured_depth_um);

    double current_energy() const { return energy_; }
    double offset_estimate() const { return offset_hat_; }

private:
    PulseParams p_;
    double energy_;      // last commanded pulse energy
    double offset_hat_;  // EWMA estimate: depth = offset + gain*energy
};

// ── Optics health / service prediction ───────────────────────────────────────

struct OpticsHealth {
    int n = 0;
    double current_power = 0.0;
    double power_slope_per_shot = 0.0;  // fitted (negative) power trend
    double rul_shots = 0.0;             // shots until power crosses the floor
    bool service_now = false;           // power already at/below floor
};

// Online monitor of delivered laser power vs shot index.
class OpticsMonitor {
public:
    void add(long shot, double power_w);

    // Health estimate; `power_floor` is the minimum acceptable delivered power.
    OpticsHealth health(double power_floor) const;

private:
    std::vector<double> k_, power_;
};

}  // namespace wproc::laser
