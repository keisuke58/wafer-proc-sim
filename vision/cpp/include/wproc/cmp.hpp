// cmp.hpp — chemical-mechanical polishing (CMP / "磨く") process intelligence.
//
// The polish leg of cut / grind / polish. Four cooperating pieces:
//
//   Preston model    removal rate = kp * pressure * velocity; the workhorse CMP
//                    law, and a polishing "plant" whose pad slowly glazes
//                    (kp drifts down) run to run.
//   PolishController run-to-run EWMA controller that trims the polish time so the
//                    removed amount holds target as the pad wears.
//   analyze_planarity  from a post-CMP height profile across patterned + field
//                    regions: dishing (Cu recess), erosion (field loss) and a
//                    planarization-uniformity verdict.
//   detect_endpoint  finds the polish endpoint (film cleared) from the friction /
//                    motor-current trace as the sharp drop after the plateau.
//
// Pure C++17, header + one TU. Mirrors apc.hpp / grind_control.hpp.
#pragma once

#include <vector>

namespace wproc::cmp {

// ── Preston removal model + plant ────────────────────────────────────────────

struct PrestonParams {
    double kp = 1.0e-3;      // Preston coefficient
    double pressure = 30.0;  // down-force pressure [kPa]
    double velocity = 0.5;   // relative pad-wafer velocity [m/s]
};

// Removal rate [nm/s] = kp * pressure * velocity.
double removal_rate(const PrestonParams& p);
// Material removed [nm] after polishing for `time_s` seconds.
double removal(const PrestonParams& p, double time_s);

// Polishing plant for simulation: removal for a given time at run k, with the
// pad glazing (kp declines) and process noise.
struct PolishPlant {
    double kp0 = 1.0e-3;
    double pressure = 30.0;
    double velocity = 0.5;
    double kp_decay_per_run = 0.0;  // pad glazing: kp -> kp0 - decay*k
    double noise_sigma = 0.0;

    double removed(double time_s, long k, double n) const {
        const double kp = kp0 - kp_decay_per_run * k;
        return kp * pressure * velocity * time_s + n;
    }
};

// ── Run-to-run polish-time controller ────────────────────────────────────────

struct PolishParams {
    double target_removal_nm = 300.0;
    double kp = 1.0e-3;         // model Preston coefficient (kp != 0)
    double pressure = 30.0;
    double velocity = 0.5;
    double lambda = 0.3;        // EWMA weight in (0, 1]
    double time0 = 20.0;        // initial polish time [s]
    double time_min = 1.0;      // actuator bounds
    double time_max = 120.0;
};

class PolishController {
public:
    explicit PolishController(PolishParams p);

    // Feed the removed amount from the last run; returns the polish time to
    // command next.
    double update(double measured_removal_nm);

    double current_time() const { return time_; }
    double kp_estimate() const { return kp_hat_; }

private:
    PolishParams p_;
    double time_;    // last commanded polish time
    double kp_hat_;  // EWMA estimate of the effective Preston coefficient
};

// ── Planarity: dishing / erosion ─────────────────────────────────────────────

struct PlanaritySpec {
    double nominal_nm = 0.0;       // reference field height (post-CMP target)
    double max_dishing_nm = 40.0;
    double max_erosion_nm = 30.0;
};

struct PlanarityStats {
    int n_field = 0, n_feature = 0;
    double field_mean_nm = 0.0;
    double feature_mean_nm = 0.0;
    double dishing_nm = 0.0;       // field_mean - feature_mean (Cu recess)
    double erosion_nm = 0.0;       // nominal - field_mean (field loss)
    double range_nm = 0.0;         // max - min over the profile
    double uniformity_pct = 0.0;   // 100*(1 - sigma_field/|field_mean|), clamped
    bool ok = true;                // dishing/erosion within spec
};

// Analyze a post-CMP height profile (nm). `is_feature[i]` marks patterned (Cu)
// samples; the rest are field. Throws std::invalid_argument on a size mismatch
// or if either region is empty.
PlanarityStats analyze_planarity(const std::vector<double>& profile_nm,
                                 const std::vector<char>& is_feature,
                                 const PlanaritySpec& spec = PlanaritySpec{});

// ── Endpoint detection ───────────────────────────────────────────────────────

struct Endpoint {
    bool detected = false;
    int index = -1;
    double time_s = 0.0;
    double drop = 0.0;       // plateau-to-endpoint signal drop
};

// Detect the CMP endpoint from a friction / motor-current trace sampled at `fs`
// Hz: after smoothing, the steepest sustained drop past the plateau. Detected
// when the drop exceeds `drop_frac` of the plateau-to-minimum range.
Endpoint detect_endpoint(const std::vector<double>& signal, double fs,
                         double drop_frac = 0.5, int smooth = 5);

}  // namespace wproc::cmp
