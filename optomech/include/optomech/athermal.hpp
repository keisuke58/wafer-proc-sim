// athermal.hpp — passive athermalization + random-vibration robustness.
//
// Two environmental-robustness design tasks for the barrel:
//
//   * Athermalization — the barrel's thermal expansion defocuses the image. A
//     compensator spacer of an opposite/higher-CTE material can be sized so its
//     expansion cancels the barrel's, giving a near-zero net focus drift with
//     temperature (passive athermalization).
//   * Random vibration — transport/handling vibration is broadband. Miles'
//     equation gives the RMS acceleration response of the first mode to a flat
//     input PSD; the 3-sigma bending stress is then checked against the
//     material's endurance limit for a fatigue margin.
//
// Reuses the barrel definition from the FEA module. Pure C++17.
#pragma once

#include "optomech/fea.hpp"

namespace optm::athermal {

struct Config {
    double comp_cte_per_C = 70e-6;  // high-CTE compensator spacer (e.g. polymer)
    double modal_Q = 25.0;          // modal amplification (= 1/(2*zeta))
    double psd_g2_per_hz = 0.04;    // flat input acceleration PSD [g^2/Hz]
    double endurance_MPa = 96.0;    // material endurance (fatigue) limit
};

struct Result {
    double uncomp_drift_um_per_C = 0.0;  // barrel alone
    double comp_len_mm = 0.0;            // compensator length for ~0 net drift
    double comp_drift_um_per_C = 0.0;    // residual with a realizable length
    double first_mode_hz = 0.0;
    double grms_response = 0.0;          // Miles RMS response acceleration [g]
    double accel_3sigma_g = 0.0;
    double rand_stress_MPa = 0.0;        // 3-sigma bending stress
    double fatigue_margin = 0.0;         // endurance / stress
    bool ok = false;                     // athermalized and fatigue margin >= 2
};

Result analyze(const fea::Barrel& b, const Config& cfg = Config{});

}  // namespace optm::athermal
