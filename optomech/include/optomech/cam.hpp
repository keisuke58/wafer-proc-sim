// cam.hpp — autofocus / zoom mechanism: cam-barrel kinematics + VCM control.
//
// A camera lens moves focus and zoom groups axially. Two mechanisms are modelled:
//
//   * Cam barrel — a rotating barrel with a machined cam groove converts barrel
//     rotation theta into axial group displacement z via a cam profile z(theta).
//     The follower kinematics (velocity, acceleration) must stay smooth to avoid
//     jerk that the user feels or that excites the actuator.
//
//   * Voice-coil-motor (VCM) autofocus — the focus group is a mass on a spring
//     (the flexure) driven by a coil force; a PID controller with optional
//     feedforward drives it to a commanded position. Key metrics: settling time,
//     overshoot, steady-state error.
//
// Pure C++17, no external dependencies.
#pragma once

#include <vector>

namespace optm::cam {

// ── Cam-barrel follower kinematics ──────────────────────────────────────────

// A cam profile z(theta) built from a cycloidal (smooth) rise: the group moves
// `stroke_mm` over a rotation of `sweep_deg`, with zero velocity/acceleration at
// both ends (cycloidal motion law) — the standard low-vibration cam.
struct CamProfile {
    double stroke_mm = 5.0;
    double sweep_deg = 90.0;
};

struct CamSample {
    double theta_deg = 0.0;
    double z_mm = 0.0;        // axial displacement
    double vel = 0.0;         // dz/dtheta [mm/deg]
    double accel = 0.0;       // d2z/dtheta2 [mm/deg^2]
};

// Sample the cam law at `n` points across the sweep. A cycloidal law keeps the
// follower acceleration finite everywhere (no jerk spikes).
std::vector<CamSample> cam_curve(const CamProfile& p, int n = 100);

// Peak follower acceleration magnitude over the sweep [mm/deg^2].
double peak_accel(const CamProfile& p, int n = 400);

// ── VCM autofocus actuator + control ────────────────────────────────────────

struct Vcm {
    double mass_g = 0.5;       // focus-group mass [g]
    double flexure_n_per_mm = 2.0;  // spring rate [N/mm]
    double damping = 0.002;    // viscous damping [N/(mm/s)] (lightly damped)
    double force_max_n = 1.2;  // coil force at full command [N] (saturates)
};

struct Pid {
    double kp = 0.0, ki = 0.0, kd = 0.0;
    bool feedforward = false;  // model-based force feedforward
};

struct StepResult {
    double settling_ms = 0.0;  // time to stay within 2% of target
    double overshoot_pct = 0.0;
    double ss_error_um = 0.0;  // final steady-state error
    std::vector<double> t_ms;
    std::vector<double> pos_um;   // response (µm)
    std::vector<double> target_um;
};

// Simulate a step of `target_um` on the VCM under PID control for `t_end_ms`.
StepResult step_response(const Vcm& v, const Pid& pid, double target_um,
                         double t_end_ms = 60.0, double dt_ms = 0.02);

}  // namespace optm::cam
