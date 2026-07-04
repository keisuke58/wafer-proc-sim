// motion.hpp — real-time stage motion: S-curve trajectory + servo tracking.
//
// A dicing/grinding machine indexes the wafer on a precision XY stage. Two
// pieces of the real-time control problem:
//
//   S-curve trajectory  a jerk-limited (7-segment) point-to-point profile — the
//                       jerk cap keeps the stage from exciting structural modes,
//                       which would blur the cut.
//   Servo tracking      a PID controller with model-based (velocity + accel)
//                       feedforward following that reference through a
//                       second-order stage plant; feedforward collapses the
//                       following error that feedback alone leaves.
//
// Pure C++17, header + one TU.
#pragma once

#include <vector>

namespace wproc::motion {

// ── Jerk-limited S-curve ─────────────────────────────────────────────────────

struct Limits {
    double vmax = 1.0;   // max velocity
    double amax = 10.0;  // max acceleration
    double jmax = 100.0; // max jerk
};

struct Trajectory {
    double dt = 0.0;
    double total_time = 0.0;
    std::vector<double> pos;   // reference position
    std::vector<double> vel;   // reference velocity
    std::vector<double> acc;   // reference acceleration
};

// Rest-to-rest S-curve moving `distance` (>0) under the given limits, sampled at
// `dt`. Peak velocity/acceleration are reduced automatically for short moves so
// the profile always starts and ends at rest.
Trajectory scurve(double distance, const Limits& lim, double dt);

// ── Servo tracking ───────────────────────────────────────────────────────────

// Second-order stage: mass * a = force - damping * velocity.
struct StagePlant {
    double mass = 1.0;
    double damping = 2.0;
};

struct ServoGains {
    double kp = 400.0;
    double kd = 40.0;
    double ki = 0.0;
    bool feedforward = true;  // add model-based velocity + accel feedforward
};

struct TrackResult {
    double max_error = 0.0;
    double rms_error = 0.0;
    double final_error = 0.0;
};

// Track a reference trajectory through the plant with the given servo; returns
// the following-error statistics.
TrackResult track(const Trajectory& ref, const StagePlant& plant,
                  const ServoGains& gains);

}  // namespace wproc::motion
