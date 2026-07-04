// motion.cpp — S-curve trajectory generation + servo tracking.
#include "wproc/motion.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <tuple>

namespace wproc::motion {

namespace {

// Accel-phase distance (0 -> vp -> 0 has 2x this) and its segment times for a
// symmetric jerk-limited ramp to peak velocity vp.
std::tuple<double, double, double> accel_phase(double vp, double amax,
                                               double jmax) {
    double tj, ta;
    if (vp < amax * amax / jmax) {   // amax never reached
        tj = std::sqrt(vp / jmax);
        ta = 0.0;
    } else {
        tj = amax / jmax;
        ta = vp / amax - tj;
    }
    const double Ta = 2.0 * tj + ta;      // time 0 -> vp
    const double dist = vp * Ta / 2.0;    // area under the 0->vp velocity ramp
    return {dist, tj, ta};
}

}  // namespace

Trajectory scurve(double distance, const Limits& lim, double dt) {
    Trajectory tr;
    tr.dt = dt;
    if (distance <= 0.0 || dt <= 0.0) return tr;

    const double jmax = lim.jmax, amax = lim.amax;
    double vp = lim.vmax;
    auto [Da, tj, ta] = accel_phase(vp, amax, jmax);
    double tv;
    if (2.0 * Da <= distance) {
        tv = (distance - 2.0 * Da) / vp;      // cruise at vmax
    } else {
        // Peak velocity not reached: bisect vp so accel+decel exactly spans D.
        double lo = 0.0, hi = lim.vmax;
        for (int it = 0; it < 100; ++it) {
            vp = 0.5 * (lo + hi);
            const double d = std::get<0>(accel_phase(vp, amax, jmax));
            if (2.0 * d > distance) hi = vp; else lo = vp;
        }
        vp = 0.5 * (lo + hi);
        std::tie(Da, tj, ta) = accel_phase(vp, amax, jmax);
        tv = 0.0;
    }

    // 7 segments: +j, 0(const a), -j, 0(cruise), -j, 0(const a), +j.
    const double dur[7] = {tj, ta, tj, tv, tj, ta, tj};
    const double jrk[7] = {jmax, 0.0, -jmax, 0.0, -jmax, 0.0, jmax};
    const double total = std::accumulate(std::begin(dur), std::end(dur), 0.0);
    tr.total_time = total;

    // Integrate jerk -> accel -> vel -> pos at fixed dt.
    double p = 0.0, v = 0.0, a = 0.0;
    int seg = 0;
    double seg_t = 0.0;
    const int nsteps = static_cast<int>(std::ceil(total / dt)) + 1;
    tr.pos.reserve(nsteps);
    tr.vel.reserve(nsteps);
    tr.acc.reserve(nsteps);
    for (int i = 0; i < nsteps; ++i) {
        tr.pos.push_back(p);
        tr.vel.push_back(v);
        tr.acc.push_back(a);
        // advance one dt through the (possibly multiple) segments
        double rem = dt;
        while (rem > 1e-15 && seg < 7) {
            const double seg_left = dur[seg] - seg_t;
            const double step = std::min(rem, seg_left);
            const double j = jrk[seg];
            // integrate the sub-step
            p += v * step + 0.5 * a * step * step + (1.0 / 6.0) * j * step * step * step;
            v += a * step + 0.5 * j * step * step;
            a += j * step;
            seg_t += step;
            rem -= step;
            if (seg_t >= dur[seg] - 1e-15) { seg_t = 0.0; ++seg; }
        }
    }
    // Snap the endpoint to remove integration drift.
    if (!tr.pos.empty()) {
        tr.pos.back() = distance;
        tr.vel.back() = 0.0;
        tr.acc.back() = 0.0;
    }
    return tr;
}

TrackResult track(const Trajectory& ref, const StagePlant& plant,
                  const ServoGains& gains) {
    TrackResult r;
    const std::size_t n = ref.pos.size();
    if (n == 0 || ref.dt <= 0.0 || plant.mass <= 0.0) return r;

    double p = 0.0, v = 0.0, integ = 0.0, sse = 0.0;
    const double dt = ref.dt;
    for (std::size_t i = 0; i < n; ++i) {
        const double e = ref.pos[i] - p;
        const double ed = ref.vel[i] - v;
        integ += e * dt;
        double force = gains.kp * e + gains.kd * ed + gains.ki * integ;
        if (gains.feedforward)                       // model-based feedforward
            force += plant.mass * ref.acc[i] + plant.damping * ref.vel[i];
        const double a = (force - plant.damping * v) / plant.mass;
        v += a * dt;
        p += v * dt;

        const double err = std::fabs(ref.pos[i] - p);
        r.max_error = std::max(r.max_error, err);
        sse += err * err;
        r.final_error = err;
    }
    r.rms_error = std::sqrt(sse / n);
    return r;
}

}  // namespace wproc::motion
