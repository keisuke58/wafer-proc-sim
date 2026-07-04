// test_cam.cpp — cam-barrel kinematics + VCM autofocus control.
#include <cmath>
#include <iostream>

#include "optomech/cam.hpp"

using namespace optm;

static int g_total = 0, g_failed = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        ++g_total;                                                      \
        if (!(cond)) {                                                  \
            ++g_failed;                                                 \
            std::cerr << "FAIL [" << __LINE__ << "]: " #cond "\n";      \
        }                                                               \
    } while (0)

static void test_cam_curve() {
    cam::CamProfile p; p.stroke_mm = 5.0; p.sweep_deg = 90.0;
    auto curve = cam::cam_curve(p, 200);
    CHECK(curve.size() == 200);
    // Endpoints: z goes 0 -> stroke; end velocity/accel ~ 0 (cycloidal law).
    CHECK(std::fabs(curve.front().z_mm) < 1e-9);
    CHECK(std::fabs(curve.back().z_mm - 5.0) < 1e-6);
    CHECK(std::fabs(curve.front().vel) < 1e-6 && std::fabs(curve.back().vel) < 1e-6);
    // Monotonic rise.
    for (std::size_t i = 1; i < curve.size(); ++i)
        CHECK(curve[i].z_mm >= curve[i - 1].z_mm - 1e-9);
    // Peak accel matches the analytic cycloidal value stroke/sweep^2 * 2pi.
    const double analytic = 5.0 / (90.0 * 90.0) * 6.283185307179586;
    CHECK(std::fabs(cam::peak_accel(p) - analytic) / analytic < 0.01);
}

static void test_vcm_control() {
    cam::Vcm v;  // defaults
    const double tgt = 300.0;  // 300 µm focus move

    // Feedforward cancels the spring load: fast settle, negligible error.
    cam::Pid ff; ff.kp = 0.02; ff.kd = 2e-5; ff.feedforward = true;
    cam::StepResult r_ff = cam::step_response(v, ff, tgt);
    std::cout << "FF: settle=" << r_ff.settling_ms << "ms over="
              << r_ff.overshoot_pct << "% ss=" << r_ff.ss_error_um << "um\n";
    CHECK(r_ff.settling_ms < 12.0);
    CHECK(r_ff.ss_error_um < 2.0);
    CHECK(r_ff.overshoot_pct < 25.0);
    CHECK(r_ff.pos_um.size() == r_ff.t_ms.size());

    // Same gains without feedforward leave a large steady-state focus error
    // (the coil must fight the spring with proportional error alone).
    cam::Pid noff = ff; noff.feedforward = false;
    cam::StepResult r_no = cam::step_response(v, noff, tgt);
    std::cout << "no-FF: ss=" << r_no.ss_error_um << "um\n";
    CHECK(r_no.ss_error_um > 10.0);
    CHECK(r_no.ss_error_um > r_ff.ss_error_um);
}

int main() {
    test_cam_curve();
    test_vcm_control();
    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) { std::cerr << g_failed << " FAILED\n"; return 1; }
    std::cout << "CAM OK\n";
    return 0;
}
