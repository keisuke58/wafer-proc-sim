// test_vcal.cpp — calibration fit recovery, distortion, and Gage R&R.
#include <cmath>
#include <iostream>
#include <vector>

#include "wproc/vcal.hpp"

using namespace wproc;

static int g_total = 0, g_failed = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        ++g_total;                                                      \
        if (!(cond)) {                                                  \
            ++g_failed;                                                 \
            std::cerr << "FAIL [" << __LINE__ << "]: " #cond "\n";      \
        }                                                               \
    } while (0)

// Recover a known pixel→stage affine from noiseless fiducials.
static void test_calibration() {
    // Ground truth: 5 µm/px, 2° rotation, offset (10 mm, 5 mm).
    const double s = 0.005;  // mm/px
    const double th = 2.0 * M_PI / 180.0;
    const double a00 = s * std::cos(th), a01 = -s * std::sin(th);
    const double a10 = s * std::sin(th), a11 = s * std::cos(th);
    std::vector<vcal::Correspondence> pts;
    for (int gy = 0; gy < 4; ++gy)
        for (int gx = 0; gx < 4; ++gx) {
            const double px = gx * 300.0, py = gy * 300.0;
            vcal::Correspondence c;
            c.px = px; c.py = py;
            c.sx = a00 * px + a01 * py + 10.0;
            c.sy = a10 * px + a11 * py + 5.0;
            pts.push_back(c);
        }
    const vcal::CalibResult r = vcal::calibrate(pts);
    std::cout << "calib rms=" << r.rms_um << "um  um/px=" << r.um_per_px
              << "  rot=" << r.rotation_deg << "\n";
    CHECK(r.ok);
    CHECK(r.rms_um < 0.01);                       // near-perfect fit
    CHECK(std::fabs(r.um_per_px - 5.0) < 0.05);   // 5 µm/px recovered
    CHECK(std::fabs(r.rotation_deg - 2.0) < 0.05);

    // A known point maps correctly.
    CHECK(std::fabs(r.transform.map_x(0, 0) - 10.0) < 1e-6);
    CHECK(std::fabs(r.transform.map_y(0, 0) - 5.0) < 1e-6);

    // Too few points → not ok.
    CHECK(!vcal::calibrate({pts[0], pts[1]}).ok);
}

static void test_distortion() {
    vcal::Distortion d;
    d.cx = 100; d.cy = 100; d.f = 100; d.k1 = 0.1;
    double ux = 0, uy = 0;
    vcal::undistort_point(d, 100, 100, ux, uy);   // centre is a fixed point
    CHECK(std::fabs(ux - 100.0) < 1e-9 && std::fabs(uy - 100.0) < 1e-9);
    vcal::undistort_point(d, 200, 100, ux, uy);   // pushes outward for k1>0
    CHECK(ux > 200.0);
}

// A synthetic study where equipment noise is small and parts differ a lot →
// GRR should be acceptable with many distinct categories.
static void test_gage_capable() {
    vcal::GageStudy s;
    s.parts = 5; s.operators = 3; s.trials = 3;
    const double part_true[5] = {1.0, 3.0, 5.0, 7.0, 9.0};   // wide part spread
    const double op_bias[3] = {0.0, 0.05, -0.05};            // tiny operator bias
    // Deterministic pseudo-noise so the test is reproducible.
    unsigned st = 7u;
    auto noise = [&]() {
        st = st * 1664525u + 1013904223u;
        return (static_cast<double>(st >> 9) / 8388608.0 - 0.5) * 0.06;  // ±0.03
    };
    s.values.resize(s.operators);
    for (int o = 0; o < s.operators; ++o) {
        s.values[o].resize(s.parts);
        for (int p = 0; p < s.parts; ++p)
            for (int t = 0; t < s.trials; ++t)
                s.values[o][p].push_back(part_true[p] + op_bias[o] + noise());
    }
    const vcal::GageResult g = vcal::gage_rr(s);
    std::cout << "GRR%=" << g.pct_grr << " ndc=" << g.ndc
              << " EV=" << g.repeatability << " AV=" << g.reproducibility
              << " PV=" << g.part_variation << "\n";
    CHECK(g.ok);
    CHECK(g.pct_grr < 30.0);
    CHECK(g.ndc >= 5);
    CHECK(g.acceptable);
    CHECK(g.part_variation > g.grr);   // parts dominate a capable system
}

// A study where equipment noise is huge → system should be rejected.
static void test_gage_incapable() {
    vcal::GageStudy s;
    s.parts = 4; s.operators = 2; s.trials = 3;
    unsigned st = 99u;
    auto noise = [&]() {
        st = st * 1664525u + 1013904223u;
        return (static_cast<double>(st >> 9) / 8388608.0 - 0.5) * 6.0;  // ±3
    };
    const double part_true[4] = {5.0, 5.2, 4.9, 5.1};   // parts barely differ
    s.values.resize(s.operators);
    for (int o = 0; o < s.operators; ++o) {
        s.values[o].resize(s.parts);
        for (int p = 0; p < s.parts; ++p)
            for (int t = 0; t < s.trials; ++t)
                s.values[o][p].push_back(part_true[p] + noise());
    }
    const vcal::GageResult g = vcal::gage_rr(s);
    std::cout << "incapable GRR%=" << g.pct_grr << " ndc=" << g.ndc << "\n";
    CHECK(g.ok);
    CHECK(!g.acceptable);

    // Rendering produces a sized canvas.
    std::vector<vcal::Correspondence> pts = {
        {0, 0, 0, 0}, {100, 0, 0.5, 0}, {0, 100, 0, 0.5}, {100, 100, 0.5, 0.5}};
    const vcal::CalibResult cal = vcal::calibrate(pts);
    ColorImage img = vcal::render_residuals(pts, cal, 200, 200);
    CHECK(img.width() == 200 && img.height() == 200 && !img.empty());
}

int main() {
    test_calibration();
    test_distortion();
    test_gage_capable();
    test_gage_incapable();
    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "VCAL OK\n";
    return 0;
}
