// test_metrology.cpp — repeatability statistics and pixel-scale calibration.
#include <cmath>
#include <iostream>

#include "wproc/filters.hpp"
#include "wproc/image.hpp"
#include "wproc/metrology.hpp"

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
#define CHECK_NEAR(a, b, tol)                                                  \
    do {                                                                       \
        ++g_total;                                                             \
        double _d = std::fabs(double(a) - double(b));                          \
        if (_d > (tol)) {                                                      \
            ++g_failed;                                                        \
            std::cerr << "FAIL [" << __LINE__ << "]: |" << (a) << " - " << (b) \
                      << "| = " << _d << " > " << (tol) << "\n";               \
        }                                                                      \
    } while (0)

static Image clean_kerf(int left, int right) {
    Image img(400, 200, 150.0f);
    for (int y = 0; y < 200; ++y)
        for (int x = left; x <= right; ++x) img.at(x, y) = 40.0f;
    return gaussian_blur(img, 1.0f);
}

static void test_repeatability() {
    Image base = clean_kerf(160, 240);  // ~40 um at 0.5 um/px
    KerfInspector::Params p;
    p.um_per_px = 0.5;
    KerfInspector insp(p);

    metro::Repeatability r = metro::repeatability(base, insp, 40, 6.0f, /*seed=*/7);
    CHECK(r.n >= 38);                       // nearly all trials find the kerf
    CHECK_NEAR(r.width_mean_um, 40.0, 1.5);  // unbiased about the true width
    CHECK(r.width_sd_um > 0.0);              // noise produces real spread
    CHECK(r.width_sd_um < 1.5);              // but the measurement is precise
    CHECK(r.width_min_um <= r.width_mean_um && r.width_mean_um <= r.width_max_um);
    // determinism: same seed -> identical statistics
    metro::Repeatability r2 = metro::repeatability(base, insp, 40, 6.0f, 7);
    CHECK_NEAR(r.width_sd_um, r2.width_sd_um, 1e-9);
    std::cout << "repeatability: mean=" << r.width_mean_um
              << " sd=" << r.width_sd_um << " cv%=" << r.width_cv_pct << "\n";
}

static void test_calibration() {
    // Vertical line grating: full pitch = 24 px (12 bright, 12 dark).
    const int W = 384, H = 64, pitch_px = 24;
    Image target(W, H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            target.at(x, y) = (x % pitch_px) < pitch_px / 2 ? 200.0f : 40.0f;
    target = gaussian_blur(target, 1.0f);

    // Known true pitch = 12 um  =>  expected scale 12/24 = 0.5 um/px.
    metro::Calibration c = metro::estimate_um_per_px(target, 12.0);
    CHECK(c.ok);
    CHECK_NEAR(c.pixel_pitch_px, 24.0, 0.5);
    CHECK_NEAR(c.um_per_px, 0.5, 0.02);
    std::cout << "calibration: pitch=" << c.pixel_pitch_px
              << "px  um/px=" << c.um_per_px << "\n";
}

int main() {
    test_repeatability();
    test_calibration();
    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "METROLOGY OK\n";
    return 0;
}
