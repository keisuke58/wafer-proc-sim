// test_stealth.cpp — stealth-dicing subsurface layer & crack detection.
#include <cmath>
#include <iostream>

#include "wproc/filters.hpp"
#include "wproc/image.hpp"
#include "wproc/stealth_ir.hpp"

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

// IR cross-section: dark silicon with three bright SD layers, optional crack.
static Image make_sd(bool with_crack, int crack_top_y) {
    Image img(200, 120, 60.0f);
    const int depths[3] = {30, 50, 70};
    for (int d : depths)
        for (int y = d - 1; y <= d + 1; ++y)
            for (int x = 0; x < 200; ++x) img.at(x, y) = 200.0f;
    if (with_crack)  // bright vertical crack from the top layer toward surface
        for (int y = crack_top_y; y <= 30; ++y)
            for (int x = 99; x <= 101; ++x) img.at(x, y) = 200.0f;
    return gaussian_blur(img, 1.0f);
}

int main() {
    // Three layers, no crack.
    stealth::SdResult r = stealth::inspect_stealth(make_sd(false, 0));
    CHECK(r.layer_count == 3);
    if (r.layer_count == 3) {
        CHECK_NEAR(r.layer_depths_px[0], 30.0, 1.5);
        CHECK_NEAR(r.layer_depths_px[1], 50.0, 1.5);
        CHECK_NEAR(r.layer_depths_px[2], 70.0, 1.5);
    }
    CHECK_NEAR(r.mean_pitch_px, 20.0, 1.0);
    CHECK(r.max_crack_px < 5.0);              // no crack present
    CHECK(!r.crack_reaches_surface);
    std::cout << "layers=" << r.layer_count << " pitch=" << r.mean_pitch_px
              << " crack=" << r.max_crack_px << "\n";

    // Crack from the top layer up to y=12 (does not reach the surface).
    stealth::SdResult rc = stealth::inspect_stealth(make_sd(true, 12));
    CHECK(rc.layer_count == 3);
    CHECK(rc.max_crack_px > 12.0);           // ~18 px crack detected
    CHECK(!rc.crack_reaches_surface);        // stops well above y=0
    std::cout << "with crack: crack_len=" << rc.max_crack_px
              << " reaches_surface=" << rc.crack_reaches_surface << "\n";

    // Crack all the way to the surface (y=1).
    stealth::SdResult rs = stealth::inspect_stealth(make_sd(true, 1));
    CHECK(rs.crack_reaches_surface);
    std::cout << "surface crack: reaches_surface=" << rs.crack_reaches_surface
              << " len=" << rs.max_crack_px << "\n";

    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "STEALTH OK\n";
    return 0;
}
