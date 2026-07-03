// test_grind_metrology.cpp — TTV/WARP/BOW recovery on synthetic thickness maps.
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "wproc/grind_metrology.hpp"
#include "wproc/image.hpp"

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

// Build a thickness map on a disk: base + tilt*x + a radial bowl of amplitude
// `amp` (centre thinner by amp). Cells outside the disk are left at 0 (no-data).
static Image make_map(int n, double base, double tilt, double amp) {
    Image img(n, n, 0.0f);
    const double cx = (n - 1) * 0.5, cy = (n - 1) * 0.5;
    const double R = n * 0.5 * 0.98;
    const double R2 = R * R;
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            const double dx = x - cx, dy = y - cy;
            const double d2 = dx * dx + dy * dy;
            if (d2 > R2) continue;
            const double bowl = -amp * (1.0 - d2 / R2);  // centre = -amp
            img.at(x, y) = static_cast<float>(base + tilt * dx + bowl);
        }
    }
    return img;
}

static void test_pure_tilt() {
    // Only a linear tilt: the plane fit should absorb it, leaving WARP ~ 0.
    Image m = make_map(128, 100.0, 0.05, 0.0);
    grind::ThicknessStats s = grind::analyze_thickness(m);
    std::cout << "tilt: TTV=" << s.ttv_um << " WARP=" << s.warp_um
              << " BOW=" << s.bow_um << " unif=" << s.uniformity_pct << "%\n";
    CHECK(s.n_inside > 1000);
    CHECK(s.ttv_um > 3.0);          // tilt spans a real thickness range
    CHECK(s.warp_um < 0.5);         // de-tilted surface is flat
    CHECK(std::fabs(s.bow_um) < 0.5);
    CHECK(s.uniformity_pct > 95.0);
}

static void test_tilt_plus_bowl() {
    const double amp = 4.0;
    Image m = make_map(128, 100.0, 0.05, amp);
    grind::ThicknessStats s = grind::analyze_thickness(m);
    std::cout << "bowl: TTV=" << s.ttv_um << " WARP=" << s.warp_um
              << " BOW=" << s.bow_um << " mean=" << s.mean_um << "\n";
    // WARP is the PV of the de-tilted surface ~ the bowl amplitude.
    CHECK(s.warp_um > 3.0 && s.warp_um < 5.5);
    // Centre is thinner than the reference plane => negative bow.
    CHECK(s.bow_um < -0.5);
    // TTV includes tilt + bowl, so it is at least the bowl PV.
    CHECK(s.ttv_um >= s.warp_um - 1e-6);
    CHECK(s.mean_um > 90.0 && s.mean_um < 100.0);
}

static void test_invalid() {
    Image empty_disk(64, 64, 0.0f);  // all no-data
    bool threw = false;
    try {
        grind::analyze_thickness(empty_disk);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_render() {
    Image m = make_map(96, 100.0, 0.0, 3.0);
    ColorImage c = grind::render_thickness(m);
    CHECK(c.width() == 96 && c.height() == 96);
    CHECK(!c.empty());
}

int main() {
    test_pure_tilt();
    test_tilt_plus_bowl();
    test_invalid();
    test_render();
    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "GRIND METROLOGY OK\n";
    return 0;
}
