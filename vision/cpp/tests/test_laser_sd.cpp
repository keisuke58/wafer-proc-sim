// test_laser_sd.cpp — stealth SD-layer + HAZ quality, surface-crack risk.
#include <cmath>
#include <iostream>

#include "wproc/image.hpp"
#include "wproc/laser_sd.hpp"

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

// Cross-section IR image: bright horizontal SD bands (rows) on a dark wafer,
// evenly pitched. Optionally a bright vertical crack rising toward the surface.
static Image make_sd(int w, int h, bool crack) {
    Image img(w, h, 30.0f);                    // wafer body (dark in IR)
    const int rows[3] = {40, 60, 80};          // SD layers, pitch 20
    for (int r : rows)
        for (int y = r - 2; y <= r + 2; ++y)
            for (int x = 0; x < w; ++x)
                if (y >= 0 && y < h) img.at(x, y) = 220.0f;
    if (crack) {                                // crack from top layer to surface
        const int x = w / 2;                    // 3 px wide so it survives blur
        for (int y = 2; y <= 39; ++y)
            for (int dx = -1; dx <= 1; ++dx) img.at(x + dx, y) = 220.0f;
    }
    return img;
}

static void test_clean_sd() {
    Image img = make_sd(80, 120, /*crack=*/false);
    laser::SdQuality q = laser::inspect_sd_haz(img);
    std::cout << "clean SD: layers=" << q.layer_count << " pitch=" << q.mean_pitch_px
              << " unif=" << q.pitch_uniformity_pct << "% haz=" << q.haz_thickness_px
              << " risk=" << q.surface_risk << " ok=" << q.ok << "\n";
    CHECK(q.layer_count == 3);
    CHECK(std::fabs(q.mean_pitch_px - 20.0) < 2.0);
    CHECK(q.pitch_uniformity_pct > 95.0);
    CHECK(q.haz_thickness_px > 0.0 && q.haz_thickness_px <= 12.0);
    CHECK(!q.crack_reaches_surface);
    CHECK(q.ok);
}

static void test_surface_crack() {
    Image img = make_sd(80, 120, /*crack=*/true);
    laser::SdQuality q = laser::inspect_sd_haz(img);
    std::cout << "cracked SD: max_crack=" << q.max_crack_px << " reaches="
              << q.crack_reaches_surface << " risk=" << q.surface_risk
              << " ok=" << q.ok << "\n";
    CHECK(q.crack_reaches_surface);
    CHECK(q.max_crack_px > 20.0);
    CHECK(q.surface_risk > 0.9);
    CHECK(!q.ok);
}

int main() {
    test_clean_sd();
    test_surface_crack();
    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "LASER SD OK\n";
    return 0;
}
