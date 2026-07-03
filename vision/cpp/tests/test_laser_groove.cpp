// test_laser_groove.cpp — ablation-groove width / HAZ / debris inspection.
#include <cmath>
#include <iostream>
#include <random>

#include "wproc/image.hpp"
#include "wproc/laser_groove.hpp"

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

// Vertical groove: dark floor, an intermediate HAZ band on each side, bright
// substrate, plus a sprinkle of bright debris specks on the substrate.
static Image make_groove(int w, int h, int half_width_px, int haz_px, int debris) {
    Image img(w, h, 150.0f);          // substrate
    const int c = w / 2;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const int d = std::abs(x - c);
            if (d < half_width_px) img.at(x, y) = 40.0f;                 // groove floor
            else if (d < half_width_px + haz_px) img.at(x, y) = 95.0f;   // HAZ band
        }
    // Debris specks on the substrate (deterministic).
    std::mt19937 gen(21);
    std::uniform_int_distribution<int> ux(0, w - 1), uy(0, h - 1);
    for (int i = 0; i < debris; ++i) {
        int x = ux(gen), y = uy(gen);
        if (std::abs(x - c) >= half_width_px + haz_px + 3) img.at(x, y) = 235.0f;
    }
    return img;
}

static void test_clean_groove() {
    // half-width 20 px ⇒ 40 px wide ⇒ 20 µm at 0.5 µm/px; HAZ 8 px ⇒ 4 µm/side.
    Image g = make_groove(200, 120, 20, 8, 30);
    laser::GrooveSpec spec;  // target 20 µm ± 4, HAZ ≤ 6, debris ≤ 0.02
    laser::GrooveResult r = laser::inspect_groove(g, spec);
    std::cout << "clean: found=" << r.found << " width=" << r.width_um
              << "um haz=" << r.haz_um << "um depth=" << r.depth_index
              << " debris=" << r.debris_index << " ok=" << r.ok << "\n";
    CHECK(r.found);
    CHECK(std::fabs(r.width_um - 20.0) <= 4.0);
    CHECK(r.depth_index > 0.5);
    CHECK(r.haz_um > 2.0 && r.haz_um < 6.0);
    CHECK(r.debris_index > 0.0);          // debris detected
    CHECK(r.debris_index < 0.02);
    CHECK(r.ok);
}

static void test_bad_groove_and_render() {
    // Too wide (70 px half ⇒ 70 µm) ⇒ out of width spec ⇒ FAIL.
    Image wide = make_groove(240, 120, 35, 8, 10);
    laser::GrooveResult r = laser::inspect_groove(wide);
    std::cout << "wide: width=" << r.width_um << "um ok=" << r.ok << "\n";
    CHECK(r.found);
    CHECK(r.width_um > 30.0);
    CHECK(!r.ok);

    // Flat image ⇒ no groove.
    Image flat(120, 80, 150.0f);
    laser::GrooveResult fr = laser::inspect_groove(flat);
    CHECK(!fr.found);

    // Annotated overlay keeps the image size.
    ColorImage a = laser::annotate_groove(make_groove(160, 100, 16, 6, 12),
                                          laser::inspect_groove(make_groove(160, 100, 16, 6, 12)));
    CHECK(a.width() == 160 && a.height() == 100);
    CHECK(!a.empty());
}

int main() {
    test_clean_groove();
    test_bad_groove_and_render();
    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "LASER GROOVE OK\n";
    return 0;
}
