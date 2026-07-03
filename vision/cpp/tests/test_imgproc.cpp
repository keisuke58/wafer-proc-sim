// test_imgproc.cpp — self-contained unit tests (no framework, ctest-friendly).
// Exit code 0 = all passed, 1 = one or more failures.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "wproc/connected.hpp"
#include "wproc/filters.hpp"
#include "wproc/image.hpp"
#include "wproc/io.hpp"
#include "wproc/kerf_inspect.hpp"
#include "wproc/morphology.hpp"
#include "wproc/threshold.hpp"

using namespace wproc;

static int g_total = 0;
static int g_failed = 0;

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

// Build a synthetic dark-kerf image (no noise) with optional chips.
static Image make_kerf(int W, int H, int left, int right, bool with_chip) {
    Image img(W, H, 150.0f);
    for (int y = 0; y < H; ++y)
        for (int x = left; x <= right; ++x) img.at(x, y) = 40.0f;
    if (with_chip)  // dark chip protruding left of the left wall
        for (int y = 40; y < 60; ++y)
            for (int x = left - 8; x < left; ++x) img.at(x, y) = 40.0f;
    return gaussian_blur(img, 1.0f);
}

static void test_gaussian_constant() {
    Image img(20, 20, 77.0f);
    Image b = gaussian_blur(img, 1.5f);
    CHECK_NEAR(b.at(10, 10), 77.0, 1e-3);  // constant stays constant
    CHECK_NEAR(b.at(0, 0), 77.0, 1e-3);    // clamped border stays constant
}

static void test_sobel_step_edge() {
    const int W = 30, H = 20;
    Image img(W, H, 0.0f);
    for (int y = 0; y < H; ++y)
        for (int x = 15; x < W; ++x) img.at(x, y) = 100.0f;  // step at x=15
    Gradient g = sobel(img);
    // Strong horizontal gradient at the edge, ~zero vertical gradient anywhere.
    float max_gx = 0.0f, max_gy = 0.0f;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            max_gx = std::max(max_gx, std::fabs(g.gx.at(x, y)));
            max_gy = std::max(max_gy, std::fabs(g.gy.at(x, y)));
        }
    CHECK(max_gx > 100.0f);
    CHECK(max_gy < 1e-3f);
}

static void test_otsu_bimodal() {
    Image img(100, 1, 20.0f);
    for (int x = 50; x < 100; ++x) img.at(x, 0) = 200.0f;  // two clusters
    float t = otsu_threshold(img);
    // Otsu returns the first maximizer over the equal-variance plateau [20,199];
    // any value here separates the two clusters correctly.
    CHECK(t >= 20.0f && t < 200.0f);
    Image bin = binarize(img, t, /*invert=*/false);
    CHECK_NEAR(bin.at(10, 0), 0.0, 1e-6);
    CHECK_NEAR(bin.at(90, 0), 255.0, 1e-6);
}

static void test_connected_components() {
    Image bin(20, 20, 0.0f);
    // two separated 3x3 blobs
    for (int y = 2; y < 5; ++y)
        for (int x = 2; x < 5; ++x) bin.at(x, y) = 255.0f;
    for (int y = 12; y < 15; ++y)
        for (int x = 12; x < 15; ++x) bin.at(x, y) = 255.0f;
    auto comps = connected_components(bin, 1);
    CHECK(comps.size() == 2);
    if (comps.size() == 2) {
        CHECK(comps[0].area == 9);
        CHECK(comps[1].area == 9);
    }
}

static void test_morphology_open_removes_speck() {
    Image bin(20, 20, 0.0f);
    bin.at(10, 10) = 255.0f;  // single-pixel speck
    for (int y = 2; y < 8; ++y)
        for (int x = 2; x < 8; ++x) bin.at(x, y) = 255.0f;  // solid block
    Image opened = morph_open(bin, 1);
    CHECK_NEAR(opened.at(10, 10), 0.0, 1e-6);  // speck removed
    CHECK_NEAR(opened.at(5, 5), 255.0, 1e-6);  // block survives
}

static void test_kerf_width_measurement() {
    Image img = make_kerf(400, 200, 160, 240, /*with_chip=*/false);
    KerfInspector::Params p;
    p.um_per_px = 0.5;
    KerfInspector insp(p);
    KerfResult r = insp.inspect(img);
    CHECK(r.found);
    CHECK(r.status == "ok");
    // walls ~160/240 -> 80 px -> 40 um
    CHECK_NEAR(r.left_wall_px, 160.0, 2.0);
    CHECK_NEAR(r.right_wall_px, 240.0, 2.0);
    CHECK_NEAR(r.width_um, 40.0, 1.5);
    CHECK(r.confidence > 0.5);
}

static void test_kerf_chip_detection() {
    Image img = make_kerf(400, 200, 160, 240, /*with_chip=*/true);
    KerfInspector::Params p;
    p.um_per_px = 0.5;
    p.max_allowed_chip_um = 2.0;  // 8px chip * 0.5 = ~4um > 2um -> should fail
    KerfInspector insp(p);
    KerfResult r = insp.inspect(img);
    CHECK(r.found);
    CHECK(r.chips.size() >= 1);
    CHECK(r.max_chip_um > 2.0);
    CHECK(r.chipping_in_spec == false);
    CHECK(r.pass == false);
    std::string js = to_json(r);
    CHECK(js.find("width_um") != std::string::npos);
    CHECK(js.find("chips") != std::string::npos);
}

static void test_dust_blob_not_counted_as_chip() {
    // Dark blob far from both walls (dust/fiducial) must not be reported as
    // chipping — previously it was measured back to the nearest wall and
    // produced a huge bogus protrusion.
    Image img(400, 200, 150.0f);
    for (int y = 0; y < 200; ++y)
        for (int x = 160; x <= 240; ++x) img.at(x, y) = 40.0f;  // kerf
    for (int y = 100; y < 108; ++y)
        for (int x = 20; x < 28; ++x) img.at(x, y) = 40.0f;     // dust at x~20
    img = gaussian_blur(img, 1.0f);

    KerfInspector::Params p;
    p.um_per_px = 0.5;
    KerfInspector insp(p);
    KerfResult r = insp.inspect(img);
    CHECK(r.found);
    CHECK(r.chips.empty());          // dust ignored
    CHECK(r.chipping_in_spec);
    CHECK(r.pass);                   // width 40um in spec, no chipping
}

static void test_pgm_crlf_header() {
    // P5 with CRLF line endings (Windows tooling): the '\n' after maxval must
    // not be consumed as the first pixel.
    const char* path = "test_crlf.pgm";
    {
        std::ofstream f(path, std::ios::binary);
        f << "P5\r\n3 2\r\n255\r\n";
        const unsigned char px[6] = {10, 20, 30, 40, 50, 60};
        f.write(reinterpret_cast<const char*>(px), 6);
    }
    Image img = read_pgm(path);
    std::remove(path);
    CHECK(img.width() == 3);
    CHECK(img.height() == 2);
    CHECK_NEAR(img.at(0, 0), 10.0, 1e-6);  // not shifted by the stray '\n'
    CHECK_NEAR(img.at(2, 1), 60.0, 1e-6);
}

static void test_invalid_construction_and_params() {
    // ColorImage must reject negative dimensions with invalid_argument, not
    // attempt a wrapped-size_t giant allocation.
    bool threw = false;
    try {
        ColorImage bad(-1, 5);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    // Oversized sigma must be rejected before the float->int radius cast.
    threw = false;
    try {
        Image img(8, 8, 1.0f);
        gaussian_blur(img, 1e30f);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_no_kerf_on_flat() {
    Image img(100, 100, 128.0f);  // no channel
    KerfInspector insp;
    KerfResult r = insp.inspect(img);
    CHECK(!r.found);
    CHECK(r.status == "no_kerf_found");
}

int main() {
    test_gaussian_constant();
    test_sobel_step_edge();
    test_otsu_bimodal();
    test_connected_components();
    test_morphology_open_removes_speck();
    test_kerf_width_measurement();
    test_kerf_chip_detection();
    test_dust_blob_not_counted_as_chip();
    test_pgm_crlf_header();
    test_invalid_construction_and_params();
    test_no_kerf_on_flat();

    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " check(s) FAILED\n";
        return 1;
    }
    std::cout << "ALL TESTS PASSED\n";
    return 0;
}
