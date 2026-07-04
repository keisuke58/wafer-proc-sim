// test_regi.cpp — phase-correlation shift recovery and die-to-die diff.
#include <cmath>
#include <iostream>

#include "wproc/image.hpp"
#include "wproc/regi.hpp"

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

// A textured "die" pattern so phase correlation has a sharp peak.
static Image make_die(int w, int h) {
    Image d(w, h, 60.0f);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            // Grid of bond pads plus some structure.
            if ((x % 16 < 6) && (y % 16 < 6)) d.at(x, y) = 210.0f;
            if (((x + y) % 24) < 3) d.at(x, y) += 30.0f;
        }
    return d;
}

static void stamp_defect(Image& img, int cx, int cy, int r, float val) {
    for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx)
            if (dx * dx + dy * dy <= r * r) {
                const int x = cx + dx, y = cy + dy;
                if (x >= 0 && x < img.width() && y >= 0 && y < img.height())
                    img.at(x, y) = val;
            }
}

static void test_shift_recovery() {
    const Image ref = make_die(96, 96);
    const Image moved = regi::shift_image(ref, 5.0, -3.0);
    const regi::Shift s = regi::phase_correlate(ref, moved);
    std::cout << "shift dx=" << s.dx << " dy=" << s.dy << " resp=" << s.response << "\n";
    CHECK(std::fabs(s.dx - 5.0) < 0.7);
    CHECK(std::fabs(s.dy + 3.0) < 0.7);
}

static void test_diff_finds_defect() {
    const Image ref = make_die(96, 96);
    // The test die is shifted AND has a defect the reference lacks.
    Image test = regi::shift_image(ref, 4.0, 2.0);
    stamp_defect(test, 60, 40, 4, 255.0f);
    const regi::DiffResult r = regi::diff_inspect(ref, test);
    std::cout << "diff: shift(" << r.shift.dx << "," << r.shift.dy << ") thr="
              << r.threshold << " defects=" << r.defect_count << "\n";
    CHECK(std::fabs(r.shift.dx - 4.0) < 0.8);
    CHECK(std::fabs(r.shift.dy - 2.0) < 0.8);
    CHECK(r.defect_count >= 1);
    // The largest flagged blob sits near the injected defect.
    bool near = false;
    for (const auto& c : r.defects)
        if (std::fabs(c.cx - 60.0) < 6.0 && std::fabs(c.cy - 40.0) < 6.0) near = true;
    CHECK(near);

    ColorImage ann = regi::annotate_diff(test, r);
    CHECK(ann.width() == 96 && ann.height() == 96 && !ann.empty());
}

static void test_clean_pair_is_quiet() {
    const Image ref = make_die(96, 96);
    const Image test = regi::shift_image(ref, 2.0, -1.0);  // aligned, no defect
    const regi::DiffResult r = regi::diff_inspect(ref, test);
    std::cout << "clean pair defects=" << r.defect_count << "\n";
    CHECK(r.defect_count == 0);
}

int main() {
    test_shift_recovery();
    test_diff_finds_defect();
    test_clean_pair_is_quiet();
    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "REGI OK\n";
    return 0;
}
