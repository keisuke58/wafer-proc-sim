// test_align.cpp — NCC template matching: shift recovery, tilt, rejection.
#include <cmath>
#include <iostream>
#include <vector>

#include "wproc/align.hpp"
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

constexpr int TW = 48, TH = 48;

// A distinctive constellation of bright blobs, so both halves are unambiguous.
static const int BLOBS[][2] = {{6, 8}, {30, 6}, {12, 34}, {38, 28}, {22, 20}};

// Stamp the constellation into `img` with its top-left at (px, py); columns in
// the right half get an extra `tilt` vertical shift (to synthesize street tilt).
static void stamp(Image& img, int px, int py, int tilt = 0) {
    const int n_blobs = static_cast<int>(sizeof(BLOBS) / sizeof(BLOBS[0]));
    for (int bi = 0; bi < n_blobs; ++bi) {
        const int bx = BLOBS[bi][0], by = BLOBS[bi][1];
        const int extra = (bx >= TW / 2) ? tilt : 0;
        for (int dy = 0; dy < 5; ++dy)
            for (int dx = 0; dx < 5; ++dx) {
                const int x = px + bx + dx, y = py + by + dy + extra;
                if (x >= 0 && x < img.width() && y >= 0 && y < img.height())
                    img.at(x, y) = 220.0f;
            }
    }
}

static Image make_template() {
    Image t(TW, TH, 40.0f);
    stamp(t, 0, 0);
    return t;
}

static void test_exact_and_shift() {
    const Image templ = make_template();

    // Scene with the pattern at (80, 60).
    Image a(220, 180, 40.0f);
    stamp(a, 80, 60);
    align::Match m = align::match_ncc(a, templ, 0, 0, a.width(), a.height());
    std::cout << "exact: (" << m.x << "," << m.y << ") score=" << m.score << "\n";
    CHECK(m.found);
    CHECK(m.score > 0.95);
    CHECK(std::fabs(m.x - 80.0) < 0.6 && std::fabs(m.y - 60.0) < 0.6);

    // Shifted by (+15, -8): pattern top-left at (95, 52).
    Image b(220, 180, 40.0f);
    stamp(b, 95, 52);
    align::Registration r = align::register_pattern(b, templ, 80, 60, 30);
    std::cout << "shift: dx=" << r.dx << " dy=" << r.dy << " ang=" << r.angle_deg
              << " score=" << r.score << " ok=" << r.ok << "\n";
    CHECK(r.ok);
    CHECK(std::fabs(r.dx - 15.0) < 0.6);
    CHECK(std::fabs(r.dy + 8.0) < 0.6);
    CHECK(std::fabs(r.angle_deg) < 3.0);   // no false tilt for a pure shift
}

static void test_tilt_and_reject() {
    const Image templ = make_template();

    // Right half shifted down by 4 px → tilt = atan2(4, 24) ≈ 9.5°.
    Image t(220, 180, 40.0f);
    stamp(t, 90, 70, /*tilt=*/4);
    align::Registration r = align::register_pattern(t, templ, 80, 60, 30);
    std::cout << "tilt: angle=" << r.angle_deg << "\n";
    CHECK(r.ok);
    CHECK(r.angle_deg > 5.0 && r.angle_deg < 14.0);

    // A scene without the pattern → low score, not ok.
    Image empty(120, 100, 40.0f);
    align::Registration re = align::register_pattern(empty, templ, 50, 40, 20);
    CHECK(!re.ok);

    // Annotated overlay keeps the image size.
    Image a(160, 120, 40.0f);
    stamp(a, 60, 40);
    align::Match m = align::match_ncc(a, templ, 0, 0, a.width(), a.height());
    ColorImage ann = align::annotate_match(a, m, TW, TH);
    CHECK(ann.width() == 160 && ann.height() == 120 && !ann.empty());
}

int main() {
    test_exact_and_shift();
    test_tilt_and_reject();
    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "ALIGN OK\n";
    return 0;
}
