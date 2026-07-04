// test_pkg.cpp — bump-array inspection: count, pitch, missing, coplanarity.
#include <cmath>
#include <iostream>
#include <vector>

#include "wproc/image.hpp"
#include "wproc/pkg.hpp"

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

// 6x6 bump grid (pitch 20 px, radius 5). `drop` removes interior bumps and
// `dim` dims a few (lower brightness → worse coplanarity).
static Image make_array(bool drop, bool dim) {
    Image img(170, 170, 40.0f);
    const int p = 20, r = 5, x0 = 25, y0 = 25;
    for (int j = 0; j < 6; ++j)
        for (int i = 0; i < 6; ++i) {
            if (drop && ((i == 2 && j == 3) || (i == 3 && j == 2))) continue;  // missing
            float val = 220.0f;
            if (dim && ((i + j) % 5 == 0)) val = 160.0f;                        // short bump
            const int cx = x0 + i * p, cy = y0 + j * p;
            for (int dy = -r; dy <= r; ++dy)
                for (int dx = -r; dx <= r; ++dx)
                    if (dx * dx + dy * dy <= r * r) img.at(cx + dx, cy + dy) = val;
        }
    return img;
}

static void test_good_array() {
    Image img = make_array(/*drop=*/false, /*dim=*/false);
    pkg::BumpSpec spec;  // 10 um target, pitch inferred
    pkg::BumpResult r = pkg::inspect_bumps(img, spec);
    std::cout << "good: n=" << r.count << " pitch=" << r.pitch_um << " dCV=" << r.diameter_cv
              << " copl=" << r.coplanarity << " miss=" << r.missing
              << " place=" << r.placement_rms_um << " ok=" << r.ok << "\n";
    CHECK(r.count == 36);
    CHECK(std::fabs(r.pitch_um - 20.0) < 2.0);
    CHECK(std::fabs(r.mean_diameter_um - 10.0) < 2.0);
    CHECK(r.diameter_cv < 0.12);
    CHECK(r.coplanarity > 0.95);
    CHECK(r.missing == 0);
    CHECK(r.placement_rms_um < 1.5);
    CHECK(r.ok);
}

static void test_defective_array() {
    Image good = make_array(false, false);
    pkg::BumpResult gr = pkg::inspect_bumps(good);

    Image bad = make_array(/*drop=*/true, /*dim=*/true);
    pkg::BumpResult r = pkg::inspect_bumps(bad);
    std::cout << "bad: n=" << r.count << " miss=" << r.missing
              << " copl=" << r.coplanarity << " ok=" << r.ok << "\n";
    CHECK(r.count == 34);              // two removed
    CHECK(r.missing == 2);
    CHECK(r.coplanarity < gr.coplanarity);   // dimmer bumps hurt coplanarity
    CHECK(!r.ok);

    ColorImage ann = pkg::annotate_bumps(bad, r);
    CHECK(ann.width() == bad.width() && ann.height() == bad.height() && !ann.empty());
}

int main() {
    test_good_array();
    test_defective_array();
    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "PKG OK\n";
    return 0;
}
