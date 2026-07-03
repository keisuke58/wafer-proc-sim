// test_grind_surface.cpp — grinding-mark orientation, roughness, SSD index.
#include <cmath>
#include <iostream>
#include <random>

#include "wproc/grind_surface.hpp"
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

constexpr double kPi = 3.14159265358979323846;

// Sinusoidal stripes. axis=0 -> value varies along x (vertical marks);
// axis=1 -> value varies along y (horizontal marks).
static Image stripes(int n, double amp, double period, int axis) {
    Image img(n, n, 128.0f);
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
            const double t = (axis == 0 ? x : y);
            img.at(x, y) = static_cast<float>(128.0 + amp * std::sin(2 * kPi * t / period));
        }
    return img;
}

static void test_marks_orientation() {
    // Horizontal marks: intensity varies along y -> gradient in y.
    grind::SurfaceQuality h = grind::analyze_surface(stripes(128, 50.0, 8.0, 1));
    std::cout << "h-marks: coh=" << h.coherence << " angle=" << h.mark_angle_deg
              << " sev=" << h.mark_severity << " Ra=" << h.roughness_ra << "\n";
    CHECK(h.coherence > 0.7);
    CHECK(h.mark_angle_deg < 15.0 || h.mark_angle_deg > 165.0);  // ~horizontal
    CHECK(h.mark_severity > 0.35);   // strong marks fail the default spec
    CHECK(!h.ok);

    // Vertical marks: intensity varies along x -> mark direction ~90 deg.
    grind::SurfaceQuality v = grind::analyze_surface(stripes(128, 50.0, 8.0, 0));
    std::cout << "v-marks: coh=" << v.coherence << " angle=" << v.mark_angle_deg << "\n";
    CHECK(v.coherence > 0.7);
    CHECK(std::fabs(v.mark_angle_deg - 90.0) < 15.0);  // ~vertical
}

static void test_isotropic_smooth() {
    // Isotropic noise: no dominant orientation -> low coherence.
    Image noise(128, 128, 128.0f);
    std::mt19937 gen(5);
    std::normal_distribution<double> nd(0.0, 8.0);
    for (int y = 0; y < 128; ++y)
        for (int x = 0; x < 128; ++x)
            noise.at(x, y) = static_cast<float>(128.0 + nd(gen));
    grind::SurfaceQuality q = grind::analyze_surface(noise);
    std::cout << "noise: coh=" << q.coherence << " ssd=" << q.ssd_index << "\n";
    CHECK(q.coherence < 0.35);

    // A smooth low-frequency ramp: negligible roughness / SSD -> passes spec.
    Image ramp(128, 128, 0.0f);
    for (int y = 0; y < 128; ++y)
        for (int x = 0; x < 128; ++x)
            ramp.at(x, y) = static_cast<float>(100.0 + 0.1 * x);
    grind::SurfaceQuality r = grind::analyze_surface(ramp);
    std::cout << "ramp: Ra=" << r.roughness_ra << " sev=" << r.mark_severity
              << " ssd=" << r.ssd_index << " ok=" << r.ok << "\n";
    CHECK(r.roughness_ra < 1.0);
    CHECK(r.mark_severity < 0.1);
    CHECK(r.ok);
}

static void test_roughness_ordering_and_render() {
    // Rougher (higher-amplitude, higher-frequency) surface -> larger Ra & SSD.
    grind::SurfaceQuality mild = grind::analyze_surface(stripes(128, 8.0, 16.0, 1));
    grind::SurfaceQuality harsh = grind::analyze_surface(stripes(128, 60.0, 4.0, 1));
    CHECK(harsh.roughness_ra > mild.roughness_ra);
    CHECK(harsh.ssd_index > mild.ssd_index);

    ColorImage heat = grind::render_ssd_heatmap(stripes(96, 40.0, 6.0, 1));
    CHECK(heat.width() == 96 && heat.height() == 96);
    CHECK(!heat.empty());
}

int main() {
    test_marks_orientation();
    test_isotropic_smooth();
    test_roughness_ordering_and_render();
    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "GRIND SURFACE OK\n";
    return 0;
}
