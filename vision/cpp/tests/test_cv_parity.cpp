// test_cv_parity.cpp — cross-validate the from-scratch kernels against OpenCV.
//
// Built only when OpenCV is available (WPROC_WITH_OPENCV). Asserts that the
// hand-written Gaussian, Sobel, and Otsu implementations agree with OpenCV's
// on a realistic synthetic kerf image — interior pixels for the convolutions
// (border policies differ in ULP-level details), 1 bin of slack for Otsu.
#include <cmath>
#include <iostream>
#include <random>

#include "wproc/cv_backend.hpp"
#include "wproc/filters.hpp"
#include "wproc/image.hpp"
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

// Synthetic kerf scene with noise — same structure the inspector sees.
static Image make_scene() {
    Image img(200, 120, 150.0f);
    for (int y = 0; y < img.height(); ++y)
        for (int x = 80; x <= 120; ++x) img.at(x, y) = 40.0f;
    std::mt19937 gen(7);
    std::normal_distribution<float> noise(0.0f, 5.0f);
    for (std::size_t i = 0; i < img.size(); ++i) img.data()[i] += noise(gen);
    return img;
}

// Max |a-b| over the interior (skip `margin` pixels at each border where
// replicate-border conventions can differ between implementations).
static float max_interior_diff(const Image& a, const Image& b, int margin) {
    float worst = 0.0f;
    for (int y = margin; y < a.height() - margin; ++y)
        for (int x = margin; x < a.width() - margin; ++x)
            worst = std::max(worst, std::abs(a.at(x, y) - b.at(x, y)));
    return worst;
}

int main() {
    Image scene = make_scene();

    // Gaussian: same sigma, same radius rule; interior must agree tightly
    // relative to the 0..255 signal range.
    for (float sigma : {0.8f, 1.2f, 2.5f}) {
        Image ours = gaussian_blur(scene, sigma);
        Image cvs = cv_backend::gaussian_blur(scene, sigma);
        int margin = static_cast<int>(std::ceil(3.0f * sigma)) + 1;
        float d = max_interior_diff(ours, cvs, margin);
        CHECK(d < 0.05f);  // < 0.02% of full scale
        std::cout << "gaussian sigma=" << sigma << " max|diff|=" << d << "\n";
    }

    // Sobel 3x3: identical definition; interior should match to float noise.
    {
        Gradient ours = sobel(scene);
        Gradient cvs = cv_backend::sobel(scene);
        float dx = max_interior_diff(ours.gx, cvs.gx, 2);
        float dy = max_interior_diff(ours.gy, cvs.gy, 2);
        CHECK(dx < 1e-3f);
        CHECK(dy < 1e-3f);
        std::cout << "sobel max|diff| gx=" << dx << " gy=" << dy << "\n";
    }

    // Otsu: identical histogram method; allow 1 bin for rounding differences.
    {
        float ours = otsu_threshold(scene);
        float cvs = cv_backend::otsu_threshold(scene);
        CHECK(std::abs(ours - cvs) <= 1.0f);
        std::cout << "otsu ours=" << ours << " opencv=" << cvs << "\n";
    }

    std::cout << (g_total - g_failed) << "/" << g_total
              << " parity checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " parity check(s) FAILED\n";
        return 1;
    }
    std::cout << "OPENCV PARITY OK\n";
    return 0;
}
