// test_simd_parity.cpp — SIMD/threaded filters must match the scalar path.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

#include "wproc/filters.hpp"
#include "wproc/image.hpp"
#include "wproc/simd_filters.hpp"

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

static float max_diff(const Image& a, const Image& b) {
    float w = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i)
        w = std::max(w, std::fabs(a.data()[i] - b.data()[i]));
    return w;
}

static Image noisy(int W, int H, unsigned seed) {
    Image img(W, H, 150.0f);
    for (int y = 0; y < H; ++y)
        for (int x = W / 3; x < 2 * W / 3; ++x) img.at(x, y) = 40.0f;
    std::mt19937 gen(seed);
    std::normal_distribution<float> n(0.0f, 6.0f);
    for (std::size_t i = 0; i < img.size(); ++i) img.data()[i] += n(gen);
    return img;
}

int main() {
    std::cout << "AVX2 code path: " << (simd::avx2_enabled() ? "yes" : "no")
              << "\n";

    // Non-multiple-of-8 dimensions stress the SIMD remainder/border handling.
    for (int W : {37, 64, 200, 253}) {
        Image img = noisy(W, 91, static_cast<unsigned>(W));
        for (float sigma : {0.8f, 1.2f, 2.5f}) {
            Image ref = gaussian_blur(img, sigma);
            float d1 = max_diff(ref, simd::gaussian_blur(img, sigma));
            float d2 = max_diff(ref, simd::gaussian_blur_mt(img, sigma, 4));
            CHECK(d1 < 1e-3f);
            CHECK(d2 < 1e-3f);
        }
        Gradient ref = sobel(img);
        Gradient s1 = simd::sobel(img);
        Gradient s4 = simd::sobel_mt(img, 4);
        CHECK(max_diff(ref.gx, s1.gx) < 1e-3f);
        CHECK(max_diff(ref.gy, s1.gy) < 1e-3f);
        CHECK(max_diff(ref.gx, s4.gx) < 1e-3f);
        CHECK(max_diff(ref.gy, s4.gy) < 1e-3f);
    }

    std::cout << (g_total - g_failed) << "/" << g_total << " parity checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "SIMD PARITY OK\n";
    return 0;
}
