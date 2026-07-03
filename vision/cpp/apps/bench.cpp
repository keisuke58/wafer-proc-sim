// bench.cpp — throughput benchmark for the wproc filters.
//
// Times the scalar, AVX2, and multithreaded implementations of Gaussian blur
// and Sobel on a large synthetic image and reports megapixels/second plus the
// equivalent camera frame rate at a target resolution. Establishes the
// "real-time image-processing" numbers for the inspection pipeline.
//
// Usage: bench [--width W] [--height H] [--sigma S] [--iters N] [--threads T]
#include <chrono>
#include <cstdio>
#include <functional>
#include <vector>

#include "cli_args.hpp"
#include "wproc/filters.hpp"
#include "wproc/image.hpp"
#include "wproc/simd_filters.hpp"

using namespace wproc;
using wproc_cli::arg_double;

namespace {

Image make_image(int W, int H) {
    Image img(W, H, 150.0f);
    for (int y = 0; y < H; ++y)
        for (int x = W / 3; x < 2 * W / 3; ++x) img.at(x, y) = 40.0f;
    return img;
}

volatile float g_sink = 0.0f;  // keeps results from being optimized away

// Median wall-clock seconds of `iters` runs of fn (fn returns a checksum).
double time_median(int iters, const std::function<float()>& fn) {
    std::vector<double> ts;
    ts.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        g_sink = fn();
        auto t1 = std::chrono::steady_clock::now();
        ts.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
}

float sink_gx(const Gradient& g) { return g.gx.data()[g.gx.size() / 2]; }
float sink_img(const Image& im) { return im.data()[im.size() / 2]; }

void report(const char* name, double secs, double mpix) {
    double mpps = mpix / secs;
    // frame rate at a typical 2448x2048 (5 MP) line-scan/area camera frame.
    double fps_5mp = mpps / 5.0;
    std::printf("  %-26s %8.2f ms   %8.1f MPix/s   %7.1f fps@5MP\n", name,
                secs * 1e3, mpps, fps_5mp);
}

}  // namespace

int main(int argc, char** argv) {
    const int W = static_cast<int>(arg_double(argc, argv, "--width", 2048));
    const int H = static_cast<int>(arg_double(argc, argv, "--height", 2048));
    const float sigma = static_cast<float>(arg_double(argc, argv, "--sigma", 1.2));
    const int iters = static_cast<int>(arg_double(argc, argv, "--iters", 7));
    const int threads = static_cast<int>(arg_double(argc, argv, "--threads", 0));

    Image img = make_image(W, H);
    const double mpix = static_cast<double>(W) * H / 1e6;

    std::printf("wproc benchmark — %dx%d (%.1f MPix), sigma=%.2f, iters=%d, "
                "AVX2=%s\n",
                W, H, mpix, sigma, iters, simd::avx2_enabled() ? "on" : "off");

    std::printf("Gaussian blur:\n");
    report("scalar", time_median(iters, [&] { return sink_img(gaussian_blur(img, sigma)); }), mpix);
    report("avx2 (1 thread)", time_median(iters, [&] { return sink_img(simd::gaussian_blur(img, sigma)); }), mpix);
    report("avx2 + threads", time_median(iters, [&] { return sink_img(simd::gaussian_blur_mt(img, sigma, threads)); }), mpix);

    std::printf("Sobel 3x3:\n");
    report("scalar", time_median(iters, [&] { return sink_gx(sobel(img)); }), mpix);
    report("avx2 (1 thread)", time_median(iters, [&] { return sink_gx(simd::sobel(img)); }), mpix);
    report("avx2 + threads", time_median(iters, [&] { return sink_gx(simd::sobel_mt(img, threads)); }), mpix);
    return 0;
}
