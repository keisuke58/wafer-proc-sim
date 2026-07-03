#include "wproc/threshold.hpp"

#include <array>
#include <cmath>

namespace wproc {

namespace {
int bin_of(float v) {
    int b = static_cast<int>(std::lround(v));
    if (b < 0) b = 0;
    if (b > 255) b = 255;
    return b;
}
}  // namespace

float otsu_threshold(const Image& src) {
    std::array<long, 256> hist{};
    for (std::size_t i = 0; i < src.size(); ++i)
        ++hist[bin_of(src.data()[i])];

    const long total = static_cast<long>(src.size());
    if (total == 0) return 127.5f;

    double sum_all = 0.0;
    for (int t = 0; t < 256; ++t) sum_all += static_cast<double>(t) * hist[t];

    double sum_bg = 0.0;
    long w_bg = 0;
    double best_var = -1.0;
    int best_t = 127;

    for (int t = 0; t < 256; ++t) {
        w_bg += hist[t];
        if (w_bg == 0) continue;
        long w_fg = total - w_bg;
        if (w_fg == 0) break;

        sum_bg += static_cast<double>(t) * hist[t];
        double mean_bg = sum_bg / w_bg;
        double mean_fg = (sum_all - sum_bg) / w_fg;
        double diff = mean_bg - mean_fg;
        double between = static_cast<double>(w_bg) * w_fg * diff * diff;
        if (between > best_var) {
            best_var = between;
            best_t = t;
        }
    }
    return static_cast<float>(best_t);
}

Image binarize(const Image& src, float thr, bool invert) {
    Image out(src.width(), src.height());
    for (std::size_t i = 0; i < src.size(); ++i) {
        bool fg = invert ? (src.data()[i] <= thr) : (src.data()[i] > thr);
        out.data()[i] = fg ? 255.0f : 0.0f;
    }
    return out;
}

}  // namespace wproc
