#include "wproc/filters.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace wproc {

Image gaussian_blur(const Image& src, float sigma) {
    if (sigma <= 0.0f)
        throw std::invalid_argument("gaussian_blur: sigma must be > 0");
    if (src.empty()) return src;

    const int radius = static_cast<int>(std::ceil(3.0f * sigma));
    std::vector<float> kernel(2 * radius + 1);
    float sum = 0.0f;
    const float inv2s2 = 1.0f / (2.0f * sigma * sigma);
    for (int k = -radius; k <= radius; ++k) {
        float w = std::exp(-static_cast<float>(k) * k * inv2s2);
        kernel[k + radius] = w;
        sum += w;
    }
    std::transform(kernel.begin(), kernel.end(), kernel.begin(),
                   [sum](float w) { return w / sum; });  // normalize

    const int W = src.width(), H = src.height();

    // Horizontal pass -> tmp
    Image tmp(W, H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float acc = 0.0f;
            for (int k = -radius; k <= radius; ++k)
                acc += kernel[k + radius] * src.at_clamped(x + k, y);
            tmp.at(x, y) = acc;
        }

    // Vertical pass -> out
    Image out(W, H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float acc = 0.0f;
            for (int k = -radius; k <= radius; ++k)
                acc += kernel[k + radius] * tmp.at_clamped(x, y + k);
            out.at(x, y) = acc;
        }
    return out;
}

Gradient sobel(const Image& src) {
    const int W = src.width(), H = src.height();
    Gradient g{Image(W, H), Image(W, H)};
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float tl = src.at_clamped(x - 1, y - 1);
            float tc = src.at_clamped(x, y - 1);
            float tr = src.at_clamped(x + 1, y - 1);
            float ml = src.at_clamped(x - 1, y);
            float mr = src.at_clamped(x + 1, y);
            float bl = src.at_clamped(x - 1, y + 1);
            float bc = src.at_clamped(x, y + 1);
            float br = src.at_clamped(x + 1, y + 1);
            g.gx.at(x, y) = (tr + 2.0f * mr + br) - (tl + 2.0f * ml + bl);
            g.gy.at(x, y) = (bl + 2.0f * bc + br) - (tl + 2.0f * tc + tr);
        }
    return g;
}

Image gradient_magnitude(const Gradient& g) {
    const int W = g.gx.width(), H = g.gx.height();
    if (g.gy.width() != W || g.gy.height() != H)
        throw std::invalid_argument("gradient_magnitude: gx/gy size mismatch");
    Image out(W, H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float dx = g.gx.at(x, y), dy = g.gy.at(x, y);
            out.at(x, y) = std::sqrt(dx * dx + dy * dy);
        }
    return out;
}

}  // namespace wproc
