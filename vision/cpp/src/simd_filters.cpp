#include "wproc/simd_filters.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace wproc::simd {

namespace {

std::vector<float> gaussian_kernel(float sigma, int& radius) {
    radius = static_cast<int>(std::ceil(3.0f * sigma));
    std::vector<float> k(2 * radius + 1);
    float sum = 0.0f;
    const float inv2s2 = 1.0f / (2.0f * sigma * sigma);
    for (int i = -radius; i <= radius; ++i) {
        float w = std::exp(-static_cast<float>(i) * i * inv2s2);
        k[i + radius] = w;
        sum += w;
    }
    std::transform(k.begin(), k.end(), k.begin(),
                   [sum](float w) { return w / sum; });
    return k;
}

void check_sigma(float sigma) {
    if (sigma <= 0.0f)
        throw std::invalid_argument("simd gaussian_blur: sigma must be > 0");
    if (!(sigma <= 1000.0f))
        throw std::invalid_argument("simd gaussian_blur: sigma too large");
}

// ── Horizontal Gaussian pass over rows [y0, y1) ─────────────────────────────
// out[y,x] = sum_i k[i] * src(x+i, y) with x clamped to [0, W).
void gauss_h_rows(const float* in, float* out, int W, int /*H*/, const float* k,
                  int radius, int y0, int y1) {
    // Interior [lo, hi): every tap x+i lands in [0, W), so no clamping needed.
    const int lo = std::min(radius, W);
    const int hi = std::max(lo, W - radius);
    auto scalar_at = [&](const float* row, int x) {
        float acc = 0.0f;
        for (int i = -radius; i <= radius; ++i) {
            int xx = x + i;
            xx = xx < 0 ? 0 : (xx >= W ? W - 1 : xx);
            acc += k[i + radius] * row[xx];
        }
        return acc;
    };
    for (int y = y0; y < y1; ++y) {
        const float* row = in + static_cast<std::size_t>(y) * W;
        float* orow = out + static_cast<std::size_t>(y) * W;
        for (int x = 0; x < lo; ++x) orow[x] = scalar_at(row, x);  // left border
        int x = lo;
#if defined(__AVX2__)
        for (; x + 8 <= hi; x += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int i = -radius; i <= radius; ++i) {
                __m256 kk = _mm256_set1_ps(k[i + radius]);
                __m256 v = _mm256_loadu_ps(row + x + i);
                acc = _mm256_fmadd_ps(kk, v, acc);
            }
            _mm256_storeu_ps(orow + x, acc);
        }
#endif
        for (; x < hi; ++x) {  // interior remainder (taps in-bounds)
            float acc = 0.0f;
            for (int i = -radius; i <= radius; ++i)
                acc += k[i + radius] * row[x + i];
            orow[x] = acc;
        }
        for (; x < W; ++x) orow[x] = scalar_at(row, x);  // right border
    }
}

// ── Vertical Gaussian pass over rows [y0, y1) ───────────────────────────────
// out[y,x] = sum_i k[i] * src(x, y+i) with y clamped. Vectorized across x.
void gauss_v_rows(const float* in, float* out, int W, int H, const float* k,
                  int radius, int y0, int y1) {
    for (int y = y0; y < y1; ++y) {
        float* orow = out + static_cast<std::size_t>(y) * W;
        int x = 0;
#if defined(__AVX2__)
        for (; x + 8 <= W; x += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int i = -radius; i <= radius; ++i) {
                int yy = y + i;
                yy = yy < 0 ? 0 : (yy >= H ? H - 1 : yy);
                __m256 kk = _mm256_set1_ps(k[i + radius]);
                __m256 v = _mm256_loadu_ps(in + static_cast<std::size_t>(yy) * W + x);
                acc = _mm256_fmadd_ps(kk, v, acc);
            }
            _mm256_storeu_ps(orow + x, acc);
        }
#endif
        for (; x < W; ++x) {
            float acc = 0.0f;
            for (int i = -radius; i <= radius; ++i) {
                int yy = y + i;
                yy = yy < 0 ? 0 : (yy >= H ? H - 1 : yy);
                acc += k[i + radius] * in[static_cast<std::size_t>(yy) * W + x];
            }
            orow[x] = acc;
        }
    }
}

// ── Sobel over rows [y0, y1) ────────────────────────────────────────────────
void sobel_rows(const float* in, float* gx, float* gy, int W, int H, int y0,
                int y1) {
    auto cl = [](int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); };
    for (int y = y0; y < y1; ++y) {
        int ym = cl(y - 1, H - 1), yp = cl(y + 1, H - 1);
        const float* r0 = in + static_cast<std::size_t>(ym) * W;
        const float* r1 = in + static_cast<std::size_t>(y) * W;
        const float* r2 = in + static_cast<std::size_t>(yp) * W;
        float* ogx = gx + static_cast<std::size_t>(y) * W;
        float* ogy = gy + static_cast<std::size_t>(y) * W;
        int x = 0;
        // x=0 border scalar
        {
            int xm = 0, xp = std::min(1, W - 1);
            ogx[0] = (r0[xp] + 2 * r1[xp] + r2[xp]) - (r0[xm] + 2 * r1[xm] + r2[xm]);
            ogy[0] = (r2[xm] + 2 * r2[0] + r2[xp]) - (r0[xm] + 2 * r0[0] + r0[xp]);
        }
        x = 1;
#if defined(__AVX2__)
        const __m256 two = _mm256_set1_ps(2.0f);
        for (; x + 8 <= W - 1; x += 8) {
            __m256 tl = _mm256_loadu_ps(r0 + x - 1), tc = _mm256_loadu_ps(r0 + x),
                   tr = _mm256_loadu_ps(r0 + x + 1);
            __m256 ml = _mm256_loadu_ps(r1 + x - 1), mr = _mm256_loadu_ps(r1 + x + 1);
            __m256 bl = _mm256_loadu_ps(r2 + x - 1), bc = _mm256_loadu_ps(r2 + x),
                   br = _mm256_loadu_ps(r2 + x + 1);
            __m256 gxr = _mm256_sub_ps(
                _mm256_add_ps(_mm256_add_ps(tr, br), _mm256_mul_ps(two, mr)),
                _mm256_add_ps(_mm256_add_ps(tl, bl), _mm256_mul_ps(two, ml)));
            __m256 gyr = _mm256_sub_ps(
                _mm256_add_ps(_mm256_add_ps(bl, br), _mm256_mul_ps(two, bc)),
                _mm256_add_ps(_mm256_add_ps(tl, tr), _mm256_mul_ps(two, tc)));
            _mm256_storeu_ps(ogx + x, gxr);
            _mm256_storeu_ps(ogy + x, gyr);
        }
#endif
        for (; x < W - 1; ++x) {
            ogx[x] = (r0[x + 1] + 2 * r1[x + 1] + r2[x + 1]) -
                     (r0[x - 1] + 2 * r1[x - 1] + r2[x - 1]);
            ogy[x] = (r2[x - 1] + 2 * r2[x] + r2[x + 1]) -
                     (r0[x - 1] + 2 * r0[x] + r0[x + 1]);
        }
        if (W > 1) {  // x = W-1 border scalar
            int xr = W - 1, xm = W - 2, xp = W - 1;
            ogx[xr] = (r0[xp] + 2 * r1[xp] + r2[xp]) - (r0[xm] + 2 * r1[xm] + r2[xm]);
            ogy[xr] = (r2[xm] + 2 * r2[xr] + r2[xp]) - (r0[xm] + 2 * r0[xr] + r0[xp]);
        }
    }
}

int resolve_threads(int threads, int H) {
    if (threads <= 0) {
        threads = static_cast<int>(std::thread::hardware_concurrency());
        if (threads <= 0) threads = 1;
    }
    return std::max(1, std::min(threads, H));
}

template <class Fn>
void run_tiled(int H, int threads, Fn&& fn) {
    threads = resolve_threads(threads, H);
    if (threads == 1) {
        fn(0, H);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(threads);
    int chunk = (H + threads - 1) / threads;
    for (int t = 0; t < threads; ++t) {
        int y0 = t * chunk, y1 = std::min(H, y0 + chunk);
        if (y0 >= y1) break;
        pool.emplace_back([&fn, y0, y1] { fn(y0, y1); });
    }
    for (auto& th : pool) th.join();
}

Image gaussian_impl(const Image& src, float sigma, int threads) {
    check_sigma(sigma);
    if (src.empty()) return src;
    int radius;
    auto k = gaussian_kernel(sigma, radius);
    const int W = src.width(), H = src.height();
    Image tmp(W, H), out(W, H);
    run_tiled(H, threads, [&](int y0, int y1) {
        gauss_h_rows(src.data(), tmp.data(), W, H, k.data(), radius, y0, y1);
    });
    run_tiled(H, threads, [&](int y0, int y1) {
        gauss_v_rows(tmp.data(), out.data(), W, H, k.data(), radius, y0, y1);
    });
    return out;
}

Gradient sobel_impl(const Image& src, int threads) {
    const int W = src.width(), H = src.height();
    Gradient g{Image(W, H), Image(W, H)};
    if (src.empty()) return g;
    run_tiled(H, threads, [&](int y0, int y1) {
        sobel_rows(src.data(), g.gx.data(), g.gy.data(), W, H, y0, y1);
    });
    return g;
}

}  // namespace

bool avx2_enabled() {
#if defined(__AVX2__)
    return true;
#else
    return false;
#endif
}

Image gaussian_blur(const Image& src, float sigma) {
    return gaussian_impl(src, sigma, 1);
}
Gradient sobel(const Image& src) { return sobel_impl(src, 1); }

Image gaussian_blur_mt(const Image& src, float sigma, int threads) {
    return gaussian_impl(src, sigma, threads);
}
Gradient sobel_mt(const Image& src, int threads) {
    return sobel_impl(src, threads);
}

}  // namespace wproc::simd
