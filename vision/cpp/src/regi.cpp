// regi.cpp — phase-correlation registration + difference inspection.
#include "wproc/regi.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "wproc/draw.hpp"
#include "wproc/morphology.hpp"
#include "wproc/spindle.hpp"  // reuse the in-house radix-2 FFT

namespace wproc::regi {
namespace {

using Cplx = std::complex<double>;

int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// 2-D FFT (in place, row-major, size W*H both powers of two) via 1-D FFTs over
// rows then columns using the spindle radix-2 transform.
void fft2(std::vector<Cplx>& a, int W, int H, bool inverse) {
    std::vector<Cplx> line;
    line.resize(static_cast<std::size_t>(std::max(W, H)));
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) line[x] = a[static_cast<std::size_t>(y) * W + x];
        std::vector<Cplx> row(line.begin(), line.begin() + W);
        spindle::fft(row, inverse);
        for (int x = 0; x < W; ++x) a[static_cast<std::size_t>(y) * W + x] = row[x];
    }
    for (int x = 0; x < W; ++x) {
        std::vector<Cplx> col(static_cast<std::size_t>(H));
        for (int y = 0; y < H; ++y) col[y] = a[static_cast<std::size_t>(y) * W + x];
        spindle::fft(col, inverse);
        for (int y = 0; y < H; ++y) a[static_cast<std::size_t>(y) * W + x] = col[y];
    }
}

// Pack an image into a padded complex buffer with a raised-cosine (Hann) window
// so the spectral leakage from the non-periodic borders does not swamp the peak.
std::vector<Cplx> pack(const Image& im, int W, int H) {
    std::vector<Cplx> buf(static_cast<std::size_t>(W) * H, Cplx(0.0, 0.0));
    const int w = im.width(), h = im.height();
    for (int y = 0; y < h; ++y) {
        const double wy = 0.5 - 0.5 * std::cos(2.0 * M_PI * y / std::max(1, h - 1));
        for (int x = 0; x < w; ++x) {
            const double wx = 0.5 - 0.5 * std::cos(2.0 * M_PI * x / std::max(1, w - 1));
            buf[static_cast<std::size_t>(y) * W + x] = Cplx(im.at(x, y) * wx * wy, 0.0);
        }
    }
    return buf;
}

// Parabolic sub-pixel offset from three samples around a peak.
double parab(double a, double b, double c) {
    const double denom = a - 2.0 * b + c;
    if (std::fabs(denom) < 1e-12) return 0.0;
    return 0.5 * (a - c) / denom;
}

}  // namespace

Shift phase_correlate(const Image& ref, const Image& img) {
    Shift s;
    if (ref.empty() || img.empty()) return s;
    const int W = next_pow2(std::max(ref.width(), img.width()));
    const int H = next_pow2(std::max(ref.height(), img.height()));

    std::vector<Cplx> F = pack(ref, W, H);
    std::vector<Cplx> G = pack(img, W, H);
    fft2(F, W, H, false);
    fft2(G, W, H, false);

    // Cross-power spectrum R = F * conj(G) / |F * conj(G)|.
    std::vector<Cplx> R(F.size());
    for (std::size_t i = 0; i < F.size(); ++i) {
        const Cplx cp = F[i] * std::conj(G[i]);
        const double mag = std::abs(cp);
        R[i] = mag > 1e-12 ? cp / mag : Cplx(0.0, 0.0);
    }
    fft2(R, W, H, true);  // inverse: peak at the shift

    // Locate the peak of the real part.
    int px = 0, py = 0;
    double peak = -1e300, sum = 0.0;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const double v = R[static_cast<std::size_t>(y) * W + x].real();
            sum += std::fabs(v);
            if (v > peak) { peak = v; px = x; py = y; }
        }

    auto real_at = [&](int x, int y) {
        x = (x % W + W) % W; y = (y % H + H) % H;
        return R[static_cast<std::size_t>(y) * W + x].real();
    };
    const double sub_x = parab(real_at(px - 1, py), peak, real_at(px + 1, py));
    const double sub_y = parab(real_at(px, py - 1), peak, real_at(px, py + 1));

    // Unwrap the circular peak index to a signed shift. The peak of
    // IFFT(F·conj(G)) sits at the shift that maps `img` onto `ref`; negate it so
    // a positive dx means `img`'s content lies dx px to the right of `ref`.
    double dx = px + sub_x, dy = py + sub_y;
    if (dx > W / 2.0) dx -= W;
    if (dy > H / 2.0) dy -= H;
    s.dx = -dx; s.dy = -dy;
    // Peak sharpness relative to the mean magnitude — a registration-quality cue.
    const double mean = sum / static_cast<double>(R.size());
    s.response = mean > 1e-12 ? std::min(1.0, peak / (mean * W)) : 0.0;
    return s;
}

Image shift_image(const Image& src, double dx, double dy) {
    Image out(src.width(), src.height());
    for (int y = 0; y < src.height(); ++y)
        for (int x = 0; x < src.width(); ++x) {
            const double sxp = x - dx, syp = y - dy;
            const int x0 = static_cast<int>(std::floor(sxp));
            const int y0 = static_cast<int>(std::floor(syp));
            const double fx = sxp - x0, fy = syp - y0;
            const double v00 = src.at_clamped(x0, y0);
            const double v10 = src.at_clamped(x0 + 1, y0);
            const double v01 = src.at_clamped(x0, y0 + 1);
            const double v11 = src.at_clamped(x0 + 1, y0 + 1);
            const double top = v00 * (1 - fx) + v10 * fx;
            const double bot = v01 * (1 - fx) + v11 * fx;
            out.at(x, y) = static_cast<float>(top * (1 - fy) + bot * fy);
        }
    return out;
}

DiffResult diff_inspect(const Image& ref, const Image& test,
                        const DiffParams& params) {
    DiffResult r;
    r.shift = phase_correlate(ref, test);
    // Bring the test image back onto the reference frame.
    const Image aligned = shift_image(test, -r.shift.dx, -r.shift.dy);

    const int W = std::min(ref.width(), aligned.width());
    const int H = std::min(ref.height(), aligned.height());
    Image absdiff(W, H, 0.0f);
    double sum = 0.0, sumsq = 0.0;
    long n = 0;
    const int g = params.guard;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const double d = std::fabs(aligned.at(x, y) - ref.at(x, y));
            absdiff.at(x, y) = static_cast<float>(d);
            if (x >= g && y >= g && x < W - g && y < H - g) {
                sum += d; sumsq += d * d; ++n;
            }
        }
    const double mean = n ? sum / n : 0.0;
    const double var = n ? std::max(0.0, sumsq / n - mean * mean) : 0.0;
    r.residual_rms = std::sqrt(var + mean * mean);
    r.threshold = mean + params.k_sigma * std::sqrt(var);

    Image mask(W, H, 0.0f);
    for (int y = g; y < H - g; ++y)
        for (int x = g; x < W - g; ++x)
            if (absdiff.at(x, y) > r.threshold) mask.at(x, y) = 255.0f;
    if (params.open_radius > 0) mask = morph_open(mask, params.open_radius);

    r.defects = connected_components(mask, params.min_area);
    r.defect_count = static_cast<int>(r.defects.size());
    for (const auto& c : r.defects) r.defect_area += c.area;
    return r;
}

ColorImage annotate_diff(const Image& test, const DiffResult& r) {
    ColorImage out = to_color(test);
    for (const auto& c : r.defects) {
        draw_rect(out, c.min_x - 1, c.min_y - 1, c.max_x + 1, c.max_y + 1,
                  {220, 50, 50});
    }
    return out;
}

}  // namespace wproc::regi
