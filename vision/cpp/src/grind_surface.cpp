// grind_surface.cpp — grinding-mark & subsurface-damage surface analysis.
#include "wproc/grind_surface.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "wproc/filters.hpp"

namespace wproc::grind {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Mean and variance of an image's pixels.
void mean_var(const Image& img, double& mean, double& var) {
    const std::size_t n = img.size();
    if (n == 0) { mean = var = 0.0; return; }
    const float* p = img.data();
    double sum = 0.0, sumsq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double v = p[i];
        sum += v; sumsq += v * v;
    }
    mean = sum / n;
    var = std::max(0.0, sumsq / n - mean * mean);
}

}  // namespace

SurfaceQuality analyze_surface(const Image& gray, float hp_sigma,
                               const SurfaceSpec& spec) {
    SurfaceQuality q;
    if (gray.empty()) return q;

    // Structure tensor from Sobel gradients: J = sum of [gx^2, gx*gy; ., gy^2].
    const Gradient g = sobel(gray);
    const std::size_t n = gray.size();
    const float* gx = g.gx.data();
    const float* gy = g.gy.data();
    double jxx = 0.0, jyy = 0.0, jxy = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        jxx += double(gx[i]) * gx[i];
        jyy += double(gy[i]) * gy[i];
        jxy += double(gx[i]) * gy[i];
    }
    const double trace = jxx + jyy;
    const double aniso = std::sqrt((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy);
    q.coherence = trace > 0 ? std::clamp(aniso / trace, 0.0, 1.0) : 0.0;

    // Dominant gradient orientation; grinding marks run perpendicular to it.
    const double grad_angle = 0.5 * std::atan2(2.0 * jxy, jxx - jyy);
    double mark = grad_angle + kPi / 2.0;  // perpendicular = mark direction
    double deg = mark * 180.0 / kPi;
    deg = std::fmod(deg, 180.0);
    if (deg < 0) deg += 180.0;
    q.mark_angle_deg = deg;

    // Relative gradient energy: mean gradient magnitude normalized by a mid-gray
    // reference, saturating at 1 — so severity blends "how oriented" with
    // "how strong" the texture is.
    const double mean_grad_energy = trace / (n > 0 ? n : 1);
    const double rel_energy = std::clamp(std::sqrt(mean_grad_energy) / 64.0, 0.0, 1.0);
    q.mark_severity = std::clamp(q.coherence * rel_energy, 0.0, 1.0);

    // High-pass residual = gray - Gaussian(gray): roughness Ra and SSD index.
    const Image low = gaussian_blur(gray, hp_sigma);
    const float* pg = gray.data();
    const float* pl = low.data();
    double abs_sum = 0.0, res_sumsq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double r = double(pg[i]) - pl[i];
        abs_sum += std::fabs(r);
        res_sumsq += r * r;
    }
    q.roughness_ra = abs_sum / n;

    double gmean = 0.0, gvar = 0.0;
    mean_var(gray, gmean, gvar);
    const double res_var = res_sumsq / n;  // residual is ~zero-mean
    q.ssd_index = gvar > 1e-9 ? std::clamp(res_var / gvar, 0.0, 1.0) : 0.0;

    q.ok = q.mark_severity <= spec.max_mark_severity &&
           q.roughness_ra <= spec.max_roughness_ra &&
           q.ssd_index <= spec.max_ssd_index;
    return q;
}

ColorImage render_ssd_heatmap(const Image& gray, float hp_sigma,
                              float pool_sigma) {
    const int w = gray.width(), h = gray.height();
    ColorImage out(w, h);
    if (gray.empty()) return out;

    // Local high-frequency energy = pool( residual^2 ).
    const Image low = gaussian_blur(gray, hp_sigma);
    Image energy(w, h);
    const std::size_t n = gray.size();
    const float* pg = gray.data();
    const float* pl = low.data();
    float* pe = energy.data();
    for (std::size_t i = 0; i < n; ++i) {
        const float r = pg[i] - pl[i];
        pe[i] = r * r;
    }
    const Image pooled = gaussian_blur(energy, pool_sigma);

    // Normalize by the 99th-ish percentile (use max for a robust-enough scale).
    const float* pp = pooled.data();
    float emax = 0.0f;
    for (std::size_t i = 0; i < n; ++i) emax = std::max(emax, pp[i]);
    const double scale = emax > 0 ? 1.0 / emax : 0.0;

    // Black -> red -> yellow -> white ramp.
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const double t = std::clamp(pp[std::size_t(y) * w + x] * scale, 0.0, 1.0);
            std::uint8_t r, gg, bb;
            if (t < 0.5) {
                r = static_cast<std::uint8_t>(std::lround(255 * (t / 0.5)));
                gg = 0; bb = 0;
            } else if (t < 0.85) {
                r = 255;
                gg = static_cast<std::uint8_t>(std::lround(255 * ((t - 0.5) / 0.35)));
                bb = 0;
            } else {
                r = 255; gg = 255;
                bb = static_cast<std::uint8_t>(std::lround(255 * ((t - 0.85) / 0.15)));
            }
            out.set(x, y, Rgb{r, gg, bb});
        }
    }
    return out;
}

}  // namespace wproc::grind
