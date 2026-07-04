// pkg.cpp — bump / pillar array inspection for advanced packaging.
#include "wproc/pkg.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "wproc/connected.hpp"
#include "wproc/threshold.hpp"

namespace wproc::pkg {

namespace {
constexpr double kPi = 3.14159265358979323846;

// Median of a copy of v (v may be reordered).
double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const std::size_t m = v.size() / 2;
    return (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

// Mean brightness over a component's bounding box foreground (pixels > thr).
double bump_brightness(const Image& gray, const Component& c, float thr) {
    double sum = 0.0;
    long n = 0;
    for (int y = c.min_y; y <= c.max_y; ++y)
        for (int x = c.min_x; x <= c.max_x; ++x)
            if (gray.at(x, y) > thr) { sum += gray.at(x, y); ++n; }
    return n > 0 ? sum / n : 0.0;
}

}  // namespace

BumpResult inspect_bumps(const Image& gray, const BumpSpec& spec) {
    BumpResult r;
    if (gray.empty()) return r;

    const float thr = otsu_threshold(gray);
    const Image bin = binarize(gray, thr, /*invert=*/false);   // bright = bumps
    const std::vector<Component> comps = connected_components(bin, spec.min_area_px);
    if (comps.empty()) return r;

    std::vector<double> diam_um, xs, ys, bright;
    for (const Component& c : comps) {
        const double d_px = 2.0 * std::sqrt(static_cast<double>(c.area) / kPi);
        r.bumps.push_back(Bump{c.cx, c.cy, d_px * spec.um_per_px, d_px,
                               bump_brightness(gray, c, thr), false});
        diam_um.push_back(d_px * spec.um_per_px);
        xs.push_back(c.cx);
        ys.push_back(c.cy);
        bright.push_back(r.bumps.back().brightness);
    }
    r.count = static_cast<int>(r.bumps.size());

    // Diameter mean / CV and per-bump verdict.
    const double dsum = std::accumulate(diam_um.begin(), diam_um.end(), 0.0);
    r.mean_diameter_um = dsum / r.count;
    const double dmean = r.mean_diameter_um;
    double dvar = std::accumulate(diam_um.begin(), diam_um.end(), 0.0,
        [dmean](double a, double d) { return a + (d - dmean) * (d - dmean); });
    dvar /= r.count;
    r.diameter_cv = r.mean_diameter_um > 0 ? std::sqrt(dvar) / r.mean_diameter_um : 0.0;
    for (Bump& b : r.bumps)
        b.ok = std::fabs(b.diameter_um - spec.target_diameter_um) <= spec.diameter_tol_um;

    // Coplanarity proxy from brightness CV.
    const double bmean =
        std::accumulate(bright.begin(), bright.end(), 0.0) / r.count;
    double bvar = std::accumulate(bright.begin(), bright.end(), 0.0,
        [bmean](double a, double v) { return a + (v - bmean) * (v - bmean); });
    bvar /= r.count;
    const double bcv = bmean > 0 ? std::sqrt(bvar) / bmean : 0.0;
    r.coplanarity = std::clamp(1.0 - bcv, 0.0, 1.0);

    // Pitch = median nearest-neighbour distance.
    std::vector<double> nn;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        double best = 1e30;
        for (std::size_t j = 0; j < xs.size(); ++j) {
            if (i == j) continue;
            const double dx = xs[i] - xs[j], dy = ys[i] - ys[j];
            best = std::min(best, std::sqrt(dx * dx + dy * dy));
        }
        if (best < 1e29) nn.push_back(best);
    }
    const double pitch_px = median(nn);
    r.pitch_um = pitch_px * spec.um_per_px;

    // Placement RMS: residual of each centroid from a regular grid whose phase
    // is the mean fractional offset (per axis), independent of the origin.
    if (pitch_px > 1.0) {
        auto axis_resid = [&](const std::vector<double>& a) {
            const double phase_sum = std::accumulate(a.begin(), a.end(), 0.0,
                [&](double s, double v) { return s + std::fmod(v, pitch_px); });
            const double phase = phase_sum / a.size();
            double ss = 0.0;
            for (double v : a) {
                const double node = std::round((v - phase) / pitch_px) * pitch_px + phase;
                ss += (v - node) * (v - node);
            }
            return ss;
        };
        const double ss = axis_resid(xs) + axis_resid(ys);
        r.placement_rms_um = std::sqrt(ss / r.count) * spec.um_per_px;

        // Missing bumps vs the inferred rectangular array extent.
        const double xmin = *std::min_element(xs.begin(), xs.end());
        const double xmax = *std::max_element(xs.begin(), xs.end());
        const double ymin = *std::min_element(ys.begin(), ys.end());
        const double ymax = *std::max_element(ys.begin(), ys.end());
        const int ncol = static_cast<int>(std::round((xmax - xmin) / pitch_px)) + 1;
        const int nrow = static_cast<int>(std::round((ymax - ymin) / pitch_px)) + 1;
        r.missing = std::max(0, ncol * nrow - r.count);
    }

    // A uniformly over/under-sized array has a low CV, so also require the mean
    // diameter itself to sit within the target tolerance — otherwise every bump
    // being the wrong size would still pass.
    r.ok = r.diameter_cv <= spec.max_diameter_cv &&
           std::fabs(r.mean_diameter_um - spec.target_diameter_um) <= spec.diameter_tol_um &&
           r.coplanarity >= spec.min_coplanarity &&
           r.missing <= spec.max_missing;
    return r;
}

ColorImage annotate_bumps(const Image& gray, const BumpResult& r) {
    const int w = gray.width(), h = gray.height();
    ColorImage out(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const int v = static_cast<int>(std::lround(std::clamp(gray.at(x, y), 0.0f, 255.0f)));
            out.set(x, y, Rgb{static_cast<std::uint8_t>(v), static_cast<std::uint8_t>(v),
                              static_cast<std::uint8_t>(v)});
        }
    for (const Bump& b : r.bumps) {
        const Rgb c = b.ok ? Rgb{40, 200, 90} : Rgb{230, 60, 60};
        const int cx = static_cast<int>(std::lround(b.x)), cy = static_cast<int>(std::lround(b.y));
        const int rad = std::max(3, static_cast<int>(std::lround(b.diameter_px / 2.0)));
        for (int t = -rad; t <= rad; ++t) {
            out.set(cx + t, cy - rad, c); out.set(cx + t, cy + rad, c);
            out.set(cx - rad, cy + t, c); out.set(cx + rad, cy + t, c);
        }
    }
    return out;
}

}  // namespace wproc::pkg
