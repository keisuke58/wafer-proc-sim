// grind_metrology.cpp — thickness-map metrology (TTV/WARP/BOW/uniformity).
#include "wproc/grind_metrology.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace wproc::grind {

namespace {

// Geometry of the inscribed wafer disk for a w x h grid.
struct Disk {
    double cx, cy, r2;  // centre and squared radius (grid units)
};

Disk disk_of(int w, int h, double radius_frac) {
    const double cx = (w - 1) * 0.5;
    const double cy = (h - 1) * 0.5;
    const double r = std::min(w, h) * 0.5 * radius_frac;
    return {cx, cy, r * r};
}

bool inside(const Disk& d, int x, int y) {
    const double dx = x - d.cx, dy = y - d.cy;
    return dx * dx + dy * dy <= d.r2;
}

}  // namespace

ThicknessStats analyze_thickness(const Image& thickness_um, double radius_frac,
                                 double invalid_below_um) {
    const int w = thickness_um.width(), h = thickness_um.height();
    const Disk disk = disk_of(w, h, radius_frac);

    // First pass: least-squares plane fit thickness ≈ a*x + b*y + c over valid
    // disk cells. Accumulate the normal-equation moments.
    double s1 = 0, sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    double sz = 0, sxz = 0, syz = 0;
    int n = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (!inside(disk, x, y)) continue;
            const double z = thickness_um.at(x, y);
            if (z <= invalid_below_um) continue;
            const double fx = x, fy = y;
            s1 += 1; sx += fx; sy += fy;
            sxx += fx * fx; sxy += fx * fy; syy += fy * fy;
            sz += z; sxz += fx * z; syz += fy * z;
            ++n;
        }
    }
    if (n < 3)
        throw std::invalid_argument(
            "analyze_thickness: fewer than 3 valid disk cells");

    // Solve the 3x3 symmetric normal system for (a, b, c) via Cramer's rule.
    const double m00 = sxx, m01 = sxy, m02 = sx;
    const double m11 = syy, m12 = sy, m22 = s1;
    const double det = m00 * (m11 * m22 - m12 * m12) -
                       m01 * (m01 * m22 - m12 * m02) +
                       m02 * (m01 * m12 - m11 * m02);
    double a = 0, b = 0, c = sz / s1;  // fall back to a flat mean plane
    if (std::fabs(det) > 1e-9) {
        const double d0 = sxz * (m11 * m22 - m12 * m12) -
                          m01 * (syz * m22 - m12 * sz) +
                          m02 * (syz * m12 - m11 * sz);
        const double d1 = m00 * (syz * m22 - m12 * sz) -
                          sxz * (m01 * m22 - m12 * m02) +
                          m02 * (m01 * sz - syz * m02);
        const double d2 = m00 * (m11 * sz - syz * m12) -
                          m01 * (m01 * sz - syz * m02) +
                          sxz * (m01 * m12 - m11 * m02);
        a = d0 / det; b = d1 / det; c = d2 / det;
    }

    // Second pass: thickness stats + de-tilted surface deviation for WARP/BOW.
    ThicknessStats st;
    st.n_inside = n;
    st.plane_a = a; st.plane_b = b; st.plane_c = c;

    double tmin = 0, tmax = 0, dmin = 0, dmax = 0, sum = 0, sumsq = 0;
    bool first = true;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (!inside(disk, x, y)) continue;
            const double z = thickness_um.at(x, y);
            if (z <= invalid_below_um) continue;
            const double dev = z - (a * x + b * y + c);
            sum += z; sumsq += z * z;
            if (first) {
                tmin = tmax = z; dmin = dmax = dev; first = false;
            } else {
                tmin = std::min(tmin, z); tmax = std::max(tmax, z);
                dmin = std::min(dmin, dev); dmax = std::max(dmax, dev);
            }
        }
    }

    st.mean_um = sum / n;
    const double var = std::max(0.0, sumsq / n - st.mean_um * st.mean_um);
    st.sigma_um = std::sqrt(var);
    st.min_um = tmin; st.max_um = tmax;
    st.ttv_um = tmax - tmin;
    st.warp_um = dmax - dmin;
    // BOW: de-tilted deviation at the disk centre (nearest valid cell).
    const int cxg = static_cast<int>(std::lround(disk.cx));
    const int cyg = static_cast<int>(std::lround(disk.cy));
    const double zc = thickness_um.at_clamped(cxg, cyg);
    st.bow_um = zc - (a * cxg + b * cyg + c);
    st.uniformity_pct =
        st.mean_um > 0
            ? std::clamp(100.0 * (1.0 - st.sigma_um / st.mean_um), 0.0, 100.0)
            : 0.0;
    return st;
}

ColorImage render_thickness(const Image& thickness_um, double radius_frac,
                            double invalid_below_um, double span_um) {
    const int w = thickness_um.width(), h = thickness_um.height();
    const Disk disk = disk_of(w, h, radius_frac);

    // Mean and sigma of valid cells (for centring / default span).
    double sum = 0, sumsq = 0;
    int n = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (!inside(disk, x, y)) continue;
            const double z = thickness_um.at(x, y);
            if (z <= invalid_below_um) continue;
            sum += z; sumsq += z * z; ++n;
        }
    }
    const double mean = n > 0 ? sum / n : 0.0;
    if (span_um <= 0.0) {
        const double var = n > 0 ? std::max(0.0, sumsq / n - mean * mean) : 0.0;
        span_um = std::max(1e-6, 3.0 * std::sqrt(var));
    }

    const Rgb neutral{60, 64, 70};
    ColorImage out(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const double z = thickness_um.at(x, y);
            if (!inside(disk, x, y) || z <= invalid_below_um) {
                out.set(x, y, neutral);
                continue;
            }
            // Diverging blue(thin) -> white(mean) -> red(thick).
            const double t = std::clamp((z - mean) / span_um, -1.0, 1.0);
            std::uint8_t r, g, bl;
            if (t >= 0) {
                const double u = 1.0 - t;  // t=0 white, t=1 red
                r = 255;
                g = static_cast<std::uint8_t>(std::lround(255 * u));
                bl = static_cast<std::uint8_t>(std::lround(255 * u));
            } else {
                const double u = 1.0 + t;  // t=0 white, t=-1 blue
                r = static_cast<std::uint8_t>(std::lround(255 * u));
                g = static_cast<std::uint8_t>(std::lround(255 * u));
                bl = 255;
            }
            out.set(x, y, Rgb{r, g, bl});
        }
    }
    return out;
}

}  // namespace wproc::grind
