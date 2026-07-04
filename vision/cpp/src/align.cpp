// align.cpp — NCC template matching for dicing-street registration.
#include "wproc/align.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace wproc::align {

namespace {

// Precomputed zero-mean template and its norm, for repeated NCC evaluation.
struct TemplateStats {
    int w = 0, h = 0;
    std::vector<double> zm;   // template minus its mean
    double norm = 0.0;        // sqrt(sum(zm^2))
};

TemplateStats prep_template(const Image& t) {
    TemplateStats ts;
    ts.w = t.width();
    ts.h = t.height();
    const int n = ts.w * ts.h;
    double sum = 0.0;
    for (int y = 0; y < ts.h; ++y)
        for (int x = 0; x < ts.w; ++x) sum += t.at(x, y);
    const double mean = n > 0 ? sum / n : 0.0;
    ts.zm.resize(n);
    double ss = 0.0;
    for (int y = 0; y < ts.h; ++y)
        for (int x = 0; x < ts.w; ++x) {
            const double v = t.at(x, y) - mean;
            ts.zm[static_cast<std::size_t>(y) * ts.w + x] = v;
            ss += v * v;
        }
    ts.norm = std::sqrt(ss);
    return ts;
}

// NCC between the template and the img patch whose top-left is (u, v).
double ncc_at(const Image& img, const TemplateStats& ts, int u, int v) {
    const int n = ts.w * ts.h;
    double sum = 0.0;
    for (int y = 0; y < ts.h; ++y)
        for (int x = 0; x < ts.w; ++x) sum += img.at(u + x, v + y);
    const double mean = sum / n;
    double cross = 0.0, iss = 0.0;
    for (int y = 0; y < ts.h; ++y)
        for (int x = 0; x < ts.w; ++x) {
            const double iv = img.at(u + x, v + y) - mean;
            cross += iv * ts.zm[static_cast<std::size_t>(y) * ts.w + x];
            iss += iv * iv;
        }
    const double denom = ts.norm * std::sqrt(iss);
    return denom > 1e-12 ? cross / denom : 0.0;
}

// Parabolic sub-pixel offset from three samples (peak at the middle).
double parabolic(double ym1, double y0, double yp1) {
    const double denom = ym1 - 2.0 * y0 + yp1;
    if (std::fabs(denom) < 1e-12) return 0.0;
    return std::clamp(0.5 * (ym1 - yp1) / denom, -0.5, 0.5);
}

}  // namespace

Match match_ncc(const Image& img, const Image& templ, int sx0, int sy0,
                int sw, int sh) {
    Match m;
    const TemplateStats ts = prep_template(templ);
    if (ts.w < 1 || ts.h < 1 || ts.norm <= 0.0) return m;

    // Valid top-left positions keep the whole template inside the image.
    const int umin = std::max(0, sx0);
    const int vmin = std::max(0, sy0);
    const int umax = std::min(img.width() - ts.w, sx0 + sw - 1);
    const int vmax = std::min(img.height() - ts.h, sy0 + sh - 1);
    if (umax < umin || vmax < vmin) return m;

    int bu = umin, bv = vmin;
    double best = -2.0;
    for (int v = vmin; v <= vmax; ++v)
        for (int u = umin; u <= umax; ++u) {
            const double s = ncc_at(img, ts, u, v);
            if (s > best) { best = s; bu = u; bv = v; }
        }

    m.found = true;
    m.score = best;
    // Sub-pixel refinement (guarded to interior peaks).
    double dx = 0.0, dy = 0.0;
    if (bu > umin && bu < umax)
        dx = parabolic(ncc_at(img, ts, bu - 1, bv), best, ncc_at(img, ts, bu + 1, bv));
    if (bv > vmin && bv < vmax)
        dy = parabolic(ncc_at(img, ts, bu, bv - 1), best, ncc_at(img, ts, bu, bv + 1));
    m.x = bu + dx;
    m.y = bv + dy;
    return m;
}

Registration register_pattern(const Image& img, const Image& templ, int ex,
                              int ey, int radius, double min_score) {
    Registration r;
    const int tw = templ.width(), th = templ.height();
    if (tw < 2 || th < 2) return r;

    const Match full = match_ncc(img, templ, ex - radius, ey - radius,
                                 2 * radius + tw, 2 * radius + th);
    if (!full.found) return r;
    r.score = full.score;
    r.dx = full.x - ex;
    r.dy = full.y - ey;
    r.ok = full.score >= min_score;

    // Tilt: match the left and right halves independently; a vertical shift
    // between them over their horizontal separation is the street tilt.
    const int hw = tw / 2;
    if (hw >= 2) {
        Image left(hw, th), right(hw, th);
        for (int y = 0; y < th; ++y)
            for (int x = 0; x < hw; ++x) {
                left.at(x, y) = templ.at(x, y);
                right.at(x, y) = templ.at(x + (tw - hw), y);
            }
        const Match ml = match_ncc(img, left, ex - radius, ey - radius,
                                   2 * radius + hw, 2 * radius + th);
        const Match mr = match_ncc(img, right, ex + (tw - hw) - radius, ey - radius,
                                   2 * radius + hw, 2 * radius + th);
        if (ml.found && mr.found) {
            const double sep = static_cast<double>(tw - hw);  // centre separation
            if (sep > 0.0)
                r.angle_deg = std::atan2(mr.y - ml.y, sep) * 180.0 / 3.14159265358979323846;
        }
    }
    return r;
}

ColorImage annotate_match(const Image& img, const Match& m, int tw, int th) {
    const int w = img.width(), h = img.height();
    ColorImage out(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const int v = static_cast<int>(std::lround(std::clamp(img.at(x, y), 0.0f, 255.0f)));
            out.set(x, y, Rgb{static_cast<std::uint8_t>(v), static_cast<std::uint8_t>(v),
                              static_cast<std::uint8_t>(v)});
        }
    if (!m.found) return out;
    const int x0 = static_cast<int>(std::lround(m.x)), y0 = static_cast<int>(std::lround(m.y));
    const Rgb g{40, 200, 90};
    for (int x = x0; x < x0 + tw; ++x) { out.set(x, y0, g); out.set(x, y0 + th - 1, g); }
    for (int y = y0; y < y0 + th; ++y) { out.set(x0, y, g); out.set(x0 + tw - 1, y, g); }
    return out;
}

}  // namespace wproc::align
