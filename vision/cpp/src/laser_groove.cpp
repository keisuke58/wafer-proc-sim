// laser_groove.cpp — laser ablation-groove inspection (top-view).
#include "wproc/laser_groove.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace wproc::laser {

namespace {

// Column-intensity profile and the levels/thresholds a vertical groove induces.
struct Profile {
    std::vector<double> col_mean;
    double sub = 0.0;   // substrate level (bright)
    double dark = 0.0;  // groove-floor level (dark)
    double thr_g = 0.0; // below ⇒ groove
    double thr_h = 0.0; // [thr_g, thr_h) ⇒ HAZ band
};

Profile profile_of(const Image& img) {
    const int w = img.width(), h = img.height();
    Profile p;
    p.col_mean.assign(w, 0.0);
    for (int x = 0; x < w; ++x) {
        double s = 0.0;
        for (int y = 0; y < h; ++y) s += img.at(x, y);
        p.col_mean[x] = h > 0 ? s / h : 0.0;
    }
    std::vector<double> sorted = p.col_mean;
    std::sort(sorted.begin(), sorted.end());
    // Substrate = high quantile (robust to the groove dip); floor = min.
    p.sub = sorted.empty() ? 0.0 : sorted[static_cast<int>(0.85 * (sorted.size() - 1))];
    p.dark = sorted.empty() ? 0.0 : sorted.front();
    const double span = p.sub - p.dark;
    p.thr_g = p.sub - 0.75 * span;  // clearly-groove
    p.thr_h = p.sub - 0.25 * span;  // edge of the HAZ band
    return p;
}

}  // namespace

GrooveResult inspect_groove(const Image& gray, const GrooveSpec& spec) {
    GrooveResult r;
    const int w = gray.width(), h = gray.height();
    if (w == 0 || h == 0) return r;

    const Profile p = profile_of(gray);
    const double span = p.sub - p.dark;
    r.depth_index = p.sub > 0 ? std::clamp(span / p.sub, 0.0, 1.0) : 0.0;
    // Too little contrast ⇒ no real groove.
    if (span < 8.0) return r;

    // Classify columns; groove width + centre from the groove columns.
    int groove_cnt = 0, haz_cnt = 0;
    double center_acc = 0.0;
    for (int x = 0; x < w; ++x) {
        const double v = p.col_mean[x];
        if (v < p.thr_g) { ++groove_cnt; center_acc += x; }
        else if (v < p.thr_h) { ++haz_cnt; }
    }
    if (groove_cnt == 0) return r;

    r.found = true;
    r.width_px = groove_cnt;
    r.center_px = center_acc / groove_cnt;
    r.width_um = r.width_px * spec.um_per_px;
    r.haz_um = (haz_cnt / 2.0) * spec.um_per_px;  // split across the two sides

    // Debris/spatter: substrate pixels thrown off their column level. Estimate
    // the substrate residual spread, then flag strong outliers on substrate
    // columns (those clear of both the groove and the HAZ band).
    double sq = 0.0;
    long sub_px = 0;
    for (int x = 0; x < w; ++x) {
        if (p.col_mean[x] < p.thr_h) continue;  // groove/HAZ column
        for (int y = 0; y < h; ++y) {
            const double d = gray.at(x, y) - p.col_mean[x];
            sq += d * d;
            ++sub_px;
        }
    }
    if (sub_px > 0) {
        const double sigma = std::sqrt(sq / sub_px);
        const double thr = std::max(4.0 * sigma, 15.0);
        long debris = 0;
        for (int x = 0; x < w; ++x) {
            if (p.col_mean[x] < p.thr_h) continue;
            for (int y = 0; y < h; ++y)
                if (std::fabs(gray.at(x, y) - p.col_mean[x]) > thr) ++debris;
        }
        r.debris_index = static_cast<double>(debris) / sub_px;
    }

    r.ok = std::fabs(r.width_um - spec.target_width_um) <= spec.width_tol_um &&
           r.haz_um <= spec.max_haz_um &&
           r.debris_index <= spec.max_debris_index &&
           r.depth_index >= spec.min_depth_index;
    return r;
}

ColorImage annotate_groove(const Image& gray, const GrooveResult& r,
                           const GrooveSpec& spec) {
    const int w = gray.width(), h = gray.height();
    ColorImage out(w, h);
    // Grayscale base.
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const int v = static_cast<int>(std::lround(std::clamp(gray.at(x, y), 0.0f, 255.0f)));
            out.set(x, y, Rgb{static_cast<std::uint8_t>(v), static_cast<std::uint8_t>(v),
                              static_cast<std::uint8_t>(v)});
        }
    if (!r.found) return out;

    const int gl = static_cast<int>(std::lround(r.center_px - r.width_px / 2.0));
    const int gr = static_cast<int>(std::lround(r.center_px + r.width_px / 2.0));
    const int haz_px = static_cast<int>(std::lround(r.haz_um / spec.um_per_px));
    auto vline = [&](int x, Rgb c) {
        if (x < 0 || x >= w) return;
        for (int y = 0; y < h; ++y) out.set(x, y, c);
    };
    // Amber HAZ band edges, then green groove edges on top.
    vline(gl - haz_px, Rgb{240, 170, 40});
    vline(gr + haz_px, Rgb{240, 170, 40});
    vline(gl, Rgb{40, 200, 90});
    vline(gr, Rgb{40, 200, 90});

    // Debris pixels in red (recompute against the column profile).
    const Profile p = profile_of(gray);
    double sq = 0.0; long sub_px = 0;
    for (int x = 0; x < w; ++x) {
        if (p.col_mean[x] < p.thr_h) continue;
        for (int y = 0; y < h; ++y) {
            const double d = gray.at(x, y) - p.col_mean[x];
            sq += d * d; ++sub_px;
        }
    }
    if (sub_px > 0) {
        const double thr = std::max(4.0 * std::sqrt(sq / sub_px), 15.0);
        for (int x = 0; x < w; ++x) {
            if (p.col_mean[x] < p.thr_h) continue;
            for (int y = 0; y < h; ++y)
                if (std::fabs(gray.at(x, y) - p.col_mean[x]) > thr)
                    out.set(x, y, Rgb{230, 60, 60});
        }
    }
    return out;
}

}  // namespace wproc::laser
