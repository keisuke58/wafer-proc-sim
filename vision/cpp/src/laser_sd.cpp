// laser_sd.cpp — stealth SD-layer + HAZ analysis (extends stealth_ir).
#include "wproc/laser_sd.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace wproc::laser {

namespace {

// Row-intensity projection (mean over x); brighter row = more SD-modified.
std::vector<double> row_projection(const Image& img) {
    const int w = img.width(), h = img.height();
    std::vector<double> r(h, 0.0);
    for (int y = 0; y < h; ++y) {
        double s = 0.0;
        for (int x = 0; x < w; ++x) s += img.at(x, y);
        r[y] = w > 0 ? s / w : 0.0;
    }
    return r;
}

}  // namespace

SdQuality inspect_sd_haz(const Image& img, const SdQualitySpec& spec) {
    SdQuality q;
    const stealth::SdResult sd = stealth::inspect_stealth(img, spec.sd);
    q.layer_count = sd.layer_count;
    q.mean_pitch_px = sd.mean_pitch_px;
    q.max_crack_px = sd.max_crack_px;
    q.crack_reaches_surface = sd.crack_reaches_surface;

    // Pitch uniformity from consecutive layer spacings.
    if (sd.layer_depths_px.size() >= 3) {
        std::vector<double> pitch;
        for (std::size_t i = 1; i < sd.layer_depths_px.size(); ++i)
            pitch.push_back(sd.layer_depths_px[i] - sd.layer_depths_px[i - 1]);
        const double mean =
            std::accumulate(pitch.begin(), pitch.end(), 0.0) / pitch.size();
        const double sq = std::inner_product(pitch.begin(), pitch.end(),
                                             pitch.begin(), 0.0);
        const double var = std::max(0.0, sq / pitch.size() - mean * mean);
        const double cv = mean > 0 ? std::sqrt(var) / mean : 0.0;
        q.pitch_uniformity_pct = std::clamp(100.0 * (1.0 - cv), 0.0, 100.0);
    }

    // HAZ thickness: the contiguous band around the top SD layer whose row
    // brightness exceeds the half-height between the baseline and the peak.
    if (!sd.layer_depths_px.empty()) {
        const std::vector<double> rp = row_projection(img);
        const int h = static_cast<int>(rp.size());
        int top = static_cast<int>(std::lround(sd.layer_depths_px.front()));
        top = std::clamp(top, 0, h - 1);
        const double peak = rp[top];
        const double base = *std::min_element(rp.begin(), rp.end());
        const double half = base + 0.5 * (peak - base);
        int lo = top, hi = top;
        while (lo - 1 >= 0 && rp[lo - 1] >= half) --lo;
        while (hi + 1 < h && rp[hi + 1] >= half) ++hi;
        q.haz_thickness_px = static_cast<double>(hi - lo + 1);

        // Surface-crack risk: fraction of the depth-to-top-layer already bridged
        // by the longest crack; forced high if it reaches the surface.
        const double top_depth = std::max(1.0, sd.layer_depths_px.front());
        double risk = std::clamp(sd.max_crack_px / top_depth, 0.0, 1.0);
        if (sd.crack_reaches_surface) risk = 1.0;
        q.surface_risk = risk;
    }

    q.ok = !q.crack_reaches_surface &&
           q.haz_thickness_px <= spec.max_haz_px &&
           (q.layer_count < 3 ||
            q.pitch_uniformity_pct >= spec.min_pitch_uniformity_pct);
    return q;
}

}  // namespace wproc::laser
