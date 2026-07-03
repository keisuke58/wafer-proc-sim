#include "wproc/stealth_ir.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "wproc/filters.hpp"

namespace wproc::stealth {

namespace {
std::vector<double> smooth3(const std::vector<double>& v) {
    const int n = static_cast<int>(v.size());
    std::vector<double> s(v);
    for (int i = 1; i < n - 1; ++i) s[i] = (v[i - 1] + v[i] + v[i + 1]) / 3.0;
    return s;
}
}  // namespace

SdResult inspect_stealth(const Image& img, SdParams p) {
    SdResult r;
    const int W = img.width(), H = img.height();
    if (W < 3 || H < 5) return r;

    Image b = gaussian_blur(img, p.blur_sigma);

    // Row projection: SD layers are bright horizontal bands, so the mean
    // brightness per row peaks at each layer depth.
    std::vector<double> row(H, 0.0);
    for (int y = 0; y < H; ++y) {
        double s = 0.0;
        for (int x = 0; x < W; ++x) s += b.at(x, y);
        row[y] = s / W;
    }
    auto rs = smooth3(row);
    double rmin = *std::min_element(rs.begin(), rs.end());
    double rmax = *std::max_element(rs.begin(), rs.end());
    if (rmax - rmin < 1e-6) return r;
    double thr_row = rmin + p.min_layer_prominence * (rmax - rmin);

    // Local maxima above the prominence threshold = layer depths (top-first).
    std::vector<int> candidates;
    for (int y = 1; y < H - 1; ++y)
        if (rs[y] > thr_row && rs[y] >= rs[y - 1] && rs[y] >= rs[y + 1])
            candidates.push_back(y);

    // Merge adjacent/plateaued candidates so a single wide (or saturated) SD
    // band isn't reported as several layers, which would corrupt the pitch.
    for (std::size_t i = 0; i < candidates.size();) {
        std::size_t j = i;
        while (j + 1 < candidates.size() && candidates[j + 1] - candidates[j] <= 1)
            ++j;
        double sum = 0.0;
        for (std::size_t k = i; k <= j; ++k) sum += candidates[k];
        r.layer_depths_px.push_back(sum / static_cast<double>(j - i + 1));
        i = j + 1;
    }
    r.layer_count = static_cast<int>(r.layer_depths_px.size());

    if (r.layer_count >= 2) {
        double sum = 0.0;
        for (int i = 1; i < r.layer_count; ++i)
            sum += r.layer_depths_px[i] - r.layer_depths_px[i - 1];
        r.mean_pitch_px = sum / (r.layer_count - 1);
    }
    if (r.layer_count == 0) return r;

    // Crack propagation: look above the topmost layer for a bright vertical
    // run (a crack heading toward the wafer surface at y=0). For each column,
    // walk upward from just above the top layer while the pixel stays bright.
    const int top = static_cast<int>(std::round(r.layer_depths_px.front()));
    double imin = 1e300, imax = -1e300;
    for (std::size_t i = 0; i < b.size(); ++i) {
        imin = std::min(imin, static_cast<double>(b.data()[i]));
        imax = std::max(imax, static_cast<double>(b.data()[i]));
    }
    double thr_pix = 0.5 * (imin + imax);

    int best_len = 0, best_top_y = top;
    for (int x = 0; x < W; ++x) {
        int y = top - 1, len = 0;
        while (y >= 0 && b.at(x, y) > thr_pix) { ++len; --y; }
        if (len > best_len) {
            best_len = len;
            best_top_y = y + 1;  // shallowest bright row reached
        }
    }
    r.max_crack_px = static_cast<double>(best_len);
    r.crack_reaches_surface = best_len > 0 && best_top_y <= p.surface_margin_px;
    return r;
}

}  // namespace wproc::stealth
