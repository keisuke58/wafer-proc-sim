#include "wproc/metrology.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "wproc/filters.hpp"

namespace wproc::metro {

Repeatability repeatability(const Image& base, const KerfInspector& insp, int n,
                            float noise_sigma, unsigned seed) {
    Repeatability r;
    std::mt19937 gen(seed);
    std::normal_distribution<float> noise(0.0f, noise_sigma);

    std::vector<double> widths, chips;
    widths.reserve(n);
    for (int t = 0; t < n; ++t) {
        Image trial = base;  // copy
        for (std::size_t i = 0; i < trial.size(); ++i) trial.data()[i] += noise(gen);
        KerfResult res = insp.inspect(trial);
        if (!res.found) continue;
        widths.push_back(res.width_um);
        chips.push_back(res.max_chip_um);
    }

    r.n = static_cast<int>(widths.size());
    if (r.n == 0) return r;

    auto stats = [](const std::vector<double>& v, double& mean, double& sd) {
        double s = 0.0;
        for (double x : v) s += x;
        mean = s / v.size();
        double ss = 0.0;
        for (double x : v) ss += (x - mean) * (x - mean);
        sd = v.size() > 1 ? std::sqrt(ss / (v.size() - 1)) : 0.0;  // sample sd
    };

    stats(widths, r.width_mean_um, r.width_sd_um);
    r.width_min_um = *std::min_element(widths.begin(), widths.end());
    r.width_max_um = *std::max_element(widths.begin(), widths.end());
    r.width_cv_pct =
        r.width_mean_um != 0.0 ? r.width_sd_um / r.width_mean_um * 100.0 : 0.0;
    stats(chips, r.chip_mean_um, r.chip_sd_um);
    return r;
}

Calibration estimate_um_per_px(const Image& target, double known_pitch_um,
                               int min_period_px, int max_period_px) {
    Calibration c;
    if (target.width() < 8 || known_pitch_um <= 0.0) return c;

    // Signed Sobel-x, projected over rows: a periodic grating gives a
    // projection whose fundamental period equals the grating pitch in pixels.
    Gradient g = sobel(target);
    const int W = target.width(), H = target.height();
    std::vector<double> proj(W, 0.0);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) proj[x] += g.gx.at(x, y);

    // Detrend (remove DC) so autocorrelation reflects periodicity, not offset.
    double mean = 0.0;
    for (double v : proj) mean += v;
    mean /= W;
    for (double& v : proj) v -= mean;

    if (max_period_px <= 0) max_period_px = W / 2;
    max_period_px = std::min(max_period_px, W - 1);
    if (min_period_px < 1) min_period_px = 1;
    if (min_period_px >= max_period_px) return c;

    // Autocorrelation; the lag maximizing it (above the trivial lag 0) is the
    // dominant period.
    double best_val = -1e300;
    int best_lag = -1;
    for (int lag = min_period_px; lag <= max_period_px; ++lag) {
        double acc = 0.0;
        for (int x = 0; x + lag < W; ++x) acc += proj[x] * proj[x + lag];
        if (acc > best_val) {
            best_val = acc;
            best_lag = lag;
        }
    }
    if (best_lag <= 0 || best_val <= 0.0) return c;

    c.ok = true;
    c.pixel_pitch_px = static_cast<double>(best_lag);
    c.um_per_px = known_pitch_um / c.pixel_pitch_px;
    return c;
}

}  // namespace wproc::metro
