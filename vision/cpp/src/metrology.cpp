#include "wproc/metrology.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
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
        mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double ss = std::accumulate(
            v.begin(), v.end(), 0.0,
            [mean](double a, double x) { return a + (x - mean) * (x - mean); });
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
    double mean = std::accumulate(proj.begin(), proj.end(), 0.0) / W;
    std::transform(proj.begin(), proj.end(), proj.begin(),
                   [mean](double v) { return v - mean; });

    if (max_period_px <= 0) max_period_px = W / 2;
    max_period_px = std::min(max_period_px, W - 2);
    if (min_period_px < 2) min_period_px = 2;
    if (min_period_px >= max_period_px) return c;

    // Overlap-normalized autocorrelation r[lag] = mean over the overlap. The
    // per-lag mean removes the small-lag bias that raw sums have (short lags
    // sum more terms). We then pick the strongest *local* maximum rather than
    // the global max, so a smooth/defocused target does not collapse onto the
    // trivial near-zero lags where neighbouring samples are still correlated.
    const int hi = std::min(max_period_px + 1, W - 1);
    std::vector<double> r(hi + 1, 0.0);
    for (int lag = 1; lag <= hi; ++lag) {
        double acc = 0.0;
        int cnt = 0;
        for (int x = 0; x + lag < W; ++x) {
            acc += proj[x] * proj[x + lag];
            ++cnt;
        }
        r[lag] = cnt > 0 ? acc / cnt : 0.0;
    }

    double best_val = 0.0;
    int best_lag = -1;
    for (int lag = min_period_px; lag <= max_period_px; ++lag) {
        if (r[lag] > 0.0 && r[lag] >= r[lag - 1] && r[lag] >= r[lag + 1] &&
            r[lag] > best_val) {
            best_val = r[lag];
            best_lag = lag;
        }
    }
    if (best_lag < 0) return c;  // no periodic structure found

    c.ok = true;
    c.pixel_pitch_px = static_cast<double>(best_lag);
    c.um_per_px = known_pitch_um / c.pixel_pitch_px;
    return c;
}

}  // namespace wproc::metro
