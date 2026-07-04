// cmp.cpp — CMP removal, polish-time APC, planarity, endpoint detection.
#include "wproc/cmp.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace wproc::cmp {

// ── Preston model ────────────────────────────────────────────────────────────

double removal_rate(const PrestonParams& p) {
    return p.kp * p.pressure * p.velocity;
}
double removal(const PrestonParams& p, double time_s) {
    return removal_rate(p) * time_s;
}

// ── PolishController ─────────────────────────────────────────────────────────

PolishController::PolishController(PolishParams p) : p_(p), time_(p.time0) {
    if (p_.kp == 0.0)
        throw std::invalid_argument("PolishController: kp must be non-zero");
    if (!(p_.lambda > 0.0 && p_.lambda <= 1.0))
        throw std::invalid_argument("PolishController: lambda must be in (0,1]");
    if (p_.time_min > p_.time_max)
        throw std::invalid_argument("PolishController: time_min > time_max");
    kp_hat_ = p_.kp;
}

double PolishController::update(double measured_removal_nm) {
    // Invert removal = kp*P*V*time to estimate the effective kp, then EWMA it.
    const double pv_t = p_.pressure * p_.velocity * time_;
    if (pv_t > 0.0) {
        const double kp_meas = measured_removal_nm / pv_t;
        kp_hat_ = p_.lambda * kp_meas + (1.0 - p_.lambda) * kp_hat_;
    }
    // Set the next time so the predicted removal lands on target.
    const double rate = kp_hat_ * p_.pressure * p_.velocity;
    double next = rate > 0.0 ? p_.target_removal_nm / rate : p_.time_max;
    next = std::clamp(next, p_.time_min, p_.time_max);
    time_ = next;
    return time_;
}

// ── Planarity ────────────────────────────────────────────────────────────────

PlanarityStats analyze_planarity(const std::vector<double>& profile_nm,
                                 const std::vector<char>& is_feature,
                                 const PlanaritySpec& spec) {
    if (profile_nm.size() != is_feature.size())
        throw std::invalid_argument("analyze_planarity: size mismatch");
    if (profile_nm.empty())
        throw std::invalid_argument("analyze_planarity: empty profile");

    PlanarityStats st;
    double field_sum = 0.0, field_sumsq = 0.0, feat_sum = 0.0;
    double lo = profile_nm[0], hi = profile_nm[0];
    for (std::size_t i = 0; i < profile_nm.size(); ++i) {
        const double h = profile_nm[i];
        lo = std::min(lo, h);
        hi = std::max(hi, h);
        if (is_feature[i]) {
            feat_sum += h;
            ++st.n_feature;
        } else {
            field_sum += h;
            field_sumsq += h * h;
            ++st.n_field;
        }
    }
    if (st.n_field == 0 || st.n_feature == 0)
        throw std::invalid_argument("analyze_planarity: need both field and feature samples");

    st.field_mean_nm = field_sum / st.n_field;
    st.feature_mean_nm = feat_sum / st.n_feature;
    st.dishing_nm = st.field_mean_nm - st.feature_mean_nm;  // Cu recess below field
    st.erosion_nm = spec.nominal_nm - st.field_mean_nm;     // field height loss
    st.range_nm = hi - lo;

    const double var = std::max(0.0, field_sumsq / st.n_field -
                                         st.field_mean_nm * st.field_mean_nm);
    const double denom = std::fabs(st.field_mean_nm);
    st.uniformity_pct =
        denom > 1e-9
            ? std::clamp(100.0 * (1.0 - std::sqrt(var) / denom), 0.0, 100.0)
            : 0.0;

    st.ok = std::fabs(st.dishing_nm) <= spec.max_dishing_nm &&
            st.erosion_nm <= spec.max_erosion_nm;
    return st;
}

// ── Endpoint detection ───────────────────────────────────────────────────────

Endpoint detect_endpoint(const std::vector<double>& signal, double fs,
                         double drop_frac, int smooth) {
    Endpoint e;
    const int n = static_cast<int>(signal.size());
    if (n < 3 || fs <= 0.0) return e;

    // Moving-average smoothing (box filter, clamped edges).
    const int r = std::max(1, smooth);
    std::vector<double> s(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double acc = 0.0;
        int cnt = 0;
        for (int j = i - r; j <= i + r; ++j) {
            const int jj = std::clamp(j, 0, n - 1);
            acc += signal[jj];
            ++cnt;
        }
        s[i] = acc / cnt;
    }

    // Plateau = maximum of the smoothed trace; the endpoint is the steepest
    // sustained descent after that plateau.
    const int peak_i = static_cast<int>(std::max_element(s.begin(), s.end()) - s.begin());
    const double plateau = s[peak_i];
    const double smin = *std::min_element(s.begin() + peak_i, s.end());
    const double span = plateau - smin;
    if (span <= 0.0) return e;

    // First index after the peak where the signal has dropped drop_frac of span.
    const double thr = plateau - drop_frac * span;
    for (int i = peak_i; i < n; ++i) {
        if (s[i] <= thr) {
            e.detected = true;
            e.index = i;
            e.time_s = i / fs;
            e.drop = plateau - s[i];
            break;
        }
    }
    return e;
}

}  // namespace wproc::cmp
