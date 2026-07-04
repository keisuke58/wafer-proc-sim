// spindle.cpp — FFT-based spindle vibration / chatter analysis.
#include "wproc/spindle.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace wproc::spindle {

namespace {
constexpr double kPi = 3.14159265358979323846;

std::size_t next_pow2(std::size_t n) {
    std::size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}
}  // namespace

void fft(std::vector<std::complex<double>>& a, bool inverse) {
    const std::size_t n = a.size();
    if (n < 2) return;

    // Bit-reversal permutation.
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    // Butterflies.
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double ang = 2.0 * kPi / static_cast<double>(len) * (inverse ? 1.0 : -1.0);
        const std::complex<double> wlen(std::cos(ang), std::sin(ang));
        for (std::size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (std::size_t k = 0; k < len / 2; ++k) {
                const std::complex<double> u = a[i + k];
                const std::complex<double> v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

Spectrum power_spectrum(const std::vector<double>& signal, double fs) {
    Spectrum s;
    s.fs = fs;
    const std::size_t n0 = signal.size();
    if (n0 < 2 || fs <= 0.0) return s;

    const std::size_t n = next_pow2(n0);
    std::vector<std::complex<double>> buf(n, {0.0, 0.0});

    // Hann window over the original samples (reduces spectral leakage).
    double win_sum_sq = 0.0;
    for (std::size_t i = 0; i < n0; ++i) {
        const double w = 0.5 * (1.0 - std::cos(2.0 * kPi * i / (n0 - 1)));
        buf[i] = std::complex<double>(signal[i] * w, 0.0);
        win_sum_sq += w * w;
    }
    fft(buf, /*inverse=*/false);

    const std::size_t half = n / 2;
    s.df = fs / static_cast<double>(n);
    s.freq.resize(half);
    s.power.resize(half);
    // Normalize so a sinusoid's power is amplitude-consistent; window-corrected.
    const double norm = (win_sum_sq > 0.0) ? 1.0 / win_sum_sq : 0.0;
    for (std::size_t k = 0; k < half; ++k) {
        s.freq[k] = static_cast<double>(k) * s.df;
        double p = std::norm(buf[k]) * norm;      // |X|^2, window-normalized
        if (k > 0) p *= 2.0;                       // fold negative frequencies
        s.power[k] = p;
    }
    return s;
}

ChatterResult analyze_vibration(const std::vector<double>& signal, double fs,
                                double spindle_rpm, double harmonic_tol_hz,
                                double ratio_threshold) {
    ChatterResult r;
    if (signal.empty() || fs <= 0.0) return r;

    // Time-domain RMS.
    const double ss = std::inner_product(signal.begin(), signal.end(),
                                         signal.begin(), 0.0);
    r.rms = std::sqrt(ss / signal.size());
    r.spindle_freq = spindle_rpm / 60.0;

    const Spectrum sp = power_spectrum(signal, fs);
    if (sp.power.empty()) return r;

    auto is_harmonic = [&](double f) {
        if (r.spindle_freq <= 0.0) return false;
        const double ratio = f / r.spindle_freq;
        const double nearest = std::round(ratio);
        if (nearest < 1.0) return false;
        return std::fabs(f - nearest * r.spindle_freq) <= harmonic_tol_hz;
    };

    // Scan local maxima (skip the DC bin); split into harmonic vs non-harmonic.
    double dom_power = 0.0;
    for (std::size_t k = 1; k + 1 < sp.power.size(); ++k) {
        const double p = sp.power[k];
        if (!(p >= sp.power[k - 1] && p >= sp.power[k + 1])) continue;  // local peak
        const double f = sp.freq[k];
        if (p > dom_power) { dom_power = p; r.dominant_freq = f; }
        if (is_harmonic(f)) {
            r.harmonic_power += p;
        } else if (p > r.chatter_power) {
            r.chatter_power = p;
            r.chatter_freq = f;
        }
    }

    r.chatter_ratio =
        r.harmonic_power > 0.0 ? r.chatter_power / r.harmonic_power : 0.0;
    r.chatter = r.chatter_ratio >= ratio_threshold;
    return r;
}

}  // namespace wproc::spindle
