// test_spindle.cpp — FFT correctness + spindle chatter detection.
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <random>
#include <vector>

#include "wproc/spindle.hpp"

using namespace wproc;

static int g_total = 0, g_failed = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        ++g_total;                                                      \
        if (!(cond)) {                                                  \
            ++g_failed;                                                 \
            std::cerr << "FAIL [" << __LINE__ << "]: " #cond "\n";      \
        }                                                               \
    } while (0)

constexpr double kPi = 3.14159265358979323846;

static void test_fft_roundtrip() {
    std::vector<std::complex<double>> x(64);
    std::mt19937 gen(1);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::generate(x.begin(), x.end(),
                  [&] { return std::complex<double>(u(gen), u(gen)); });
    std::vector<std::complex<double>> orig = x;

    spindle::fft(x, false);
    spindle::fft(x, true);
    double max_err = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] /= static_cast<double>(x.size());   // unnormalized inverse
        max_err = std::max(max_err, std::abs(x[i] - orig[i]));
    }
    std::cout << "FFT roundtrip max err = " << max_err << "\n";
    CHECK(max_err < 1e-9);

    // A pure tone must peak in the expected bin.
    const double fs = 20000.0;
    std::vector<double> tone(4096);
    for (std::size_t i = 0; i < tone.size(); ++i)
        tone[i] = std::sin(2 * kPi * 500.0 * i / fs);
    spindle::Spectrum s = spindle::power_spectrum(tone, fs);
    std::size_t kmax = 1;
    for (std::size_t k = 1; k < s.power.size(); ++k)
        if (s.power[k] > s.power[kmax]) kmax = k;
    std::cout << "tone peak at " << s.freq[kmax] << " Hz\n";
    CHECK(std::fabs(s.freq[kmax] - 500.0) < 6.0);
}

static std::vector<double> harmonic_signal(double fs, int n, double f0,
                                           unsigned seed, double chatter_amp,
                                           double chatter_f) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> noise(0.0, 0.05);
    std::vector<double> sig(n);
    for (int i = 0; i < n; ++i) {
        const double t = i / fs;
        double v = 1.0 * std::sin(2 * kPi * f0 * t) +
                   0.5 * std::sin(2 * kPi * 2 * f0 * t) +
                   0.3 * std::sin(2 * kPi * 3 * f0 * t) + noise(gen);
        if (chatter_amp > 0.0) v += chatter_amp * std::sin(2 * kPi * chatter_f * t);
        sig[i] = v;
    }
    return sig;
}

static void test_chatter() {
    const double fs = 20000.0;
    const double rpm = 30000.0;      // 500 Hz spindle
    const int n = 4096;

    // Healthy: harmonics of 500 Hz + noise, no chatter tone.
    auto healthy = harmonic_signal(fs, n, 500.0, 7, 0.0, 0.0);
    auto h = spindle::analyze_vibration(healthy, fs, rpm);
    std::cout << "healthy: dom=" << h.dominant_freq << " chatter_f=" << h.chatter_freq
              << " ratio=" << h.chatter_ratio << " chatter=" << h.chatter << "\n";
    CHECK(std::fabs(h.dominant_freq - 500.0) < 8.0);   // dominated by spindle order
    CHECK(!h.chatter);
    CHECK(h.rms > 0.0);

    // Chatter: add a strong non-harmonic 3200 Hz resonance (dominates harmonics).
    auto bad = harmonic_signal(fs, n, 500.0, 7, 1.2, 3200.0);
    auto c = spindle::analyze_vibration(bad, fs, rpm);
    std::cout << "chatter: chatter_f=" << c.chatter_freq << " ratio=" << c.chatter_ratio
              << " chatter=" << c.chatter << "\n";
    CHECK(c.chatter);
    CHECK(std::fabs(c.chatter_freq - 3200.0) < 12.0);
    CHECK(c.chatter_ratio > h.chatter_ratio);
}

int main() {
    test_fft_roundtrip();
    test_chatter();
    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "SPINDLE OK\n";
    return 0;
}
