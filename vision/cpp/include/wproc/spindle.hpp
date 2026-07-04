// spindle.hpp — spindle vibration spectrum + chatter detection.
//
// A dicing/grinding spindle runs on an air bearing at tens of thousands of RPM.
// Healthy cutting shows vibration energy concentrated at the spindle rotation
// frequency and its harmonics; a rising, non-harmonic peak (a structural
// resonance excited by the cut) is *chatter*, which scars the kerf. From a
// vibration time series this computes the power spectrum with an in-house
// radix-2 FFT, then flags chatter by comparing the strongest non-harmonic peak
// against the harmonic content.
//
// Pure C++17, no external dependencies (the FFT is hand-written).
#pragma once

#include <complex>
#include <vector>

namespace wproc::spindle {

// In-place iterative radix-2 Cooley-Tukey FFT. `a.size()` must be a power of 2.
// `inverse` computes the unnormalized inverse transform.
void fft(std::vector<std::complex<double>>& a, bool inverse = false);

struct Spectrum {
    double fs = 0.0;                 // sample rate [Hz]
    double df = 0.0;                 // frequency bin width [Hz]
    std::vector<double> freq;        // bin centre frequencies [Hz], length N/2
    std::vector<double> power;       // single-sided power per bin (Hann-windowed)
};

// One-sided power spectrum of a real signal. The signal is Hann-windowed and
// zero-padded to the next power of two. Returns the DC..Nyquist bins.
Spectrum power_spectrum(const std::vector<double>& signal, double fs);

struct ChatterResult {
    double rms = 0.0;                 // time-domain RMS of the signal
    double spindle_freq = 0.0;        // rpm/60 [Hz]
    double dominant_freq = 0.0;       // frequency of the largest spectral peak
    double harmonic_power = 0.0;      // power at spindle harmonics
    double chatter_freq = 0.0;        // strongest non-harmonic peak frequency
    double chatter_power = 0.0;       // its power
    double chatter_ratio = 0.0;       // chatter_power / harmonic_power
    bool chatter = false;             // ratio exceeds the threshold
};

// Analyze a vibration signal for chatter given the spindle speed. A peak within
// `harmonic_tol_hz` of any k*spindle_freq counts as harmonic; the largest peak
// outside all harmonics is the chatter candidate. Chatter is flagged when its
// power relative to the harmonic content exceeds `ratio_threshold`.
ChatterResult analyze_vibration(const std::vector<double>& signal, double fs,
                                double spindle_rpm,
                                double harmonic_tol_hz = 50.0,
                                double ratio_threshold = 0.5);

}  // namespace wproc::spindle
