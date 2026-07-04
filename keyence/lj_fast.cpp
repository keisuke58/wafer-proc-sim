// lj_fast.cpp — the robust sub-pixel extractor, ported to C++17 for real time.
//
// A 2D laser profiler runs the extractor once per detector COLUMN, thousands
// of columns per profile, thousands of profiles per second. The Python
// reference (keyence/lj_signal_chain.py) proves the algorithm; this port
// proves it fits a real-time budget.
//
// Identical algorithm to Python extract_robust():
//   1. threshold at 0.3 x column max
//   2. connected lobes above threshold
//   3. candidates = lobes with >= 30 % of the strongest lobe's energy
//   4. keep the SHORTEST-OPTICAL-PATH lobe (smallest mean row) — multipath
//      always lands at longer path
//   5. thresholded centroid of that lobe (saturation plateau is symmetric,
//      the unsaturated wings carry the sub-pixel information)
//
// Build:  g++ -O3 -march=native -std=c++17 -Wall -Wextra -o lj_fast lj_fast.cpp
// Run:    ./lj_fast            (prints a JSON block)
#include <algorithm>
#include <chrono>
#include <numeric>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr int   kNPix     = 256;
constexpr float kUmPerPix = 4.0f;
constexpr float kSpotSig  = 3.2f;
constexpr float kSpeckle  = 0.18f;
constexpr float kReadNoise= 0.012f;

// xorshift128+ — fast deterministic RNG for the synthesis side.
struct Rng {
    uint64_t s0, s1;
    explicit Rng(uint64_t seed) : s0(seed ^ 0x9E3779B97F4A7C15ULL),
                                  s1(seed * 0xBF58476D1CE4E5B9ULL + 1) {}
    uint64_t next() {
        uint64_t x = s0, y = s1;
        s0 = y; x ^= x << 23;
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        return s1 + y;
    }
    float uniform() { return static_cast<float>(next() >> 11) * 0x1.0p-53f; }
    float gauss() {                       // Box-Muller (one value per call)
        float u1 = std::max(uniform(), 1e-12f), u2 = uniform();
        return std::sqrt(-2.0f * std::log(u1)) *
               std::cos(6.28318530718f * u2);
    }
};

// Synthesize one detector column (same physics as the Python renderer).
void render_column(float z_um, float reflectance, float multipath,
                   Rng& rng, float* col) {
    const float center = kNPix / 2.0f + z_um / kUmPerPix;
    const float gain = 1.6f;
    for (int r = 0; r < kNPix; ++r) {
        float d = (static_cast<float>(r) - center) / kSpotSig;
        float sig = std::exp(-0.5f * d * d);
        if (multipath > 0.0f) {
            float dm = (static_cast<float>(r) - center - 18.0f) / (kSpotSig * 1.8f);
            sig += multipath * std::exp(-0.5f * dm * dm);
        }
        sig *= reflectance * gain;
        sig *= 1.0f + kSpeckle * rng.gauss();
        sig += kReadNoise * rng.gauss();
        col[r] = std::clamp(sig, 0.0f, 1.0f);
    }
}

// The product-grade extractor — single pass over lobes, no allocations.
float extract_robust(const float* col) {
    float cmax = 0.0f;
    for (int r = 0; r < kNPix; ++r) cmax = std::max(cmax, col[r]);
    const float thr = 0.3f * cmax;

    // Pass 1: find lobes, track each lobe's energy and the strongest energy.
    struct Lobe { int lo, hi; float sum; };
    Lobe lobes[16]; int nl = 0;
    float smax = 0.0f;
    int r = 0;
    while (r < kNPix && nl < 16) {
        if (col[r] > thr) {
            int lo = r; float sum = 0.0f;
            while (r < kNPix && col[r] > thr) { sum += col[r]; ++r; }
            lobes[nl++] = {lo, r - 1, sum};
            smax = std::max(smax, sum);
        } else {
            ++r;
        }
    }
    if (nl == 0) return NAN;

    // Pass 2: earliest credible lobe = shortest optical path (multipath physics).
    // The strongest lobe always satisfies sum >= 0.3*smax, so a credible lobe
    // always exists; default to lobe 0 to make that invariant explicit.
    const Lobe* best = &lobes[0];
    for (int i = 0; i < nl; ++i)
        if (lobes[i].sum >= 0.3f * smax) { best = &lobes[i]; break; }

    // Thresholded centroid of the chosen lobe.
    float wsum = 0.0f, xsum = 0.0f;
    for (int k = best->lo; k <= best->hi; ++k) {
        float w = col[k] - thr;
        wsum += w; xsum += static_cast<float>(k) * w;
    }
    return (xsum / wsum - kNPix / 2.0f) * kUmPerPix;
}

double rms(const std::vector<float>& v) {
    const double s = std::accumulate(v.begin(), v.end(), 0.0,
        [](double acc, float x) { return acc + static_cast<double>(x) * x; });
    return std::sqrt(s / static_cast<double>(v.size()));
}

}  // namespace

int main() {
    // ── accuracy parity vs the Python reference ─────────────────────────────
    struct Scen { const char* name; float refl, mp; };
    const Scen scens[] = {{"bright_clean", 1.0f, 0.0f},
                          {"dark_multipath", 0.15f, 0.45f}};
    float col[kNPix];
    std::printf("{\n  \"accuracy_rms_um\": {");
    for (size_t s = 0; s < 2; ++s) {
        Rng rng(42 + s);
        std::vector<float> errs;
        errs.reserve(4000);
        for (int i = 0; i < 4000; ++i) {
            float z = (rng.uniform() * 300.0f) - 150.0f;
            render_column(z, scens[s].refl, scens[s].mp, rng, col);
            float zh = extract_robust(col);
            if (std::isfinite(zh)) errs.push_back(zh - z);
        }
        std::printf("%s\"%s\": %.3f", s ? ", " : "", scens[s].name, rms(errs));
    }
    std::printf("},\n");

    // ── throughput: synthesize once, time extraction alone ─────────────────
    constexpr int kCols = 2'000'000;
    std::vector<float> bank(static_cast<size_t>(kCols) * kNPix);
    {
        Rng rng(7);
        for (int i = 0; i < kCols; ++i) {
            float z = (rng.uniform() * 300.0f) - 150.0f;
            float mp = (i % 3 == 0) ? 0.45f : 0.0f;
            float rf = (i % 2 == 0) ? 1.0f : 0.15f;
            render_column(z, rf, mp, rng, &bank[static_cast<size_t>(i) * kNPix]);
        }
    }
    volatile float sink = 0.0f;                 // defeat dead-code elimination
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kCols; ++i)
        sink = sink + extract_robust(&bank[static_cast<size_t>(i) * kNPix]);
    const auto t1 = std::chrono::steady_clock::now();
    const double ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const double ns_per_col = ns / kCols;
    const double cols_per_s = 1e9 / ns_per_col;
    const double profiles_per_s_3200 = cols_per_s / 3200.0;  // LJ-X class width

    std::printf("  \"columns_timed\": %d,\n", kCols);
    std::printf("  \"ns_per_column\": %.1f,\n", ns_per_col);
    std::printf("  \"megacolumns_per_s\": %.2f,\n", cols_per_s / 1e6);
    std::printf("  \"profiles_per_s_at_3200pt\": %.0f,\n", profiles_per_s_3200);
    std::printf("  \"checksum\": %.3f\n}\n", static_cast<double>(sink));
    return 0;
}
