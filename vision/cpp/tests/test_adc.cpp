// test_adc.cpp — ADC pipeline: classify defects, build classified wafer map.
//
// Uses the CNN weights + labelled patches from ml/defect_cnn_train.py. Skips
// cleanly if those artifacts are absent.
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "wproc/adc.hpp"
#include "wproc/cnn.hpp"
#include "wproc/image.hpp"

#ifndef WPROC_RESULTS_DIR
#define WPROC_RESULTS_DIR "results"
#endif

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

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : std::string(WPROC_RESULTS_DIR);
    cnn::DefectNet net;
    try {
        net.load(dir + "/defect_cnn_weights.bin");
    } catch (const std::exception& e) {
        std::cout << "SKIP: weights not found (" << e.what() << ")\n";
        return 0;
    }
    std::ifstream tf(dir + "/defect_cnn_testset.bin", std::ios::binary);
    if (!tf) { std::cout << "SKIP: test set not found\n"; return 0; }

    std::int32_t n = 0, h = 0, w = 0;
    tf.read(reinterpret_cast<char*>(&n), 4);
    tf.read(reinterpret_cast<char*>(&h), 4);
    tf.read(reinterpret_cast<char*>(&w), 4);
    std::vector<float> pix(static_cast<std::size_t>(n) * h * w);
    tf.read(reinterpret_cast<char*>(pix.data()),
            static_cast<std::streamsize>(pix.size() * sizeof(float)));
    std::vector<std::int32_t> labels(n);
    tf.read(reinterpret_cast<char*>(labels.data()),
            static_cast<std::streamsize>(n * sizeof(std::int32_t)));

    auto patch_of = [&](int i) {
        Image p(w, h);
        const float* q = &pix[static_cast<std::size_t>(i) * h * w];
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) p.at(x, y) = q[y * w + x];
        return p;
    };

    // ── General run: first 120 patches spread over a 12x12 die grid ──────────
    adc::Params params;  // 12x12, killer = {good:no, chipping/crack/contam:yes}
    std::vector<adc::Observation> obs;
    const int M = 120;
    for (int i = 0; i < M && i < n; ++i)
        obs.push_back({i % params.cols, (i / params.cols) % params.rows, patch_of(i)});
    adc::Result r = adc::run(net, obs, params);
    std::cout << "adc: n=" << r.n_defects << " counts=[" << r.class_counts[0] << ","
              << r.class_counts[1] << "," << r.class_counts[2] << "," << r.class_counts[3]
              << "] top=" << cnn::defect_name(r.top_defect_class)
              << " yield=" << r.yield_pct << "%\n";
    CHECK(r.n_defects == M);
    const int sum = std::accumulate(r.class_counts.begin(), r.class_counts.end(), 0);
    CHECK(sum == M);
    CHECK(r.top_defect_class >= 1 && r.top_defect_class <= 3);
    CHECK(r.yield_pct >= 0.0 && r.yield_pct <= 100.0);
    CHECK(r.wafer.total_inside > 0);

    // ── All-good dies pass; all-crack dies fail ──────────────────────────────
    std::vector<adc::Observation> good_obs, crack_obs;
    int gi = 0, ci = 0;
    for (int i = 0; i < n; ++i) {
        if (labels[i] == 0 && gi < 120) good_obs.push_back({gi % 12, gi / 12, patch_of(i)}), ++gi;
        if (labels[i] == 2 && ci < 120) crack_obs.push_back({ci % 12, ci / 12, patch_of(i)}), ++ci;
    }
    adc::Result gr = adc::run(net, good_obs, params);
    adc::Result cr = adc::run(net, crack_obs, params);
    std::cout << "good-only yield=" << gr.yield_pct << "%  crack-only yield="
              << cr.yield_pct << "%\n";
    CHECK(gr.yield_pct > 90.0);      // good patches don't kill dies
    CHECK(cr.yield_pct < 40.0);      // crack patches kill their dies
    CHECK(gr.yield_pct > cr.yield_pct);
    // Pareto: crack-only run is dominated by the crack class.
    CHECK(cr.top_defect_class == static_cast<int>(cnn::Defect::Crack));

    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "ADC OK\n";
    return 0;
}
