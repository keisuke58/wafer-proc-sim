// test_cnn.cpp — from-scratch C++ CNN inference matches the trained model.
//
// Loads the weights + labelled test set exported by ml/defect_cnn_train.py and
// checks the C++ inference engine reproduces the training-time accuracy. If the
// data files are absent (e.g. a checkout without the trained artifacts) the
// test skips cleanly rather than failing.
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

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
    const std::string wpath = dir + "/defect_cnn_weights.bin";
    const std::string tpath = dir + "/defect_cnn_testset.bin";

    std::ifstream tf(tpath, std::ios::binary);
    cnn::DefectNet net;
    try {
        net.load(wpath);
    } catch (const std::exception& e) {
        std::cout << "SKIP: weights not found (" << e.what() << ")\n";
        return 0;
    }
    if (!tf) {
        std::cout << "SKIP: test set not found at " << tpath << "\n";
        return 0;
    }

    std::int32_t n = 0, h = 0, w = 0;
    tf.read(reinterpret_cast<char*>(&n), 4);
    tf.read(reinterpret_cast<char*>(&h), 4);
    tf.read(reinterpret_cast<char*>(&w), 4);
    CHECK(h == cnn::DefectNet::kIn && w == cnn::DefectNet::kIn);
    CHECK(n > 0);

    std::vector<float> pix(static_cast<std::size_t>(n) * h * w);
    tf.read(reinterpret_cast<char*>(pix.data()),
            static_cast<std::streamsize>(pix.size() * sizeof(float)));
    std::vector<std::int32_t> labels(n);
    tf.read(reinterpret_cast<char*>(labels.data()),
            static_cast<std::streamsize>(n * sizeof(std::int32_t)));

    int correct = 0;
    for (int i = 0; i < n; ++i) {
        Image patch(w, h);
        const float* p = &pix[static_cast<std::size_t>(i) * h * w];
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) patch.at(x, y) = p[y * w + x];
        cnn::Prediction pr = net.classify(patch);
        if (pr.cls == labels[i]) ++correct;
    }
    const double acc = static_cast<double>(correct) / n;
    std::cout << "CNN C++ inference accuracy = " << acc * 100.0 << "% (" << correct
              << "/" << n << ")\n";
    CHECK(acc > 0.90);   // matches the ~97% trained accuracy

    // Sanity: probabilities sum to 1 and confidence is the max.
    Image z(cnn::DefectNet::kIn, cnn::DefectNet::kIn, 130.0f);
    cnn::Prediction p0 = net.classify(z);
    const float sum = std::accumulate(p0.probs.begin(), p0.probs.end(), 0.0f);
    CHECK(std::abs(sum - 1.0f) < 1e-4f);
    CHECK(p0.confidence == p0.probs[p0.cls]);

    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "CNN OK\n";
    return 0;
}
