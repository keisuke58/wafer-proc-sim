// cnn.cpp — from-scratch CNN inference (conv/relu/maxpool/dense/softmax).
#include "wproc/cnn.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace wproc::cnn {

namespace {
constexpr int C1 = 8;    // conv1 filters
constexpr int K = 5;     // conv1 kernel
constexpr int CONV_OUT = DefectNet::kIn - K + 1;  // 16
constexpr int POOL = 4;
constexpr int POOL_OUT = CONV_OUT / POOL;         // 4
constexpr int FLAT = C1 * POOL_OUT * POOL_OUT;    // 128
constexpr int NC = DefectNet::kClasses;           // 4
}  // namespace

const char* defect_name(int cls) {
    switch (cls) {
        case 0: return "good";
        case 1: return "chipping";
        case 2: return "crack";
        case 3: return "contamination";
        default: return "unknown";
    }
}

void DefectNet::load(const std::string& weights_path) {
    std::ifstream f(weights_path, std::ios::binary);
    if (!f) throw std::runtime_error("DefectNet::load: cannot open " + weights_path);

    const std::size_t n_conv1_w = std::size_t(C1) * 1 * K * K;  // 200
    const std::size_t n_fc_w = std::size_t(NC) * FLAT;          // 512
    const std::size_t total = n_conv1_w + C1 + n_fc_w + NC;     // 724

    std::vector<float> all(total);
    f.read(reinterpret_cast<char*>(all.data()),
           static_cast<std::streamsize>(total * sizeof(float)));
    if (static_cast<std::size_t>(f.gcount()) != total * sizeof(float))
        throw std::runtime_error("DefectNet::load: unexpected weight file size");

    std::size_t o = 0;
    conv1_w_.assign(all.begin() + o, all.begin() + o + n_conv1_w); o += n_conv1_w;
    conv1_b_.assign(all.begin() + o, all.begin() + o + C1);        o += C1;
    fc_w_.assign(all.begin() + o, all.begin() + o + n_fc_w);       o += n_fc_w;
    fc_b_.assign(all.begin() + o, all.begin() + o + NC);
    loaded_ = true;
}

std::array<float, 4> DefectNet::forward(const Image& patch) const {
    if (!loaded_) throw std::runtime_error("DefectNet::forward: weights not loaded");
    if (patch.width() != kIn || patch.height() != kIn)
        throw std::runtime_error("DefectNet::forward: patch must be 20x20");

    // Scale input to [0,1].
    float in[kIn][kIn];
    for (int y = 0; y < kIn; ++y)
        for (int x = 0; x < kIn; ++x)
            in[y][x] = std::clamp(patch.at(x, y) / 255.0f, 0.0f, 1.0f);

    // Conv1 (valid) + ReLU, then max-pool 4x4 -> pooled[C1][POOL_OUT][POOL_OUT].
    std::vector<float> pooled(FLAT, 0.0f);
    for (int c = 0; c < C1; ++c) {
        const float* w = &conv1_w_[std::size_t(c) * K * K];
        const float bias = conv1_b_[c];
        for (int py = 0; py < POOL_OUT; ++py) {
            for (int px = 0; px < POOL_OUT; ++px) {
                float best = -1e30f;
                for (int qy = 0; qy < POOL; ++qy) {
                    for (int qx = 0; qx < POOL; ++qx) {
                        const int oy = py * POOL + qy;  // conv output coords
                        const int ox = px * POOL + qx;
                        float acc = bias;
                        for (int ky = 0; ky < K; ++ky)
                            for (int kx = 0; kx < K; ++kx)
                                acc += w[ky * K + kx] * in[oy + ky][ox + kx];
                        const float relu = acc > 0.0f ? acc : 0.0f;
                        best = std::max(best, relu);
                    }
                }
                // channel-major flatten index: c,py,px
                pooled[(std::size_t(c) * POOL_OUT + py) * POOL_OUT + px] = best;
            }
        }
    }

    // Dense -> logits -> softmax.
    std::array<float, NC> logit{};
    for (int o = 0; o < NC; ++o) {
        float acc = fc_b_[o];
        const float* w = &fc_w_[std::size_t(o) * FLAT];
        for (int k = 0; k < FLAT; ++k) acc += w[k] * pooled[k];
        logit[o] = acc;
    }
    const float m = *std::max_element(logit.begin(), logit.end());
    std::array<float, 4> prob{};
    float sum = 0.0f;
    for (int o = 0; o < NC; ++o) { prob[o] = std::exp(logit[o] - m); sum += prob[o]; }
    for (int o = 0; o < NC; ++o) prob[o] /= (sum > 0.0f ? sum : 1.0f);
    return prob;
}

Prediction DefectNet::classify(const Image& patch) const {
    Prediction p;
    p.probs = forward(patch);
    p.cls = static_cast<int>(std::max_element(p.probs.begin(), p.probs.end()) -
                             p.probs.begin());
    p.confidence = p.probs[p.cls];
    return p;
}

}  // namespace wproc::cnn
