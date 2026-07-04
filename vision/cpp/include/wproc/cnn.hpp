// cnn.hpp — from-scratch CNN inference for wafer-defect classification.
//
// A compact convolutional classifier for 20x20 grayscale defect patches, run
// with a hand-written inference engine (conv / ReLU / max-pool / dense /
// softmax) — no deep-learning framework. Weights are trained offline in Python
// and loaded from a flat float32 blob; this is the realistic edge-deployment
// split (train in Python, infer in C++).
//
// Architecture:
//   input   1 x 20 x 20  (pixel values scaled to [0,1])
//   conv1   8 filters, 1x5x5, valid  -> 8 x 16 x 16, ReLU
//   pool1   max 4x4, stride 4        -> 8 x 4 x 4
//   flatten                          -> 128   (channel-major: c,y,x)
//   dense   128 -> 4
//   softmax                          -> class probabilities
//
// Classes: 0 good, 1 chipping, 2 crack, 3 contamination.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "wproc/image.hpp"

namespace wproc::cnn {

enum class Defect { Good = 0, Chipping = 1, Crack = 2, Contamination = 3 };
const char* defect_name(int cls);

struct Prediction {
    int cls = 0;
    float confidence = 0.0f;       // softmax probability of the winning class
    std::array<float, 4> probs{};  // full class distribution
};

// The trained defect classifier. Load weights from the flat float32 blob
// exported by ml/defect_cnn_train.py.
class DefectNet {
public:
    static constexpr int kIn = 20, kClasses = 4;

    // Load weights (throws std::runtime_error on I/O or size mismatch).
    void load(const std::string& weights_path);
    bool loaded() const { return loaded_; }

    // Classify a 20x20 grayscale patch (values ~0..255; scaled internally).
    Prediction classify(const Image& patch20) const;

    // Raw forward pass returning the 4 class probabilities.
    std::array<float, 4> forward(const Image& patch20) const;

private:
    bool loaded_ = false;
    std::vector<float> conv1_w_;  // [8][1][5][5]
    std::vector<float> conv1_b_;  // [8]
    std::vector<float> fc_w_;     // [4][128]
    std::vector<float> fc_b_;     // [4]
};

}  // namespace wproc::cnn
