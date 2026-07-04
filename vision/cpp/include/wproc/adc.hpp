// adc.hpp — automatic defect classification (ADC) pipeline.
//
// The end-to-end inspection workflow that ties the AI classifier to a product
// decision: take detected defect patches with their die coordinates, classify
// each with the from-scratch CNN, then aggregate into
//
//   * a defect-type Pareto (counts per class),
//   * a classified wafer map — a die fails if it carries a "killer" defect —
//     built on the existing wafer-map / yield tooling,
//   * overall and centre-vs-edge yield.
//
// This is the capstone: CNN inference + connected-components-style observations
// + wafer-map yield, composed into one pipeline. Pure C++17.
#pragma once

#include <array>
#include <vector>

#include "wproc/cnn.hpp"
#include "wproc/image.hpp"
#include "wproc/wafer_map.hpp"

namespace wproc::adc {

// One detected defect: its die cell and the image patch to classify.
struct Observation {
    int col = 0;
    int row = 0;
    Image patch;   // 20x20 grayscale
};

struct Params {
    int cols = 12;
    int rows = 12;
    // Which classes kill a die (index by cnn::Defect; good=0 is never a killer).
    std::array<bool, 4> killer = {false, true, true, true};
    double center_zone_frac = 0.66;
};

struct Result {
    int n_defects = 0;
    std::array<int, 4> class_counts = {0, 0, 0, 0};  // Pareto (good..contamination)
    int top_defect_class = 0;      // dominant defect class (1..3), 0 if none
    wafer::WaferResult wafer;      // die pass/fail map (killer defect => fail)
    double yield_pct = 0.0;
    std::vector<int> defect_class; // classification per observation
};

// Run the ADC pipeline: classify every observation with `net`, then bin per die
// and build the classified wafer map + yield.
Result run(const cnn::DefectNet& net, const std::vector<Observation>& obs,
           const Params& params = Params{});

}  // namespace wproc::adc
