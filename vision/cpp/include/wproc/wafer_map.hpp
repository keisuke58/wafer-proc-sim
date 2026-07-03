// wafer_map.hpp — full-wafer defect map and die yield binning.
//
// Aggregates per-die inspection results (e.g. the max chipping measured on each
// die's streets) into a wafer map: which grid cells fall inside the round wafer,
// which dies pass/fail against a spec, the overall yield, and a centre-vs-edge
// zone breakdown that surfaces edge-ring defect signatures. Renders both an
// ASCII map and a color wafer map. Pure C++ (no OpenCV).
#pragma once

#include <string>
#include <vector>

#include "wproc/image.hpp"

namespace wproc::wafer {

struct Die {
    int col = 0, row = 0;
    bool inside = false;   // within the wafer circle
    bool pass = false;     // inside && value within spec
    double norm_r = 0.0;   // normalized radius 0 (centre) .. 1 (edge)
    double value = 0.0;    // measured quantity (e.g. max chip um)
};

struct WaferResult {
    int cols = 0, rows = 0;
    std::vector<Die> dies;         // row-major, size cols*rows
    int total_inside = 0;
    int good = 0;
    double yield_pct = 0.0;
    double center_yield_pct = 0.0; // dies with norm_r <= center_zone_frac
    double edge_yield_pct = 0.0;   // dies with norm_r >  center_zone_frac
};

// Classify a cols x rows die grid from per-die values (row-major). A die passes
// if it lies inside the inscribed wafer circle and value <= limit.
WaferResult classify(int cols, int rows, const std::vector<double>& values,
                     double limit, double center_zone_frac = 0.66);

// ASCII map: '.' outside, 'o' pass, 'X' fail. One row per grid row.
std::string ascii_map(const WaferResult& w);

// Color wafer map: gray outside, green pass, red fail, each die a cell_px block.
ColorImage render_map(const WaferResult& w, int cell_px = 12);

}  // namespace wproc::wafer
