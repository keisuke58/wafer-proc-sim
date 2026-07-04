// pkg.hpp — advanced-packaging bump / pillar array inspection.
//
// After dicing/grinding, dies go to advanced packaging (flip-chip, 3D-IC,
// hybrid bonding) where solder-bump / Cu-pillar arrays must be inspected before
// bonding. From a top-view image of a bump array this recovers, reusing the
// from-scratch threshold + connected-components tools:
//
//   * bump count, diameter distribution (mean, CV),
//   * array pitch and a placement-accuracy RMS (residual from the ideal grid),
//   * a coplanarity proxy from per-bump brightness (a short/missing bump reads
//     dimmer),
//   * missing-bump count against the inferred rectangular grid.
//
// Renders an annotated overlay and returns a PASS/FAIL verdict.
#pragma once

#include "wproc/image.hpp"
#include <vector>

namespace wproc::pkg {

struct BumpSpec {
    double um_per_px = 1.0;
    double target_diameter_um = 10.0;
    double diameter_tol_um = 2.5;     // per-bump size tolerance
    double max_diameter_cv = 0.12;    // array size-uniformity limit
    double min_coplanarity = 0.85;    // 1 - brightness CV
    int max_missing = 0;
    long min_area_px = 8;             // ignore specks below this
};

struct Bump {
    double x = 0.0, y = 0.0;       // centroid [px]
    double diameter_um = 0.0;
    double brightness = 0.0;       // mean intensity (coplanarity proxy)
    bool ok = false;               // within the diameter tolerance
};

struct BumpResult {
    int count = 0;
    double mean_diameter_um = 0.0;
    double diameter_cv = 0.0;
    double pitch_um = 0.0;
    double coplanarity = 0.0;      // 1 - brightness CV, clamped to [0,1]
    double placement_rms_um = 0.0; // RMS residual from the fitted grid
    int missing = 0;               // grid nodes with no bump
    bool ok = false;               // meets the spec
    std::vector<Bump> bumps;
};

// Inspect a top-view bump-array image (bright bumps on a darker substrate).
BumpResult inspect_bumps(const Image& gray, const BumpSpec& spec = BumpSpec{});

// Annotated overlay: green = good bump, red = out-of-tolerance / dim bump.
ColorImage annotate_bumps(const Image& gray, const BumpResult& r);

}  // namespace wproc::pkg
