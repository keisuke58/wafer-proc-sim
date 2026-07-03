// laser_groove.hpp — laser ablation-groove inspection (top-view).
//
// Laser grooving (DFL-style ablation) cuts a channel into the wafer's top
// dielectric/metal stack, often ahead of blade dicing on low-k devices. From a
// top-view grayscale image of a straight vertical groove this recovers, with no
// external dependencies:
//
//   * groove width (px and µm via a supplied scale) and its centre column;
//   * a depth proxy — how dark the groove floor is relative to the substrate
//     (deeper ablation reads darker in transmission/reflection);
//   * the heat-affected-zone (HAZ) width — the discoloured transition band
//     flanking the groove;
//   * a debris / spatter index — the fraction of substrate pixels thrown off
//     the local level by recast/ejecta.
//
// It renders an annotated overlay (groove edges, HAZ band, debris) and returns a
// PASS/FAIL verdict against a spec.
#pragma once

#include "wproc/image.hpp"

namespace wproc::laser {

struct GrooveSpec {
    double um_per_px = 0.5;
    double target_width_um = 20.0;
    double width_tol_um = 4.0;      // |width - target| must be within this
    double max_haz_um = 6.0;        // per-side HAZ limit
    double max_debris_index = 0.02; // fraction of substrate pixels
    double min_depth_index = 0.30;  // contrast floor (too shallow ⇒ under-ablated)
};

struct GrooveResult {
    bool found = false;
    double width_px = 0.0;
    double width_um = 0.0;
    double center_px = 0.0;
    double depth_index = 0.0;   // 0 (no groove) .. 1 (fully dark floor)
    double haz_um = 0.0;        // per-side HAZ width
    double debris_index = 0.0;  // fraction of substrate pixels flagged as ejecta
    bool ok = false;            // meets the spec
};

// Inspect a top-view image of a single vertical ablation groove (dark groove on
// a brighter substrate).
GrooveResult inspect_groove(const Image& gray, const GrooveSpec& spec = GrooveSpec{});

// Annotated overlay: green = groove edges, amber = HAZ band, red = debris pixels.
ColorImage annotate_groove(const Image& gray, const GrooveResult& r,
                           const GrooveSpec& spec = GrooveSpec{});

}  // namespace wproc::laser
