// stealth_ir.hpp — stealth-dicing (SD) subsurface inspection from IR images.
//
// Silicon is transparent in the near-IR, so a transmission image of a stealth
// cut shows the internal modified layers (SD layers) the laser writes below the
// surface, and any cracks propagating from them toward the wafer face. This
// module analyses a cross-section IR image (y = depth into the wafer, x =
// lateral position): it locates the SD layers, measures their count/pitch, and
// quantifies crack propagation toward the surface.
#pragma once

#include <vector>

#include "wproc/image.hpp"

namespace wproc::stealth {

struct SdResult {
    int layer_count = 0;
    std::vector<double> layer_depths_px;  // SD-layer depths (rows), top-first
    double mean_pitch_px = 0.0;           // mean spacing between layers
    double max_crack_px = 0.0;            // longest crack above the top layer
    bool crack_reaches_surface = false;   // crack within surface_margin of y=0
};

struct SdParams {
    float blur_sigma = 1.0f;
    double min_layer_prominence = 0.25;  // peak height vs projection range
    int surface_margin_px = 3;           // "reaches surface" tolerance
};

// Inspect a cross-section IR image (brighter = SD-modified / crack).
SdResult inspect_stealth(const Image& img, SdParams params = SdParams{});

}  // namespace wproc::stealth
