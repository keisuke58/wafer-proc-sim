// laser_sd.hpp — stealth-dicing SD-layer + heat-affected-zone (HAZ) analysis.
//
// Extends the stealth-dicing subsurface inspector (`wproc/stealth_ir.hpp`) with
// the quality metrics a stealth-dicing process engineer actually gates on:
//
//   * SD-layer count and inter-layer pitch uniformity (multi-pass stacks must be
//     evenly spaced for a clean break);
//   * HAZ thickness — the vertical extent of the modified/bright band around the
//     top SD layer (too thick ⇒ excess thermal damage);
//   * surface-crack risk — does a crack from the SD layer reach the wafer face,
//     and how close, scaled to a 0..1 risk.
//
// Reuses `stealth::inspect_stealth` for layer/crack detection, then adds the HAZ
// and uniformity measurements. Pure C++17.
#pragma once

#include "wproc/image.hpp"
#include "wproc/stealth_ir.hpp"

namespace wproc::laser {

struct SdQuality {
    int layer_count = 0;
    double mean_pitch_px = 0.0;
    double pitch_uniformity_pct = 100.0;  // 100*(1 - cv of inter-layer pitch)
    double haz_thickness_px = 0.0;        // vertical thickness of top modified band
    double max_crack_px = 0.0;
    bool crack_reaches_surface = false;
    double surface_risk = 0.0;            // 0 (clear) .. 1 (crack at surface)
    bool ok = true;                       // meets the spec
};

struct SdQualitySpec {
    stealth::SdParams sd;      // passed through to the subsurface inspector
    double max_haz_px = 12.0;  // HAZ thickness limit
    double min_pitch_uniformity_pct = 90.0;
};

// Inspect a cross-section IR image (y = depth, brighter = SD-modified / crack)
// and return SD-layer + HAZ quality with a PASS/FAIL verdict.
SdQuality inspect_sd_haz(const Image& img, const SdQualitySpec& spec = SdQualitySpec{});

}  // namespace wproc::laser
