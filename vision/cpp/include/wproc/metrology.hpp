// metrology.hpp — measurement-quality tooling for the kerf inspector.
//
// Two capabilities a metrology tool is expected to have:
//   1. Repeatability / uncertainty: run the inspection over many noisy
//      realizations of the same scene and report the 1-sigma spread of the
//      measured kerf width (Gage-R&R-style precision).
//   2. Pixel-scale calibration: recover microns-per-pixel from an image of a
//      target with a known feature pitch, so width results are traceable.
#pragma once

#include "wproc/image.hpp"
#include "wproc/kerf_inspect.hpp"

namespace wproc::metro {

struct Repeatability {
    int n = 0;              // trials that found a kerf
    double width_mean_um = 0.0;
    double width_sd_um = 0.0;   // 1-sigma (sample standard deviation)
    double width_min_um = 0.0;
    double width_max_um = 0.0;
    double width_cv_pct = 0.0;  // coefficient of variation = sd/mean * 100
    double chip_mean_um = 0.0;
    double chip_sd_um = 0.0;
};

// Add zero-mean Gaussian noise (given sigma) to `base` `n` times and inspect
// each realization; aggregate the width/chip statistics. Deterministic in seed.
Repeatability repeatability(const Image& base, const KerfInspector& insp, int n,
                            float noise_sigma, unsigned seed = 1);

struct Calibration {
    bool ok = false;
    double pixel_pitch_px = 0.0;  // detected feature period in pixels
    double um_per_px = 0.0;       // known_pitch_um / pixel_pitch_px
};

// Estimate microns-per-pixel from a periodic calibration target (e.g. a line
// grating) whose true pitch is `known_pitch_um`. Finds the dominant period of
// the signed Sobel-x column projection by autocorrelation.
Calibration estimate_um_per_px(const Image& target, double known_pitch_um,
                               int min_period_px = 4, int max_period_px = 0);

}  // namespace wproc::metro
