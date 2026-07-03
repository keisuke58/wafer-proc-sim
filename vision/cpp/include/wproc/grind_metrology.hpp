// grind_metrology.hpp — back-grind / thinning thickness-map metrology.
//
// After back-grinding (DGP/DFG, or the DBG/TAIKO thin-wafer flows) the key
// quality question is how flat and uniform the thinned wafer is. Given a
// thickness map sampled over the wafer disk (microns, one value per grid cell),
// this computes the classic geometry metrics:
//
//   TTV   total thickness variation = max - min thickness inside the disk
//   WARP  peak-to-valley of the surface after removing best-fit tilt
//   BOW   signed centre deflection of that de-tilted surface
//   sigma 1-sigma thickness spread; uniformity = 100*(1 - sigma/mean)
//
// The reference is a least-squares plane fit over the disk, so WARP/BOW measure
// true non-planarity rather than a stage tilt. Pure C++17 (no OpenCV).
#pragma once

#include "wproc/image.hpp"

namespace wproc::grind {

struct ThicknessStats {
    int n_inside = 0;          // grid cells inside the wafer disk
    double mean_um = 0.0;
    double sigma_um = 0.0;     // 1-sigma thickness spread
    double min_um = 0.0;
    double max_um = 0.0;
    double ttv_um = 0.0;       // max - min
    double warp_um = 0.0;      // PV of de-tilted surface
    double bow_um = 0.0;       // centre deflection of de-tilted surface (signed)
    double uniformity_pct = 0.0;  // 100*(1 - sigma/mean), clamped to [0, 100]
    // Best-fit reference plane thickness(x,y) ≈ a*x + b*y + c (grid coords).
    double plane_a = 0.0, plane_b = 0.0, plane_c = 0.0;
};

// Analyze a thickness map (values in microns). Only cells within the inscribed
// wafer circle of radius `radius_frac` * (min(w,h)/2) are considered; cells at
// or below `invalid_below_um` are treated as no-data and skipped (e.g. a 0 mask
// outside the wafer). Throws std::invalid_argument if the map has no valid disk
// cells.
ThicknessStats analyze_thickness(const Image& thickness_um,
                                 double radius_frac = 0.98,
                                 double invalid_below_um = 1.0);

// Render a thickness map as a diverging blue→white→red color image centred on
// the mean (blue = thinner, red = thicker), with out-of-disk / no-data cells
// drawn in neutral gray. `span_um` sets the ± range mapped to full saturation;
// if <= 0 it defaults to 3*sigma of the valid cells.
ColorImage render_thickness(const Image& thickness_um, double radius_frac = 0.98,
                            double invalid_below_um = 1.0, double span_um = 0.0);

}  // namespace wproc::grind
