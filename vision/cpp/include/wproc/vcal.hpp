// vcal.hpp — vision calibration + alignment-accuracy verification (Gage R&R).
//
// Before a machine-vision alignment system can be trusted, two things must be
// established:
//
//   1. Calibration — the map from camera *pixels* to stage *millimetres*. Here
//      that is an affine transform (scale + rotation + shear + translation)
//      fitted by least squares from fiducial correspondences, plus an optional
//      radial lens-distortion correction (Brown k1/k2) applied first.
//
//   2. Measurement-system capability — even a calibrated system is useless if it
//      cannot repeat. Gage R&R decomposes the observed variation of an alignment
//      measurement into repeatability (equipment), reproducibility (operator/
//      setup) and part-to-part, and reports %GRR and the number of distinct
//      categories (ndc) — the standard AIAG acceptance metrics.
//
// Pure C++17, no external dependencies (the linear algebra is hand-written).
#pragma once

#include <array>
#include <vector>

#include "wproc/image.hpp"

namespace wproc::vcal {

// ── Camera → stage calibration ──────────────────────────────────────────────

// A 2-D affine map [sx; sy] = A [px; py] + b, stored row-major as
// {a00,a01,b0, a10,a11,b1}.
struct Affine {
    std::array<double, 6> m = {1, 0, 0, 0, 1, 0};

    // Map a pixel coordinate to stage coordinates.
    double map_x(double px, double py) const { return m[0] * px + m[1] * py + m[2]; }
    double map_y(double px, double py) const { return m[3] * px + m[4] * py + m[5]; }
};

// One fiducial: its pixel location in the camera and its true stage position.
struct Correspondence {
    double px = 0.0, py = 0.0;   // pixel
    double sx = 0.0, sy = 0.0;   // stage [mm]
};

struct CalibResult {
    bool ok = false;
    Affine transform;
    double rms_um = 0.0;      // residual RMS over the fiducials [µm]
    double max_um = 0.0;      // worst residual [µm]
    double um_per_px = 0.0;   // mean pixel pitch implied by the fit [µm/px]
    double rotation_deg = 0.0;
};

// Fit the pixel→stage affine by least squares (needs >= 3 non-collinear points).
// Stage coordinates are in mm; residuals are reported in µm.
CalibResult calibrate(const std::vector<Correspondence>& pts);

// Brown radial distortion. Given the distortion centre (cx,cy) in pixels and a
// normalizing focal length `f` (px), undistort a point: pinhole = center +
// (1 + k1 r^2 + k2 r^4) * (p - center), with r = |p-center| / f.
struct Distortion {
    double cx = 0.0, cy = 0.0, f = 1.0, k1 = 0.0, k2 = 0.0;
};
void undistort_point(const Distortion& d, double px, double py,
                     double& ux, double& uy);

// ── Gage R&R (average-and-range method, AIAG) ───────────────────────────────

// A designed study: `parts` distinct wafers/fiducials, each measured by
// `operators` operators, `trials` times each. `values[o][p][t]` is the alignment
// error read (e.g. residual placement in µm). Layout is validated on use.
struct GageStudy {
    int parts = 0;
    int operators = 0;
    int trials = 0;
    // Indexed [operator][part][trial].
    std::vector<std::vector<std::vector<double>>> values;
};

struct GageResult {
    bool ok = false;
    double repeatability = 0.0;   // EV — equipment variation (1 sigma)
    double reproducibility = 0.0; // AV — appraiser variation (1 sigma)
    double grr = 0.0;             // sqrt(EV^2 + AV^2)
    double part_variation = 0.0;  // PV
    double total_variation = 0.0; // TV = sqrt(GRR^2 + PV^2)
    double pct_grr = 0.0;         // 100 * GRR / TV  (< 10 good, < 30 marginal)
    double pct_ev = 0.0;
    double pct_av = 0.0;
    double pct_pv = 0.0;
    int ndc = 0;                  // number of distinct categories = 1.41*PV/GRR
    bool acceptable = false;      // pct_grr < 30 and ndc >= 5
};

// Compute a Gage R&R from the study using the average-and-range method.
GageResult gage_rr(const GageStudy& s);

// ── Rendering ───────────────────────────────────────────────────────────────

// A simple calibration-residual quiver: fiducials as crosses with a magnified
// error vector to their fitted position (green small error, red large).
ColorImage render_residuals(const std::vector<Correspondence>& pts,
                            const CalibResult& cal, int w, int h,
                            double mag = 40.0);

}  // namespace wproc::vcal
