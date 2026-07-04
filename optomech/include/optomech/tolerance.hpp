// tolerance.hpp — optomechanical tolerance stack-up for a lens barrel.
//
// The barrel holds each lens element to a finite mechanical tolerance: it can be
// laterally decentered, tilted, and axially mis-spaced. Each error moves the
// image (lateral shift = boresight error / MTF loss; axial = defocus). This
// module propagates per-element 1-sigma tolerances to the image plane two ways:
//
//   * RSS (root-sum-square) — the analytic first-order budget, and
//   * Monte Carlo — sampling all errors together for the real distribution,
//
// then ranks each (element, error-source) by its contribution so the designer
// knows which single tolerance to tighten first (the "critical tolerance").
//
// Built on the paraxial optics core. Pure C++17.
#pragma once

#include <string>
#include <vector>

#include "optomech/optics.hpp"

namespace optm::tol {

// Per-element 1-sigma manufacturing/assembly tolerances.
struct Tolerance {
    double decenter_um = 0.0;   // lateral element decenter [µm]
    double tilt_mrad = 0.0;     // element tilt [milliradian]
    double spacing_um = 0.0;    // axial spacing error [µm]
};

// One error source's contribution to the image-plane lateral budget.
struct Contribution {
    std::size_t element = 0;
    std::string source;        // "decenter" | "tilt" | "spacing"
    double sigma_um = 0.0;     // this source's 1-sigma image-plane effect [µm]
    double variance_pct = 0.0; // share of the total variance [%]
};

struct Result {
    // Lateral image-plane boresight error (dominant for MTF / pixel registration).
    double lateral_rss_um = 0.0;   // analytic root-sum-square 1-sigma
    double lateral_mc_rms_um = 0.0;// Monte Carlo RMS
    double lateral_mc_p99_um = 0.0;// Monte Carlo 99th percentile
    // Axial defocus.
    double focus_rss_um = 0.0;
    double focus_mc_rms_um = 0.0;
    std::vector<Contribution> ranking;  // lateral sources, largest first
    double efl_mm = 0.0;
    double budget_um = 0.0;
    bool ok = false;               // lateral_mc_p99 within budget
    // Monte Carlo image-plane scatter (lateral shift samples, µm) for plotting.
    std::vector<double> mc_lateral_um;
    std::vector<double> mc_focus_um;
};

// Propagate `tols` (aligned with sys.elements) to the image plane. `budget_um`
// is the allowed image-plane lateral error (e.g. one sensor pixel). Runs
// `n_samples` Monte Carlo trials with a fixed seed for reproducibility.
Result analyze(const LensSystem& sys, const std::vector<Tolerance>& tols,
               double budget_um, int n_samples = 20000, unsigned seed = 7);

}  // namespace optm::tol
