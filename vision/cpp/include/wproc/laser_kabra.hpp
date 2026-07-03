// laser_kabra.hpp — laser SiC-ingot slicing (KABRA-style) economics + layer QC.
//
// SiC is hard and expensive, and traditional diamond-wire slicing wastes a wide
// kerf as sawdust. A laser forms a thin subsurface separation layer inside the
// ingot; the wafer is then cleaved off along it, so the "kerf loss" is far
// smaller and far more of the boule becomes usable wafers. This module:
//
//   * turns ingot length + wafer thickness + per-slice loss into wafers-per-
//     ingot and material utilisation, for laser vs diamond-wire, and the wafers
//     gained / material saved by going laser;
//   * rolls the ingot cost into a per-wafer cost for each method;
//   * grades the laser separation layer from a depth profile (uniform, on-target
//     depth ⇒ a clean cleave).
//
// Pure C++17, header + one TU.
#pragma once

#include <vector>

namespace wproc::laser {

// ── Slice yield / material utilisation ───────────────────────────────────────

struct SliceInputs {
    double ingot_length_mm = 30.0;    // usable boule length
    double wafer_thickness_um = 350.0;  // finished wafer thickness
    double loss_per_slice_um = 100.0;   // kerf + grind allowance consumed per wafer
};

struct SliceYield {
    int wafers = 0;
    double used_um = 0.0;            // material that became wafers
    double waste_um = 0.0;           // material lost to kerf/allowance
    double utilization_pct = 0.0;    // used / ingot length
    double pitch_um = 0.0;           // thickness + loss per slice
};

// Wafers obtained from one ingot and how much of it became product.
SliceYield slice_yield(const SliceInputs& in);

struct SliceComparison {
    SliceYield laser;
    SliceYield saw;
    int extra_wafers = 0;            // laser - saw
    double extra_wafers_pct = 0.0;   // relative gain
    double material_saved_pct = 0.0; // utilisation gain (percentage points)
};

// Compare laser slicing (small loss) with diamond-wire sawing (large kerf) for
// the same ingot and wafer thickness.
SliceComparison compare_slice(double ingot_length_mm, double wafer_thickness_um,
                              double laser_loss_um, double saw_kerf_um);

struct SliceEcon {
    double cost_per_wafer_laser = 0.0;
    double cost_per_wafer_saw = 0.0;
    double savings_per_wafer = 0.0;   // saw - laser
    double savings_pct = 0.0;
};

// Roll the ingot cost (+ per-wafer processing) into per-wafer cost for both
// methods, using the wafer counts from `cmp`.
SliceEcon slice_economics(const SliceComparison& cmp, double ingot_cost,
                          double proc_cost_per_wafer_laser,
                          double proc_cost_per_wafer_saw);

// ── Separation-layer quality ─────────────────────────────────────────────────

struct SepLayerStats {
    int n = 0;
    double mean_depth_um = 0.0;
    double sigma_um = 0.0;
    double max_dev_um = 0.0;        // worst deviation from target
    double uniformity_pct = 0.0;    // 100*(1 - sigma/mean), clamped
    bool ok = true;                 // within tolerance of target
};

// Grade the laser separation-layer depth profile (one depth per lateral sample)
// against a target depth and tolerance. A uniform, on-target layer cleaves clean.
SepLayerStats analyze_separation(const std::vector<double>& depths_um,
                                 double target_depth_um, double tol_um);

}  // namespace wproc::laser
