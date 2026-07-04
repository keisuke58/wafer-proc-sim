// fea.hpp — first-order structural analysis of a lens barrel.
//
// The barrel is the structural spine of the lens. Three questions decide the
// mechanical design, all reduced here to closed-form / SDOF models on a thin
// tube holding a lens group:
//
//   * Thermal focus drift — the barrel expands with temperature (CTE), moving
//     the sensor-to-lens distance and defocusing the image (µm per °C).
//   * Stiffness & first mode — a soft barrel resonates and blurs; the first
//     bending frequency (cantilever tube + tip lens mass) must sit well above
//     the disturbance band.
//   * Drop shock — a dropped camera applies a short acceleration pulse; the
//     dynamic force (peak acceleration × a shock-response amplification factor)
//     must keep the bending stress below yield.
//
// Pure C++17, no external dependencies (the shock-response factor is computed by
// integrating a single-degree-of-freedom oscillator).
#pragma once

namespace optm::fea {

struct Material {
    double youngs_GPa = 68.9;   // e.g. Al 6061
    double density = 2700.0;    // kg/m^3
    double yield_MPa = 276.0;
    double cte_per_C = 23.6e-6;
};

struct Barrel {
    double outer_dia_mm = 30.0;
    double wall_mm = 2.0;
    double length_mm = 40.0;
    double lens_mass_g = 8.0;   // supported lens-group mass at the tip
    Material mat;
};

struct Result {
    double thermal_focus_drift_um_per_C = 0.0;
    double first_mode_hz = 0.0;        // first bending resonance
    double axial_stiffness_N_per_um = 0.0;
    double shock_g_peak = 0.0;         // peak input acceleration [g]
    double daf = 0.0;                  // dynamic amplification factor (SRS)
    double drop_shock_stress_MPa = 0.0;// peak bending stress under shock
    double safety_factor = 0.0;        // yield / stress
    bool ok = false;                   // safety_factor >= 2
};

// Analyze the barrel. `drop_height_m` sets the impact velocity; `shock_ms` is
// the half-sine contact-pulse duration.
Result analyze(const Barrel& b, double drop_height_m = 1.0, double shock_ms = 0.5);

}  // namespace optm::fea
