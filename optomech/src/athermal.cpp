// athermal.cpp — passive athermalization + Miles' random-vibration check.
#include "optomech/athermal.hpp"

#include <cmath>

namespace optm::athermal {
namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kG = 9.80665;
}

Result analyze(const fea::Barrel& b, const Config& cfg) {
    Result r;
    // Uncompensated: the barrel length expansion moves the image plane.
    r.uncomp_drift_um_per_C = b.mat.cte_per_C * b.length_mm * 1e3;

    // Compensator spacer sized so comp_cte * L_c == barrel_cte * L_barrel.
    if (cfg.comp_cte_per_C > 0.0) {
        const double ideal = b.mat.cte_per_C * b.length_mm / cfg.comp_cte_per_C;
        r.comp_len_mm = std::round(ideal * 10.0) / 10.0;  // realizable to 0.1 mm
        const double mismatch =
            b.mat.cte_per_C * b.length_mm - cfg.comp_cte_per_C * r.comp_len_mm;
        r.comp_drift_um_per_C = std::fabs(mismatch) * 1e3;
    }

    // Geometry for the stress calc (the first mode comes from fea::analyze).
    const double Do = b.outer_dia_mm * 1e-3;
    const double Di = (b.outer_dia_mm - 2.0 * b.wall_mm) * 1e-3;
    const double L = b.length_mm * 1e-3;
    if (Do <= 0.0 || Di < 0.0 || Di >= Do || L <= 0.0) return r;
    const double I = kPi / 64.0 * (std::pow(Do, 4) - std::pow(Di, 4));
    const double c = Do / 2.0;

    r.first_mode_hz = fea::analyze(b).first_mode_hz;

    // Miles' equation: RMS response acceleration to a flat PSD.
    r.grms_response =
        std::sqrt(kPi / 2.0 * r.first_mode_hz * cfg.modal_Q * cfg.psd_g2_per_hz);
    r.accel_3sigma_g = 3.0 * r.grms_response;

    // 3-sigma bending stress (tip lens mass under lateral acceleration).
    const double force = b.lens_mass_g * 1e-3 * r.accel_3sigma_g * kG;  // N
    const double stress = I > 0.0 ? force * L * c / I : 0.0;            // Pa
    r.rand_stress_MPa = stress * 1e-6;
    r.fatigue_margin = r.rand_stress_MPa > 0.0
                           ? cfg.endurance_MPa / r.rand_stress_MPa : 0.0;
    r.ok = r.comp_drift_um_per_C < 0.1 && r.fatigue_margin >= 2.0;
    return r;
}

}  // namespace optm::athermal
