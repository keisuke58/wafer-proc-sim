/*
 * _frame_stiffness_kernel.cpp
 *
 * Euler-Bernoulli beam static stiffness + first natural frequency for
 * DISCO-style dicing saw machine frame.
 *
 * Exposed to Python via pybind11:
 *   hollow_rect_inertia(b, h, t)          -> I_mm4
 *   hollow_circle_inertia(D, t)            -> I_mm4
 *   i_beam_inertia(H, B, tf, tw)           -> I_mm4
 *   beam_deflection(F_N, L_mm, E_GPa, I_mm4, bc) -> {delta_um, k_N_um}
 *   first_natural_freq(k_N_m, mass_kg)    -> freq_Hz
 *   MachineFrame class                    -> multi-segment frame analysis
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

// ── cross-section helpers ────────────────────────────────────────────────────

// Hollow rectangular tube: width b, height h, wall thickness t
double hollow_rect_inertia(double b_mm, double h_mm, double t_mm) {
    double b_i = b_mm - 2.0 * t_mm;
    double h_i = h_mm - 2.0 * t_mm;
    if (b_i <= 0.0 || h_i <= 0.0)
        throw std::invalid_argument("wall thickness too large for given section");
    return (b_mm * std::pow(h_mm, 3) - b_i * std::pow(h_i, 3)) / 12.0;
}

// Hollow circular tube: outer diameter D, wall thickness t
double hollow_circle_inertia(double D_mm, double t_mm) {
    double d = D_mm - 2.0 * t_mm;
    if (d <= 0.0)
        throw std::invalid_argument("wall thickness too large for given section");
    return M_PI / 64.0 * (std::pow(D_mm, 4) - std::pow(d, 4));
}

// Symmetric I-beam: total height H, flange width B, flange thick tf, web thick tw
double i_beam_inertia(double H_mm, double B_mm, double tf_mm, double tw_mm) {
    // I = I_outer_rect - 2 * I_flange_cutout
    double h_w = H_mm - 2.0 * tf_mm;  // web height
    return (B_mm * std::pow(H_mm, 3) - (B_mm - tw_mm) * std::pow(h_w, 3)) / 12.0;
}

// ── beam deflection (Euler-Bernoulli) ────────────────────────────────────────

// bc: "cantilever" (fixed-free, tip load) or "simply_supported" (mid-span load)
// Returns {delta_um, k_N_um}
py::dict beam_deflection(double F_N, double L_mm, double E_GPa, double I_mm4,
                         const std::string& bc) {
    if (L_mm <= 0.0 || E_GPa <= 0.0 || I_mm4 <= 0.0)
        throw std::invalid_argument("L, E, I must be > 0");

    double E_MPa = E_GPa * 1000.0;   // GPa -> MPa = N/mm²
    double EI = E_MPa * I_mm4;       // N·mm²

    double delta_mm;
    if (bc == "cantilever") {
        // δ = F L³ / (3 E I)
        delta_mm = F_N * std::pow(L_mm, 3) / (3.0 * EI);
    } else if (bc == "simply_supported") {
        // δ = F L³ / (48 E I)  (central point load)
        delta_mm = F_N * std::pow(L_mm, 3) / (48.0 * EI);
    } else {
        throw std::invalid_argument("bc must be 'cantilever' or 'simply_supported'");
    }

    double delta_um = delta_mm * 1000.0;
    double k_N_um   = (delta_um > 0.0) ? F_N / delta_um : 1e30;

    py::dict result;
    result["delta_um"] = delta_um;
    result["k_N_um"]   = k_N_um;
    return result;
}

// ── natural frequency ────────────────────────────────────────────────────────

// k_N_m: stiffness in N/m, mass_kg: effective tip mass
double first_natural_freq(double k_N_m, double mass_kg) {
    if (k_N_m <= 0.0 || mass_kg <= 0.0)
        throw std::invalid_argument("k and mass must be > 0");
    return std::sqrt(k_N_m / mass_kg) / (2.0 * M_PI);
}

// ── MachineFrame class ────────────────────────────────────────────────────────

struct Segment {
    std::string name;
    double L_mm;
    double E_GPa;
    double I_mm4;
    std::string bc;
};

class MachineFrame {
public:
    std::vector<Segment> segments;

    MachineFrame() = default;

    void add_segment(const std::string& name, double L_mm, double E_GPa,
                     double I_mm4, const std::string& bc = "cantilever") {
        segments.push_back({name, L_mm, E_GPa, I_mm4, bc});
    }

    // Total tip deflection under F_N: sum of cantilever contributions (series compliance).
    // Each segment treated as cantilever with same load — conservative (worst-case).
    py::dict analyze(double F_N) const {
        if (segments.empty())
            throw std::runtime_error("no segments added");

        double total_delta_um = 0.0;
        py::list seg_results;

        for (const auto& seg : segments) {
            auto r = beam_deflection(F_N, seg.L_mm, seg.E_GPa, seg.I_mm4, seg.bc);
            double d = r["delta_um"].cast<double>();
            total_delta_um += d;

            py::dict sr;
            sr["name"]     = seg.name;
            sr["delta_um"] = d;
            sr["k_N_um"]   = r["k_N_um"].cast<double>();
            seg_results.append(sr);
        }

        double k_total_N_um = (total_delta_um > 0.0) ? F_N / total_delta_um : 1e30;

        py::dict result;
        result["total_delta_um"] = total_delta_um;
        result["total_k_N_um"]   = k_total_N_um;
        result["segments"]       = seg_results;
        return result;
    }

    // Natural frequency at tip given effective spindle+blade mass
    double natural_freq_hz(double F_N, double mass_kg) const {
        auto r = analyze(F_N);
        double k_N_um = r["total_k_N_um"].cast<double>();
        double k_N_m  = k_N_um * 1e6;  // N/µm -> N/m
        return first_natural_freq(k_N_m, mass_kg);
    }

    // Factory: typical DISCO dicing saw column (granite base + steel column + spindle overhang)
    static MachineFrame disco_dicing_saw() {
        MachineFrame f;
        // Granite base: E=60 GPa, hollow rect 200x200 t=30 mm, column height 600 mm
        double I_granite = hollow_rect_inertia(200.0, 200.0, 30.0);
        f.add_segment("granite_column", 600.0, 60.0, I_granite, "cantilever");

        // Steel spindle arm: E=200 GPa, hollow circle D=80 t=8 mm, overhang 150 mm
        double I_steel = hollow_circle_inertia(80.0, 8.0);
        f.add_segment("spindle_overhang", 150.0, 200.0, I_steel, "cantilever");

        return f;
    }
};

// ── module ────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(_frame_stiffness_kernel, m) {
    m.doc() = "Machine frame static stiffness and natural frequency (Euler-Bernoulli)";

    m.def("hollow_rect_inertia",   &hollow_rect_inertia,
          py::arg("b_mm"), py::arg("h_mm"), py::arg("t_mm"),
          "Second moment of area for hollow rectangular tube [mm^4]");

    m.def("hollow_circle_inertia", &hollow_circle_inertia,
          py::arg("D_mm"), py::arg("t_mm"),
          "Second moment of area for hollow circular tube [mm^4]");

    m.def("i_beam_inertia",        &i_beam_inertia,
          py::arg("H_mm"), py::arg("B_mm"), py::arg("tf_mm"), py::arg("tw_mm"),
          "Second moment of area for symmetric I-beam [mm^4]");

    m.def("beam_deflection",       &beam_deflection,
          py::arg("F_N"), py::arg("L_mm"), py::arg("E_GPa"), py::arg("I_mm4"),
          py::arg("bc") = "cantilever",
          "Euler-Bernoulli tip deflection. bc: 'cantilever' or 'simply_supported'");

    m.def("first_natural_freq",    &first_natural_freq,
          py::arg("k_N_m"), py::arg("mass_kg"),
          "Undamped natural frequency [Hz] = sqrt(k/m) / (2 pi)");

    py::class_<MachineFrame>(m, "MachineFrame")
        .def(py::init<>())
        .def("add_segment",      &MachineFrame::add_segment,
             py::arg("name"), py::arg("L_mm"), py::arg("E_GPa"), py::arg("I_mm4"),
             py::arg("bc") = "cantilever")
        .def("analyze",          &MachineFrame::analyze,
             py::arg("F_N"),
             "Tip deflection and per-segment stiffness under load F_N [N]")
        .def("natural_freq_hz",  &MachineFrame::natural_freq_hz,
             py::arg("F_N"), py::arg("mass_kg"),
             "Estimated first natural frequency [Hz] given spindle mass [kg]")
        .def_static("disco_dicing_saw", &MachineFrame::disco_dicing_saw,
             "Factory: DISCO-like dicing saw column geometry");
}
