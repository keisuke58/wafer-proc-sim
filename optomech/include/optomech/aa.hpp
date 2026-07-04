// aa.hpp — active alignment (AA) of a lens element.
//
// Modern lens assembly does not rely on machining tolerances alone: one element
// is held in a gripper and actively moved in several DOF while the image quality
// is measured, until it is optimized ("active alignment"). This simulates that
// process — an element carries a built-in assembly error (decenter + tilt +
// focus), and a 3-DOF search (decenter x/y + focus) minimizes a spot-size merit
// with a downhill-simplex (Nelder-Mead) optimizer, exactly as the real AA
// station's feedback loop does.
//
// Built on the paraxial optics core. Pure C++17.
#pragma once

#include <vector>

#include "optomech/optics.hpp"

namespace optm::aa {

// The as-assembled error of the element to be aligned.
struct Assembly {
    std::size_t element = 0;
    double decenter_x_um = 0.0;
    double decenter_y_um = 0.0;
    double tilt_mrad = 0.0;    // not correctable by a 3-DOF (x,y,focus) move
    double focus_um = 0.0;     // axial / best-focus offset
};

struct Result {
    double merit_before_um = 0.0;  // spot-size merit at the as-assembled state
    double merit_after_um = 0.0;   // after active alignment
    int iterations = 0;
    double adj_x_um = 0.0;         // applied compensating move
    double adj_y_um = 0.0;
    double adj_focus_um = 0.0;
    bool converged = false;
    std::vector<double> merit_history;  // best merit per iteration
};

// Actively align the errored element by minimizing the spot merit over
// (decenter x, decenter y, focus).
Result align(const LensSystem& sys, const Assembly& err, int max_iter = 250);

}  // namespace optm::aa
