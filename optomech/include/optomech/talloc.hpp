// talloc.hpp — cost-optimal tolerance allocation for a lens barrel.
//
// The tolerance module answers "does this budget pass?"; this one answers the
// design question "what is the *cheapest* set of tolerances that guarantees the
// budget?". Tightening a tolerance costs money (cost ~ a / sigma), and every
// element/error source has a different image sensitivity, so equal tolerances
// waste money on insensitive elements. Minimizing total cost subject to the
// image-plane budget has a closed-form Lagrange solution (sigma_i proportional
// to (a_i / S_i^2)^(1/3)); this implements it and compares against an
// equal-tolerance baseline.
//
// Built on the paraxial optics core. Pure C++17.
#pragma once

#include <string>
#include <vector>

#include "optomech/optics.hpp"

namespace optm::talloc {

// Relative cost to hold a decenter / tilt tolerance (cost = coeff / sigma).
struct CostModel {
    double decenter_coeff = 1.0;   // per element, decenter (cost·µm)
    double tilt_coeff = 1.6;       // per element, tilt is harder to hold (cost·mrad)
};

struct Alloc {
    std::size_t element = 0;
    std::string kind;          // "decenter" | "tilt"
    double sensitivity = 0.0;  // image µm per source unit
    double sigma = 0.0;        // allocated 1-sigma tolerance (µm or mrad)
    double cost = 0.0;
};

struct Result {
    bool ok = false;
    double budget_um = 0.0;
    double opt_cost = 0.0;      // minimized total cost
    double equal_cost = 0.0;    // equal-tolerance baseline at the same budget
    double savings_pct = 0.0;   // 100*(equal-opt)/equal
    std::vector<Alloc> allocation;  // sorted tightest-first
};

// Allocate decenter+tilt tolerances across the elements of `sys` so the image-
// plane boresight RSS equals `budget_um` at minimum total cost.
Result allocate(const LensSystem& sys, double budget_um,
                const CostModel& cost = CostModel{});

}  // namespace optm::talloc
