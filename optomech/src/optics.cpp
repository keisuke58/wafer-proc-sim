// optics.cpp — paraxial ray-transfer implementation.
#include "optomech/optics.hpp"

#include <cmath>

namespace optm {
namespace {

ABCD lens_matrix(double f) {
    // Power phi = 1/f; a nearly-flat element (huge f) has ~zero power.
    const double phi = std::fabs(f) > 1e-12 ? 1.0 / f : 0.0;
    return ABCD{1.0, 0.0, -phi, 1.0};
}

ABCD gap_matrix(double t) { return ABCD{1.0, t, 0.0, 1.0}; }

}  // namespace

ABCD mul(const ABCD& lhs, const ABCD& rhs) {
    return ABCD{lhs.a * rhs.a + lhs.b * rhs.c, lhs.a * rhs.b + lhs.b * rhs.d,
                lhs.c * rhs.a + lhs.d * rhs.c, lhs.c * rhs.b + lhs.d * rhs.d};
}

ABCD system_matrix(const LensSystem& sys) {
    ABCD m;  // identity
    // Compose object->image: for each element apply lens then its trailing gap.
    // Ray propagates left to right, so the newest matrix multiplies on the left.
    for (const Element& e : sys.elements) {
        m = mul(lens_matrix(e.focal_mm), m);
        m = mul(gap_matrix(e.gap_mm), m);
    }
    return m;
}

FirstOrder first_order(const LensSystem& sys) {
    FirstOrder fo;
    if (sys.elements.empty()) return fo;
    // System power comes from the elements + inter-element gaps only (exclude the
    // final back gap, which is a propagation, not part of the lens power).
    ABCD m;
    for (std::size_t i = 0; i < sys.elements.size(); ++i) {
        m = mul(lens_matrix(sys.elements[i].focal_mm), m);
        if (i + 1 < sys.elements.size())
            m = mul(gap_matrix(sys.elements[i].gap_mm), m);
    }
    fo.system_matrix_c = m.c;
    fo.efl_mm = std::fabs(m.c) > 1e-12 ? -1.0 / m.c : 0.0;
    // Back focal distance: distance after the last vertex where a ray parallel to
    // the axis (y=1,u=0) crosses the axis: y' = A, u' = C => bfd = -A/C.
    fo.bfd_mm = std::fabs(m.c) > 1e-12 ? -m.a / m.c : 0.0;
    return fo;
}

double focus_shift_from_spacing(const LensSystem& sys, std::size_t idx,
                                double d_gap_mm) {
    if (idx >= sys.elements.size()) return 0.0;
    const double bfd0 = first_order(sys).bfd_mm;
    LensSystem pert = sys;
    pert.elements[idx].gap_mm += d_gap_mm;
    const double bfd1 = first_order(pert).bfd_mm;
    // Moving an inter-element gap changes both power spacing and the focus; the
    // net image-plane defocus is the change in where the focus lands relative to
    // the (also-moved) sensor. Report the BFD change minus the sensor motion for
    // the last gap; for inner gaps the sensor is fixed.
    if (idx + 1 == sys.elements.size()) return (bfd1 - bfd0) - d_gap_mm;
    return bfd1 - bfd0;
}

namespace {

// ABCD from just after element `idx` (its own plane) to the image plane,
// i.e. the trailing gap of idx and everything after it.
ABCD downstream_matrix(const LensSystem& sys, std::size_t idx) {
    ABCD m;
    for (std::size_t i = idx; i < sys.elements.size(); ++i) {
        // The trailing gap of element i, then the next element (if any).
        m = mul(gap_matrix(sys.elements[i].gap_mm), m);
        if (i + 1 < sys.elements.size())
            m = mul(lens_matrix(sys.elements[i + 1].focal_mm), m);
    }
    return m;
}

}  // namespace

double lateral_sensitivity_decenter(const LensSystem& sys, std::size_t idx) {
    if (idx >= sys.elements.size()) return 0.0;
    const double phi = std::fabs(sys.elements[idx].focal_mm) > 1e-12
                           ? 1.0 / sys.elements[idx].focal_mm
                           : 0.0;
    // Angular kick per unit decenter is +phi; image shift = B_downstream * kick.
    return downstream_matrix(sys, idx).b * phi;
}

double lateral_sensitivity_tilt(const LensSystem& sys, std::size_t idx) {
    if (idx >= sys.elements.size()) return 0.0;
    // A tilt theta ~ decenter of f*theta => image shift = B_downstream * theta.
    return downstream_matrix(sys, idx).b;
}

}  // namespace optm
