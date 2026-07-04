// optics.hpp — paraxial (first-order) optical model of a camera lens.
//
// A camera lens is a stack of elements held by the barrel. To reason about how
// mechanical errors (element decenter, tilt, axial spacing) move the image, we
// use the paraxial ray-transfer (ABCD) formalism: each thin element is a 2x2
// matrix [[1,0],[-phi,1]] (phi = 1/f) and each air gap is [[1,t],[0,1]]. The
// product maps a ray (height y, angle u) from object to image space, giving the
// effective focal length and back focal distance in closed form.
//
// This is the shared optical core; the tolerance and OIS modules build on it.
// Pure C++17, no external dependencies.
#pragma once

#include <array>
#include <vector>

namespace optm {

// A 2x2 ray-transfer matrix, row-major {A, B, C, D}: [y'; u'] = M [y; u].
struct ABCD {
    double a = 1.0, b = 0.0, c = 0.0, d = 1.0;
};

// Matrix product M = lhs * rhs (apply rhs first, then lhs).
ABCD mul(const ABCD& lhs, const ABCD& rhs);

// A thin lens element in the barrel: focal length and the air gap that follows
// it (to the next element or to the nominal image plane).
struct Element {
    double focal_mm = 0.0;   // element focal length [mm] (inf-power = large)
    double gap_mm = 0.0;     // axial spacing to the next element / image [mm]
    double aperture_mm = 0.0;// clear semi-aperture [mm] (bookkeeping)
};

// A centered lens system in air. Elements are ordered from object to image; the
// last element's gap is the nominal flange/back distance to the sensor.
struct LensSystem {
    std::vector<Element> elements;
};

struct FirstOrder {
    double efl_mm = 0.0;   // effective focal length
    double bfd_mm = 0.0;   // back focal distance (last vertex -> paraxial focus)
    double system_matrix_c = 0.0;  // -power of the whole system
};

// Full-system ABCD from the first element vertex to the last element's back gap.
ABCD system_matrix(const LensSystem& sys);

// Effective focal length and back focal distance of the system.
FirstOrder first_order(const LensSystem& sys);

// Image-plane defocus [mm] when the axial spacing of element `idx` changes by
// `d_gap_mm` (positive = element pushed toward the sensor). Computed by finite
// difference of the back focal distance.
double focus_shift_from_spacing(const LensSystem& sys, std::size_t idx,
                                double d_gap_mm);

// First-order lateral image shift [mm] per unit lateral decenter of element
// `idx`. A lens decentered by delta deflects a marginal ray by -delta/f at that
// plane; propagating that angular kick through the downstream matrix to the
// image gives image_shift = B_downstream * (decenter / f). Returns the
// sensitivity (image mm per mm of decenter, dimensionless).
double lateral_sensitivity_decenter(const LensSystem& sys, std::size_t idx);

// First-order lateral image shift [mm] per radian of element tilt about its
// vertex. A tilted thin lens acts like a decenter of (tilt * f) at first order,
// so the sensitivity is B_downstream (image mm per radian).
double lateral_sensitivity_tilt(const LensSystem& sys, std::size_t idx);

}  // namespace optm
