// street_detect.hpp — dicing-street localization: skew estimation, deskew,
// and multi-street detection. Built on OpenCV (Canny + Hough) as part of the
// wproc_cv backend. Header stays OpenCV-free.
//
// Real scribe-line images are rarely axis-aligned: the wafer is mounted with a
// small rotation, and a die street may repeat many times across the field.
// This turns the single-kerf inspector into a full machine-vision front-end —
// find the street orientation, straighten the image, then locate every street.
#pragma once

#include <vector>

#include "wproc/image.hpp"

namespace wproc::vision {

// Dominant skew of near-vertical streets, in degrees (positive = rotated
// clockwise from vertical). Returns 0 if no consistent lines are found.
double estimate_skew_deg(const Image& img);

// Rotate the image about its centre by -skew_deg so streets become vertical
// (replicated borders). Inverse of the skew measured by estimate_skew_deg.
Image deskew(const Image& img, double skew_deg);

// X-coordinates (sub-pixel) of the centres of dark vertical streets, left to
// right. min_width_px ignores thin noise runs; the image should be deskewed
// first for best results.
std::vector<double> detect_streets(const Image& img, int min_width_px = 3);

}  // namespace wproc::vision
