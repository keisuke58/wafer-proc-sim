// threshold.hpp — global thresholding (Otsu) and binarization.
#pragma once

#include "wproc/image.hpp"

namespace wproc {

// Compute the optimal global threshold by Otsu's method (maximizes between-class
// variance over a 256-bin histogram of pixel values clamped to [0, 255]).
float otsu_threshold(const Image& src);

// Binarize into a 0/255 image. If invert is false, pixels > thr become 255
// (foreground = bright); if invert is true, pixels <= thr become 255
// (foreground = dark), which is what dark-defect detection needs.
Image binarize(const Image& src, float thr, bool invert = false);

}  // namespace wproc
