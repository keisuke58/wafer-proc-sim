// cv_backend.hpp — optional OpenCV implementations of the core filters.
//
// Built only when CMake finds OpenCV (WPROC_WITH_OPENCV). Provides the same
// operations as the from-scratch kernels so results can be cross-validated:
// the parity unit test asserts that both implementations agree within
// tolerance on synthetic wafer images. Header stays OpenCV-free so callers
// don't need OpenCV includes.
#pragma once

#include "wproc/filters.hpp"
#include "wproc/image.hpp"

namespace wproc::cv_backend {

// cv::GaussianBlur with an equivalent kernel (radius ceil(3*sigma), BORDER_REPLICATE).
Image gaussian_blur(const Image& src, float sigma);

// cv::Sobel 3x3, signed output, BORDER_REPLICATE.
Gradient sobel(const Image& src);

// cv::threshold(THRESH_OTSU) over the image clamped to 8-bit.
float otsu_threshold(const Image& src);

}  // namespace wproc::cv_backend
