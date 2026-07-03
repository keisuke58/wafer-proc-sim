// filters.hpp — spatial filters: Gaussian smoothing and Sobel gradients.
#pragma once

#include "wproc/image.hpp"

namespace wproc {

// Separable Gaussian blur. sigma > 0; the kernel radius is ceil(3*sigma).
// Border pixels use clamped (replicated) edge handling.
Image gaussian_blur(const Image& src, float sigma);

// Result of a Sobel gradient pass.
struct Gradient {
    Image gx;   // horizontal derivative (signed)
    Image gy;   // vertical derivative (signed)
};

// 3x3 Sobel operator. Returns signed gx and gy (clamped border handling).
Gradient sobel(const Image& src);

// Gradient magnitude sqrt(gx^2 + gy^2) as a new image.
Image gradient_magnitude(const Gradient& g);

}  // namespace wproc
