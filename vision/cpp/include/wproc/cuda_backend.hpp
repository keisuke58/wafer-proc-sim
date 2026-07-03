// cuda_backend.hpp — optional CUDA implementations of the core filters.
//
// Built only when CMake finds a CUDA toolkit (WPROC_WITH_CUDA). Same
// operations and conventions as the CPU kernels (clamped borders, radius
// ceil(3*sigma)) so outputs are directly comparable; useful when scaling the
// inspection to full-wafer scan resolutions where the separable Gaussian
// dominates runtime. Header stays CUDA-free so callers need no CUDA headers.
#pragma once

#include "wproc/filters.hpp"
#include "wproc/image.hpp"

namespace wproc::cuda_backend {

// True if at least one CUDA device is usable at runtime.
bool available();

// Separable Gaussian blur on the GPU (throws std::runtime_error on CUDA errors).
Image gaussian_blur(const Image& src, float sigma);

// 3x3 Sobel on the GPU, signed gx/gy.
Gradient sobel(const Image& src);

}  // namespace wproc::cuda_backend
