// simd_filters.hpp — AVX2 and multithreaded implementations of the hot filters.
//
// Same operations and border conventions as wproc/filters.hpp (clamped edges,
// Gaussian radius = ceil(3*sigma)) so results are bit-comparable to the scalar
// path within float rounding. Interior pixels use AVX2 (8 floats/lane); the
// thin border is handled with the scalar kernel. The parallel variants tile
// rows across std::thread and call the SIMD kernel per tile.
//
// If the translation unit is compiled without AVX2, the simd_* functions
// transparently fall back to the scalar implementations, so callers always get
// a correct result.
#pragma once

#include "wproc/filters.hpp"
#include "wproc/image.hpp"

namespace wproc::simd {

// True if this build compiled the AVX2 code path (else scalar fallback).
bool avx2_enabled();

// AVX2 (single-thread) separable Gaussian blur and 3x3 Sobel.
Image gaussian_blur(const Image& src, float sigma);
Gradient sobel(const Image& src);

// Multithreaded (row-tiled) variants; threads<=0 uses hardware concurrency.
Image gaussian_blur_mt(const Image& src, float sigma, int threads = 0);
Gradient sobel_mt(const Image& src, int threads = 0);

}  // namespace wproc::simd
