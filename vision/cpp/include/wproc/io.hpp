// io.hpp — Netpbm (PGM/PPM) image I/O. No external dependencies.
//
// Supported on read:  P2 (ASCII gray), P5 (binary gray).
// Supported on write: P5 (binary gray), P6 (binary RGB).
//
// PGM/PPM is chosen deliberately: it is a trivially portable, dependency-free
// container, so the toolkit builds and runs anywhere with just a C++ compiler.
#pragma once

#include <string>

#include "wproc/image.hpp"

namespace wproc {

// Read a PGM (P2 or P5) file into a float Image. Throws std::runtime_error
// on I/O errors or malformed headers.
Image read_pgm(const std::string& path);

// Write a float Image as a binary PGM (P5). Pixel values are rounded and
// clamped to [0, 255].
void write_pgm(const std::string& path, const Image& img);

// Write an RGB image as a binary PPM (P6).
void write_ppm(const std::string& path, const ColorImage& img);

}  // namespace wproc
