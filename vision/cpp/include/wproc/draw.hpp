// draw.hpp — annotation primitives for producing inspection overlays.
#pragma once

#include "wproc/image.hpp"

namespace wproc {

// Convert a grayscale image to RGB (values clamped to [0,255]) so colored
// annotations can be drawn on top.
ColorImage to_color(const Image& img);

void draw_hline(ColorImage& img, int y, int x0, int x1, Rgb c);
void draw_vline(ColorImage& img, int x, int y0, int y1, Rgb c);

// Hollow rectangle outline (inclusive corners).
void draw_rect(ColorImage& img, int x0, int y0, int x1, int y1, Rgb c);

// Small plus-shaped marker centered at (cx, cy).
void draw_cross(ColorImage& img, int cx, int cy, int arm, Rgb c);

}  // namespace wproc
