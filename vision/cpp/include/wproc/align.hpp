// align.hpp — dicing-street auto-alignment by template matching.
//
// Before every cut the machine must register the wafer: find the taught street
// pattern in the live camera image and report the sub-pixel offset (and small
// rotation) from where it is expected. This does that with normalized
// cross-correlation (NCC) — illumination-invariant template matching — plus a
// parabolic sub-pixel peak fit, and estimates street tilt by matching the
// pattern's left and right halves independently.
//
// Pure C++17 (no OpenCV).
#pragma once

#include "wproc/image.hpp"

namespace wproc::align {

struct Match {
    bool found = false;
    double x = 0.0;      // sub-pixel template top-left x in the search image
    double y = 0.0;      // sub-pixel template top-left y
    double score = 0.0;  // peak NCC in [-1, 1]
};

// Best NCC location of `templ` within the axis-aligned search window
// [sx0, sx0+sw) x [sy0, sy0+sh) of `img` (window clamped to valid positions).
// The returned x/y are refined to sub-pixel by a parabolic fit on the NCC peak.
Match match_ncc(const Image& img, const Image& templ, int sx0, int sy0,
                int sw, int sh);

struct Registration {
    bool ok = false;
    double dx = 0.0;         // offset from expected position (px, sub-pixel)
    double dy = 0.0;
    double angle_deg = 0.0;  // estimated street tilt (small angle)
    double score = 0.0;      // NCC of the full-template match
};

// Register `templ` near its expected top-left (ex, ey), searching ±radius. dx/dy
// are the found offset from (ex, ey); the tilt is estimated from the vertical
// shift between the template's left- and right-half matches. `ok` when the peak
// NCC clears `min_score`.
Registration register_pattern(const Image& img, const Image& templ,
                              int ex, int ey, int radius,
                              double min_score = 0.5);

// Draw the matched template box (green) on a grayscale copy of `img`.
ColorImage annotate_match(const Image& img, const Match& m, int tw, int th);

}  // namespace wproc::align
