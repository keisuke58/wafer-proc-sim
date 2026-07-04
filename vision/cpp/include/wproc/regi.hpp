// regi.hpp — image registration + die-to-die difference inspection.
//
// Defect *review* and die-to-die inspection both rest on the same operation:
// take a reference die image and a device-under-test die image, align them to
// sub-pixel accuracy, then flag whatever remains after subtraction. Two
// neighbouring dies are nominally identical, so a real difference is a defect.
//
// Alignment uses phase correlation — the cross-power spectrum of the two images
// has a single sharp peak whose location is the translation between them. It is
// computed with the in-house radix-2 FFT (shared with the spindle module) run
// over rows then columns, and refined to sub-pixel by a parabolic fit. The
// aligned reference is then subtracted, the residual thresholded, cleaned with
// morphology, and labelled into defect blobs.
//
// Pure C++17, no external dependencies.
#pragma once

#include <vector>

#include "wproc/connected.hpp"
#include "wproc/image.hpp"

namespace wproc::regi {

// Estimated translation of `img` relative to `ref` (a positive dx means the
// content of `img` sits dx pixels to the right of `ref`).
struct Shift {
    double dx = 0.0;
    double dy = 0.0;
    double response = 0.0;  // normalized phase-correlation peak height (0..1)
};

// Sub-pixel translation between two equally-sized images by phase correlation.
Shift phase_correlate(const Image& ref, const Image& img);

// Resample `src` shifted by (dx, dy) with bilinear interpolation (clamped edges).
Image shift_image(const Image& src, double dx, double dy);

struct DiffParams {
    double k_sigma = 4.0;    // threshold = mean + k*sigma of the residual
    int open_radius = 1;     // morphological opening to drop single-pixel noise
    long min_area = 4;       // ignore blobs smaller than this (px)
    int guard = 3;           // ignore a border band (alignment ringing) [px]
};

struct DiffResult {
    Shift shift;                        // alignment used
    double residual_rms = 0.0;          // RMS of the aligned residual
    double threshold = 0.0;             // absolute-difference threshold applied
    int defect_count = 0;
    long defect_area = 0;               // total flagged area [px]
    std::vector<Component> defects;     // labelled difference blobs
};

// Register `test` to `ref`, subtract, and label the residual defects.
DiffResult diff_inspect(const Image& ref, const Image& test,
                        const DiffParams& params = DiffParams{});

// Overlay the flagged defect boxes (red) on a grayscale copy of `test`.
ColorImage annotate_diff(const Image& test, const DiffResult& r);

}  // namespace wproc::regi
