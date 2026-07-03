// kerf_inspect.hpp — dicing-kerf inspection for wafer scribe-line images.
//
// Domain: DISCO-style dicing leaves a cut "street" (kerf) through the wafer.
// Two quantities matter for process control:
//   1. Kerf width  — distance between the two cut walls (must hit spec).
//   2. Chipping    — dark chips/cracks protruding from the walls into the die.
//
// Image model: a vertical dark kerf channel on a brighter silicon substrate.
//   - Walls are the strong vertical gradients bounding the channel; located via
//     the signed Sobel-x column projection (left wall = strongest negative
//     slope, right wall = strongest positive slope), refined to sub-pixel by
//     parabolic interpolation.
//   - Chips are dark connected components outside the kerf band that reach a
//     wall (within wall_attach_margin_px); each is measured for protrusion
//     depth and area. Isolated dark marks away from the walls are ignored.
#pragma once

#include <string>
#include <vector>

#include "wproc/image.hpp"

namespace wproc {

struct Chip {
    int id = 0;
    int side = 0;            // -1 = left wall, +1 = right wall
    double cx = 0.0, cy = 0.0;
    long area_px = 0;
    double area_um2 = 0.0;
    double protrusion_um = 0.0;  // depth beyond the nearest wall into the die
    int min_x = 0, min_y = 0, max_x = 0, max_y = 0;
};

struct KerfResult {
    bool found = false;
    double left_wall_px = -1.0;
    double right_wall_px = -1.0;
    double width_px = 0.0;
    double width_um = 0.0;
    double confidence = 0.0;   // 0..1, wall signal-to-noise
    std::vector<Chip> chips;
    double max_chip_um = 0.0;
    double total_chip_area_um2 = 0.0;
    bool width_in_spec = false;
    bool chipping_in_spec = false;
    bool pass = false;         // width_in_spec && chipping_in_spec
    std::string status = "not_run";
};

class KerfInspector {
public:
    struct Params {
        double um_per_px = 0.5;       // metrology scale (microns per pixel)
        float blur_sigma = 1.2f;      // pre-smoothing
        float dark_threshold = -1.0f; // <0 -> Otsu automatic
        int morph_radius = 1;         // opening radius to drop specks
        long min_chip_area_px = 8;    // ignore blobs smaller than this
        // A blob only counts as chipping if it reaches within this many pixels
        // of a kerf wall; isolated dark marks (dust, fiducials) elsewhere on
        // the substrate are ignored. Allows for blur + morphological erosion
        // pulling a wall-attached blob slightly off the wall.
        double wall_attach_margin_px = 4.0;
        double min_wall_snr = 8.0;    // projection peak / mean for confidence=1
        // Spec limits for pass/fail (set width_tol_um < 0 to skip the check).
        double nominal_width_um = 40.0;
        double width_tol_um = 5.0;
        double max_allowed_chip_um = 5.0;
    };

    KerfInspector() = default;
    explicit KerfInspector(Params params) : p_(params) {}
    const Params& params() const { return p_; }

    KerfResult inspect(const Image& img) const;

    // Render the original image with wall lines and chip boxes drawn on top.
    ColorImage annotate(const Image& img, const KerfResult& r) const;

private:
    Params p_;
};

// Serialize a result as a JSON object (pretty-printed, 2-space indent).
std::string to_json(const KerfResult& r);

}  // namespace wproc
