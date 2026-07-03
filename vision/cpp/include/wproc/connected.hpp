// connected.hpp — connected-component labeling for binary images.
//
// Two-pass union-find (8-connectivity). Used to isolate individual chipping
// defects along a dicing kerf so each can be measured independently.
#pragma once

#include <vector>

#include "wproc/image.hpp"

namespace wproc {

struct Component {
    int label = 0;         // 1-based component id
    long area = 0;         // pixel count
    double cx = 0.0;       // centroid x
    double cy = 0.0;       // centroid y
    int min_x = 0, min_y = 0, max_x = 0, max_y = 0;  // bounding box (inclusive)

    int bbox_width() const { return max_x - min_x + 1; }
    int bbox_height() const { return max_y - min_y + 1; }
};

// Label foreground (pixel > 0) components with 8-connectivity.
// Returns the list of components (background excluded), sorted by descending
// area. min_area drops components smaller than the threshold (noise).
std::vector<Component> connected_components(const Image& binary, long min_area = 1);

}  // namespace wproc
