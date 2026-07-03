// morphology.hpp — binary morphology with a square structuring element.
//
// Input images are treated as binary: pixel > 0 is foreground, else background.
// Output images use 0 / 255. Border handled as background (erosion shrinks at
// edges), which is the conventional and safe choice for defect blobs.
#pragma once

#include "wproc/image.hpp"

namespace wproc {

Image erode(const Image& src, int radius);
Image dilate(const Image& src, int radius);

// open = erode then dilate (removes small bright specks / noise)
Image morph_open(const Image& src, int radius);

// close = dilate then erode (fills small holes, joins nearby blobs)
Image morph_close(const Image& src, int radius);

}  // namespace wproc
