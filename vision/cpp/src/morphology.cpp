#include "wproc/morphology.hpp"

#include <algorithm>

namespace wproc {

namespace {

// erode: foreground survives only if the whole (2r+1)^2 neighbourhood is
// foreground. dilate: background flips to foreground if any neighbour is
// foreground. `want_all` selects between the two.
Image morph_pass(const Image& src, int radius, bool want_all) {
    if (radius <= 0) return src;
    const int W = src.width(), H = src.height();
    Image out(W, H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            bool all_fg = true, any_fg = false;
            for (int dy = -radius; dy <= radius; ++dy)
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = x + dx, ny = y + dy;
                    bool fg;
                    if (nx < 0 || nx >= W || ny < 0 || ny >= H)
                        fg = false;  // border = background
                    else
                        fg = src.at(nx, ny) > 0.0f;
                    all_fg = all_fg && fg;
                    any_fg = any_fg || fg;
                }
            bool keep = want_all ? all_fg : any_fg;
            out.at(x, y) = keep ? 255.0f : 0.0f;
        }
    return out;
}

}  // namespace

Image erode(const Image& src, int radius) {
    return morph_pass(src, radius, /*want_all=*/true);
}

Image dilate(const Image& src, int radius) {
    return morph_pass(src, radius, /*want_all=*/false);
}

Image morph_open(const Image& src, int radius) {
    return dilate(erode(src, radius), radius);
}

Image morph_close(const Image& src, int radius) {
    return erode(dilate(src, radius), radius);
}

}  // namespace wproc
