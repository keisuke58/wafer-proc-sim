#include "wproc/draw.hpp"

#include <algorithm>
#include <cmath>

namespace wproc {

ColorImage to_color(const Image& img) {
    ColorImage out(img.width(), img.height());
    for (int y = 0; y < img.height(); ++y)
        for (int x = 0; x < img.width(); ++x) {
            float v = img.at(x, y);
            long r = std::lround(v);
            if (r < 0) r = 0;
            if (r > 255) r = 255;
            auto u = static_cast<std::uint8_t>(r);
            out.set(x, y, Rgb{u, u, u});
        }
    return out;
}

void draw_hline(ColorImage& img, int y, int x0, int x1, Rgb c) {
    if (x0 > x1) std::swap(x0, x1);
    for (int x = x0; x <= x1; ++x) img.set(x, y, c);
}

void draw_vline(ColorImage& img, int x, int y0, int y1, Rgb c) {
    if (y0 > y1) std::swap(y0, y1);
    for (int y = y0; y <= y1; ++y) img.set(x, y, c);
}

void draw_rect(ColorImage& img, int x0, int y0, int x1, int y1, Rgb c) {
    if (x0 > x1) std::swap(x0, x1);
    if (y0 > y1) std::swap(y0, y1);
    draw_hline(img, y0, x0, x1, c);
    draw_hline(img, y1, x0, x1, c);
    draw_vline(img, x0, y0, y1, c);
    draw_vline(img, x1, y0, y1, c);
}

void draw_cross(ColorImage& img, int cx, int cy, int arm, Rgb c) {
    draw_hline(img, cy, cx - arm, cx + arm, c);
    draw_vline(img, cx, cy - arm, cy + arm, c);
}

}  // namespace wproc
