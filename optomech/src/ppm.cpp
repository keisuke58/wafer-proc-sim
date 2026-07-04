// ppm.cpp — Canvas drawing + binary PPM writer.
#include "optomech/ppm.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>

namespace optm {

Canvas::Canvas(int w, int h, Rgb fill)
    : w_(w < 0 ? 0 : w), h_(h < 0 ? 0 : h),
      d_(static_cast<std::size_t>(w_) * h_ * 3) {
    for (std::size_t i = 0; i < d_.size(); i += 3) {
        d_[i] = fill.r; d_[i + 1] = fill.g; d_[i + 2] = fill.b;
    }
}

void Canvas::set(int x, int y, Rgb c) {
    if (x < 0 || y < 0 || x >= w_ || y >= h_) return;
    const std::size_t i = (static_cast<std::size_t>(y) * w_ + x) * 3;
    d_[i] = c.r; d_[i + 1] = c.g; d_[i + 2] = c.b;
}

void Canvas::line(int x0, int y0, int x1, int y1, Rgb c) {
    int dx = std::abs(x1 - x0), dy = -std::abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1, err = dx + dy;
    for (int guard = 0; guard < 100000; ++guard) {
        set(x0, y0, c);
        if (x0 == x1 && y0 == y1) break;
        const int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

void Canvas::rect(int x0, int y0, int x1, int y1, Rgb c) {
    line(x0, y0, x1, y0, c); line(x0, y1, x1, y1, c);
    line(x0, y0, x0, y1, c); line(x1, y0, x1, y1, c);
}

void Canvas::disk(int cx, int cy, int r, Rgb c) {
    for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx)
            if (dx * dx + dy * dy <= r * r) set(cx + dx, cy + dy, c);
}

bool write_ppm(const std::string& path, const Canvas& c) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << "P6\n" << c.width() << " " << c.height() << "\n255\n";
    f.write(reinterpret_cast<const char*>(c.data()),
            static_cast<std::streamsize>(c.width()) * c.height() * 3);
    return static_cast<bool>(f);
}

}  // namespace optm
