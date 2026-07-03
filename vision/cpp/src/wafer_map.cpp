#include "wproc/wafer_map.hpp"

#include <cmath>
#include <string>

namespace wproc::wafer {

WaferResult classify(int cols, int rows, const std::vector<double>& values,
                     double limit, double center_zone_frac) {
    WaferResult w;
    w.cols = cols;
    w.rows = rows;
    if (cols <= 0 || rows <= 0 ||
        values.size() != static_cast<std::size_t>(cols) * rows)
        return w;

    int c_in = 0, c_good = 0, e_in = 0, e_good = 0;
    w.dies.reserve(static_cast<std::size_t>(cols) * rows);
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < cols; ++col) {
            Die d;
            d.col = col;
            d.row = row;
            // Normalized die-centre position in [-1, 1]; inscribed wafer circle
            // has radius 1.
            double nx = (col + 0.5) / cols * 2.0 - 1.0;
            double ny = (row + 0.5) / rows * 2.0 - 1.0;
            d.norm_r = std::sqrt(nx * nx + ny * ny);
            d.value = values[static_cast<std::size_t>(row) * cols + col];
            d.inside = d.norm_r <= 1.0;
            d.pass = d.inside && d.value <= limit;
            if (d.inside) {
                ++w.total_inside;
                if (d.pass) ++w.good;
                if (d.norm_r <= center_zone_frac) {
                    ++c_in;
                    if (d.pass) ++c_good;
                } else {
                    ++e_in;
                    if (d.pass) ++e_good;
                }
            }
            w.dies.push_back(d);
        }

    w.yield_pct = w.total_inside ? 100.0 * w.good / w.total_inside : 0.0;
    w.center_yield_pct = c_in ? 100.0 * c_good / c_in : 0.0;
    w.edge_yield_pct = e_in ? 100.0 * e_good / e_in : 0.0;
    return w;
}

std::string ascii_map(const WaferResult& w) {
    std::string s;
    s.reserve(static_cast<std::size_t>(w.rows) * (w.cols + 1));
    for (int row = 0; row < w.rows; ++row) {
        for (int col = 0; col < w.cols; ++col) {
            const Die& d = w.dies[static_cast<std::size_t>(row) * w.cols + col];
            s += d.inside ? (d.pass ? 'o' : 'X') : '.';
        }
        s += '\n';
    }
    return s;
}

ColorImage render_map(const WaferResult& w, int cell_px) {
    if (cell_px < 1) cell_px = 1;
    ColorImage img(w.cols * cell_px, w.rows * cell_px);
    const Rgb outside{60, 62, 66}, pass{30, 170, 90}, fail{220, 60, 60};
    for (int row = 0; row < w.rows; ++row)
        for (int col = 0; col < w.cols; ++col) {
            const Die& d = w.dies[static_cast<std::size_t>(row) * w.cols + col];
            Rgb c = d.inside ? (d.pass ? pass : fail) : outside;
            for (int dy = 0; dy < cell_px; ++dy)
                for (int dx = 0; dx < cell_px; ++dx)
                    img.set(col * cell_px + dx, row * cell_px + dy, c);
        }
    return img;
}

}  // namespace wproc::wafer
