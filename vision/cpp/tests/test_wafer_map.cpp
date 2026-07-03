// test_wafer_map.cpp — die binning, yield, and centre/edge zone analysis.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "wproc/wafer_map.hpp"

using namespace wproc;

static int g_total = 0, g_failed = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        ++g_total;                                                      \
        if (!(cond)) {                                                  \
            ++g_failed;                                                 \
            std::cerr << "FAIL [" << __LINE__ << "]: " #cond "\n";      \
        }                                                               \
    } while (0)

int main() {
    const int C = 40, R = 40;
    // Edge-heavy chipping signature: chip grows with radius^2 (edge ring worse).
    std::mt19937 gen(5);
    std::normal_distribution<double> noise(0.0, 0.4);
    std::vector<double> chips(static_cast<std::size_t>(C) * R);
    for (int row = 0; row < R; ++row)
        for (int col = 0; col < C; ++col) {
            double nx = (col + 0.5) / C * 2 - 1, ny = (row + 0.5) / R * 2 - 1;
            double r = std::sqrt(nx * nx + ny * ny);
            chips[static_cast<std::size_t>(row) * C + col] =
                2.0 + 5.0 * r * r + noise(gen);  // um
        }

    wafer::WaferResult w = wafer::classify(C, R, chips, /*limit=*/5.0);

    // Dies inside the inscribed circle ~ (pi/4) * C*R = ~1256.
    CHECK(w.total_inside > 1150 && w.total_inside < 1320);
    // Edge ring fails, centre passes: strong zone gradient.
    CHECK(w.center_yield_pct > 90.0);
    CHECK(w.edge_yield_pct < w.center_yield_pct - 20.0);
    CHECK(w.yield_pct > w.edge_yield_pct && w.yield_pct < w.center_yield_pct);
    std::cout << "inside=" << w.total_inside << " yield=" << w.yield_pct
              << "% center=" << w.center_yield_pct << "% edge="
              << w.edge_yield_pct << "%\n";

    // ASCII map dimensions and legend characters.
    std::string a = ascii_map(w);
    int nl = static_cast<int>(std::count(a.begin(), a.end(), '\n'));
    CHECK(nl == R);
    CHECK(a.find('.') != std::string::npos);  // outside cells exist
    CHECK(a.find('o') != std::string::npos);  // some pass
    CHECK(a.find('X') != std::string::npos);  // some fail

    // Color map is the right size.
    ColorImage img = wafer::render_map(w, 8);
    CHECK(img.width() == C * 8 && img.height() == R * 8);

    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "WAFER MAP OK\n";
    return 0;
}
