// test_route.cpp — A* path planning + transport-job sequencing.
#include <cmath>
#include <iostream>
#include <vector>

#include "wproc/route.hpp"

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

// Straight-line path on an empty grid has the octile length.
static void test_astar_open() {
    route::Grid g(20, 20);
    const route::Path p = route::astar(g, {0, 0}, {9, 0});
    CHECK(p.found);
    CHECK(std::fabs(p.length - 9.0) < 1e-6);
    CHECK(p.cells.front().x == 0 && p.cells.back().x == 9);
}

// A wall forces a detour; the path must go around and stay off obstacles.
static void test_astar_obstacle() {
    route::Grid g(20, 20);
    for (int y = 0; y < 15; ++y) g.block(10, y);   // vertical wall with a gap
    const route::Path p = route::astar(g, {2, 7}, {18, 7});
    CHECK(p.found);
    for (const route::Cell& c : p.cells) CHECK(!g.blocked(c.x, c.y));
    // Detour is longer than the blocked straight shot (16).
    CHECK(p.length > 16.0);

    // Fully walled goal is unreachable.
    route::Grid g2(10, 10);
    for (int y = 0; y < 10; ++y) g2.block(5, y);
    const route::Path np = route::astar(g2, {1, 1}, {8, 1});
    CHECK(!np.found);
}

// Optimized sequencing must not beat FIFO by making things worse, and on a
// deliberately bad FIFO order it should improve travel substantially.
static void test_sequencing() {
    // Stations laid out so FIFO zig-zags but a good order is monotone.
    std::vector<route::Station> st = {
        {"L0", 0, 0},   {"P1", 100, 0}, {"P2", 100, 100}, {"P3", 0, 100},
        {"P4", 50, 50}, {"P5", 200, 0}, {"P6", 200, 100}};
    // Jobs pick from a load port and place at a chamber; FIFO order is scrambled.
    std::vector<route::Job> jobs = {
        {0, 5}, {0, 1}, {0, 6}, {0, 2}, {0, 4}, {0, 3}};
    route::Station home{"home", 0, 0};

    const route::Schedule fifo = route::plan_fifo(st, jobs, home);
    const route::Schedule opt = route::optimize(st, jobs, home);
    std::cout << "FIFO travel=" << fifo.travel << " opt travel=" << opt.travel
              << " makespan " << fifo.makespan_s << "->" << opt.makespan_s << "\n";
    CHECK(opt.order.size() == jobs.size());
    CHECK(opt.travel <= fifo.travel + 1e-9);   // never worse
    CHECK(opt.travel < fifo.travel);           // strictly better here
    CHECK(opt.makespan_s < fifo.makespan_s);

    // Order is a valid permutation.
    std::vector<int> seen(jobs.size(), 0);
    for (int j : opt.order) {
        const bool valid = j >= 0 && j < static_cast<int>(jobs.size());
        CHECK(valid);
        if (valid) seen[static_cast<std::size_t>(j)]++;
    }
    for (int s : seen) CHECK(s == 1);

    // Rendering yields a sized canvas.
    route::Grid g(16, 16);
    const route::Path p = route::astar(g, {1, 1}, {14, 14});
    ColorImage img = route::render_path(g, p, 6);
    CHECK(img.width() == 96 && img.height() == 96 && !img.empty());
}

int main() {
    test_astar_open();
    test_astar_obstacle();
    test_sequencing();
    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "ROUTE OK\n";
    return 0;
}
