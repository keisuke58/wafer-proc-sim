// route.hpp — wafer-transport robot: collision-free paths + job sequencing.
//
// A cluster tool moves wafers between load ports, aligners, process chambers and
// metrology with a handling robot. Two optimization problems decide its
// throughput:
//
//   1. Path planning — get the end-effector from A to B without hitting a
//      chamber or another arm. Solved here with A* on an occupancy grid
//      (octile heuristic, 8-connected), which returns the shortest collision-
//      free path.
//
//   2. Job sequencing — given a batch of pick→place transport jobs, choose the
//      order that minimizes total travel (hence maximizes wafers/hour). Solved
//      with a nearest-neighbour construction refined by 2-opt, and compared
//      against naive FIFO order.
//
// Pure C++17, no external dependencies.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "wproc/image.hpp"

namespace wproc::route {

// ── A* path planning on an occupancy grid ───────────────────────────────────

struct Cell {
    int x = 0;
    int y = 0;
};

// Occupancy grid: occ[y*w + x] != 0 marks a blocked cell.
struct Grid {
    int w = 0;
    int h = 0;
    std::vector<std::uint8_t> occ;

    Grid() = default;
    Grid(int width, int height) : w(width), h(height), occ(static_cast<std::size_t>(width) * height, 0) {}
    bool blocked(int x, int y) const {
        return x < 0 || y < 0 || x >= w || y >= h ||
               occ[static_cast<std::size_t>(y) * w + x] != 0;
    }
    void block(int x, int y) {
        if (x >= 0 && y >= 0 && x < w && y < h) occ[static_cast<std::size_t>(y) * w + x] = 1;
    }
};

struct Path {
    bool found = false;
    double length = 0.0;         // path length in cells (diagonals cost sqrt2)
    int expanded = 0;            // nodes popped (search effort)
    std::vector<Cell> cells;     // start..goal inclusive
};

// Shortest 8-connected collision-free path from `start` to `goal`. Diagonal
// moves are disallowed when they would cut a blocked corner.
Path astar(const Grid& g, Cell start, Cell goal);

// ── Transport-job sequencing ────────────────────────────────────────────────

struct Station {
    std::string name;
    double x = 0.0;
    double y = 0.0;
};

// A transport job: pick a wafer at station `pick`, place it at station `place`.
struct Job {
    int pick = 0;
    int place = 0;
};

struct Schedule {
    std::vector<int> order;   // permutation of job indices
    double travel = 0.0;      // total end-effector travel distance
    double makespan_s = 0.0;  // travel/speed + handling per job
};

struct SeqParams {
    double speed = 500.0;     // mm/s
    double handling_s = 1.5;  // fixed pick+place time per job
};

// Serve the jobs in the given `order`, starting from `home`, and score it.
Schedule evaluate(const std::vector<Station>& st, const std::vector<Job>& jobs,
                  const std::vector<int>& order, const Station& home,
                  const SeqParams& p = SeqParams{});

// FIFO baseline: serve jobs in their original order.
Schedule plan_fifo(const std::vector<Station>& st, const std::vector<Job>& jobs,
                   const Station& home, const SeqParams& p = SeqParams{});

// Optimized: nearest-neighbour construction refined by 2-opt.
Schedule optimize(const std::vector<Station>& st, const std::vector<Job>& jobs,
                  const Station& home, const SeqParams& p = SeqParams{});

// Draw the grid (obstacles) and a planned path (green) as a color image, scaled
// up by `cell_px`.
ColorImage render_path(const Grid& g, const Path& path, int cell_px = 8);

}  // namespace wproc::route
