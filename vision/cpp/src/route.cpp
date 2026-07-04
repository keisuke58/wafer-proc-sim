// route.cpp — A* path planning + transport-job sequencing.
#include "wproc/route.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <vector>

#include "wproc/draw.hpp"

namespace wproc::route {
namespace {

constexpr double kSqrt2 = 1.4142135623730951;

double octile(int x0, int y0, int x1, int y1) {
    const int dx = std::abs(x1 - x0), dy = std::abs(y1 - y0);
    return (dx + dy) + (kSqrt2 - 2.0) * std::min(dx, dy);
}

struct Node {
    int idx;
    double f;
    bool operator>(const Node& o) const { return f > o.f; }
};

double dist(const Station& a, const Station& b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

}  // namespace

Path astar(const Grid& g, Cell start, Cell goal) {
    Path result;
    if (g.w <= 0 || g.h <= 0) return result;
    if (g.blocked(start.x, start.y) || g.blocked(goal.x, goal.y)) return result;

    const std::size_t N = static_cast<std::size_t>(g.w) * g.h;
    auto id = [&](int x, int y) { return static_cast<std::size_t>(y) * g.w + x; };
    std::vector<double> gscore(N, 1e300);
    std::vector<int> came(N, -1);
    std::vector<std::uint8_t> closed(N, 0);
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open;

    gscore[id(start.x, start.y)] = 0.0;
    open.push({static_cast<int>(id(start.x, start.y)),
               octile(start.x, start.y, goal.x, goal.y)});

    const int dxs[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    const int dys[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    const int gid = static_cast<int>(id(goal.x, goal.y));

    while (!open.empty()) {
        const Node cur = open.top();
        open.pop();
        if (closed[cur.idx]) continue;
        closed[cur.idx] = 1;
        ++result.expanded;
        if (cur.idx == gid) break;

        const int cx = cur.idx % g.w, cy = cur.idx / g.w;
        for (int k = 0; k < 8; ++k) {
            const int nx = cx + dxs[k], ny = cy + dys[k];
            if (g.blocked(nx, ny)) continue;
            const bool diag = k >= 4;
            // Disallow corner-cutting: both orthogonal neighbours must be free.
            if (diag && (g.blocked(cx + dxs[k], cy) || g.blocked(cx, cy + dys[k])))
                continue;
            const std::size_t nid = id(nx, ny);
            if (closed[nid]) continue;
            const double step = diag ? kSqrt2 : 1.0;
            const double tentative = gscore[cur.idx] + step;
            if (tentative < gscore[nid]) {
                gscore[nid] = tentative;
                came[nid] = cur.idx;
                open.push({static_cast<int>(nid),
                           tentative + octile(nx, ny, goal.x, goal.y)});
            }
        }
    }

    if (came[gid] == -1 && gid != static_cast<int>(id(start.x, start.y)))
        return result;
    result.found = true;
    result.length = gscore[gid];
    for (int at = gid; at != -1; at = came[at])
        result.cells.push_back({at % g.w, at / g.w});
    std::reverse(result.cells.begin(), result.cells.end());
    return result;
}

Schedule evaluate(const std::vector<Station>& st, const std::vector<Job>& jobs,
                  const std::vector<int>& order, const Station& home,
                  const SeqParams& p) {
    Schedule s;
    s.order = order;
    Station cur = home;
    for (int j : order) {
        const Station& pick = st[jobs[j].pick];
        const Station& place = st[jobs[j].place];
        s.travel += dist(cur, pick) + dist(pick, place);
        cur = place;
    }
    s.makespan_s = s.travel / p.speed + static_cast<double>(order.size()) * p.handling_s;
    return s;
}

Schedule plan_fifo(const std::vector<Station>& st, const std::vector<Job>& jobs,
                   const Station& home, const SeqParams& p) {
    std::vector<int> order(jobs.size());
    for (std::size_t i = 0; i < jobs.size(); ++i) order[i] = static_cast<int>(i);
    return evaluate(st, jobs, order, home, p);
}

Schedule optimize(const std::vector<Station>& st, const std::vector<Job>& jobs,
                  const Station& home, const SeqParams& p) {
    const int n = static_cast<int>(jobs.size());
    if (n == 0) return evaluate(st, jobs, {}, home, p);

    // Nearest-neighbour construction: repeatedly pick the job whose pickup is
    // closest to the robot's current position.
    std::vector<int> order;
    std::vector<std::uint8_t> used(n, 0);
    Station cur = home;
    for (int step = 0; step < n; ++step) {
        int best = -1;
        double bd = 1e300;
        for (int j = 0; j < n; ++j) {
            if (used[j]) continue;
            const double d = dist(cur, st[jobs[j].pick]);
            if (d < bd) { bd = d; best = j; }
        }
        if (best < 0) break;  // all jobs served (defensive; loop bounds ensure one)
        used[best] = 1;
        order.push_back(best);
        cur = st[jobs[best].place];
    }

    // 2-opt refinement: reverse order segments while the total travel drops.
    double best_travel = evaluate(st, jobs, order, home, p).travel;
    bool improved = true;
    while (improved) {
        improved = false;
        for (int i = 0; i < n - 1; ++i) {
            for (int kk = i + 1; kk < n; ++kk) {
                std::vector<int> cand = order;
                std::reverse(cand.begin() + i, cand.begin() + kk + 1);
                const double t = evaluate(st, jobs, cand, home, p).travel;
                if (t + 1e-9 < best_travel) {
                    best_travel = t;
                    order = std::move(cand);
                    improved = true;
                }
            }
        }
    }
    return evaluate(st, jobs, order, home, p);
}

ColorImage render_path(const Grid& g, const Path& path, int cell_px) {
    const int W = g.w * cell_px, H = g.h * cell_px;
    ColorImage out(W, H);
    for (int y = 0; y < g.h; ++y)
        for (int x = 0; x < g.w; ++x) {
            const Rgb c = g.blocked(x, y) ? Rgb{60, 66, 74} : Rgb{232, 238, 244};
            for (int dy = 0; dy < cell_px; ++dy)
                for (int dx = 0; dx < cell_px; ++dx)
                    out.set(x * cell_px + dx, y * cell_px + dy, c);
        }
    for (const Cell& c : path.cells) {
        const int cx = c.x * cell_px + cell_px / 2, cy = c.y * cell_px + cell_px / 2;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                out.set(cx + dx, cy + dy, {40, 165, 100});
    }
    if (path.found && !path.cells.empty()) {
        const Cell s = path.cells.front(), e = path.cells.back();
        draw_cross(out, s.x * cell_px + cell_px / 2, s.y * cell_px + cell_px / 2, 3,
                   {40, 110, 210});
        draw_cross(out, e.x * cell_px + cell_px / 2, e.y * cell_px + cell_px / 2, 3,
                   {210, 70, 70});
    }
    return out;
}

}  // namespace wproc::route
