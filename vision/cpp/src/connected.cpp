#include "wproc/connected.hpp"

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace wproc {

namespace {

// Disjoint-set (union-find) with path compression + union by rank.
class DisjointSet {
public:
    int make() {
        parent_.push_back(static_cast<int>(parent_.size()));
        rank_.push_back(0);
        return static_cast<int>(parent_.size()) - 1;
    }
    int find(int x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];  // path halving
            x = parent_[x];
        }
        return x;
    }
    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent_[b] = a;
        if (rank_[a] == rank_[b]) ++rank_[a];
    }

private:
    std::vector<int> parent_;
    std::vector<int> rank_;
};

}  // namespace

std::vector<Component> connected_components(const Image& binary, long min_area) {
    const int W = binary.width(), H = binary.height();
    std::vector<int> labels(static_cast<std::size_t>(W) * H, 0);
    DisjointSet ds;
    ds.make();  // reserve index 0 as "no provisional label"

    // First pass: assign provisional labels, record equivalences.
    // Neighbours already visited (8-conn): W, NW, N, NE.
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            if (binary.at(x, y) <= 0.0f) continue;
            int best = 0;
            auto consider = [&](int nx, int ny) {
                if (nx < 0 || nx >= W || ny < 0 || ny >= H) return;
                int l = labels[static_cast<std::size_t>(ny) * W + nx];
                if (l == 0) return;
                if (best == 0)
                    best = l;
                else
                    ds.unite(best, l);
            };
            consider(x - 1, y);
            consider(x - 1, y - 1);
            consider(x, y - 1);
            consider(x + 1, y - 1);

            if (best == 0) best = ds.make();  // new provisional label
            labels[static_cast<std::size_t>(y) * W + x] = best;
        }

    // Second pass: resolve to root labels and accumulate statistics.
    std::unordered_map<int, Component> comps;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            int l = labels[static_cast<std::size_t>(y) * W + x];
            if (l == 0) continue;
            int root = ds.find(l);
            Component& c = comps[root];
            if (c.area == 0) {
                c.label = root;
                c.min_x = c.max_x = x;
                c.min_y = c.max_y = y;
            }
            c.area += 1;
            c.cx += x;
            c.cy += y;
            c.min_x = std::min(c.min_x, x);
            c.max_x = std::max(c.max_x, x);
            c.min_y = std::min(c.min_y, y);
            c.max_y = std::max(c.max_y, y);
        }

    std::vector<Component> out;
    out.reserve(comps.size());
    for (auto& [root, c] : comps) {
        if (c.area < min_area) continue;
        c.cx /= static_cast<double>(c.area);
        c.cy /= static_cast<double>(c.area);
        out.push_back(c);
    }

    // Deterministic order: largest area first, then by position.
    std::sort(out.begin(), out.end(), [](const Component& a, const Component& b) {
        if (a.area != b.area) return a.area > b.area;
        if (a.min_y != b.min_y) return a.min_y < b.min_y;
        return a.min_x < b.min_x;
    });

    // Relabel 1..N in the sorted order for stable, human-friendly ids.
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i].label = static_cast<int>(i) + 1;
    return out;
}

}  // namespace wproc
