#include "wproc/kerf_inspect.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

#include "wproc/connected.hpp"
#include "wproc/draw.hpp"
#include "wproc/filters.hpp"
#include "wproc/morphology.hpp"
#include "wproc/threshold.hpp"

namespace wproc {

namespace {

std::vector<float> column_projection(const Image& gx) {
    std::vector<float> proj(gx.width(), 0.0f);
    for (int y = 0; y < gx.height(); ++y)
        for (int x = 0; x < gx.width(); ++x)
            proj[x] += gx.at(x, y);
    return proj;
}

std::vector<float> smooth3(const std::vector<float>& p) {
    const int n = static_cast<int>(p.size());
    std::vector<float> s(p);
    for (int i = 1; i < n - 1; ++i) s[i] = (p[i - 1] + p[i] + p[i + 1]) / 3.0f;
    return s;
}

// Parabolic sub-pixel refinement around integer index i using neighbours.
double subpixel(const std::vector<float>& p, int i) {
    const int n = static_cast<int>(p.size());
    if (i <= 0 || i >= n - 1) return static_cast<double>(i);
    double a = p[i - 1], b = p[i], c = p[i + 1];
    double denom = a - 2.0 * b + c;
    if (std::abs(denom) < 1e-9) return static_cast<double>(i);
    double offset = 0.5 * (a - c) / denom;
    if (offset < -1.0) offset = -1.0;
    if (offset > 1.0) offset = 1.0;
    return static_cast<double>(i) + offset;
}

double clamp01(double v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); }

}  // namespace

KerfResult KerfInspector::inspect(const Image& img) const {
    KerfResult r;
    if (img.width() < 5 || img.height() < 3) {
        r.status = "image_too_small";
        return r;
    }

    // 1. Denoise, then signed horizontal gradient projected over rows.
    Image blurred = gaussian_blur(img, p_.blur_sigma);
    Gradient g = sobel(blurred);
    std::vector<float> proj = smooth3(column_projection(g.gx));

    // 2. Locate walls: dark channel => negative slope at the left wall,
    //    positive slope at the right wall.
    int W = static_cast<int>(proj.size());
    int imin = 0, imax = 0;
    float vmin = proj[0], vmax = proj[0];
    double sum_abs = 0.0;
    for (int c = 0; c < W; ++c) {
        if (proj[c] < vmin) { vmin = proj[c]; imin = c; }
        if (proj[c] > vmax) { vmax = proj[c]; imax = c; }
        sum_abs += std::abs(proj[c]);
    }
    double mean_abs = sum_abs / std::max(1, W);

    // Left wall must precede the right wall for a dark central channel.
    if (imin >= imax || vmin >= 0.0f || vmax <= 0.0f) {
        r.status = "no_kerf_found";
        return r;
    }

    double left = subpixel(proj, imin);
    double right = subpixel(proj, imax);
    r.found = true;
    r.left_wall_px = left;
    r.right_wall_px = right;
    r.width_px = right - left;
    r.width_um = r.width_px * p_.um_per_px;

    double avg_peak = 0.5 * (std::abs(vmin) + std::abs(vmax));
    double snr = avg_peak / (mean_abs + 1e-9);
    r.confidence = clamp01((snr - 1.0) / (p_.min_wall_snr - 1.0));

    // 3. Chipping: dark blobs lying outside the kerf band.
    float thr = p_.dark_threshold < 0.0f ? otsu_threshold(blurred)
                                         : p_.dark_threshold;
    Image dark = binarize(blurred, thr, /*invert=*/true);
    if (p_.morph_radius > 0) dark = morph_open(dark, p_.morph_radius);

    // Zero the kerf interior so only wall-protruding chips remain.
    int lo = static_cast<int>(std::floor(left));
    int hi = static_cast<int>(std::ceil(right));
    for (int y = 0; y < dark.height(); ++y)
        for (int x = std::max(0, lo); x <= std::min(dark.width() - 1, hi); ++x)
            dark.at(x, y) = 0.0f;

    auto comps = connected_components(dark, p_.min_chip_area_px);
    const double um2 = p_.um_per_px * p_.um_per_px;
    int next_id = 1;
    for (const auto& c : comps) {
        Chip chip;
        chip.side = (c.cx < left) ? -1 : +1;
        // Chipping protrudes from a wall; a dark blob that doesn't reach the
        // wall (dust, fiducial, texture) is not a chip and must not be
        // measured back to it, which would report a huge bogus protrusion.
        bool touches_wall =
            (chip.side < 0)
                ? (static_cast<double>(c.max_x) >= left - p_.wall_attach_margin_px)
                : (static_cast<double>(c.min_x) <= right + p_.wall_attach_margin_px);
        if (!touches_wall) continue;
        double depth_px = (chip.side < 0) ? (left - c.min_x)
                                          : (c.max_x - right);
        if (depth_px < 0.0) depth_px = 0.0;
        chip.protrusion_um = depth_px * p_.um_per_px;
        chip.cx = c.cx;
        chip.cy = c.cy;
        chip.area_px = c.area;
        chip.area_um2 = static_cast<double>(c.area) * um2;
        chip.min_x = c.min_x;
        chip.min_y = c.min_y;
        chip.max_x = c.max_x;
        chip.max_y = c.max_y;
        chip.id = next_id++;
        r.max_chip_um = std::max(r.max_chip_um, chip.protrusion_um);
        r.total_chip_area_um2 += chip.area_um2;
        r.chips.push_back(chip);
    }

    // 4. Spec verdict.
    r.width_in_spec = (p_.width_tol_um < 0.0) ||
                      (std::abs(r.width_um - p_.nominal_width_um) <= p_.width_tol_um);
    r.chipping_in_spec = r.max_chip_um <= p_.max_allowed_chip_um;
    r.pass = r.width_in_spec && r.chipping_in_spec;
    r.status = "ok";
    return r;
}

ColorImage KerfInspector::annotate(const Image& img, const KerfResult& r) const {
    ColorImage out = to_color(img);
    if (!r.found) return out;

    const Rgb green{0, 220, 0};
    const Rgb red{235, 40, 40};
    const Rgb yellow{240, 220, 0};

    int lx = static_cast<int>(std::lround(r.left_wall_px));
    int rx = static_cast<int>(std::lround(r.right_wall_px));
    draw_vline(out, lx, 0, img.height() - 1, green);
    draw_vline(out, rx, 0, img.height() - 1, green);

    for (const auto& c : r.chips) {
        Rgb col = (c.protrusion_um > p_.max_allowed_chip_um) ? red : yellow;
        draw_rect(out, c.min_x, c.min_y, c.max_x, c.max_y, col);
        draw_cross(out, static_cast<int>(std::lround(c.cx)),
                   static_cast<int>(std::lround(c.cy)), 2, col);
    }
    return out;
}

std::string to_json(const KerfResult& r) {
    std::ostringstream o;
    o.setf(std::ios::fixed);
    o.precision(4);
    o << "{\n";
    o << "  \"status\": \"" << r.status << "\",\n";
    o << "  \"found\": " << (r.found ? "true" : "false") << ",\n";
    o << "  \"left_wall_px\": " << r.left_wall_px << ",\n";
    o << "  \"right_wall_px\": " << r.right_wall_px << ",\n";
    o << "  \"width_px\": " << r.width_px << ",\n";
    o << "  \"width_um\": " << r.width_um << ",\n";
    o << "  \"confidence\": " << r.confidence << ",\n";
    o << "  \"width_in_spec\": " << (r.width_in_spec ? "true" : "false") << ",\n";
    o << "  \"chipping_in_spec\": " << (r.chipping_in_spec ? "true" : "false") << ",\n";
    o << "  \"pass\": " << (r.pass ? "true" : "false") << ",\n";
    o << "  \"max_chip_um\": " << r.max_chip_um << ",\n";
    o << "  \"total_chip_area_um2\": " << r.total_chip_area_um2 << ",\n";
    o << "  \"chip_count\": " << r.chips.size() << ",\n";
    o << "  \"chips\": [";
    for (std::size_t i = 0; i < r.chips.size(); ++i) {
        const Chip& c = r.chips[i];
        o << (i == 0 ? "\n" : ",\n");
        o << "    {";
        o << "\"id\": " << c.id << ", ";
        o << "\"side\": \"" << (c.side < 0 ? "left" : "right") << "\", ";
        o << "\"cx\": " << c.cx << ", \"cy\": " << c.cy << ", ";
        o << "\"area_px\": " << c.area_px << ", ";
        o << "\"area_um2\": " << c.area_um2 << ", ";
        o << "\"protrusion_um\": " << c.protrusion_um << ", ";
        o << "\"bbox\": [" << c.min_x << ", " << c.min_y << ", " << c.max_x
          << ", " << c.max_y << "]}";
    }
    o << (r.chips.empty() ? "]\n" : "\n  ]\n");
    o << "}\n";
    return o.str();
}

}  // namespace wproc
