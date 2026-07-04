// vcal.cpp — implementation of vision calibration + Gage R&R.
#include "wproc/vcal.hpp"

#include <algorithm>
#include <cmath>

#include "wproc/draw.hpp"

namespace wproc::vcal {
namespace {

// Solve a 3x3 linear system A x = b by Gaussian elimination with partial
// pivoting. Returns false if the matrix is (near) singular.
bool solve3(std::array<std::array<double, 3>, 3> a, std::array<double, 3> b,
            std::array<double, 3>& x) {
    for (int col = 0; col < 3; ++col) {
        int piv = col;
        for (int r = col + 1; r < 3; ++r)
            if (std::fabs(a[r][col]) > std::fabs(a[piv][col])) piv = r;
        if (std::fabs(a[piv][col]) < 1e-12) return false;
        std::swap(a[piv], a[col]);
        std::swap(b[piv], b[col]);
        for (int r = 0; r < 3; ++r) {
            if (r == col) continue;
            const double f = a[r][col] / a[col][col];
            for (int c = col; c < 3; ++c) a[r][c] -= f * a[col][c];
            b[r] -= f * b[col];
        }
    }
    for (int i = 0; i < 3; ++i) x[i] = b[i] / a[i][i];
    return true;
}

// Fit [a0 a1 a2] minimizing sum (a0*px + a1*py + a2 - target)^2 via the normal
// equations. `sel` picks the target field (sx or sy) from each correspondence.
bool fit_row(const std::vector<Correspondence>& pts, bool use_sx,
             std::array<double, 3>& out) {
    std::array<std::array<double, 3>, 3> ata{};
    std::array<double, 3> atb{};
    for (const auto& p : pts) {
        const double bx[3] = {p.px, p.py, 1.0};
        const double t = use_sx ? p.sx : p.sy;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) ata[i][j] += bx[i] * bx[j];
            atb[i] += bx[i] * t;
        }
    }
    return solve3(ata, atb, out);
}

// AIAG average-and-range constants (index by count, 0/1 unused).
double k1_trials(int r) {  // 1/d2* for range of r readings
    switch (r) { case 2: return 0.8862; case 3: return 0.5908; default: break; }
    return 0.5908;
}
double k2_ops(int o) {
    switch (o) { case 2: return 0.7071; case 3: return 0.5231; default: break; }
    return 0.5231;
}
double k3_parts(int p) {
    static const double k[] = {0, 0, 0.7071, 0.5231, 0.4467, 0.4030, 0.3742,
                               0.3534, 0.3375, 0.3249, 0.3146};
    if (p >= 2 && p <= 10) return k[p];
    return 0.3146;
}

}  // namespace

CalibResult calibrate(const std::vector<Correspondence>& pts) {
    CalibResult r;
    if (pts.size() < 3) return r;
    std::array<double, 3> ax{}, ay{};
    if (!fit_row(pts, true, ax) || !fit_row(pts, false, ay)) return r;
    r.transform.m = {ax[0], ax[1], ax[2], ay[0], ay[1], ay[2]};

    double sse = 0.0, worst = 0.0;
    for (const auto& p : pts) {
        const double ex = r.transform.map_x(p.px, p.py) - p.sx;
        const double ey = r.transform.map_y(p.px, p.py) - p.sy;
        const double d2 = ex * ex + ey * ey;  // mm^2
        sse += d2;
        worst = std::max(worst, std::sqrt(d2));
    }
    r.rms_um = std::sqrt(sse / static_cast<double>(pts.size())) * 1000.0;
    r.max_um = worst * 1000.0;

    // Pixel pitch = mean length of the two linear-part columns (mm/px → µm/px).
    const double c0 = std::hypot(ax[0], ay[0]);
    const double c1 = std::hypot(ax[1], ay[1]);
    r.um_per_px = 0.5 * (c0 + c1) * 1000.0;
    r.rotation_deg = std::atan2(ay[0], ax[0]) * 180.0 / M_PI;
    r.ok = true;
    return r;
}

void undistort_point(const Distortion& d, double px, double py,
                     double& ux, double& uy) {
    const double dx = px - d.cx, dy = py - d.cy;
    const double r2 = (dx * dx + dy * dy) / (d.f * d.f);
    const double s = 1.0 + d.k1 * r2 + d.k2 * r2 * r2;
    ux = d.cx + s * dx;
    uy = d.cy + s * dy;
}

GageResult gage_rr(const GageStudy& s) {
    GageResult g;
    const int O = s.operators, P = s.parts, T = s.trials;
    // The AIAG average-and-range constants below are only tabulated for
    // 2-3 operators, 2-3 trials and 2-10 parts; reject studies outside that
    // range rather than silently substituting an approximate constant.
    if (O < 2 || O > 3 || T < 2 || T > 3 || P < 2 || P > 10 ||
        static_cast<int>(s.values.size()) != O)
        return g;

    // Ranges within each operator×part, plus operator and part averages.
    double rbar_sum = 0.0;
    int rbar_n = 0;
    std::vector<double> op_mean(O, 0.0);
    std::vector<int> op_cnt(O, 0);
    std::vector<double> part_sum(P, 0.0);
    std::vector<int> part_cnt(P, 0);
    for (int o = 0; o < O; ++o) {
        if (static_cast<int>(s.values[o].size()) != P) return g;
        for (int p = 0; p < P; ++p) {
            const auto& tr = s.values[o][p];
            if (static_cast<int>(tr.size()) != T) return g;
            double lo = tr[0], hi = tr[0];
            for (double v : tr) {
                lo = std::min(lo, v);
                hi = std::max(hi, v);
                op_mean[o] += v; op_cnt[o]++;
                part_sum[p] += v; part_cnt[p]++;
            }
            rbar_sum += hi - lo; rbar_n++;
        }
    }
    const double rbar = rbar_sum / rbar_n;
    for (int o = 0; o < O; ++o) op_mean[o] /= op_cnt[o];

    // EV (repeatability).
    g.repeatability = rbar * k1_trials(T);

    // AV (reproducibility) from the operator-average spread.
    double xbar_lo = op_mean[0], xbar_hi = op_mean[0];
    for (double m : op_mean) { xbar_lo = std::min(xbar_lo, m); xbar_hi = std::max(xbar_hi, m); }
    const double xdiff = xbar_hi - xbar_lo;
    const double av2 = std::pow(xdiff * k2_ops(O), 2) -
                       (g.repeatability * g.repeatability) / (P * T);
    g.reproducibility = av2 > 0.0 ? std::sqrt(av2) : 0.0;

    g.grr = std::hypot(g.repeatability, g.reproducibility);

    // PV from the part-average spread.
    std::vector<double> part_mean(P);
    for (int p = 0; p < P; ++p) part_mean[p] = part_sum[p] / part_cnt[p];
    double plo = part_mean[0], phi = part_mean[0];
    for (double m : part_mean) { plo = std::min(plo, m); phi = std::max(phi, m); }
    g.part_variation = (phi - plo) * k3_parts(P);

    g.total_variation = std::hypot(g.grr, g.part_variation);
    if (g.total_variation > 0.0) {
        g.pct_grr = 100.0 * g.grr / g.total_variation;
        g.pct_ev = 100.0 * g.repeatability / g.total_variation;
        g.pct_av = 100.0 * g.reproducibility / g.total_variation;
        g.pct_pv = 100.0 * g.part_variation / g.total_variation;
    }
    g.ndc = g.grr > 0.0 ? static_cast<int>(1.41 * g.part_variation / g.grr) : 0;
    g.acceptable = g.pct_grr < 30.0 && g.ndc >= 5;
    g.ok = true;
    return g;
}

ColorImage render_residuals(const std::vector<Correspondence>& pts,
                            const CalibResult& cal, int w, int h, double mag) {
    Image bg(w, h, 245.0f);
    ColorImage out = to_color(bg);
    if (!cal.ok || pts.empty()) return out;

    // Map stage coordinates into the canvas: fit a display scale from the
    // stage-coordinate bounding box with a margin.
    double slo_x = pts[0].sx, shi_x = slo_x, slo_y = pts[0].sy, shi_y = slo_y;
    for (const auto& p : pts) {
        slo_x = std::min(slo_x, p.sx); shi_x = std::max(shi_x, p.sx);
        slo_y = std::min(slo_y, p.sy); shi_y = std::max(shi_y, p.sy);
    }
    const double margin = 30.0;
    const double sx = (w - 2 * margin) / std::max(1e-9, shi_x - slo_x);
    const double sy = (h - 2 * margin) / std::max(1e-9, shi_y - slo_y);
    auto cx = [&](double x) { return static_cast<int>(margin + (x - slo_x) * sx); };
    auto cy = [&](double y) { return static_cast<int>(margin + (y - slo_y) * sy); };

    for (const auto& p : pts) {
        const double fx = cal.transform.map_x(p.px, p.py);
        const double fy = cal.transform.map_y(p.px, p.py);
        const double err_um = std::hypot(fx - p.sx, fy - p.sy) * 1000.0;
        const Rgb col = err_um > 3.0 ? Rgb{210, 60, 60} : Rgb{40, 160, 90};
        const int x0 = cx(p.sx), y0 = cy(p.sy);
        // Magnified error vector from the true position to the fitted position.
        const int x1 = cx(p.sx + (fx - p.sx) * mag);
        const int y1 = cy(p.sy + (fy - p.sy) * mag);
        draw_cross(out, x0, y0, 4, {40, 40, 40});
        // Simple Bresenham line for the error vector.
        int dx = std::abs(x1 - x0), dy = -std::abs(y1 - y0);
        int stepx = x0 < x1 ? 1 : -1, stepy = y0 < y1 ? 1 : -1, e = dx + dy;
        int xx = x0, yy = y0;
        for (int guard = 0; guard < 4000; ++guard) {
            out.set(xx, yy, col);
            if (xx == x1 && yy == y1) break;
            const int e2 = 2 * e;
            if (e2 >= dy) { e += dy; xx += stepx; }
            if (e2 <= dx) { e += dx; yy += stepy; }
        }
    }
    return out;
}

}  // namespace wproc::vcal
