// aa.cpp — active alignment via Nelder-Mead simplex.
#include "optomech/aa.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace optm::aa {
namespace {

using Vec = std::array<double, 3>;  // (decenter_x, decenter_y, focus) adjustments

}  // namespace

Result align(const LensSystem& sys, const Assembly& err, int max_iter) {
    Result r;
    const double S_dec = std::fabs(lateral_sensitivity_decenter(sys, err.element));
    const double S_tilt = std::fabs(lateral_sensitivity_tilt(sys, err.element));
    const double wf = 0.6;  // relative weight of defocus in the spot merit

    // Spot-size merit [µm] given the compensating move (ax, ay, af). The applied
    // decenter cancels the assembly decenter; the tilt term is irreducible with a
    // 3-DOF move and sets the achievable floor.
    auto merit = [&](const Vec& v) {
        const double lat_x = S_dec * (err.decenter_x_um + v[0]) + S_tilt * err.tilt_mrad;
        const double lat_y = S_dec * (err.decenter_y_um + v[1]);
        const double foc = err.focus_um + v[2];
        return std::sqrt(lat_x * lat_x + lat_y * lat_y + wf * wf * foc * foc);
    };

    r.merit_before_um = merit({0.0, 0.0, 0.0});

    // Nelder-Mead simplex over 3 parameters.
    std::array<Vec, 4> s = {Vec{0, 0, 0}, Vec{20, 0, 0}, Vec{0, 20, 0}, Vec{0, 0, 20}};
    std::array<double, 4> fv;
    for (int i = 0; i < 4; ++i) fv[i] = merit(s[i]);

    const double alpha = 1.0, gamma = 2.0, rho = 0.5, sigma = 0.5;
    int it = 0;
    for (; it < max_iter; ++it) {
        std::array<int, 4> idx = {0, 1, 2, 3};
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return fv[a] < fv[b]; });
        const int lo = idx[0], hi = idx[3], hi2 = idx[2];
        r.merit_history.push_back(fv[lo]);
        if (fv[hi] - fv[lo] < 1e-6) break;  // converged

        // Centroid of all but the worst.
        Vec cen = {0, 0, 0};
        for (int i = 0; i < 4; ++i)
            if (i != hi)
                for (int k = 0; k < 3; ++k) cen[k] += s[i][k] / 3.0;

        auto reflect = [&](double t) {
            Vec p;
            for (int k = 0; k < 3; ++k) p[k] = cen[k] + t * (cen[k] - s[hi][k]);
            return p;
        };
        Vec xr = reflect(alpha);
        double fr = merit(xr);
        if (fr < fv[lo]) {
            Vec xe = reflect(gamma);
            const double fe = merit(xe);
            if (fe < fr) { s[hi] = xe; fv[hi] = fe; } else { s[hi] = xr; fv[hi] = fr; }
        } else if (fr < fv[hi2]) {
            s[hi] = xr; fv[hi] = fr;
        } else {
            Vec xc;
            for (int k = 0; k < 3; ++k) xc[k] = cen[k] + rho * (s[hi][k] - cen[k]);
            const double fc = merit(xc);
            if (fc < fv[hi]) { s[hi] = xc; fv[hi] = fc; }
            else {
                for (int i = 0; i < 4; ++i)
                    if (i != lo) {
                        for (int k = 0; k < 3; ++k) s[i][k] = s[lo][k] + sigma * (s[i][k] - s[lo][k]);
                        fv[i] = merit(s[i]);
                    }
            }
        }
    }

    int best = 0;
    for (int i = 1; i < 4; ++i) if (fv[i] < fv[best]) best = i;
    r.merit_after_um = fv[best];
    r.adj_x_um = s[best][0];
    r.adj_y_um = s[best][1];
    r.adj_focus_um = s[best][2];
    r.iterations = it;
    r.converged = r.merit_after_um < r.merit_before_um;
    return r;
}

}  // namespace optm::aa
