#include "wproc/blade_health.hpp"

#include <algorithm>
#include <cmath>

namespace wproc::blade {

namespace {

// Ordinary least-squares slope/intercept of y vs x. Returns false if x has no
// spread (slope undefined).
bool linfit(const std::vector<double>& x, const std::vector<double>& y,
            double& slope, double& intercept) {
    const std::size_t n = x.size();
    if (n < 2) return false;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (std::size_t i = 0; i < n; ++i) {
        sx += x[i]; sy += y[i];
        sxx += x[i] * x[i]; sxy += x[i] * y[i];
    }
    double denom = n * sxx - sx * sx;
    if (std::abs(denom) < 1e-12) return false;
    slope = (n * sxy - sx * sy) / denom;
    intercept = (sy - slope * sx) / n;
    return true;
}

// Cuts until a rising trend (slope>0) carries the current fitted value to
// `limit`. Large (effectively infinite) if not trending toward the limit.
double cuts_to_limit(double last_fit, double slope, double limit) {
    if (last_fit >= limit) return 0.0;          // already over
    if (slope <= 1e-9) return 1e9;              // not degrading
    return (limit - last_fit) / slope;
}

}  // namespace

Sample Monitor::add(long cut, double width_um, double chip_um) {
    cuts_.push_back(static_cast<double>(cut));
    widths_.push_back(width_um);
    chips_.push_back(chip_um);

    spc::Point p = chart_.add(width_um);
    Sample s;
    s.cut = cut;
    s.width_um = width_um;
    s.chip_um = chip_um;
    s.z = p.z;
    s.in_control = p.in_control;
    s.violations = p.violations;
    return s;
}

Health Monitor::health() const {
    Health h;
    h.n = static_cast<int>(cuts_.size());
    if (h.n == 0) return h;

    double last_x = cuts_.back();
    double ws = 0, wi = 0, cs = 0, ci = 0;
    double width_fit = widths_.back(), chip_fit = chips_.back();
    if (linfit(cuts_, widths_, ws, wi)) {
        h.width_slope_um_per_cut = ws;
        width_fit = ws * last_x + wi;
    }
    if (linfit(cuts_, chips_, cs, ci)) {
        h.chip_slope_um_per_cut = cs;
        chip_fit = cs * last_x + ci;
    }

    // Wear index: fraction of the way from nominal to the binding limit,
    // taken as the worse of the width and chipping margins.
    double width_frac =
        (width_fit - spec_.nominal_width_um) /
        std::max(1e-9, spec_.width_usl_um - spec_.nominal_width_um);
    double chip_frac = chip_fit / std::max(1e-9, spec_.max_chip_um);
    h.wear_index = std::max({0.0, width_frac, chip_frac});

    double rul_w = cuts_to_limit(width_fit, h.width_slope_um_per_cut,
                                 spec_.width_usl_um);
    double rul_c = cuts_to_limit(chip_fit, h.chip_slope_um_per_cut,
                                 spec_.max_chip_um);
    h.rul_cuts = std::min(rul_w, rul_c);
    h.replace_now =
        width_fit >= spec_.width_usl_um || chip_fit >= spec_.max_chip_um;
    return h;
}

}  // namespace wproc::blade
