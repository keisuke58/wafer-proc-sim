// laser_kabra.cpp — laser SiC-ingot slicing economics + separation-layer QC.
#include "wproc/laser_kabra.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace wproc::laser {

SliceYield slice_yield(const SliceInputs& in) {
    SliceYield y;
    y.pitch_um = in.wafer_thickness_um + in.loss_per_slice_um;
    const double length_um = in.ingot_length_mm * 1000.0;
    if (y.pitch_um <= 0.0 || length_um <= 0.0) return y;

    y.wafers = static_cast<int>(std::floor(length_um / y.pitch_um));
    y.used_um = y.wafers * in.wafer_thickness_um;
    y.waste_um = std::max(0.0, length_um - y.used_um);
    y.utilization_pct = std::clamp(100.0 * y.used_um / length_um, 0.0, 100.0);
    return y;
}

SliceComparison compare_slice(double ingot_length_mm, double wafer_thickness_um,
                              double laser_loss_um, double saw_kerf_um) {
    SliceComparison c;
    SliceInputs li{ingot_length_mm, wafer_thickness_um, laser_loss_um};
    SliceInputs si{ingot_length_mm, wafer_thickness_um, saw_kerf_um};
    c.laser = slice_yield(li);
    c.saw = slice_yield(si);
    c.extra_wafers = c.laser.wafers - c.saw.wafers;
    c.extra_wafers_pct =
        c.saw.wafers > 0 ? 100.0 * c.extra_wafers / c.saw.wafers : 0.0;
    c.material_saved_pct = c.laser.utilization_pct - c.saw.utilization_pct;
    return c;
}

SliceEcon slice_economics(const SliceComparison& cmp, double ingot_cost,
                          double proc_cost_per_wafer_laser,
                          double proc_cost_per_wafer_saw) {
    SliceEcon e;
    if (cmp.laser.wafers > 0)
        e.cost_per_wafer_laser =
            ingot_cost / cmp.laser.wafers + proc_cost_per_wafer_laser;
    if (cmp.saw.wafers > 0)
        e.cost_per_wafer_saw =
            ingot_cost / cmp.saw.wafers + proc_cost_per_wafer_saw;
    e.savings_per_wafer = e.cost_per_wafer_saw - e.cost_per_wafer_laser;
    e.savings_pct = e.cost_per_wafer_saw > 0
                        ? 100.0 * e.savings_per_wafer / e.cost_per_wafer_saw
                        : 0.0;
    return e;
}

SepLayerStats analyze_separation(const std::vector<double>& depths_um,
                                 double target_depth_um, double tol_um) {
    SepLayerStats s;
    s.n = static_cast<int>(depths_um.size());
    if (s.n == 0) return s;

    const double sum = std::accumulate(depths_um.begin(), depths_um.end(), 0.0);
    s.mean_depth_um = sum / s.n;
    const double sumsq = std::inner_product(depths_um.begin(), depths_um.end(),
                                            depths_um.begin(), 0.0);
    const double var = std::max(0.0, sumsq / s.n - s.mean_depth_um * s.mean_depth_um);
    s.sigma_um = std::sqrt(var);

    // Worst deviation from the target depth.
    s.max_dev_um = std::accumulate(
        depths_um.begin(), depths_um.end(), 0.0,
        [target_depth_um](double a, double d) {
            return std::max(a, std::fabs(d - target_depth_um));
        });

    s.uniformity_pct =
        s.mean_depth_um > 0
            ? std::clamp(100.0 * (1.0 - s.sigma_um / s.mean_depth_um), 0.0, 100.0)
            : 0.0;
    // Clean cleave: every sample within tolerance of target.
    s.ok = s.max_dev_um <= tol_um;
    return s;
}

}  // namespace wproc::laser
