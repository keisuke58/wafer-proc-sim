// blade_health.hpp — predictive blade-wear monitoring from kerf-inspection
// history. Feeds each cut's measured kerf width and max chipping through an SPC
// chart and a running least-squares trend, then predicts remaining useful life
// (cuts until width or chipping crosses its spec limit) — a digital-twin signal
// for when to dress or replace the blade.
#pragma once

#include <vector>

#include "wproc/spc.hpp"

namespace wproc::blade {

struct Spec {
    double nominal_width_um = 40.0;
    double width_usl_um = 45.0;     // upper spec limit for kerf width
    double max_chip_um = 5.0;       // chipping limit
};

struct Sample {
    long cut = 0;         // cut index (monotonic)
    double width_um = 0.0;
    double chip_um = 0.0;
    double z = 0.0;                       // SPC z-score of width
    bool in_control = true;               // SPC verdict at this cut
    std::vector<spc::Rule> violations;
};

struct Health {
    int n = 0;
    double width_slope_um_per_cut = 0.0;  // fitted drift
    double chip_slope_um_per_cut = 0.0;
    double wear_index = 0.0;              // 0 (fresh) .. >=1 (at/over limit)
    double rul_cuts = 0.0;               // cuts until first limit crossing
    bool replace_now = false;            // any limit already exceeded
};

// Online monitor: establish control limits from an initial baseline of widths,
// then add cuts one at a time.
class Monitor {
public:
    Monitor(Spec spec, const std::vector<double>& baseline_widths)
        : spec_(spec), chart_(spc::limits_from_baseline(baseline_widths)) {}

    Sample add(long cut, double width_um, double chip_um);

    // Current wear/RUL estimate from the accumulated history.
    Health health() const;

    const spc::Limits& control_limits() const { return chart_.limits(); }

private:
    Spec spec_;
    spc::Chart chart_;
    std::vector<double> cuts_, widths_, chips_;
};

}  // namespace wproc::blade
