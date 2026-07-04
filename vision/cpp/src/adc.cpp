// adc.cpp — automatic defect classification pipeline (CNN + wafer map + yield).
#include "wproc/adc.hpp"

#include <algorithm>

namespace wproc::adc {

Result run(const cnn::DefectNet& net, const std::vector<Observation>& obs,
           const Params& params) {
    Result r;
    r.n_defects = static_cast<int>(obs.size());
    r.defect_class.reserve(obs.size());

    // Per-die killer flag (value fed to the wafer-map classifier).
    std::vector<double> die_value(static_cast<std::size_t>(params.cols) * params.rows, 0.0);

    for (const Observation& o : obs) {
        const int cls = net.classify(o.patch).cls;
        r.defect_class.push_back(cls);
        if (cls >= 0 && cls < 4) ++r.class_counts[cls];
        if (o.col >= 0 && o.col < params.cols && o.row >= 0 && o.row < params.rows) {
            if (cls >= 0 && cls < 4 && params.killer[cls])
                die_value[static_cast<std::size_t>(o.row) * params.cols + o.col] = 1.0;
        }
    }

    // Dominant defect class among 1..3 (0 = none).
    int top = 0, best = 0;
    for (int c = 1; c < 4; ++c)
        if (r.class_counts[c] > best) { best = r.class_counts[c]; top = c; }
    r.top_defect_class = top;

    // A die passes if it carries no killer defect (value <= 0.5).
    r.wafer = wafer::classify(params.cols, params.rows, die_value, /*limit=*/0.5,
                              params.center_zone_frac);
    r.yield_pct = r.wafer.yield_pct;
    return r;
}

}  // namespace wproc::adc
