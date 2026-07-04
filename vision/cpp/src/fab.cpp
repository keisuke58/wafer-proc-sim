// fab.cpp — digital-twin line analysis: throughput, bottleneck, OEE.
#include "wproc/fab.hpp"

#include <algorithm>
#include <stdexcept>

namespace wproc::fab {

LineResult analyze_line(const std::vector<Station>& stations) {
    if (stations.empty())
        throw std::invalid_argument("analyze_line: no stations");

    LineResult r;
    r.line_yield = 1.0;
    double max_eff = 0.0;
    double max_ideal = 0.0;

    // First pass: effective cycles, bottleneck, compounded yield.
    for (const Station& s : stations) {
        if (!(s.cycle_s > 0.0))
            throw std::invalid_argument("analyze_line: cycle_s must be > 0");
        if (!(s.availability > 0.0 && s.availability <= 1.0))
            throw std::invalid_argument("analyze_line: availability in (0,1]");
        if (!(s.quality > 0.0 && s.quality <= 1.0))
            throw std::invalid_argument("analyze_line: quality in (0,1]");
        // ideal_cycle_s above cycle_s just saturates performance at 1.0 (handled
        // by the std::min below), so it needs no separate guard.
        const double ideal = s.ideal_cycle_s > 0.0 ? s.ideal_cycle_s : s.cycle_s;
        const double eff = s.cycle_s / s.availability;
        if (eff > max_eff) { max_eff = eff; r.bottleneck = static_cast<int>(r.stations.size()); }
        max_ideal = std::max(max_ideal, ideal);
        r.line_yield *= s.quality;

        StationOEE o;
        o.name = s.name;
        o.availability = s.availability;
        o.performance = std::min(1.0, ideal / s.cycle_s);
        o.quality = s.quality;
        o.oee = o.availability * o.performance * o.quality;
        o.eff_cycle_s = eff;
        r.stations.push_back(o);
    }

    r.bottleneck_cycle_s = max_eff;
    r.throughput_wph = max_eff > 0.0 ? 3600.0 / max_eff : 0.0;
    r.good_wph = r.throughput_wph * r.line_yield;
    const double theoretical_wph = max_ideal > 0.0 ? 3600.0 / max_ideal : 0.0;
    r.line_oee = theoretical_wph > 0.0
                     ? std::clamp(r.good_wph / theoretical_wph, 0.0, 1.0)
                     : 0.0;

    // Second pass: utilization relative to the bottleneck, flag the bottleneck.
    for (std::size_t i = 0; i < r.stations.size(); ++i) {
        r.stations[i].utilization =
            max_eff > 0.0 ? r.stations[i].eff_cycle_s / max_eff : 0.0;
        r.stations[i].bottleneck = (static_cast<int>(i) == r.bottleneck);
    }
    return r;
}

}  // namespace wproc::fab
