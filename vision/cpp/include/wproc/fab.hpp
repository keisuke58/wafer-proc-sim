// fab.hpp — digital twin of a wafer-processing line: throughput, bottleneck, OEE.
//
// Ties the point tools into a system view. A serial line of stations
// (dice → grind → polish → inspect) each has an actual and ideal cycle time,
// an availability (uptime), and a quality (per-station yield). This computes:
//
//   * the bottleneck (largest downtime-adjusted cycle) and line throughput,
//   * the compounded line yield and good-wafer rate,
//   * OEE = Availability × Performance × Quality, per station and for the line
//     (good output ÷ theoretical maximum output),
//   * per-station utilization, to show how loaded each stage runs.
//
// Pure C++17, header + one TU.
#pragma once

#include <string>
#include <vector>

namespace wproc::fab {

struct Station {
    std::string name;
    double cycle_s = 30.0;        // actual cycle time [s/wafer]
    double ideal_cycle_s = 0.0;   // design cycle (<=cycle_s); 0 => same as cycle_s
    double availability = 1.0;    // uptime fraction, (0,1]
    double quality = 1.0;         // per-station yield, (0,1]
};

struct StationOEE {
    std::string name;
    double availability = 0.0;
    double performance = 0.0;     // ideal_cycle / cycle
    double quality = 0.0;
    double oee = 0.0;             // A * P * Q
    double eff_cycle_s = 0.0;     // cycle / availability (downtime-adjusted)
    double utilization = 0.0;     // eff_cycle / bottleneck eff_cycle
    bool bottleneck = false;
};

struct LineResult {
    int bottleneck = -1;               // index of the bottleneck station
    double bottleneck_cycle_s = 0.0;   // its downtime-adjusted cycle
    double throughput_wph = 0.0;       // wafers/hr the line sustains
    double line_yield = 0.0;           // product of station qualities
    double good_wph = 0.0;             // throughput * line_yield
    double line_oee = 0.0;             // good output / theoretical max output
    std::vector<StationOEE> stations;
};

// Analyze a serial line. Throws std::invalid_argument on empty input or a
// station with a non-positive cycle time / out-of-range availability.
LineResult analyze_line(const std::vector<Station>& stations);

}  // namespace wproc::fab
