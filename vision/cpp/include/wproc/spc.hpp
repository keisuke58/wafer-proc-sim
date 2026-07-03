// spc.hpp — statistical process control on a stream of measurements.
//
// Implements the classic Western Electric run rules on standardized values
// (z = (x - mean)/sigma). Control limits are set from a baseline sample (or
// supplied directly). Used to flag when a dicing process drifts out of control
// — e.g. kerf width creeping up as the blade wears.
#pragma once

#include <deque>
#include <string>
#include <vector>

namespace wproc::spc {

enum class Rule {
    Beyond3Sigma,   // WE-1: one point outside 3-sigma
    TwoOfThree2Sigma,  // WE-2: 2 of 3 consecutive beyond 2-sigma, same side
    FourOfFive1Sigma,  // WE-3: 4 of 5 consecutive beyond 1-sigma, same side
    EightOneSide       // WE-4: 8 consecutive on one side of the mean
};

const char* rule_name(Rule r);

struct Limits {
    double mean = 0.0;
    double sigma = 1.0;
};

// Mean and (sample) standard deviation of a baseline in-control sample.
Limits limits_from_baseline(const std::vector<double>& baseline);

struct Point {
    double value = 0.0;
    double z = 0.0;                 // standardized deviation from mean
    bool in_control = true;         // false if any rule fired
    std::vector<Rule> violations;   // rules that fired at this point
};

// Streaming control chart. Feed one measurement at a time; each returns the
// evaluated point including any run-rule violations at that sample.
class Chart {
public:
    explicit Chart(Limits lim) : lim_(lim) {
        // Guard every construction path (not just baseline-derived limits):
        // a non-positive sigma would make z = NaN/Inf and silently pass rules.
        if (!(lim_.sigma > 0.0)) lim_.sigma = 1.0;
    }
    const Limits& limits() const { return lim_; }

    Point add(double value);

private:
    Limits lim_;
    std::deque<double> zhist_;  // recent z-scores (bounded window)
};

}  // namespace wproc::spc
