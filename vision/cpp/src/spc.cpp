#include "wproc/spc.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace wproc::spc {

const char* rule_name(Rule r) {
    switch (r) {
        case Rule::Beyond3Sigma:    return "WE1: point beyond 3-sigma";
        case Rule::TwoOfThree2Sigma:return "WE2: 2 of 3 beyond 2-sigma (one side)";
        case Rule::FourOfFive1Sigma:return "WE3: 4 of 5 beyond 1-sigma (one side)";
        case Rule::EightOneSide:    return "WE4: 8 consecutive on one side";
    }
    return "unknown";
}

Limits limits_from_baseline(const std::vector<double>& b) {
    Limits lim;
    if (b.empty()) return lim;
    lim.mean = std::accumulate(b.begin(), b.end(), 0.0) / b.size();
    double ss = std::accumulate(
        b.begin(), b.end(), 0.0,
        [m = lim.mean](double a, double x) { return a + (x - m) * (x - m); });
    lim.sigma = b.size() > 1 ? std::sqrt(ss / (b.size() - 1)) : 1.0;
    if (lim.sigma <= 0.0) lim.sigma = 1.0;
    return lim;
}

Point Chart::add(double value) {
    Point p;
    p.value = value;
    p.z = (value - lim_.mean) / lim_.sigma;

    zhist_.push_back(p.z);
    if (zhist_.size() > 8) zhist_.pop_front();

    const std::size_t n = zhist_.size();
    auto recent = [&](std::size_t k) {  // last k z-scores as a small view
        return std::vector<double>(zhist_.end() - static_cast<long>(k),
                                   zhist_.end());
    };
    auto count_side = [](const std::vector<double>& v, double thr, bool upper) {
        return static_cast<int>(std::count_if(
            v.begin(), v.end(),
            [thr, upper](double z) { return upper ? (z > thr) : (z < -thr); }));
    };

    // WE1: current point beyond 3-sigma.
    if (std::abs(p.z) > 3.0) p.violations.push_back(Rule::Beyond3Sigma);

    // WE2: 2 of the last 3 beyond 2-sigma on the same side.
    if (n >= 3) {
        auto v = recent(3);
        if (count_side(v, 2.0, true) >= 2 || count_side(v, 2.0, false) >= 2)
            p.violations.push_back(Rule::TwoOfThree2Sigma);
    }
    // WE3: 4 of the last 5 beyond 1-sigma on the same side.
    if (n >= 5) {
        auto v = recent(5);
        if (count_side(v, 1.0, true) >= 4 || count_side(v, 1.0, false) >= 4)
            p.violations.push_back(Rule::FourOfFive1Sigma);
    }
    // WE4: 8 consecutive on one side of the mean.
    if (n >= 8) {
        auto v = recent(8);
        bool all_up = true, all_dn = true;
        for (double z : v) {
            all_up = all_up && (z > 0.0);
            all_dn = all_dn && (z < 0.0);
        }
        if (all_up || all_dn) p.violations.push_back(Rule::EightOneSide);
    }

    p.in_control = p.violations.empty();
    return p;
}

}  // namespace wproc::spc
