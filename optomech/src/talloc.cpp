// talloc.cpp — cost-optimal tolerance allocation (Lagrange closed form).
#include "optomech/talloc.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace optm::talloc {

Result allocate(const LensSystem& sys, double budget_um, const CostModel& cost) {
    Result r;
    r.budget_um = budget_um;
    if (sys.elements.empty() || budget_um <= 0.0) return r;

    // Collect the error sources with their image sensitivity and cost coeff.
    struct Src { std::size_t elem; std::string kind; double S; double a; };
    std::vector<Src> src;
    for (std::size_t i = 0; i < sys.elements.size(); ++i) {
        const double sd = std::fabs(lateral_sensitivity_decenter(sys, i));
        const double st = std::fabs(lateral_sensitivity_tilt(sys, i));
        if (sd > 1e-9) src.push_back({i, "decenter", sd, cost.decenter_coeff});
        if (st > 1e-9) src.push_back({i, "tilt", st, cost.tilt_coeff});
    }
    if (src.empty()) return r;

    // Optimal sigma_i = c_i * lambda^(-1/3), c_i = (a_i / (2 S_i^2))^(1/3).
    // The 2-D radial budget is 2 * sum (S_i sigma_i)^2 = B^2.
    double denom = 0.0;
    std::vector<double> c(src.size());
    for (std::size_t k = 0; k < src.size(); ++k) {
        c[k] = std::cbrt(src[k].a / (2.0 * src[k].S * src[k].S));
        denom += (src[k].S * c[k]) * (src[k].S * c[k]);
    }
    const double lambda_inv_third = budget_um / std::sqrt(2.0 * denom);

    // Equal-tolerance baseline meeting the same budget: sigma0 for all sources.
    const double sumS2 = std::accumulate(src.begin(), src.end(), 0.0,
        [](double acc, const Src& s) { return acc + s.S * s.S; });
    const double sigma0 = budget_um / std::sqrt(2.0 * sumS2);

    for (std::size_t k = 0; k < src.size(); ++k) {
        Alloc a;
        a.element = src[k].elem;
        a.kind = src[k].kind;
        a.sensitivity = src[k].S;
        a.sigma = c[k] * lambda_inv_third;
        a.cost = src[k].a / a.sigma;
        r.opt_cost += a.cost;
        r.equal_cost += src[k].a / sigma0;
        r.allocation.push_back(a);
    }
    r.savings_pct = r.equal_cost > 0.0
                        ? 100.0 * (r.equal_cost - r.opt_cost) / r.equal_cost : 0.0;
    std::sort(r.allocation.begin(), r.allocation.end(),
              [](const Alloc& x, const Alloc& y) { return x.sigma < y.sigma; });
    r.ok = true;
    return r;
}

}  // namespace optm::talloc
