// tolerance.cpp — Monte Carlo + RSS tolerance stack-up.
#include "optomech/tolerance.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace optm::tol {

Result analyze(const LensSystem& sys, const std::vector<Tolerance>& tols,
               double budget_um, int n_samples, unsigned seed) {
    Result r;
    r.budget_um = budget_um;
    if (sys.elements.empty() || tols.size() != sys.elements.size()) return r;
    r.efl_mm = first_order(sys).efl_mm;

    // First-order sensitivities per element (image µm per unit error).
    const std::size_t N = sys.elements.size();
    std::vector<double> s_dec(N), s_tilt(N), s_focus(N);
    const double eps_um = 1.0;  // finite-difference step for focus (µm)
    for (std::size_t i = 0; i < N; ++i) {
        // decenter: image µm per µm decenter (dimensionless sensitivity).
        s_dec[i] = lateral_sensitivity_decenter(sys, i);
        // tilt: image mm per rad -> image µm per mrad = (mm/rad)*1e3 µm/mm *1e-3 rad/mrad.
        s_tilt[i] = lateral_sensitivity_tilt(sys, i);  // mm per rad == µm per mrad
        // focus: image µm defocus per µm spacing error.
        const double df_mm = focus_shift_from_spacing(sys, i, eps_um * 1e-3);
        s_focus[i] = df_mm / (eps_um * 1e-3);  // mm/mm, dimensionless
    }

    // Analytic RSS budgets and per-source contributions.
    double lat_var = 0.0, foc_var = 0.0;
    std::vector<Contribution> contribs;
    for (std::size_t i = 0; i < N; ++i) {
        const double c_dec = std::fabs(s_dec[i]) * tols[i].decenter_um;      // µm
        const double c_tilt = std::fabs(s_tilt[i]) * tols[i].tilt_mrad;      // µm
        const double c_foc = std::fabs(s_focus[i]) * tols[i].spacing_um;     // µm
        lat_var += c_dec * c_dec + c_tilt * c_tilt;
        foc_var += c_foc * c_foc;
        contribs.push_back({i, "decenter", c_dec, 0.0});
        contribs.push_back({i, "tilt", c_tilt, 0.0});
    }
    // lat_var is per-axis; the image error is the 2-D radial magnitude
    // (independent x/y), whose 1-sigma is sqrt(2) larger.
    r.lateral_rss_um = std::sqrt(2.0 * lat_var);
    r.focus_rss_um = std::sqrt(foc_var);
    for (auto& c : contribs)
        c.variance_pct = lat_var > 0 ? 100.0 * c.sigma_um * c.sigma_um / lat_var : 0.0;
    std::sort(contribs.begin(), contribs.end(),
              [](const Contribution& a, const Contribution& b) {
                  return a.sigma_um > b.sigma_um;
              });
    r.ranking = std::move(contribs);

    // Monte Carlo: sample all sources together for the true distribution.
    std::mt19937 gen(seed);
    std::normal_distribution<double> z(0.0, 1.0);
    r.mc_lateral_um.reserve(static_cast<std::size_t>(n_samples));
    r.mc_focus_um.reserve(static_cast<std::size_t>(n_samples));
    double lat_ss = 0.0, foc_ss = 0.0;
    for (int k = 0; k < n_samples; ++k) {
        double lx = 0.0, ly = 0.0, fz = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            const double dec = tols[i].decenter_um * z(gen);   // µm, one axis
            const double dec_y = tols[i].decenter_um * z(gen);
            const double tlt = tols[i].tilt_mrad * z(gen);     // mrad
            const double tlt_y = tols[i].tilt_mrad * z(gen);
            const double sp = tols[i].spacing_um * z(gen);     // µm
            lx += s_dec[i] * dec + s_tilt[i] * tlt;
            ly += s_dec[i] * dec_y + s_tilt[i] * tlt_y;
            fz += s_focus[i] * sp;
        }
        const double lat = std::sqrt(lx * lx + ly * ly);
        r.mc_lateral_um.push_back(lat);
        r.mc_focus_um.push_back(fz);
        lat_ss += lat * lat;
        foc_ss += fz * fz;
    }
    r.lateral_mc_rms_um = std::sqrt(lat_ss / n_samples);
    r.focus_mc_rms_um = std::sqrt(foc_ss / n_samples);

    std::vector<double> sorted = r.mc_lateral_um;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t p99 = static_cast<std::size_t>(0.99 * (sorted.size() - 1));
    r.lateral_mc_p99_um = sorted[p99];
    r.ok = r.lateral_mc_p99_um <= budget_um;
    return r;
}

}  // namespace optm::tol
