// reliability_kernel.cpp — Reliability / RASIS analysis for dicing equipment
//
// AP試験 concepts demonstrated (信頼性・RASIS):
//   - MTBF / MTTR / 稼働率 (Availability = MTBF / (MTBF + MTTR))
//   - 直列システム: λ_sys = Σλi  →  MTBF_sys = 1 / Σλi
//   - 並列システム: R_sys(t) = 1 − Π(1−Ri(t))  (fault-tolerant redundancy)
//   - Weibull 分布: R(t)=exp(−(t/η)^β), h(t)=(β/η)·(t/η)^(β−1)
//       β < 1 → 初期故障 (infant mortality)
//       β = 1 → 偶発故障 (exponential = constant rate)
//       β > 1 → 摩耗故障 (wear-out, e.g. blade fatigue)
//   - OEE = Availability × Performance × Quality  (総合設備効率)
//   - モンテカルロ故障シミュレーション (verify analytic MTBF)
//
// Application: DISCO DFL7160 blade service life planning.
//   Blade: β≈2 (wear-out), η≈72 h → predict replacement interval.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

// ── Gamma function (Lanczos approximation) ───────────────────────────────────
// Used for Weibull MTTF = η · Γ(1 + 1/β).
static double gamma_func(double z) {
    static const double g = 7.0;
    static const double c[] = {
        0.99999999999980993,
        676.5203681218851,
       -1259.1392167224028,
        771.32342877765313,
       -176.61502916214059,
        12.507343278686905,
       -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    };
    if (z < 0.5)
        return M_PI / (std::sin(M_PI * z) * gamma_func(1.0 - z));
    z -= 1.0;
    double x = c[0];
    for (int i = 1; i < 9; ++i) x += c[i] / (z + i);
    double t = z + g + 0.5;
    return std::sqrt(2.0 * M_PI) * std::pow(t, z + 0.5) * std::exp(-t) * x;
}

// ── Series system ─────────────────────────────────────────────────────────────
// R_sys(t) = Π Ri(t).  For constant-rate (exponential): λ_sys = Σλi.
py::dict series_system(const std::vector<double>& mtbf_hours) {
    if (mtbf_hours.empty()) throw std::invalid_argument("empty mtbf list");
    double lambda_sum = 0.0;
    for (double m : mtbf_hours) {
        if (m <= 0.0) throw std::invalid_argument("MTBF must be > 0");
        lambda_sum += 1.0 / m;
    }
    double mtbf_sys = 1.0 / lambda_sum;
    py::dict r;
    r["mtbf_hours"]         = mtbf_sys;
    r["failure_rate"]       = lambda_sum;
    r["n_components"]       = static_cast<int>(mtbf_hours.size());
    r["weakest_component"]  = *std::min_element(mtbf_hours.begin(), mtbf_hours.end());
    return r;
}

// ── Parallel (redundant) system ───────────────────────────────────────────────
// MTTF_sys for n identical units (hot standby, exponential):
//   MTTF = (1/λ) · Σ_{k=1}^{n} 1/k
py::dict parallel_system(const std::vector<double>& mtbf_hours) {
    if (mtbf_hours.empty()) throw std::invalid_argument("empty mtbf list");
    int n = static_cast<int>(mtbf_hours.size());
    double avg_mtbf = 0.0;
    for (double m : mtbf_hours) avg_mtbf += m;
    avg_mtbf /= n;
    // Use average MTBF assumption (identical units).
    double harmonic = 0.0;
    for (int k = 1; k <= n; ++k) harmonic += 1.0 / k;
    double mttf_sys = avg_mtbf * harmonic;
    py::dict r;
    r["mttf_hours"]   = mttf_sys;
    r["n_components"] = n;
    r["avg_unit_mtbf"]= avg_mtbf;
    r["improvement"]  = mttf_sys / avg_mtbf;  // improvement factor vs single unit
    return r;
}

// ── Availability & OEE ────────────────────────────────────────────────────────
py::dict availability_oee(
    double mtbf_hours,
    double mttr_hours,
    double performance_rate,  // 0..1 (actual speed / rated speed)
    double quality_rate       // 0..1 (good parts / total parts)
) {
    if (mtbf_hours <= 0.0 || mttr_hours < 0.0)
        throw std::invalid_argument("MTBF > 0 and MTTR >= 0 required");
    double avail = mtbf_hours / (mtbf_hours + mttr_hours);
    double oee   = avail * performance_rate * quality_rate;
    double uptime_frac  = mtbf_hours / (mtbf_hours + mttr_hours);
    py::dict r;
    r["availability"]      = avail;
    r["performance_rate"]  = performance_rate;
    r["quality_rate"]      = quality_rate;
    r["oee"]               = oee;
    r["uptime_fraction"]   = uptime_frac;
    r["downtime_fraction"] = 1.0 - uptime_frac;
    // RASIS labels
    r["reliability_label"] = std::string(mtbf_hours > 1000 ? "High" : mtbf_hours > 100 ? "Medium" : "Low");
    r["oee_label"]         = std::string(oee >= 0.85 ? "World-class" : oee >= 0.65 ? "Typical" : "Low");
    return r;
}

// ── Weibull distribution ──────────────────────────────────────────────────────
py::dict weibull_analysis(
    const std::vector<double>& t_hours,  // time points to evaluate
    double beta,                          // shape: <1 infant, =1 random, >1 wear
    double eta                            // scale (characteristic life) [hours]
) {
    if (beta <= 0.0 || eta <= 0.0)
        throw std::invalid_argument("beta > 0 and eta > 0 required");
    if (t_hours.empty()) throw std::invalid_argument("t_hours is empty");

    double mttf = eta * gamma_func(1.0 + 1.0 / beta);

    py::list r_list, h_list, f_list;
    for (double t : t_hours) {
        if (t < 0.0) { r_list.append(1.0); h_list.append(0.0); f_list.append(0.0); continue; }
        double tb = std::pow(t / eta, beta);
        double R  = std::exp(-tb);                          // reliability
        double h  = (beta / eta) * std::pow(t / eta, beta - 1.0);  // hazard rate
        double F  = 1.0 - R;                               // CDF (probability of failure)
        r_list.append(R);
        h_list.append(h);
        f_list.append(F);
    }

    std::string phase = (beta < 0.99) ? "infant_mortality"
                      : (beta < 1.01) ? "random_failure"
                                      : "wear_out";

    py::dict r;
    r["reliability"]    = r_list;
    r["hazard_rate"]    = h_list;
    r["cdf"]            = f_list;
    r["mttf_hours"]     = mttf;
    r["beta"]           = beta;
    r["eta"]            = eta;
    r["failure_phase"]  = phase;
    return r;
}

// ── Blade life simulation (Monte Carlo) ──────────────────────────────────────
// Simulate n_blades blade lives from Weibull(β, η).  Verify analytic MTTF.
// Uses simple Box-Muller / inverse-CDF method for Weibull random variates:
//   T = η · (−ln U)^(1/β),  U ~ Uniform(0,1)
//
// NOTE: uses a deterministic LCG (seed-based) so results are reproducible
//       without std::mt19937 (which requires <random> and C++11 headers that
//       pybind11 environment may not expose cleanly in all build configs).
py::dict simulate_blade_life(
    int    n_blades,
    double beta,
    double eta,
    double mttr_hours,
    uint32_t seed
) {
    if (n_blades <= 0) throw std::invalid_argument("n_blades > 0 required");
    if (beta <= 0.0 || eta <= 0.0) throw std::invalid_argument("beta,eta > 0 required");

    // LCG parameters (Numerical Recipes)
    uint64_t state = static_cast<uint64_t>(seed) ^ 0xdeadbeefULL;
    auto lcg_uniform = [&]() -> double {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>((state >> 33) & 0x7FFFFFFFULL) / static_cast<double>(0x7FFFFFFFULL);
    };

    double life_sum = 0.0, life_sq = 0.0;
    double life_min = 1e18, life_max = 0.0;
    double total_down = 0.0;

    for (int i = 0; i < n_blades; ++i) {
        double u = std::max(1e-12, lcg_uniform());
        double life = eta * std::pow(-std::log(u), 1.0 / beta);
        life_sum += life;
        life_sq  += life * life;
        if (life < life_min) life_min = life;
        if (life > life_max) life_max = life;
        total_down += mttr_hours;
    }

    double mean_life = life_sum / n_blades;
    double var_life  = life_sq / n_blades - mean_life * mean_life;
    double std_life  = std::sqrt(std::max(0.0, var_life));
    double analytic_mttf = eta * gamma_func(1.0 + 1.0 / beta);

    py::dict r;
    r["n_blades"]        = n_blades;
    r["mean_life_hours"] = mean_life;
    r["std_life_hours"]  = std_life;
    r["min_life_hours"]  = life_min;
    r["max_life_hours"]  = life_max;
    r["analytic_mttf"]   = analytic_mttf;
    r["mttf_error_pct"]  = 100.0 * std::abs(mean_life - analytic_mttf) / analytic_mttf;
    r["total_downtime_hours"] = total_down;
    r["availability"]    = mean_life / (mean_life + mttr_hours);
    return r;
}

// ── pybind11 module ──────────────────────────────────────────────────────────

PYBIND11_MODULE(_reliability_kernel, m) {
    m.doc() =
        "Reliability / RASIS analysis for semiconductor dicing equipment.\n"
        "\n"
        "AP試験 topics: MTBF/MTTR/稼働率, 直列/並列システム信頼性,\n"
        "  Weibull分布, OEE, バスタブ曲線(初期/偶発/摩耗故障)\n"
        "\n"
        "RASIS = Reliability · Availability · Serviceability · Integrity · Security";

    m.def("series_system", &series_system,
          py::arg("mtbf_hours"),
          "Series system MTBF: 1/λ_sys = 1/Σ(1/MTBFi).\n"
          "System fails when ANY component fails.\n"
          "Returns: mtbf_hours, failure_rate, n_components, weakest_component.");

    m.def("parallel_system", &parallel_system,
          py::arg("mtbf_hours"),
          "Parallel (hot-standby) system MTTF: η·Σ(1/k) for k=1..n.\n"
          "System fails only when ALL components fail.\n"
          "Returns: mttf_hours, n_components, avg_unit_mtbf, improvement.");

    m.def("availability_oee", &availability_oee,
          py::arg("mtbf_hours"),
          py::arg("mttr_hours"),
          py::arg("performance_rate") = 1.0,
          py::arg("quality_rate")     = 1.0,
          "Compute availability and OEE (Overall Equipment Effectiveness).\n"
          "  Availability = MTBF / (MTBF + MTTR)\n"
          "  OEE = Availability × Performance × Quality\n"
          "Returns: availability, performance_rate, quality_rate, oee,\n"
          "         uptime_fraction, downtime_fraction, reliability_label, oee_label.");

    m.def("weibull_analysis", &weibull_analysis,
          py::arg("t_hours"),
          py::arg("beta"),
          py::arg("eta"),
          "Evaluate Weibull reliability R(t), hazard h(t), CDF F(t).\n"
          "  R(t) = exp(-(t/η)^β)   — survival probability\n"
          "  h(t) = (β/η)·(t/η)^(β-1) — instantaneous failure rate\n"
          "  β < 1: infant mortality;  β = 1: random;  β > 1: wear-out\n"
          "Returns: reliability, hazard_rate, cdf (lists), mttf_hours, failure_phase.");

    m.def("simulate_blade_life", &simulate_blade_life,
          py::arg("n_blades")    = 1000,
          py::arg("beta")        = 2.0,    // wear-out regime typical for dicing blade
          py::arg("eta")         = 72.0,   // characteristic life 72 h (example)
          py::arg("mttr_hours")  = 0.5,    // 30-min blade change
          py::arg("seed")        = 42,
          "Monte Carlo blade life simulation (Weibull variates via inverse CDF).\n"
          "Verifies analytic MTTF = η·Γ(1+1/β) empirically.\n"
          "Returns: mean/std/min/max life, analytic_mttf, mttf_error_pct, availability.");
}
