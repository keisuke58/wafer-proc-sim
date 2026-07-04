// spc_monitor.cpp — Statistical Process Control for kerf width monitoring
//
// AP試験 concepts demonstrated (品質管理):
//   - X̄-R 管理図 (Shewhart control chart)
//       UCL_x = X̄̄ + A2·R̄,  LCL_x = X̄̄ − A2·R̄
//       UCL_r = D4·R̄,        LCL_r = D3·R̄
//   - 工程能力指数 Cp, Cpk
//       σ̂ = R̄/d2  (estimated from R chart)
//       Cp  = (USL−LSL) / (6·σ̂)
//       Cpk = min( (USL−X̄̄)/(3·σ̂),  (X̄̄−LSL)/(3·σ̂) )
//   - Western Electric (WECO) rules — 4 main detection patterns
//   - EWMA chart: Zt = λ·xt + (1−λ)·Zt-1,  control limits ±3σ·√(λ/(2−λ))
//
// Application: real-time kerf width QC from detect_kerf_width() measurements.
//   Cpk < 1.33 → yield risk → trigger recipe correction.
//   WECO alarm  → blade wear event or setup shift → maintenance alert.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <numeric>

namespace py = pybind11;

// ── Shewhart constants ───────────────────────────────────────────────────────
// Indexed by subgroup_size n (valid n = 2..10).
// Source: ASTM E2281, AIAG SPC 2nd ed.
struct ShewhartConst {
    double A2, d2, D3, D4;
};
static const ShewhartConst SHEWHART[] = {
    // n=0 placeholder
    {0.0,  0.0,   0.0, 0.0},
    // n=1 placeholder
    {0.0,  0.0,   0.0, 0.0},
    {1.880, 1.128, 0.0,   3.267},  // n=2
    {1.023, 1.693, 0.0,   2.574},  // n=3
    {0.729, 2.059, 0.0,   2.282},  // n=4
    {0.577, 2.326, 0.0,   2.114},  // n=5
    {0.483, 2.534, 0.0,   2.004},  // n=6
    {0.419, 2.704, 0.076, 1.924},  // n=7
    {0.373, 2.847, 0.136, 1.864},  // n=8
    {0.337, 2.970, 0.184, 1.816},  // n=9
    {0.308, 3.078, 0.223, 1.777},  // n=10
};

// ── Western Electric (WECO) rules ────────────────────────────────────────────
// Returns bitmask of which rules triggered (bits 0-3 = rules 1-4).
// Rule 1: any point outside 3σ (UCL/LCL)
// Rule 2: 2 of 3 consecutive points beyond 2σ (same side)
// Rule 3: 4 of 5 consecutive points beyond 1σ (same side)
// Rule 4: 8 consecutive points on same side of centre
static uint32_t weco_check(
    const std::vector<double>& z,   // z-scores: (xi − CL) / σ
    size_t idx                      // index of latest observation
) {
    uint32_t flags = 0;

    if (idx >= 1) {  // need at least current point
        // Rule 1: current point beyond ±3
        if (std::abs(z[idx]) > 3.0) flags |= (1u << 0);
    }

    // Rule 2: 2 of last 3 points beyond ±2 (same side)
    if (idx >= 2) {
        int pos = 0, neg = 0;
        for (int k = 0; k <= 2; ++k) {
            size_t j = idx - k;
            if (z[j] >  2.0) ++pos;
            if (z[j] < -2.0) ++neg;
        }
        if (pos >= 2 || neg >= 2) flags |= (1u << 1);
    }

    // Rule 3: 4 of last 5 points beyond ±1 (same side)
    if (idx >= 4) {
        int pos = 0, neg = 0;
        for (int k = 0; k <= 4; ++k) {
            size_t j = idx - k;
            if (z[j] >  1.0) ++pos;
            if (z[j] < -1.0) ++neg;
        }
        if (pos >= 4 || neg >= 4) flags |= (1u << 2);
    }

    // Rule 4: 8 consecutive points on same side of 0
    if (idx >= 7) {
        int pos = 0, neg = 0;
        for (int k = 0; k <= 7; ++k) {
            size_t j = idx - k;
            if (z[j] > 0.0) ++pos;
            if (z[j] < 0.0) ++neg;
        }
        if (pos == 8 || neg == 8) flags |= (1u << 3);
    }

    return flags;
}

// ── Main SPC function ────────────────────────────────────────────────────────

py::dict spc_update(
    const std::vector<double>& measurements,  // raw individual measurements [mm]
    double usl,                               // upper spec limit [mm]
    double lsl,                               // lower spec limit [mm]
    int    subgroup_size,                     // n (2..10)
    double ewma_lambda                        // EWMA smoothing factor (0 < λ ≤ 1)
) {
    if (subgroup_size < 2 || subgroup_size > 10)
        throw std::invalid_argument("subgroup_size must be 2..10");
    if (measurements.empty())
        throw std::invalid_argument("measurements is empty");

    const auto& c = SHEWHART[subgroup_size];

    // ── Compute subgroup X̄ and R ─────────────────────────────────────────────
    int n_grps = static_cast<int>(measurements.size()) / subgroup_size;
    if (n_grps < 1)
        throw std::invalid_argument("fewer measurements than one subgroup");

    std::vector<double> xbar(n_grps), rng(n_grps);
    for (int g = 0; g < n_grps; ++g) {
        double sum = 0.0, mn = measurements[g * subgroup_size], mx = mn;
        for (int j = 0; j < subgroup_size; ++j) {
            double v = measurements[g * subgroup_size + j];
            sum += v;
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        xbar[g] = sum / subgroup_size;
        rng[g]  = mx - mn;
    }

    double xbar_bar = 0.0, r_bar = 0.0;
    for (int g = 0; g < n_grps; ++g) { xbar_bar += xbar[g]; r_bar += rng[g]; }
    xbar_bar /= n_grps;
    r_bar    /= n_grps;

    // ── Control limits (X̄ chart) ──────────────────────────────────────────────
    double ucl_x = xbar_bar + c.A2 * r_bar;
    double lcl_x = xbar_bar - c.A2 * r_bar;
    double ucl_r = c.D4 * r_bar;
    double lcl_r = c.D3 * r_bar;

    // ── σ̂ estimate from R chart ───────────────────────────────────────────────
    double sigma_hat = (c.d2 > 0.0) ? r_bar / c.d2 : 0.0;

    // ── Process capability ────────────────────────────────────────────────────
    double cp  = (sigma_hat > 0.0) ? (usl - lsl) / (6.0 * sigma_hat) : 0.0;
    double cpu = (sigma_hat > 0.0) ? (usl - xbar_bar) / (3.0 * sigma_hat) : 0.0;
    double cpl = (sigma_hat > 0.0) ? (xbar_bar - lsl) / (3.0 * sigma_hat) : 0.0;
    double cpk = std::min(cpu, cpl);

    // ── WECO z-scores for X̄ chart ─────────────────────────────────────────────
    std::vector<double> z_x(n_grps);
    for (int g = 0; g < n_grps; ++g)
        z_x[g] = (sigma_hat > 0.0) ? (xbar[g] - xbar_bar) / (sigma_hat / std::sqrt(subgroup_size)) : 0.0;

    std::vector<uint32_t> weco_flags(n_grps, 0);
    for (int g = 0; g < n_grps; ++g)
        weco_flags[g] = weco_check(z_x, static_cast<size_t>(g));

    const bool any_alarm = std::any_of(weco_flags.begin(), weco_flags.end(),
                                       [](uint32_t f) { return f != 0; });

    // ── EWMA chart ────────────────────────────────────────────────────────────
    // Applied to individual measurements (not subgroup means)
    int M = static_cast<int>(measurements.size());
    std::vector<double> ewma(M);
    const double mu_all =
        std::accumulate(measurements.begin(), measurements.end(), 0.0) / M;

    ewma[0] = measurements[0];
    for (int k = 1; k < M; ++k)
        ewma[k] = ewma_lambda * measurements[k] + (1.0 - ewma_lambda) * ewma[k-1];

    // EWMA steady-state control limits
    double ewma_sigma = (sigma_hat > 0.0 && ewma_lambda > 0.0)
        ? sigma_hat * std::sqrt(ewma_lambda / (2.0 - ewma_lambda))
        : 0.0;
    double ucl_ewma = mu_all + 3.0 * ewma_sigma;
    double lcl_ewma = mu_all - 3.0 * ewma_sigma;

    // ── Build result ──────────────────────────────────────────────────────────
    py::list xbar_list, r_list, z_list, flags_list, ewma_list;
    for (int g = 0; g < n_grps; ++g) {
        xbar_list.append(xbar[g]);
        r_list.append(rng[g]);
        z_list.append(z_x[g]);
        flags_list.append(weco_flags[g]);
    }
    for (int k = 0; k < M; ++k) ewma_list.append(ewma[k]);

    py::dict res;
    res["xbar_bar"]      = xbar_bar;
    res["r_bar"]         = r_bar;
    res["sigma_hat"]     = sigma_hat;
    res["ucl_x"]         = ucl_x;
    res["lcl_x"]         = lcl_x;
    res["ucl_r"]         = ucl_r;
    res["lcl_r"]         = lcl_r;
    res["cp"]            = cp;
    res["cpk"]           = cpk;
    res["cpu"]           = cpu;
    res["cpl"]           = cpl;
    res["usl"]           = usl;
    res["lsl"]           = lsl;
    res["xbar"]          = xbar_list;
    res["r"]             = r_list;
    res["z_scores"]      = z_list;
    res["weco_flags"]    = flags_list;
    res["weco_alarm"]    = any_alarm;
    res["ewma"]          = ewma_list;
    res["ucl_ewma"]      = ucl_ewma;
    res["lcl_ewma"]      = lcl_ewma;
    res["n_subgroups"]   = n_grps;
    res["subgroup_size"] = subgroup_size;
    return res;
}

PYBIND11_MODULE(_spc_monitor_kernel, m) {
    m.doc() =
        "Statistical Process Control (SPC) for kerf width monitoring.\n"
        "\n"
        "AP試験 topics: X̄-R管理図, Cp/Cpk工程能力指数, WECO判定ルール, EWMA管理図\n"
        "\n"
        "Feed detect_kerf_width() measurements into spc_update().\n"
        "Cpk < 1.33 triggers recipe correction; WECO alarm triggers maintenance.";

    m.def("spc_update", &spc_update,
          py::arg("measurements"),
          py::arg("usl"),
          py::arg("lsl"),
          py::arg("subgroup_size") = 5,
          py::arg("ewma_lambda")   = 0.2,
          "Compute X̄-R chart, Cpk, WECO rules, and EWMA from a list of measurements.\n"
          "\n"
          "measurements : list[float]  — individual measurements (e.g. kerf widths [mm])\n"
          "usl / lsl    : float        — spec limits [mm]\n"
          "subgroup_size: int (2..10)  — rational subgroup size n\n"
          "ewma_lambda  : float (0,1]  — EWMA weight (0.2 = sensitive to 1σ shifts)\n"
          "\n"
          "Returns dict:\n"
          "  xbar_bar, r_bar, sigma_hat          — chart statistics\n"
          "  ucl_x, lcl_x, ucl_r, lcl_r         — X̄ and R chart limits\n"
          "  cp, cpk, cpu, cpl                   — process capability indices\n"
          "  xbar, r, z_scores, weco_flags        — per-subgroup arrays\n"
          "  weco_alarm                           — True if any WECO rule fired\n"
          "  ewma, ucl_ewma, lcl_ewma             — EWMA chart values and limits\n"
          "  n_subgroups, subgroup_size");
}
