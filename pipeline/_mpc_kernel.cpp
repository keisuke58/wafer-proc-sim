// MPC kernel with GP surrogate for Advanced Process Control (APC).
//
// WHY C++: mpc_optimize() performs an exhaustive n_feed × n_rpm grid search
// (up to 400 candidates) × H horizon steps, each requiring two GP dot-products
// over N training points.  A Python loop over that triple nest would be ~50×
// slower than this inner-loop-fused C++ version; keeping it here lets the APC
// layer call optimize() once per wafer without blocking.
//
// State x0 = [blade_wear_um, material_hardness, depth_um]        (3D)
// Control u = [feed_rate_mms, spindle_rpm]                       (2D)
// GP input  = [blade_wear_um, material_hardness, depth_um,
//              feed_rate_mms, spindle_rpm]                        (5D)
//
// Both GP surrogates (quality, wear) share length_scales / signal_var.
//
// Build: bash build_all_kernels.sh

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <limits>
#include <algorithm>

namespace py = pybind11;
using arr2d = py::array_t<double, py::array::c_style | py::array::forcecast>;
using arr1d = py::array_t<double, py::array::c_style | py::array::forcecast>;

// ── MPCParams ─────────────────────────────────────────────────────────────────

struct MPCParams {
    int   horizon   = 10;
    int   n_feed    = 20;
    int   n_rpm     = 20;
    float feed_min  = 10.0f,    feed_max  = 200.0f;
    float rpm_min   = 10000.0f, rpm_max   = 35000.0f;
    float w_quality = 1.0f;
    float w_wear    = 0.5f;
    float w_delta_u = 0.1f;
};

// ── Inline RBF dot-product  k(x_aug, X_tr) @ alpha ───────────────────────────
//
// x_aug: pointer to 5 doubles [blade_wear, hardness, depth, feed, rpm]
// Xtr:   unchecked<2> proxy, shape (N, 5)
// ls:    unchecked<1> proxy, shape (5,)
// Returns GP predictive mean for a single test point.
//
// Inlining avoids the pybind11 FFI overhead inside the tight grid-search loop.
static inline double gp_mean_single(
    const double* __restrict__ x_aug,
    const py::detail::unchecked_reference<double, 2>& Xtr,
    const py::detail::unchecked_reference<double, 1>& alpha,
    const py::detail::unchecked_reference<double, 1>& ls,
    double signal_var,
    ssize_t N, ssize_t D)
{
    double mean = 0.0;
    for (ssize_t j = 0; j < N; ++j) {
        double s = 0.0;
        for (ssize_t d = 0; d < D; ++d) {
            double diff = (x_aug[d] - Xtr(j, d)) / ls(d);
            s += diff * diff;
        }
        mean += signal_var * std::exp(-0.5 * s) * alpha(j);
    }
    return mean;
}

// ── Shared argument validation ────────────────────────────────────────────────

static void validate_gp_args(
    const arr1d& x0,
    const arr2d& Xtr_q, const arr1d& alpha_q,
    const arr2d& Xtr_w, const arr1d& alpha_w,
    const arr1d& ls)
{
    if (x0.shape(0) != 3)
        throw std::invalid_argument("x0 must have shape (3,)");
    if (Xtr_q.shape(1) != 5 || Xtr_w.shape(1) != 5)
        throw std::invalid_argument("GP training inputs must have 5 columns");
    if (alpha_q.shape(0) != Xtr_q.shape(0))
        throw std::invalid_argument("alpha_quality length must match Xtr_quality rows");
    if (alpha_w.shape(0) != Xtr_w.shape(0))
        throw std::invalid_argument("alpha_wear length must match Xtr_wear rows");
    if (ls.shape(0) != 5)
        throw std::invalid_argument("length_scales must have shape (5,)");
}

// ── mpc_optimize ─────────────────────────────────────────────────────────────
//
// Exhaustive grid search over n_feed × n_rpm candidates.
// The horizon loop accumulates cost assuming the same (feed, rpm) is held
// constant over H steps — a receding-horizon approximation appropriate when
// the process dynamics are slow relative to the control period.
//
// Wear-state propagation per step: blade_wear += |wear_rate| * feed_rate * dt
// where dt = 1 step (dimensionless; wear_rate in um/mm × feed in mm/s gives
// um/s, but the GP already encodes the net wear magnitude per control action).
// We treat each horizon step as 1 s so the increment is numerically consistent.

py::dict mpc_optimize(
    arr1d x0,
    arr2d Xtr_quality, arr1d alpha_quality,
    arr2d Xtr_wear,    arr1d alpha_wear,
    arr1d length_scales,
    double signal_var,
    double u_prev_feed,
    double u_prev_rpm,
    MPCParams params)
{
    validate_gp_args(x0, Xtr_quality, alpha_quality,
                     Xtr_wear, alpha_wear, length_scales);

    auto x0_r    = x0.unchecked<1>();
    auto Xtr_q   = Xtr_quality.unchecked<2>();
    auto alph_q  = alpha_quality.unchecked<1>();
    auto Xtr_w   = Xtr_wear.unchecked<2>();
    auto alph_w  = alpha_wear.unchecked<1>();
    auto ls      = length_scales.unchecked<1>();

    const ssize_t Nq = Xtr_q.shape(0);
    const ssize_t Nw = Xtr_w.shape(0);
    constexpr ssize_t D = 5;

    const int H      = params.horizon;
    const int nf     = params.n_feed;
    const int nr     = params.n_rpm;
    const double wq  = params.w_quality;
    const double ww  = params.w_wear;
    const double wu  = params.w_delta_u;

    // Normalise delta-u penalty to control ranges so weights are scale-free.
    const double feed_range = static_cast<double>(params.feed_max - params.feed_min);
    const double rpm_range  = static_cast<double>(params.rpm_max  - params.rpm_min);

    double best_cost = std::numeric_limits<double>::infinity();
    double best_feed = static_cast<double>(params.feed_min);
    double best_rpm  = static_cast<double>(params.rpm_min);

    double x_aug[D];

    for (int fi = 0; fi < nf; ++fi) {
        double feed = static_cast<double>(params.feed_min)
                    + fi * feed_range / std::max(nf - 1, 1);

        for (int ri = 0; ri < nr; ++ri) {
            double rpm = static_cast<double>(params.rpm_min)
                       + ri * rpm_range / std::max(nr - 1, 1);

            // Smoothness penalty: normalised Euclidean distance in u-space.
            double df   = (feed - u_prev_feed) / feed_range;
            double dr   = (rpm  - u_prev_rpm)  / rpm_range;
            double cost_smooth = wu * (df * df + dr * dr);

            // Propagate state over H steps with constant (feed, rpm).
            double wear = x0_r(0);
            double hard = x0_r(1);
            double dep  = x0_r(2);

            double cost_q = 0.0;
            double cost_w = 0.0;

            for (int h = 0; h < H; ++h) {
                x_aug[0] = wear;
                x_aug[1] = hard;
                x_aug[2] = dep;
                x_aug[3] = feed;
                x_aug[4] = rpm;

                double ra       = gp_mean_single(x_aug, Xtr_q, alph_q, ls,
                                                 signal_var, Nq, D);
                double wear_rate = gp_mean_single(x_aug, Xtr_w, alph_w, ls,
                                                  signal_var, Nw, D);

                // Accumulate stage costs; clip GP output to >= 0.
                cost_q += wq * std::max(ra, 0.0);
                cost_w += ww * std::max(wear_rate, 0.0);

                // State transition: wear accumulates at rate |wear_rate|·feed.
                // wear_rate is in um/mm; feed in mm/s → um/s per step (1 s).
                wear += std::abs(wear_rate) * feed;
            }

            double total = cost_q + cost_w + cost_smooth;
            if (total < best_cost) {
                best_cost = total;
                best_feed = feed;
                best_rpm  = rpm;
            }
        }
    }

    py::dict result;
    result["feed_rate_mms"] = best_feed;
    result["spindle_rpm"]   = best_rpm;
    result["total_cost"]    = best_cost;
    return result;
}

// ── mpc_evaluate_action ───────────────────────────────────────────────────────

py::dict mpc_evaluate_action(
    arr1d x0,
    arr2d Xtr_quality, arr1d alpha_quality,
    arr2d Xtr_wear,    arr1d alpha_wear,
    arr1d length_scales,
    double signal_var,
    double feed_rate,
    double spindle_rpm,
    MPCParams params)
{
    validate_gp_args(x0, Xtr_quality, alpha_quality,
                     Xtr_wear, alpha_wear, length_scales);

    auto x0_r   = x0.unchecked<1>();
    auto Xtr_q  = Xtr_quality.unchecked<2>();
    auto alph_q = alpha_quality.unchecked<1>();
    auto Xtr_w  = Xtr_wear.unchecked<2>();
    auto alph_w = alpha_wear.unchecked<1>();
    auto ls     = length_scales.unchecked<1>();

    const ssize_t Nq = Xtr_q.shape(0);
    const ssize_t Nw = Xtr_w.shape(0);
    constexpr ssize_t D = 5;

    const double feed_range = static_cast<double>(params.feed_max - params.feed_min);
    const double rpm_range  = static_cast<double>(params.rpm_max  - params.rpm_min);

    double wear = x0_r(0);
    double hard = x0_r(1);
    double dep  = x0_r(2);

    double cost_q = 0.0;
    double cost_w = 0.0;
    double x_aug[D];

    for (int h = 0; h < params.horizon; ++h) {
        x_aug[0] = wear;
        x_aug[1] = hard;
        x_aug[2] = dep;
        x_aug[3] = feed_rate;
        x_aug[4] = spindle_rpm;

        double ra        = gp_mean_single(x_aug, Xtr_q, alph_q, ls,
                                          signal_var, Nq, D);
        double wear_rate = gp_mean_single(x_aug, Xtr_w, alph_w, ls,
                                          signal_var, Nw, D);

        cost_q += params.w_quality * std::max(ra, 0.0);
        cost_w += params.w_wear    * std::max(wear_rate, 0.0);
        wear   += std::abs(wear_rate) * feed_rate;
    }

    // Smoothness penalty against the midpoint of the control range
    // (used when no u_prev is available in single-action evaluation).
    double df_mid = (feed_rate   - 0.5 * (params.feed_min + params.feed_max)) / feed_range;
    double dr_mid = (spindle_rpm - 0.5 * (params.rpm_min  + params.rpm_max))  / rpm_range;
    double cost_smooth = params.w_delta_u * (df_mid * df_mid + dr_mid * dr_mid);

    py::dict result;
    result["quality_cost"]    = cost_q;
    result["wear_cost"]       = cost_w;
    result["smoothness_cost"] = cost_smooth;
    result["total_cost"]      = cost_q + cost_w + cost_smooth;
    return result;
}

// ── mpc_pareto_front ──────────────────────────────────────────────────────────
//
// Returns all non-dominated (feed, rpm, quality_cost, wear_cost) tuples.
// Dominance: point A dominates B iff A.q <= B.q AND A.w <= B.w (at least
// one strict).  The search is O((n_feed × n_rpm)²) but grids are small (≤400).

py::list mpc_pareto_front(
    arr1d x0,
    arr2d Xtr_quality, arr1d alpha_quality,
    arr2d Xtr_wear,    arr1d alpha_wear,
    arr1d length_scales,
    double signal_var,
    MPCParams params)
{
    validate_gp_args(x0, Xtr_quality, alpha_quality,
                     Xtr_wear, alpha_wear, length_scales);

    auto x0_r   = x0.unchecked<1>();
    auto Xtr_q  = Xtr_quality.unchecked<2>();
    auto alph_q = alpha_quality.unchecked<1>();
    auto Xtr_w  = Xtr_wear.unchecked<2>();
    auto alph_w = alpha_wear.unchecked<1>();
    auto ls     = length_scales.unchecked<1>();

    const ssize_t Nq = Xtr_q.shape(0);
    const ssize_t Nw = Xtr_w.shape(0);
    constexpr ssize_t D = 5;

    const int nf = params.n_feed;
    const int nr = params.n_rpm;
    const double feed_range = static_cast<double>(params.feed_max - params.feed_min);
    const double rpm_range  = static_cast<double>(params.rpm_max  - params.rpm_min);

    struct Candidate {
        double feed, rpm, cq, cw;
    };
    std::vector<Candidate> cands;
    cands.reserve(nf * nr);

    double x_aug[D];
    x_aug[1] = x0_r(1);   // hardness and depth are constant over the grid sweep
    x_aug[2] = x0_r(2);

    for (int fi = 0; fi < nf; ++fi) {
        double feed = static_cast<double>(params.feed_min)
                    + fi * feed_range / std::max(nf - 1, 1);
        for (int ri = 0; ri < nr; ++ri) {
            double rpm = static_cast<double>(params.rpm_min)
                       + ri * rpm_range / std::max(nr - 1, 1);

            x_aug[0] = x0_r(0);
            x_aug[3] = feed;
            x_aug[4] = rpm;

            double ra        = gp_mean_single(x_aug, Xtr_q, alph_q, ls,
                                              signal_var, Nq, D);
            double wear_rate = gp_mean_single(x_aug, Xtr_w, alph_w, ls,
                                              signal_var, Nw, D);

            cands.push_back({feed, rpm,
                             params.w_quality * std::max(ra, 0.0),
                             params.w_wear    * std::max(wear_rate, 0.0)});
        }
    }

    // Filter: keep candidate i if no other j dominates it.
    std::vector<bool> dominated(cands.size(), false);
    for (std::size_t i = 0; i < cands.size(); ++i) {
        if (dominated[i]) continue;
        for (std::size_t j = 0; j < cands.size(); ++j) {
            if (i == j || dominated[j]) continue;
            if (cands[j].cq <= cands[i].cq && cands[j].cw <= cands[i].cw &&
                (cands[j].cq < cands[i].cq || cands[j].cw < cands[i].cw)) {
                dominated[i] = true;
                break;
            }
        }
    }

    py::list front;
    for (std::size_t i = 0; i < cands.size(); ++i) {
        if (!dominated[i]) {
            py::dict pt;
            pt["feed_rate_mms"]  = cands[i].feed;
            pt["spindle_rpm"]    = cands[i].rpm;
            pt["quality_cost"]   = cands[i].cq;
            pt["wear_cost"]      = cands[i].cw;
            front.append(pt);
        }
    }
    return front;
}

// ── module ────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(_mpc_kernel, m) {
    m.doc() = "MPC kernel with GP surrogate for APC: grid-search optimizer, "
              "action evaluator, and Pareto-front extractor";

    py::class_<MPCParams>(m, "MPCParams")
        .def(py::init<>())
        .def_readwrite("horizon",   &MPCParams::horizon)
        .def_readwrite("n_feed",    &MPCParams::n_feed)
        .def_readwrite("n_rpm",     &MPCParams::n_rpm)
        .def_readwrite("feed_min",  &MPCParams::feed_min)
        .def_readwrite("feed_max",  &MPCParams::feed_max)
        .def_readwrite("rpm_min",   &MPCParams::rpm_min)
        .def_readwrite("rpm_max",   &MPCParams::rpm_max)
        .def_readwrite("w_quality", &MPCParams::w_quality)
        .def_readwrite("w_wear",    &MPCParams::w_wear)
        .def_readwrite("w_delta_u", &MPCParams::w_delta_u);

    m.def("mpc_optimize", &mpc_optimize,
          py::arg("x0"),
          py::arg("Xtr_quality"),  py::arg("alpha_quality"),
          py::arg("Xtr_wear"),     py::arg("alpha_wear"),
          py::arg("length_scales"), py::arg("signal_var"),
          py::arg("u_prev_feed"),  py::arg("u_prev_rpm"),
          py::arg("params") = MPCParams{},
          "Grid-search MPC optimizer. Returns dict with feed_rate_mms, "
          "spindle_rpm, total_cost.");

    m.def("mpc_evaluate_action", &mpc_evaluate_action,
          py::arg("x0"),
          py::arg("Xtr_quality"),  py::arg("alpha_quality"),
          py::arg("Xtr_wear"),     py::arg("alpha_wear"),
          py::arg("length_scales"), py::arg("signal_var"),
          py::arg("feed_rate"),    py::arg("spindle_rpm"),
          py::arg("params") = MPCParams{},
          "Evaluate MPC cost for a single (feed, rpm) action over H steps. "
          "Returns dict with quality_cost, wear_cost, smoothness_cost, total_cost.");

    m.def("mpc_pareto_front", &mpc_pareto_front,
          py::arg("x0"),
          py::arg("Xtr_quality"),  py::arg("alpha_quality"),
          py::arg("Xtr_wear"),     py::arg("alpha_wear"),
          py::arg("length_scales"), py::arg("signal_var"),
          py::arg("params") = MPCParams{},
          "Return list of non-dominated (feed, rpm, quality_cost, wear_cost) "
          "dicts from the control grid.");
}
