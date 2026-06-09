// Ensemble Kalman Filter (EnKF) — analysis step + vectorized observation models
// (pybind11).
//
// WHY C++: The APC simulation calls EnKF.update() once per wafer. The bottleneck
// is the per-ensemble-member Python loop:
//
//   hx = np.array([obs_model(ens[i]) for i in range(N)])  # N=30-100 calls
//
// Moving both the observation map and the analysis update (cross-covariance +
// Kalman gain + ensemble correction) to C++ eliminates N Python-call overheads
// and one matrix-transpose multiply, keeping only the FFI crossing per update.
//
// Two observation model variants are provided:
//   blade_wear_obs   — Lawn–Evans chipping model (pipeline/keyence_apc.py)
//   three_state_obs  — 3-state chipping model (ml/ekf_state_estimation.py)
//
// The general analysis step enkf_analysis() works for any observation model:
// compute hx in Python (or via one of the helpers above), then call enkf_analysis.
//
// Build: bash build_all_kernels.sh

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace py = pybind11;
using Arr  = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Arr2 = py::array_t<double, py::array::c_style | py::array::forcecast>;

// ── General EnKF analysis (perturbed-observations formulation) ────────────────
//
// Inputs:
//   xf  (N × n_state) forecast ensemble
//   yf  (N,)          predicted observations H(xf^i)
//   y_obs              scalar true observation
//   v   (N,)          observation noise perturbations  ~ N(0, R)
//   R                  scalar observation noise variance
//
// Returns:
//   xa  (N × n_state) analysis (updated) ensemble
//
// Algorithm (Burgers et al. 1998, stochastic EnKF):
//   x_bar = mean(xf, axis=0);  y_bar = mean(yf)
//   dX = xf - x_bar;           dY = yf - y_bar
//   Pxy = (dX^T dY) / (N-1)   [n_state vector]
//   Pyy = sum(dY^2)/(N-1) + R  [scalar]
//   K   = Pxy / Pyy            [n_state vector]
//   xa^i = xf^i + K * (y_obs + v^i - yf^i)

static py::array_t<double> enkf_analysis(
    Arr2 xf_arr, Arr yf_arr,
    double y_obs, Arr v_arr,
    double R)
{
    if (xf_arr.ndim() != 2)
        throw std::runtime_error("enkf_analysis: xf must be 2-D (N × n_state)");
    if (yf_arr.ndim() != 1 || v_arr.ndim() != 1)
        throw std::runtime_error("enkf_analysis: yf and v must be 1-D (N,)");

    const int N       = (int)xf_arr.shape(0);
    const int n_state = (int)xf_arr.shape(1);

    if ((int)yf_arr.shape(0) != N || (int)v_arr.shape(0) != N)
        throw std::runtime_error("enkf_analysis: xf, yf, v must all have N rows");
    if (N < 2)
        throw std::runtime_error("enkf_analysis: need at least 2 ensemble members");

    const auto xf = xf_arr.unchecked<2>();
    const auto yf = yf_arr.unchecked<1>();
    const auto v  = v_arr.unchecked<1>();

    // Ensemble means.
    double yf_mean = 0.0;
    std::vector<double> xf_mean(n_state, 0.0);
    for (int i = 0; i < N; ++i) {
        yf_mean += yf(i);
        for (int k = 0; k < n_state; ++k) xf_mean[k] += xf(i, k);
    }
    yf_mean /= N;
    for (int k = 0; k < n_state; ++k) xf_mean[k] /= N;

    // Cross-covariances (single pass).
    std::vector<double> Pxy(n_state, 0.0);
    double Pyy = 0.0;
    for (int i = 0; i < N; ++i) {
        const double dy = yf(i) - yf_mean;
        Pyy += dy * dy;
        for (int k = 0; k < n_state; ++k)
            Pxy[k] += (xf(i, k) - xf_mean[k]) * dy;
    }
    const double norm = 1.0 / (N - 1);
    for (int k = 0; k < n_state; ++k) Pxy[k] *= norm;
    Pyy = Pyy * norm + R;

    // Kalman gain K = Pxy / Pyy.
    std::vector<double> K(n_state);
    for (int k = 0; k < n_state; ++k) K[k] = Pxy[k] / Pyy;

    // Analysis update.
    auto xa_arr = py::array_t<double>({N, n_state});
    auto xa     = xa_arr.mutable_unchecked<2>();
    for (int i = 0; i < N; ++i) {
        const double innov = y_obs + v(i) - yf(i);
        for (int k = 0; k < n_state; ++k)
            xa(i, k) = xf(i, k) + K[k] * innov;
    }
    return xa_arr;
}


// ── Blade-wear observation model (Lawn–Evans, pipeline/keyence_apc.py) ────────
//
//   hx[i] = chip0 * (max(feed, 0.1) / feed_nom)^feed_exp
//              * (1 + wear_scale * ens[i,0])
//              * ens[i,1]
//
// ens: (N, 2)   [blade_wear, material_var]

static py::array_t<double> blade_wear_obs(
    Arr2 ens_arr,
    double feed,
    double chip0,
    double feed_nom,
    double wear_scale,
    double feed_exp)
{
    if (ens_arr.ndim() != 2 || ens_arr.shape(1) < 2)
        throw std::runtime_error("blade_wear_obs: ens must be (N, ≥2)");

    const int N    = (int)ens_arr.shape(0);
    const auto ens = ens_arr.unchecked<2>();

    const double feed_safe   = std::max(feed, 0.1);
    const double feed_factor = std::pow(feed_safe / feed_nom, feed_exp);

    auto out  = py::array_t<double>(N);
    auto bout = out.mutable_unchecked<1>();
    for (int i = 0; i < N; ++i)
        bout(i) = chip0 * feed_factor * (1.0 + wear_scale * ens(i, 0)) * ens(i, 1);
    return out;
}


// ── 3-state observation model (ml/ekf_state_estimation.py) ───────────────────
//
// State: [d_eff, alpha_w, k_ic]
//
//   actual_depth = D_NOM * clip(d_eff, 0.5, 1.5)
//   base = chipping_physics(actual_depth, blade_w)
//   hx[i] = base / (clip(alpha_w, 0.1, 2.0) * clip(k_ic, 0.5, 1.5))
//
// where chipping_physics(depth, bw) = depth_term * width_term * interaction * 3.5
//
// ens: (N, 3)   [d_eff, alpha_w, k_ic]

static inline double chipping_physics_cpp(double depth_um, double blade_w_um) {
    const double depth_term  = 1.2 + 0.035 * (depth_um - 80.0) / 310.0 * 10.0;
    const double width_term  = 0.8 + 0.4   * (blade_w_um - 20.0) / 35.0;
    const double interaction = 1.0 + 0.12  * (
        (depth_um - 80.0) / 310.0 * (blade_w_um - 20.0) / 35.0
    );
    return depth_term * width_term * interaction * 3.5;
}

static py::array_t<double> three_state_obs(
    Arr2 ens_arr,
    double D_NOM,
    double blade_w)
{
    if (ens_arr.ndim() != 2 || ens_arr.shape(1) < 3)
        throw std::runtime_error("three_state_obs: ens must be (N, ≥3)");

    const int  N   = (int)ens_arr.shape(0);
    const auto ens = ens_arr.unchecked<2>();

    auto out  = py::array_t<double>(N);
    auto bout = out.mutable_unchecked<1>();
    for (int i = 0; i < N; ++i) {
        const double d_eff  = std::max(0.5, std::min(1.5, ens(i, 0)));
        const double alpha_w = std::max(0.1, std::min(2.0, ens(i, 1)));
        const double k_ic   = std::max(0.5, std::min(1.5, ens(i, 2)));
        const double actual_depth = D_NOM * d_eff;
        const double base = chipping_physics_cpp(actual_depth, blade_w);
        bout(i) = base / (alpha_w * k_ic);
    }
    return out;
}


// ── Combined convenience: full update in one call ─────────────────────────────
//
// For BladeWearEnKF:  predict + obs + analysis in one round-trip.
// Returns: (xa, hx_mean)  — updated ensemble + mean predicted obs for logging.

static py::tuple blade_wear_update(
    Arr2 xf_arr,
    double feed,
    double y_obs,
    Arr  v_arr,
    double R,
    double chip0, double feed_nom, double wear_scale, double feed_exp,
    // physical constraints for clipping post-update
    double wear_lo, double wear_hi, double matvar_lo, double matvar_hi)
{
    auto hx  = blade_wear_obs(xf_arr, feed, chip0, feed_nom, wear_scale, feed_exp);
    auto xa  = enkf_analysis(xf_arr, hx, y_obs, v_arr, R);

    // Apply physical constraints in-place.
    const int N = (int)xa.shape(0);
    auto mxa = xa.mutable_unchecked<2>();
    for (int i = 0; i < N; ++i) {
        mxa(i, 0) = std::max(wear_lo,   std::min(wear_hi,   mxa(i, 0)));
        mxa(i, 1) = std::max(matvar_lo, std::min(matvar_hi, mxa(i, 1)));
    }

    // Mean predicted obs for logging.
    const auto bhx = hx.unchecked<1>();
    double hx_mean = 0.0;
    for (int i = 0; i < N; ++i) hx_mean += bhx(i);
    hx_mean /= N;

    return py::make_tuple(xa, hx_mean);
}


PYBIND11_MODULE(_enkf_kernel, m) {
    m.doc() = "C++ EnKF analysis step + vectorized observation models (pybind11)";

    m.def("enkf_analysis", &enkf_analysis,
          "Perturbed-observations EnKF analysis step. "
          "xf (N×n_state), yf (N,), y_obs scalar, v (N,), R scalar "
          "-> xa (N×n_state) updated ensemble",
          py::arg("xf"), py::arg("yf"),
          py::arg("y_obs"), py::arg("v"), py::arg("R"));

    m.def("blade_wear_obs", &blade_wear_obs,
          "Lawn–Evans blade-wear chipping model vectorised over ensemble. "
          "ens (N×2) [wear, mat_var], feed scalar -> hx (N,)",
          py::arg("ens"), py::arg("feed"),
          py::arg("chip0"), py::arg("feed_nom"),
          py::arg("wear_scale"), py::arg("feed_exp"));

    m.def("three_state_obs", &three_state_obs,
          "3-state chipping obs model vectorised over ensemble. "
          "ens (N×3) [d_eff, alpha_w, k_ic] -> hx (N,)",
          py::arg("ens"), py::arg("D_NOM"), py::arg("blade_w"));

    m.def("blade_wear_update", &blade_wear_update,
          "Full BladeWearEnKF update in one C++ call: obs prediction + analysis + clip. "
          "Returns (xa_updated, hx_mean).",
          py::arg("xf"), py::arg("feed"), py::arg("y_obs"), py::arg("v"), py::arg("R"),
          py::arg("chip0"), py::arg("feed_nom"), py::arg("wear_scale"), py::arg("feed_exp"),
          py::arg("wear_lo") = 0.0, py::arg("wear_hi") = 2.0,
          py::arg("matvar_lo") = 0.5, py::arg("matvar_hi") = 1.5);
}
