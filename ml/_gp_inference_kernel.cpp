/*
 * _gp_inference_kernel.cpp
 *
 * Gaussian Process inference hot loops in C++.
 * Used by the APC (Adaptive Process Control) layer for fast online prediction.
 *
 * Kernels:
 *   rbf_kernel_matrix(X, length_scales, signal_var)
 *       → K (N×N) symmetric, contiguous row-major
 *   rbf_kernel_vector(x_star, X_train, length_scales, signal_var)
 *       → k_star (N,)
 *   gp_predict_mean(X_test, X_train, alpha, length_scales, signal_var)
 *       → mean array  [hot loop: O(N_test × N_train × D)]
 *   gp_predict_full(X_test, X_train, alpha, L_chol_inv,
 *                   length_scales, signal_var)
 *       → {mean, std}  [includes posterior variance]
 *
 * Matches sklearn GaussianProcessRegressor RBF kernel convention:
 *   k(x, x') = signal_var * exp( -0.5 * sum_d ((x_d - x'_d) / l_d)² )
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace py = pybind11;
using arr2d = py::array_t<double, py::array::c_style | py::array::forcecast>;
using arr1d = py::array_t<double, py::array::c_style | py::array::forcecast>;

// ── RBF kernel matrix K(X, X) ─────────────────────────────────────────────────

arr2d rbf_kernel_matrix(const arr2d& X_arr,
                        const arr1d& ls_arr,
                        double signal_var)
{
    auto X  = X_arr.unchecked<2>();
    auto ls = ls_arr.unchecked<1>();
    ssize_t N = X.shape(0), D = X.shape(1);
    if (ls.shape(0) != D)
        throw std::invalid_argument("length_scales dim mismatch");

    auto K_arr = py::array_t<double>({N, N});
    auto K = K_arr.mutable_unchecked<2>();

    for (ssize_t i = 0; i < N; ++i) {
        K(i, i) = signal_var;
        for (ssize_t j = i + 1; j < N; ++j) {
            double s = 0.0;
            for (ssize_t d = 0; d < D; ++d) {
                double diff = (X(i,d) - X(j,d)) / ls(d);
                s += diff * diff;
            }
            double kij = signal_var * std::exp(-0.5 * s);
            K(i, j) = kij;
            K(j, i) = kij;
        }
    }
    return K_arr;
}

// ── RBF kernel vector k(x_star, X_train) ──────────────────────────────────────

arr1d rbf_kernel_vector(const arr1d& x_star_arr,
                        const arr2d& X_arr,
                        const arr1d& ls_arr,
                        double signal_var)
{
    auto xs = x_star_arr.unchecked<1>();
    auto X  = X_arr.unchecked<2>();
    auto ls = ls_arr.unchecked<1>();
    ssize_t N = X.shape(0), D = X.shape(1);
    if (xs.shape(0) != D || ls.shape(0) != D)
        throw std::invalid_argument("dimension mismatch");

    auto k_arr = py::array_t<double>(N);
    auto k = k_arr.mutable_unchecked<1>();

    for (ssize_t i = 0; i < N; ++i) {
        double s = 0.0;
        for (ssize_t d = 0; d < D; ++d) {
            double diff = (xs(d) - X(i,d)) / ls(d);
            s += diff * diff;
        }
        k(i) = signal_var * std::exp(-0.5 * s);
    }
    return k_arr;
}

// ── GP predictive mean (hot loop: M test points × N train × D dims) ──────────

arr1d gp_predict_mean(const arr2d& X_test_arr,
                      const arr2d& X_train_arr,
                      const arr1d& alpha_arr,
                      const arr1d& ls_arr,
                      double signal_var)
{
    auto Xte  = X_test_arr.unchecked<2>();
    auto Xtr  = X_train_arr.unchecked<2>();
    auto alph = alpha_arr.unchecked<1>();
    auto ls   = ls_arr.unchecked<1>();

    ssize_t M = Xte.shape(0), N = Xtr.shape(0), D = Xtr.shape(1);
    if (Xte.shape(1) != D || alph.shape(0) != N || ls.shape(0) != D)
        throw std::invalid_argument("dimension mismatch");

    auto mu_arr = py::array_t<double>(M);
    auto mu = mu_arr.mutable_unchecked<1>();

    for (ssize_t i = 0; i < M; ++i) {
        double mean = 0.0;
        for (ssize_t j = 0; j < N; ++j) {
            double s = 0.0;
            for (ssize_t d = 0; d < D; ++d) {
                double diff = (Xte(i,d) - Xtr(j,d)) / ls(d);
                s += diff * diff;
            }
            mean += signal_var * std::exp(-0.5 * s) * alph(j);
        }
        mu(i) = mean;
    }
    return mu_arr;
}

// ── GP predictive mean + std ───────────────────────────────────────────────────

// L_inv: lower-triangular Cholesky factor inverse, shape (N, N)
// std = sqrt( signal_var - sum(v²) ) where v = L_inv @ k_star
py::dict gp_predict_full(const arr2d& X_test_arr,
                         const arr2d& X_train_arr,
                         const arr1d& alpha_arr,
                         const arr2d& L_inv_arr,
                         const arr1d& ls_arr,
                         double signal_var,
                         double noise_var = 0.0)
{
    auto Xte   = X_test_arr.unchecked<2>();
    auto Xtr   = X_train_arr.unchecked<2>();
    auto alph  = alpha_arr.unchecked<1>();
    auto L_inv = L_inv_arr.unchecked<2>();
    auto ls    = ls_arr.unchecked<1>();

    ssize_t M = Xte.shape(0), N = Xtr.shape(0), D = Xtr.shape(1);
    if (Xte.shape(1)!=D || alph.shape(0)!=N || ls.shape(0)!=D
        || L_inv.shape(0)!=N || L_inv.shape(1)!=N)
        throw std::invalid_argument("dimension mismatch");

    auto mu_arr  = py::array_t<double>(M);
    auto std_arr = py::array_t<double>(M);
    auto mu  = mu_arr.mutable_unchecked<1>();
    auto sig = std_arr.mutable_unchecked<1>();

    // Reusable k_star buffer
    std::vector<double> kstar(N), v(N);

    for (ssize_t i = 0; i < M; ++i) {
        // Compute k_star
        double mean = 0.0;
        for (ssize_t j = 0; j < N; ++j) {
            double s = 0.0;
            for (ssize_t d = 0; d < D; ++d) {
                double diff = (Xte(i,d) - Xtr(j,d)) / ls(d);
                s += diff * diff;
            }
            kstar[j] = signal_var * std::exp(-0.5 * s);
            mean += kstar[j] * alph(j);
        }
        mu(i) = mean;

        // v = L_inv @ k_star  (lower-triangular matmul)
        double var = signal_var + noise_var;
        for (ssize_t r = 0; r < N; ++r) {
            double dot = 0.0;
            for (ssize_t c = 0; c <= r; ++c)
                dot += L_inv(r, c) * kstar[c];
            v[r] = dot;
            var -= dot * dot;
        }
        sig(i) = (var > 0.0) ? std::sqrt(var) : 0.0;
    }

    py::dict result;
    result["mean"] = mu_arr;
    result["std"]  = std_arr;
    return result;
}

// ── Acquisition functions (for BO / APC) ─────────────────────────────────────

// Expected Improvement: EI(x) = (µ - f_best - xi) * Φ(Z) + σ * φ(Z)
// where Z = (µ - f_best - xi) / σ
arr1d expected_improvement(const arr1d& mu_arr, const arr1d& std_arr,
                            double f_best, double xi = 0.01)
{
    auto mu  = mu_arr.unchecked<1>();
    auto sig = std_arr.unchecked<1>();
    ssize_t M = mu.shape(0);
    if (sig.shape(0) != M) throw std::invalid_argument("mu/std length mismatch");

    auto ei_arr = py::array_t<double>(M);
    auto ei = ei_arr.mutable_unchecked<1>();

    static const double inv_sqrt2   = 1.0 / std::sqrt(2.0);
    static const double inv_sqrt2pi = 1.0 / std::sqrt(2.0 * M_PI);

    for (ssize_t i = 0; i < M; ++i) {
        double s = sig(i);
        if (s < 1e-12) { ei(i) = 0.0; continue; }
        double Z   = (mu(i) - f_best - xi) / s;
        double phi = inv_sqrt2pi * std::exp(-0.5 * Z * Z);
        double Phi = 0.5 * (1.0 + std::erf(Z * inv_sqrt2));
        ei(i) = (mu(i) - f_best - xi) * Phi + s * phi;
    }
    return ei_arr;
}

// Upper Confidence Bound: UCB(x) = µ + sqrt(beta) * σ
arr1d upper_confidence_bound(const arr1d& mu_arr, const arr1d& std_arr,
                              double beta = 2.0)
{
    auto mu  = mu_arr.unchecked<1>();
    auto sig = std_arr.unchecked<1>();
    ssize_t M = mu.shape(0);
    if (sig.shape(0) != M) throw std::invalid_argument("mu/std length mismatch");

    double sq_beta = std::sqrt(beta);
    auto ucb_arr = py::array_t<double>(M);
    auto ucb = ucb_arr.mutable_unchecked<1>();
    for (ssize_t i = 0; i < M; ++i)
        ucb(i) = mu(i) + sq_beta * sig(i);
    return ucb_arr;
}

// ── module ────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(_gp_inference_kernel, m) {
    m.doc() = "GP inference hot loops: RBF kernel matrix, prediction, EI/UCB acquisition";

    m.def("rbf_kernel_matrix", &rbf_kernel_matrix,
          py::arg("X"), py::arg("length_scales"), py::arg("signal_var"),
          "Compute N×N RBF kernel matrix K(X, X)");

    m.def("rbf_kernel_vector", &rbf_kernel_vector,
          py::arg("x_star"), py::arg("X_train"),
          py::arg("length_scales"), py::arg("signal_var"),
          "Compute k_star vector k(x_star, X_train)");

    m.def("gp_predict_mean", &gp_predict_mean,
          py::arg("X_test"), py::arg("X_train"), py::arg("alpha"),
          py::arg("length_scales"), py::arg("signal_var"),
          "GP predictive mean for M test points (hot loop)");

    m.def("gp_predict_full", &gp_predict_full,
          py::arg("X_test"), py::arg("X_train"), py::arg("alpha"),
          py::arg("L_inv"), py::arg("length_scales"), py::arg("signal_var"),
          py::arg("noise_var") = 0.0,
          "GP predictive mean + std via Cholesky");

    m.def("expected_improvement", &expected_improvement,
          py::arg("mu"), py::arg("std"), py::arg("f_best"), py::arg("xi") = 0.01,
          "Expected Improvement acquisition function");

    m.def("upper_confidence_bound", &upper_confidence_bound,
          py::arg("mu"), py::arg("std"), py::arg("beta") = 2.0,
          "Upper Confidence Bound acquisition function");
}
