// recipe_optimizer_kernel.cpp — Bayesian optimization for dicing/grinding recipes
//
// AP試験 / DISCO engineer concepts:
//   - ベイズ最適化: ガウス過程 (GP) サロゲート + 獲得関数 (EI/UCB) で少ない実験で最適解探索
//     → "AP大でプロセス研究に配属→BO実装" の直結テーマ
//   - 獲得関数 EI: EI(x) = E[max(f(x) - f_best, 0)] = σ·φ(z) + (µ-f_best)·Φ(z)
//     z = (µ(x) - f_best) / σ(x)
//   - UCB: UCB(x) = µ(x) + κ·σ(x)
//   - GP カーネル: RBF k(x,x') = σ²·exp(-||x-x'||²/(2l²))
//     ハイパーパラメータ: σ²（信号分散）, l（長さスケール）, σ_n²（ノイズ）
//   - マルチ目的 BO: チッピング最小化 + スループット最大化（Pareto front）
//   - プロセス能力指数 Cp = (USL-LSL)/(6σ), Cpk = min(USL-µ,µ-LSL)/(3σ)
//
// DISCO context:
//   - AP大でプロセス研究に入った場合の最初の実装候補
//   - 砥石回転数・送り速度・切り込み深さの3パラメータ最適化
//   - 目的: Ra最小化 / スループット最大化 / チッピング抑制

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace py = pybind11;

static constexpr double PI = M_PI;

// ── LCG helpers ───────────────────────────────────────────────────────────────
static double lcg_gauss(uint64_t& s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (s >> 11) * (1.0 / (1ULL << 53));
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (s >> 11) * (1.0 / (1ULL << 53));
    u1 = std::max(u1, 1e-15);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

// ── Gaussian CDF / PDF ────────────────────────────────────────────────────────
static double phi(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}
static double phi_pdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * PI);
}

// ── RBF kernel (scalar) ───────────────────────────────────────────────────────
static double rbf(double x, double xp, double l) {
    double d = x - xp;
    return std::exp(-0.5 * d * d / (l * l));
}

// ── 1-D GP posterior (analytic, noise-free observations) ──────────────────────
// Given n observed (X, y) pairs, returns (mu, sigma) at query point x_q.
// Hyperparams: signal variance sigma2, length scale l, noise sigma2_n.
static std::pair<double, double> gp_predict(
    const std::vector<double>& X,
    const std::vector<double>& y,
    double x_q,
    double sigma2, double l, double sigma2_n
) {
    int n = static_cast<int>(X.size());
    if (n == 0) return {0.0, std::sqrt(sigma2)};

    // Build K(X,X) + σ²_n I  (n×n, row-major)
    std::vector<double> K(n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            K[i*n+j] = sigma2 * rbf(X[i], X[j], l) + (i==j ? sigma2_n : 0.0);

    // k*(X): covariance between X_obs and x_q
    std::vector<double> kstar(n);
    for (int i = 0; i < n; ++i)
        kstar[i] = sigma2 * rbf(X[i], x_q, l);

    // Cholesky  L·Lᵀ = K  (simple, no pivoting)
    std::vector<double> L(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = K[i*n+j];
            for (int k = 0; k < j; ++k) s -= L[i*n+k] * L[j*n+k];
            L[i*n+j] = (i == j) ? std::sqrt(std::max(s, 1e-12)) : s / L[j*n+j];
        }
    }

    // α = K⁻¹ y via forward/back substitution
    std::vector<double> alpha(n);
    // Forward: L·v = y
    for (int i = 0; i < n; ++i) {
        double s = y[i];
        for (int k = 0; k < i; ++k) s -= L[i*n+k] * alpha[k];
        alpha[i] = s / L[i*n+i];
    }
    // Backward: Lᵀ·α = v
    for (int i = n-1; i >= 0; --i) {
        for (int k = i+1; k < n; ++k) alpha[i] -= L[k*n+i] * alpha[k];
        alpha[i] /= L[i*n+i];
    }

    // mu* = k*ᵀ·α
    double mu = 0.0;
    for (int i = 0; i < n; ++i) mu += kstar[i] * alpha[i];

    // v = L⁻¹·k*  (for variance)
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i) {
        double s = kstar[i];
        for (int k = 0; k < i; ++k) s -= L[i*n+k] * v[k];
        v[i] = s / L[i*n+i];
    }
    double var = sigma2 - 0.0;
    for (int i = 0; i < n; ++i) var -= v[i] * v[i];
    double sigma_star = std::sqrt(std::max(var, 1e-12));

    return {mu, sigma_star};
}

// ── Expected Improvement acquisition ─────────────────────────────────────────
static double expected_improvement(double mu, double sigma, double f_best,
                                    double xi = 0.01) {
    double z = (mu - f_best - xi) / std::max(sigma, 1e-9);
    return (mu - f_best - xi) * phi(z) + sigma * phi_pdf(z);
}

// ── UCB acquisition ───────────────────────────────────────────────────────────
static double ucb(double mu, double sigma, double kappa = 2.0) {
    return mu + kappa * sigma;
}

// ── Mock objective: dicing quality score ─────────────────────────────────────
// Simulate: quality = f(spindle_krpm) = -(spindle - optimal)² / width² + noise
// True optimal depends on material.
static double mock_quality(double spindle_krpm, double noise_sigma,
                            double optimal_krpm, double width,
                            uint64_t& rng) {
    double q = -(spindle_krpm - optimal_krpm) * (spindle_krpm - optimal_krpm)
             / (width * width);
    return q + noise_sigma * lcg_gauss(rng);
}

// ── BO loop (1-D, single objective) ──────────────────────────────────────────
static py::dict bayesian_optimize_1d(
    double x_lo,          // search range low (e.g. spindle_krpm)
    double x_hi,          // search range high
    double optimal_x,     // true optimum (for the mock objective)
    int    n_init,         // initial random points
    int    n_iter,         // BO iterations
    double noise_sigma,    // observation noise
    const std::string& acq_fn,  // "EI" or "UCB"
    double gp_l,          // GP length scale
    double kappa,         // UCB kappa
    uint64_t seed
) {
    uint64_t rng = seed ? seed : 2026ULL;
    std::vector<double> X_obs, y_obs;

    // Initial random exploration
    double x_range = x_hi - x_lo;
    for (int i = 0; i < n_init; ++i) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        double x = x_lo + (rng >> 11) * (1.0 / (1ULL << 53)) * x_range;
        double width = x_range * 0.3;
        double y = mock_quality(x, noise_sigma, optimal_x, width, rng);
        X_obs.push_back(x);
        y_obs.push_back(y);
    }

    py::list iterations;

    for (int it = 0; it < n_iter; ++it) {
        double f_best = *std::max_element(y_obs.begin(), y_obs.end());

        // Grid search for best acquisition point
        int    best_idx = 0;
        double best_acq = -1e18;
        std::vector<double> grid_x, grid_mu, grid_sigma, grid_acq;
        int n_grid = 200;
        for (int g = 0; g <= n_grid; ++g) {
            double xq = x_lo + x_range * g / n_grid;
            auto [mu, sig] = gp_predict(X_obs, y_obs, xq,
                                         1.0, gp_l, noise_sigma * noise_sigma);
            double a = (acq_fn == "UCB")
                     ? ucb(mu, sig, kappa)
                     : expected_improvement(mu, sig, f_best);
            grid_x.push_back(xq);
            grid_mu.push_back(mu);
            grid_sigma.push_back(sig);
            grid_acq.push_back(a);
            if (a > best_acq) { best_acq = a; best_idx = g; }
        }

        double x_next = grid_x[best_idx];
        double width  = x_range * 0.3;
        double y_next = mock_quality(x_next, noise_sigma, optimal_x, width, rng);
        X_obs.push_back(x_next);
        y_obs.push_back(y_next);

        // Best so far
        int best_obs = static_cast<int>(
            std::max_element(y_obs.begin(), y_obs.end()) - y_obs.begin());
        py::dict entry;
        entry["iter"]      = it;
        entry["x_next"]    = x_next;
        entry["y_next"]    = y_next;
        entry["best_x"]    = X_obs[best_obs];
        entry["best_y"]    = y_obs[best_obs];
        entry["regret"]    = std::abs(X_obs[best_obs] - optimal_x);
        iterations.append(entry);
    }

    int best_obs = static_cast<int>(
        std::max_element(y_obs.begin(), y_obs.end()) - y_obs.begin());

    py::dict r;
    r["X_obs"]         = X_obs;
    r["y_obs"]         = y_obs;
    r["best_x"]        = X_obs[best_obs];
    r["best_y"]        = y_obs[best_obs];
    r["true_optimal"]  = optimal_x;
    r["final_regret"]  = std::abs(X_obs[best_obs] - optimal_x);
    r["iterations"]    = iterations;
    r["acq_fn"]        = acq_fn;
    r["converged"]     = (std::abs(X_obs[best_obs] - optimal_x) < x_range * 0.05);
    return r;
}

// ── Process capability (Cp, Cpk) ─────────────────────────────────────────────
static py::dict process_capability(
    const std::vector<double>& measurements,
    double usl,   // Upper Spec Limit
    double lsl    // Lower Spec Limit
) {
    int n = static_cast<int>(measurements.size());
    if (n < 2) return {};

    double mean = 0;
    for (double v : measurements) mean += v;
    mean /= n;

    double var = 0;
    for (double v : measurements) var += (v - mean) * (v - mean);
    double sigma = std::sqrt(var / (n - 1));

    double Cp   = (usl - lsl) / (6.0 * sigma);
    double Cpk  = std::min((usl - mean) / (3.0 * sigma),
                            (mean - lsl) / (3.0 * sigma));
    double ppm_est = 1e6 * (1.0 - (phi((usl - mean)/sigma) - phi((lsl - mean)/sigma)));

    std::string grade = (Cpk >= 1.67) ? "Six Sigma" :
                        (Cpk >= 1.33) ? "Capable (1.33)" :
                        (Cpk >= 1.00) ? "Marginal" : "Not capable";

    py::dict r;
    r["mean"]    = mean;
    r["sigma"]   = sigma;
    r["Cp"]      = Cp;
    r["Cpk"]     = Cpk;
    r["usl"]     = usl;
    r["lsl"]     = lsl;
    r["ppm_est"] = ppm_est;
    r["grade"]   = grade;
    return r;
}

// ── DISCO recipe optimization demo ───────────────────────────────────────────
// Optimize spindle speed for Si dicing quality (surface roughness proxy).
// Uses EI BO with 5 init + 15 iterations → should find optimum near 30 krpm.
static py::dict recipe_optimizer_demo() {
    auto r_ei  = bayesian_optimize_1d(10.0, 50.0, 30.0,
                                       5, 20, 0.05, "EI", 8.0, 2.0, 2026ULL);
    auto r_ucb = bayesian_optimize_1d(10.0, 50.0, 30.0,
                                       5, 20, 0.05, "UCB", 8.0, 2.0, 2026ULL);

    // Capability of best recipe
    uint64_t rng = 9999ULL;
    double best_x = r_ei["best_x"].cast<double>();
    std::vector<double> repeat_y;
    for (int i = 0; i < 30; ++i)
        repeat_y.push_back(mock_quality(best_x, 0.05, 30.0, 12.0, rng) + 0.6);
    auto cap = process_capability(repeat_y, 0.8, 0.2);

    py::dict demo;
    demo["scenario"]        = "Spindle speed optimization for Si dicing quality";
    demo["search_range"]    = "10-50 krpm";
    demo["true_optimal_krpm"] = 30.0;
    demo["ei_best_krpm"]    = r_ei["best_x"];
    demo["ucb_best_krpm"]   = r_ucb["best_x"];
    demo["ei_regret"]       = r_ei["final_regret"];
    demo["ucb_regret"]      = r_ucb["final_regret"];
    demo["ei_converged"]    = r_ei["converged"];
    demo["ucb_converged"]   = r_ucb["converged"];
    demo["best_recipe_Cpk"] = cap["Cpk"];
    demo["best_recipe_grade"] = cap["grade"];
    return demo;
}

PYBIND11_MODULE(_recipe_optimizer_kernel, m) {
    m.doc() = "Bayesian optimization for dicing/grinding recipes: GP surrogate + EI/UCB";

    m.def("bayesian_optimize_1d", &bayesian_optimize_1d,
          py::arg("x_lo")         = 10.0,
          py::arg("x_hi")         = 50.0,
          py::arg("optimal_x")    = 30.0,
          py::arg("n_init")       = 5,
          py::arg("n_iter")       = 20,
          py::arg("noise_sigma")  = 0.05,
          py::arg("acq_fn")       = "EI",
          py::arg("gp_l")         = 8.0,
          py::arg("kappa")        = 2.0,
          py::arg("seed")         = 2026ULL,
          "1-D Bayesian optimization with GP surrogate.\n"
          "acq_fn: 'EI' (Expected Improvement) or 'UCB' (Upper Confidence Bound).\n"
          "Returns: X_obs, y_obs, best_x, final_regret, iterations, converged.");

    m.def("process_capability", &process_capability,
          py::arg("measurements"),
          py::arg("usl"),
          py::arg("lsl"),
          "Compute Cp, Cpk, estimated PPM from measurement data.\n"
          "Cp = (USL-LSL)/(6σ),  Cpk = min(USL-µ, µ-LSL)/(3σ).\n"
          "Grade: Six Sigma (Cpk>=1.67) / Capable / Marginal / Not capable.");

    m.def("recipe_optimizer_demo", &recipe_optimizer_demo,
          "DISCO demo: optimize spindle speed (10-50 krpm) for Si dicing.\n"
          "EI vs UCB comparison; capability analysis at best recipe.\n"
          "Returns: best krpm per method, regret, convergence, Cpk grade.");
}
