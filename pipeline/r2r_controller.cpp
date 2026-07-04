// r2r_controller.cpp — Run-to-Run (R2R) APC controller
//
// AP試験 / DISCO engineer concepts:
//   - EWMA R2R: 指数重み付き移動平均でプロセスドリフトを推定・補正
//       d̂[n] = λ(y[n] - u[n-1]) + (1-λ)d̂[n-1]
//       u[n] = target - d̂[n]
//   - DEWMA R2R: 二重 EWMA でリニアドリフト（ブレード摩耗）に対応
//       Z₁[n] = λ·y[n] + (1-λ)Z₁[n-1]
//       Z₂[n] = λ·Z₁[n] + (1-λ)Z₂[n-1]
//       level a[n] = 2Z₁ - Z₂,  trend T[n] = λ/(1-λ)·(Z₁-Z₂)
//       u[n] = target - (a[n] + T[n])
//   - EWMA のチューニング: λ 小 → 安定・追従遅い; λ 大 → 雑音増幅
//   - R2R APC の意義: ブレード摩耗・環境変動→カーフ幅ドリフト補正
//
// DISCO context:
//   - ダイシング後カーフ幅を計測 → 次ランの切り込み量を自動補正
//   - KEYENCE 自動計測 + APC フィードバックで歩留まり向上
//   - SiC 基板: ブレード摩耗によるドリフト速度が Si の 2× → DEWMA が有効

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace py = pybind11;

// ── Pseudo-RNG (LCG — no std::mt19937 state captures in pybind dict) ─────────
static double lcg_randn(uint64_t &state) {
    // Box-Muller from two LCG draws
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (state >> 11) * (1.0 / (1ULL << 53));
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (state >> 11) * (1.0 / (1ULL << 53));
    u1 = std::max(u1, 1e-15);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

// ── Statistics helpers ────────────────────────────────────────────────────────
static double mean_vec(const std::vector<double> &v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

static double rms_vec(const std::vector<double> &v) {
    if (v.empty()) return 0.0;
    const double s = std::accumulate(v.begin(), v.end(), 0.0,
        [](double acc, double x) { return acc + x * x; });
    return std::sqrt(s / v.size());
}

// ============================================================================
// EWMA R2R controller
//   target_um     : desired output (e.g. kerf width in µm)
//   n_runs        : total number of wafer runs
//   lambda_ewma   : EWMA smoothing factor (0 < λ ≤ 1)
//   sigma_noise   : per-run measurement noise std dev (µm)
//   drift_um_run  : systematic drift per run (e.g. blade wear → kerf narrowing)
//   seed          : RNG seed
//
// Returns dict with run-by-run traces and summary statistics.
// ============================================================================
static py::dict ewma_r2r(double target_um, int n_runs, double lambda_ewma,
                          double sigma_noise_um, double drift_um_per_run,
                          uint64_t seed) {
    std::vector<double> y_out, u_ctrl, d_est, error;
    double d_hat = 0.0;   // EWMA disturbance estimate
    double u     = 0.0;   // controller output (offset from nominal)
    uint64_t rng = seed ? seed : 42ULL;

    for (int n = 0; n < n_runs; ++n) {
        // True output: nominal + drift + noise + controller correction
        double true_drift = drift_um_per_run * (n + 1);
        double noise      = sigma_noise_um * lcg_randn(rng);
        double y          = target_um + true_drift + noise + u;   // u is correction

        // EWMA disturbance update: d̂[n] = λ(y[n] - u[n-1]) + (1-λ)d̂[n-1]
        double measured_disturbance = y - u - target_um;  // y - u gives raw process output
        d_hat = lambda_ewma * measured_disturbance + (1.0 - lambda_ewma) * d_hat;

        // Controller: u[n] = target - d̂[n]   (negative feedback)
        u = -d_hat;  // correction to apply next run

        y_out.push_back(y);
        u_ctrl.push_back(u);
        d_est.push_back(d_hat);
        error.push_back(y - target_um);
    }

    std::vector<double> abs_err(error.size());
    for (size_t i = 0; i < error.size(); ++i) abs_err[i] = std::abs(error[i]);

    py::dict r;
    r["controller"]    = "EWMA";
    r["lambda"]        = lambda_ewma;
    r["target_um"]     = target_um;
    r["y_out"]         = y_out;
    r["u_ctrl"]        = u_ctrl;
    r["d_est"]         = d_est;
    r["error"]         = error;
    r["rmse_um"]       = rms_vec(error);
    r["mean_error_um"] = mean_vec(error);
    r["max_abs_err_um"]= *std::max_element(abs_err.begin(), abs_err.end());
    r["final_error_um"]= error.back();
    r["converged"]     = (std::abs(error.back()) < 1.0);   // within 1 µm
    return r;
}

// ============================================================================
// Double-EWMA (DEWMA) R2R controller
//   Handles linear drift (e.g. progressive blade wear).
//   Z₁[n] = λ·y[n] + (1-λ)Z₁[n-1]   (level smoother)
//   Z₂[n] = λ·Z₁[n] + (1-λ)Z₂[n-1]  (trend smoother)
//   level  a[n] = 2Z₁[n] - Z₂[n]
//   trend  T[n] = λ/(1-λ) · (Z₁[n] - Z₂[n])
//   control output u[n] = target - (a[n] + T[n])
// ============================================================================
static py::dict dewma_r2r(double target_um, int n_runs, double lambda,
                           double sigma_noise_um, double drift_um_per_run,
                           uint64_t seed) {
    std::vector<double> y_out, u_ctrl, d_level, d_trend, error;
    double Z1 = 0.0, Z2 = 0.0;
    double u   = 0.0;
    uint64_t rng = seed ? seed : 42ULL;
    double lam_ratio = lambda / std::max(1.0 - lambda, 1e-9);

    for (int n = 0; n < n_runs; ++n) {
        double true_drift = drift_um_per_run * (n + 1);
        double noise      = sigma_noise_um * lcg_randn(rng);
        double y          = target_um + true_drift + noise + u;

        // Detrend observation: y_raw = y - u - target
        double y_raw = y - u - target_um;

        Z1 = lambda * y_raw + (1.0 - lambda) * Z1;
        Z2 = lambda * Z1   + (1.0 - lambda) * Z2;

        double a  = 2.0 * Z1 - Z2;          // level estimate
        double T  = lam_ratio * (Z1 - Z2);   // trend (slope) estimate
        u = -(a + T);                         // feedforward + feedback

        y_out.push_back(y);
        u_ctrl.push_back(u);
        d_level.push_back(a);
        d_trend.push_back(T);
        error.push_back(y - target_um);
    }

    std::vector<double> abs_err(error.size());
    for (size_t i = 0; i < error.size(); ++i) abs_err[i] = std::abs(error[i]);

    py::dict r;
    r["controller"]    = "DEWMA";
    r["lambda"]        = lambda;
    r["target_um"]     = target_um;
    r["y_out"]         = y_out;
    r["u_ctrl"]        = u_ctrl;
    r["d_level"]       = d_level;
    r["d_trend"]       = d_trend;
    r["error"]         = error;
    r["rmse_um"]       = rms_vec(error);
    r["mean_error_um"] = mean_vec(error);
    r["max_abs_err_um"]= *std::max_element(abs_err.begin(), abs_err.end());
    r["final_error_um"]= error.back();
    r["converged"]     = (std::abs(error.back()) < 1.0);
    return r;
}

// ============================================================================
// Lambda sensitivity sweep
//   Sweep lambda from 0.05 to 0.95 and return RMSE for each.
//   Useful for finding optimal lambda for a given drift/noise ratio.
// ============================================================================
static py::list lambda_sweep(double target_um, int n_runs,
                              double sigma_noise_um, double drift_um_per_run,
                              const std::string &controller_type,
                              uint64_t seed) {
    py::list results;
    for (int i = 1; i <= 19; ++i) {
        double lam = i * 0.05;
        py::dict r;
        if (controller_type == "DEWMA") {
            r = dewma_r2r(target_um, n_runs, lam, sigma_noise_um, drift_um_per_run, seed);
        } else {
            r = ewma_r2r(target_um, n_runs, lam, sigma_noise_um, drift_um_per_run, seed);
        }
        py::dict entry;
        entry["lambda"]  = lam;
        entry["rmse_um"] = r["rmse_um"];
        entry["converged"] = r["converged"];
        results.append(entry);
    }
    return results;
}

// ============================================================================
// DISCO demo: kerf width R2R control scenario
//   Simulates a 200-wafer SiC dicing lot with blade wear drift.
//   Compares: no control, EWMA, DEWMA.
// ============================================================================
static py::dict r2r_demo() {
    double target_um      = 55.0;   // target kerf width (µm) — DISCO DFL7160 standard
    int    n_runs         = 100;    // wafers per blade
    double sigma_noise    = 0.50;   // KEYENCE measurement noise (µm std dev)
    double drift_per_run  = 0.20;   // blade wear → kerf narrowing (µm/wafer), SiC-grade
    uint64_t seed         = 2026ULL;

    // No control: raw process
    std::vector<double> y_raw;
    uint64_t rng0 = seed;
    for (int n = 0; n < n_runs; ++n) {
        double y = target_um + drift_per_run * (n + 1)
                 + sigma_noise * lcg_randn(rng0);
        y_raw.push_back(y);
    }
    std::vector<double> err0(y_raw.size());
    for (size_t i = 0; i < y_raw.size(); ++i) err0[i] = y_raw[i] - target_um;

    auto r_ewma  = ewma_r2r (target_um, n_runs, 0.3, sigma_noise, drift_per_run, seed);
    auto r_dewma = dewma_r2r(target_um, n_runs, 0.3, sigma_noise, drift_per_run, seed);

    py::dict demo;
    demo["scenario"]         = "SiC 200-wafer lot: blade wear drift correction";
    demo["target_um"]        = target_um;
    demo["drift_um_per_run"] = drift_per_run;
    demo["sigma_noise_um"]   = sigma_noise;
    demo["n_runs"]           = n_runs;
    demo["no_control_rmse"]  = rms_vec(err0);
    demo["ewma_rmse"]        = r_ewma["rmse_um"];
    demo["dewma_rmse"]       = r_dewma["rmse_um"];
    demo["ewma_converged"]   = r_ewma["converged"];
    demo["dewma_converged"]  = r_dewma["converged"];
    demo["ewma_final_err"]   = r_ewma["final_error_um"];
    demo["dewma_final_err"]  = r_dewma["final_error_um"];
    // DEWMA advantage: handles linear drift → lower RMSE at end-of-blade-life
    double ewma_rmse  = r_ewma["rmse_um"].cast<double>();
    double dewma_rmse = r_dewma["rmse_um"].cast<double>();
    demo["dewma_improvement_pct"] = (ewma_rmse - dewma_rmse) / ewma_rmse * 100.0;
    demo["y_no_ctrl"]    = y_raw;
    demo["y_ewma"]       = r_ewma["y_out"];
    demo["y_dewma"]      = r_dewma["y_out"];
    return demo;
}

// ============================================================================
// PYBIND11 module
// ============================================================================
PYBIND11_MODULE(_r2r_controller_kernel, m) {
    m.doc() = "Run-to-Run (R2R) APC: EWMA + DEWMA controllers for DISCO dicing";

    m.def("ewma_r2r", &ewma_r2r,
          py::arg("target_um")         = 55.0,
          py::arg("n_runs")            = 100,
          py::arg("lambda_ewma")       = 0.3,
          py::arg("sigma_noise_um")    = 0.8,
          py::arg("drift_um_per_run")  = 0.05,
          py::arg("seed")              = 42ULL,
          "EWMA R2R APC controller.\n"
          "d̂[n] = λ(y[n]-u[n-1]) + (1-λ)d̂[n-1];  u[n] = target - d̂[n].\n"
          "Returns: y_out, u_ctrl, d_est, error, rmse_um, converged.");

    m.def("dewma_r2r", &dewma_r2r,
          py::arg("target_um")         = 55.0,
          py::arg("n_runs")            = 100,
          py::arg("lambda_dewma")      = 0.3,
          py::arg("sigma_noise_um")    = 0.8,
          py::arg("drift_um_per_run")  = 0.05,
          py::arg("seed")              = 42ULL,
          "Double-EWMA R2R APC: handles linear drift (blade wear).\n"
          "Z1/Z2 double smoothing; level=2Z1-Z2; trend=λ/(1-λ)·(Z1-Z2).\n"
          "Returns: y_out, u_ctrl, d_level, d_trend, error, rmse_um, converged.");

    m.def("lambda_sweep", &lambda_sweep,
          py::arg("target_um")         = 55.0,
          py::arg("n_runs")            = 100,
          py::arg("sigma_noise_um")    = 0.8,
          py::arg("drift_um_per_run")  = 0.05,
          py::arg("controller_type")   = "EWMA",
          py::arg("seed")              = 42ULL,
          "Sweep lambda 0.05→0.95 and return RMSE for each. "
          "Use to find optimal lambda for given drift/noise ratio.");

    m.def("r2r_demo", &r2r_demo,
          "DISCO demo: 100-wafer SiC lot with blade wear drift.\n"
          "Compares no-control vs EWMA vs DEWMA. Returns kerf traces + RMSE comparison.");
}
