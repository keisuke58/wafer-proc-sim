// in_situ_monitor_kernel.cpp — In-process sensor fusion & feedback
//
// AP試験 / DISCO engineer concepts:
//   - AE（アコースティックエミッション）センサー: 加工中の微細亀裂・チッピングをリアルタイム検出
//     AE RMS ∝ 切削力 × 材料係数; チッピング時にバースト成分が急増
//   - スピンドル電流センサー: I_spindle ∝ P_cut / V_spindle
//     電流急増 = ブレード磨耗・目詰まりの前兆 → 先手交換判断
//   - カーフ幅インプロセス計測: ビジョン画像 → カーフ幅 [µm] → R2R フィードバック
//   - マルチセンサー異常スコア: 各センサーの z-score 合算 → 確信度付き異常判定
//   - カルマンフィルタによるドリフト推定: x̂[n] = A·x̂[n-1] + K(y[n] - C·x̂[n-1])
//     プロセスドリフト（ブレード摩耗・温度変化）をリアルタイム追跡
//
// DISCO context:
//   - KEYENCE センサー内蔵ダイサー: カーフ幅を全チップ計測してフィードバック
//   - スピンドル電流モニタリング: ブレード交換タイミングを最適化
//   - AE センサー: SiC のような脆性材でチッピング予兆検出

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace py = pybind11;

// ── LCG RNG ───────────────────────────────────────────────────────────────────
static double lcg_gauss(uint64_t& s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (s >> 11) * (1.0 / (1ULL << 53));
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (s >> 11) * (1.0 / (1ULL << 53));
    u1 = std::max(u1, 1e-15);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

static double lcg_uniform(uint64_t& s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (s >> 11) * (1.0 / (1ULL << 53));
}

// ── AE sensor simulation ──────────────────────────────────────────────────────
// Returns: AE RMS trace, chipping event indices, detection rate
static py::dict ae_sensor_sim(
    double blade_depth_um,
    double feed_mm_s,
    int    n_chips,             // number of die cuts
    double chipping_rate,       // fraction of cuts with chipping (0–1)
    double ae_threshold,        // AE RMS threshold for alarm [a.u.]
    uint64_t seed
) {
    uint64_t rng = seed ? seed : 42ULL;
    std::vector<double> ae_rms;
    std::vector<int>    chip_events;
    int true_chips = 0, detected = 0, false_alarms = 0;

    for (int i = 0; i < n_chips; ++i) {
        // Baseline AE: proportional to feed rate × depth
        double ae_base = 0.3 * (blade_depth_um / 30.0) * (feed_mm_s / 10.0)
                       + 0.05 * std::abs(lcg_gauss(rng));

        bool has_chip = (lcg_uniform(rng) < chipping_rate);
        if (has_chip) {
            ++true_chips;
            chip_events.push_back(i);
            // Burst component at chipping event
            ae_base += 1.5 + 0.3 * std::abs(lcg_gauss(rng));
        }
        ae_rms.push_back(ae_base);

        bool alarm = (ae_base >= ae_threshold);
        if (alarm && has_chip)  ++detected;
        if (alarm && !has_chip) ++false_alarms;
    }

    int missed = true_chips - detected;
    double det_rate = true_chips > 0 ? double(detected) / true_chips : 1.0;
    double fp_rate  = (n_chips - true_chips) > 0
                    ? double(false_alarms) / (n_chips - true_chips) : 0.0;

    py::dict r;
    r["ae_rms"]         = ae_rms;
    r["chip_events"]    = chip_events;
    r["n_true_chips"]   = true_chips;
    r["n_detected"]     = detected;
    r["n_missed"]       = missed;
    r["n_false_alarms"] = false_alarms;
    r["detection_rate"] = det_rate;
    r["false_alarm_rate"] = fp_rate;
    r["threshold"]      = ae_threshold;
    return r;
}

// ── Spindle current monitor ───────────────────────────────────────────────────
// Model: I[n] = I_base × (1 + wear_factor × n/n_total) + noise
// Alarm at I > threshold → blade exchange recommendation
static py::dict spindle_current_monitor(
    double I_base_A,            // nominal current at fresh blade
    int    n_runs,              // total dicing runs in this blade life
    double wear_factor,         // current increase from fresh to worn (fraction)
    double sigma_noise_A,       // measurement noise
    double alarm_threshold_A,   // current alarm level
    uint64_t seed
) {
    uint64_t rng = seed ? seed : 7ULL;
    std::vector<double> current_trace;
    int alarm_run = -1;

    for (int i = 0; i < n_runs; ++i) {
        double I = I_base_A * (1.0 + wear_factor * double(i) / n_runs)
                 + sigma_noise_A * lcg_gauss(rng);
        current_trace.push_back(I);
        if (alarm_run < 0 && I >= alarm_threshold_A)
            alarm_run = i;
    }

    double I_final = current_trace.back();
    double wear_pct = (I_final - I_base_A) / (I_base_A * wear_factor) * 100.0;

    py::dict r;
    r["current_trace"]      = current_trace;
    r["alarm_run"]          = alarm_run;
    r["alarm_triggered"]    = (alarm_run >= 0);
    r["I_base_A"]           = I_base_A;
    r["I_final_A"]          = I_final;
    r["alarm_threshold_A"]  = alarm_threshold_A;
    r["blade_wear_pct_at_alarm"] = (alarm_run >= 0)
        ? wear_factor * double(alarm_run) / n_runs * 100.0 : 100.0;
    r["remaining_life_pct"] = (alarm_run >= 0)
        ? (1.0 - double(alarm_run) / n_runs) * 100.0 : 0.0;
    return r;
}

// ── Kalman filter for kerf-width drift estimation ─────────────────────────────
// State: x = [kerf_offset_um, drift_rate_um_per_run]
// Transition: x[n] = A·x[n-1] + w  (w ~ N(0, Q))
// Observation: y[n] = C·x[n] + v   (v ~ N(0, R))
static py::dict kalman_kerf_filter(
    double target_um,
    int    n_runs,
    double drift_um_per_run,    // true process drift
    double sigma_process,       // process noise std (µm)
    double sigma_meas,          // measurement noise std (µm)
    uint64_t seed
) {
    uint64_t rng = seed ? seed : 99ULL;

    // State: x = [offset, drift]
    double x0 = 0.0, x1 = 0.0;
    // Covariance P (2×2 as 4 scalars: P00, P01, P10, P11)
    double P00 = 1.0, P01 = 0.0, P10 = 0.0, P11 = 0.01;
    // Process noise Q
    double Q00 = sigma_process*sigma_process, Q11 = 1e-4;
    // Measurement noise R
    double R_meas = sigma_meas * sigma_meas;

    std::vector<double> y_meas, x_est_offset, x_est_drift, innov;

    for (int n = 0; n < n_runs; ++n) {
        // Predict
        double x0p = x0 + x1;     // offset += drift estimate
        double x1p = x1;
        double P00p = P00 + P01 + P10 + P11 + Q00;
        double P01p = P01 + P11;
        double P10p = P10 + P11;
        double P11p = P11 + Q11;

        // True measurement: target + true_drift*n + meas_noise
        double y = target_um + drift_um_per_run * (n + 1)
                 + sigma_meas * lcg_gauss(rng);
        y_meas.push_back(y);

        // Innovation: y - C·x_pred  (C = [1, 0])
        double innov_val = y - target_um - x0p;
        innov.push_back(innov_val);

        // Kalman gain K = P·Cᵀ / (C·P·Cᵀ + R)
        double S = P00p + R_meas;
        double K0 = P00p / S;
        double K1 = P10p / S;

        // Update
        x0 = x0p + K0 * innov_val;
        x1 = x1p + K1 * innov_val;
        P00 = (1 - K0) * P00p;
        P01 = (1 - K0) * P01p;
        P10 = P10p - K1 * P00p;
        P11 = P11p - K1 * P01p;

        x_est_offset.push_back(x0);
        x_est_drift.push_back(x1);
    }

    // Drift estimate error at end
    double final_drift_est = x_est_drift.back();
    double drift_error_pct = std::abs(final_drift_est - drift_um_per_run)
                           / std::abs(drift_um_per_run + 1e-9) * 100.0;

    py::dict r;
    r["y_meas"]           = y_meas;
    r["x_est_offset"]     = x_est_offset;
    r["x_est_drift"]      = x_est_drift;
    r["innovation"]       = innov;
    r["true_drift"]       = drift_um_per_run;
    r["estimated_drift"]  = final_drift_est;
    r["drift_error_pct"]  = drift_error_pct;
    r["converged"]        = (drift_error_pct < 20.0);
    return r;
}

// ── Multi-sensor fusion: anomaly score ────────────────────────────────────────
// Fuse AE, spindle current, kerf-width into a single anomaly score per run.
// Score = weighted sum of z-scores; alarm if score > threshold.
static py::dict sensor_fusion_demo(int n_runs, double fault_run, uint64_t seed) {
    uint64_t rng = seed ? seed : 314ULL;
    std::vector<double> scores;
    std::vector<bool>   alarms;
    int true_alarms = 0, correct = 0;

    // Running stats for z-score normalization (Welford)
    double ae_mean = 0, ae_M2 = 0;
    double ci_mean = 0, ci_M2 = 0;
    double kw_mean = 0, kw_M2 = 0;

    auto welford_update = [](double x, double& mean, double& M2, int n) {
        double delta = x - mean;
        mean += delta / n;
        M2   += delta * (x - mean);
    };

    for (int n = 1; n <= n_runs; ++n) {
        bool fault = (n >= static_cast<int>(fault_run));

        double ae = 0.4 + 0.05 * std::abs(lcg_gauss(rng)) + (fault ? 1.2 : 0.0);
        double ci = 3.5 + 0.1  * lcg_gauss(rng)           + (fault ? 0.8 : 0.0);
        double kw = 55. + 0.8  * lcg_gauss(rng)            + (fault ? 2.5 : 0.0);

        welford_update(ae, ae_mean, ae_M2, n);
        welford_update(ci, ci_mean, ci_M2, n);
        welford_update(kw, kw_mean, kw_M2, n);

        double ae_std = (n > 2) ? std::sqrt(ae_M2 / (n - 1)) : 1.0;
        double ci_std = (n > 2) ? std::sqrt(ci_M2 / (n - 1)) : 1.0;
        double kw_std = (n > 2) ? std::sqrt(kw_M2 / (n - 1)) : 1.0;

        double z_ae = std::abs((ae - ae_mean) / std::max(ae_std, 1e-9));
        double z_ci = std::abs((ci - ci_mean) / std::max(ci_std, 1e-9));
        double z_kw = std::abs((kw - kw_mean) / std::max(kw_std, 1e-9));

        double score = 0.5 * z_ae + 0.3 * z_ci + 0.2 * z_kw;
        bool alarm   = (score > 2.5);

        scores.push_back(score);
        alarms.push_back(alarm);
        if (fault)  ++true_alarms;
        if (alarm && fault) ++correct;
    }

    double precision = true_alarms > 0 ? double(correct) / true_alarms : 1.0;

    py::dict r;
    r["anomaly_scores"]    = scores;
    r["alarms"]            = alarms;
    r["fault_start_run"]   = static_cast<int>(fault_run);
    r["n_true_alarm_runs"] = true_alarms;
    r["n_correct_alarms"]  = correct;
    r["detection_precision"] = precision;
    r["threshold"]         = 2.5;
    return r;
}

PYBIND11_MODULE(_in_situ_monitor_kernel, m) {
    m.doc() = "In-situ sensor fusion: AE, spindle current, Kalman kerf drift, multi-sensor anomaly";

    m.def("ae_sensor_sim", &ae_sensor_sim,
          py::arg("blade_depth_um")   = 30.0,
          py::arg("feed_mm_s")        = 10.0,
          py::arg("n_chips")          = 200,
          py::arg("chipping_rate")    = 0.05,
          py::arg("ae_threshold")     = 1.0,
          py::arg("seed")             = 42ULL,
          "Simulate AE sensor during dicing. Chipping events cause burst in AE RMS.\n"
          "Returns: ae_rms trace, chip event indices, detection/false-alarm rates.");

    m.def("spindle_current_monitor", &spindle_current_monitor,
          py::arg("I_base_A")           = 3.5,
          py::arg("n_runs")             = 200,
          py::arg("wear_factor")        = 0.4,
          py::arg("sigma_noise_A")      = 0.08,
          py::arg("alarm_threshold_A")  = 4.5,
          py::arg("seed")               = 7ULL,
          "Simulate spindle current as blade wears. Alarm triggers before catastrophic failure.\n"
          "Returns: current trace, alarm run, remaining life % at alarm.");

    m.def("kalman_kerf_filter", &kalman_kerf_filter,
          py::arg("target_um")          = 55.0,
          py::arg("n_runs")             = 100,
          py::arg("drift_um_per_run")   = 0.05,
          py::arg("sigma_process")      = 0.2,
          py::arg("sigma_meas")         = 0.8,
          py::arg("seed")               = 99ULL,
          "2-state Kalman filter tracking kerf-width offset + drift rate.\n"
          "Returns: measurements, state estimates, drift error, convergence flag.");

    m.def("sensor_fusion_demo", &sensor_fusion_demo,
          py::arg("n_runs")    = 150,
          py::arg("fault_run") = 100.0,
          py::arg("seed")      = 314ULL,
          "Multi-sensor fusion (AE + spindle current + kerf width) -> anomaly score.\n"
          "Weighted z-score combination; alarm above threshold 2.5.\n"
          "Returns: anomaly scores, alarms, detection precision.");
}
