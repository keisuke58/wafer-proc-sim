// grind_demo.cpp — end-to-end back-grind demo: thickness metrology, surface
// (grinding-mark / SSD) inspection, infeed APC, wheel health, and economics.
//
// Synthesizes repeatable data (no external inputs), runs the full wproc::grind
// stack, writes two color figures (thickness map, SSD heat map), and prints all
// metrics as a JSON blob on stdout so a dashboard can embed real tool output.
//
// Usage:
//   grind_demo [--thickness out1.ppm] [--ssd out2.ppm] [--seed N]
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cli_args.hpp"
#include "wproc/grind_control.hpp"
#include "wproc/grind_metrology.hpp"
#include "wproc/grind_surface.hpp"
#include "wproc/image.hpp"
#include "wproc/io.hpp"

using namespace wproc;
using wproc_cli::arg_double;

namespace {

constexpr double kPi = 3.14159265358979323846;

// Synthetic thinned-wafer thickness map: nominal target with a stage tilt, a
// central bow, an edge roll-off ring, and measurement noise. Cells outside the
// wafer disk are left at 0 (no-data).
Image make_thickness_map(int n, double target_um, unsigned seed) {
    Image img(n, n, 0.0f);
    const double cx = (n - 1) * 0.5, cy = (n - 1) * 0.5;
    const double R = n * 0.5 * 0.98, R2 = R * R;
    std::mt19937 gen(seed);
    std::normal_distribution<double> noise(0.0, 0.15);
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
            const double dx = x - cx, dy = y - cy, d2 = dx * dx + dy * dy;
            if (d2 > R2) continue;
            const double rr = d2 / R2;
            const double tilt = 0.02 * dx;               // stage tilt
            const double bow = -1.2 * (1.0 - rr);        // centre thinner
            const double edge = 0.9 * std::pow(rr, 6.0); // edge roll-off (thicker)
            img.at(x, y) =
                static_cast<float>(target_um + tilt + bow + edge + noise(gen));
        }
    return img;
}

// Synthetic ground back-surface: oriented grinding marks plus a rougher damaged
// patch, on a mid-gray field with sensor noise.
Image make_surface(int n, unsigned seed) {
    Image img(n, n, 128.0f);
    std::mt19937 gen(seed);
    std::normal_distribution<double> noise(0.0, 3.0);
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
            double v = 128.0 + 18.0 * std::sin(2 * kPi * y / 7.0);  // marks (horizontal)
            // A rougher subsurface-damage patch in the lower-right quadrant.
            if (x > n * 0.6 && y > n * 0.6)
                v += 26.0 * std::sin(2 * kPi * (x + y) / 3.0);
            img.at(x, y) = static_cast<float>(v + noise(gen));
        }
    return img;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string thickness_out =
        wproc_cli::arg_str(argc, argv, "--thickness", "");
    const std::string ssd_out = wproc_cli::arg_str(argc, argv, "--ssd", "");
    const unsigned seed =
        static_cast<unsigned>(arg_double(argc, argv, "--seed", 2024));

    const double target_um = 100.0;

    // 1) Thickness metrology.
    const Image tmap = make_thickness_map(256, target_um, seed);
    const grind::ThicknessStats ts = grind::analyze_thickness(tmap);
    if (!thickness_out.empty())
        write_ppm(thickness_out, grind::render_thickness(tmap));

    // 2) Surface (grinding-mark / SSD) inspection.
    const Image surf = make_surface(256, seed + 1);
    const grind::SurfaceQuality sq = grind::analyze_surface(surf);
    if (!ssd_out.empty())
        write_ppm(ssd_out, grind::render_ssd_heatmap(surf));

    // 3) Infeed APC vs open loop over a wheel-dulling run.
    grind::GrindPlant plant;
    plant.offset0_um = 780.0;
    plant.gain_um_per_infeed = 100.0;
    plant.dull_um_per_wafer = 0.15;
    plant.force_gain = 2.0;
    plant.force_dull_per_wafer = 0.05;
    plant.noise_sigma = 0.4;

    const double nominal =
        (plant.offset0_um - target_um) / plant.gain_um_per_infeed;
    const int N = 300;
    std::mt19937 gen(seed + 2);
    std::normal_distribution<double> pn(0.0, plant.noise_sigma);

    grind::GrindParams gp;
    gp.target_um = target_um;
    gp.gain = 100.0;
    gp.lambda = 0.3;
    gp.infeed0 = nominal;
    grind::InfeedController ctrl(gp);
    grind::WheelMonitor wheel;

    std::vector<double> open_t, cl_t, force_series;
    double infeed = ctrl.current_infeed();
    double sse_open = 0.0, sse_cl = 0.0;
    for (long k = 1; k <= N; ++k) {
        const double ot = plant.final_thickness(nominal, k, pn(gen));
        const double ct = plant.final_thickness(infeed, k, pn(gen));
        open_t.push_back(ot);
        cl_t.push_back(ct);
        sse_open += (ot - target_um) * (ot - target_um);
        sse_cl += (ct - target_um) * (ct - target_um);
        const double removed = plant.gain_um_per_infeed * infeed;
        const double f = plant.force(infeed, k);
        force_series.push_back(f);
        wheel.add(k, removed, f);
        infeed = ctrl.update(ct);
    }
    const double open_rms = std::sqrt(sse_open / N);
    const double cl_rms = std::sqrt(sse_cl / N);
    const grind::WheelHealth wh = wheel.health(/*force_dress_limit=*/30.0);

    // 4) Economics (yield reflects the closed-loop quality above).
    grind::EconInputs ei;
    ei.wafers_per_hour = 22.0;
    ei.dies_per_wafer = 1200.0;
    ei.yield_pct = 96.5;
    const grind::EconSummary es = grind::economics(ei);

    // JSON blob (real tool output for the dashboard).
    std::cout.setf(std::ios::fixed);
    std::cout.precision(4);
    std::cout << "{\n"
              << "  \"thickness\": {\"target_um\": " << target_um
              << ", \"mean_um\": " << ts.mean_um << ", \"ttv_um\": " << ts.ttv_um
              << ", \"warp_um\": " << ts.warp_um << ", \"bow_um\": " << ts.bow_um
              << ", \"sigma_um\": " << ts.sigma_um
              << ", \"uniformity_pct\": " << ts.uniformity_pct
              << ", \"n_inside\": " << ts.n_inside << "},\n"
              << "  \"surface\": {\"coherence\": " << sq.coherence
              << ", \"mark_angle_deg\": " << sq.mark_angle_deg
              << ", \"mark_severity\": " << sq.mark_severity
              << ", \"roughness_ra\": " << sq.roughness_ra
              << ", \"ssd_index\": " << sq.ssd_index
              << ", \"ok\": " << (sq.ok ? "true" : "false") << "},\n"
              << "  \"apc\": {\"target_um\": " << target_um
              << ", \"open_rms_um\": " << open_rms
              << ", \"closed_rms_um\": " << cl_rms
              << ", \"final_infeed\": " << ctrl.current_infeed()
              << ", \"nominal_infeed\": " << nominal << "},\n"
              << "  \"wheel\": {\"current_force_n\": " << wh.current_force
              << ", \"force_slope_per_wafer\": " << wh.force_slope_per_wafer
              << ", \"rul_wafers\": " << wh.rul_wafers
              << ", \"g_ratio\": " << wh.g_ratio
              << ", \"dress_now\": " << (wh.dress_now ? "true" : "false") << "},\n"
              << "  \"econ\": {\"throughput_wph\": " << es.throughput_wph
              << ", \"good_dies_per_hour\": " << es.good_dies_per_hour
              << ", \"cost_per_wafer\": " << es.cost_per_wafer
              << ", \"cost_per_good_die\": " << es.cost_per_good_die << "},\n";

    // Per-wafer series (real run data) so a dashboard can plot convergence.
    auto emit_series = [](const char* name, const std::vector<double>& v,
                          int prec) {
        std::cout << "  \"" << name << "\": [";
        std::cout.precision(prec);
        for (std::size_t i = 0; i < v.size(); ++i)
            std::cout << (i ? "," : "") << v[i];
        std::cout << "]";
    };
    std::cout << "  \"series\": {";
    emit_series("open_thickness_um", open_t, 3);
    std::cout << ",";
    emit_series("closed_thickness_um", cl_t, 3);
    std::cout << ",";
    emit_series("force_n", force_series, 3);
    std::cout << "}\n}\n";
    return 0;
}
