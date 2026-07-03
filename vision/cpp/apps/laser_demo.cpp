// laser_demo.cpp — end-to-end laser-processing demo: ablation-groove inspection,
// stealth SD-layer / HAZ analysis, pulse-energy APC, optics health, and KABRA
// SiC-ingot slicing economics.
//
// Synthesizes repeatable data, runs the full wproc::laser stack, writes two
// color figures (annotated groove, annotated SD cross-section), and prints all
// metrics + per-shot series as JSON so a dashboard can embed real tool output.
//
// Usage:
//   laser_demo [--groove out1.ppm] [--sd out2.ppm] [--seed N]
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cli_args.hpp"
#include "wproc/image.hpp"
#include "wproc/io.hpp"
#include "wproc/laser_control.hpp"
#include "wproc/laser_groove.hpp"
#include "wproc/laser_kabra.hpp"
#include "wproc/laser_sd.hpp"

using namespace wproc;
using wproc_cli::arg_double;

namespace {

// Top-view ablation groove: dark floor, HAZ band, bright substrate, debris.
Image make_groove(int w, int h, unsigned seed) {
    Image img(w, h, 150.0f);
    const int c = w / 2, half = 20, haz = 8;  // 20µm wide, 4µm/side HAZ @0.5µm/px
    std::mt19937 gen(seed);
    std::normal_distribution<double> noise(0.0, 3.0);
    std::uniform_int_distribution<int> ux(0, w - 1), uy(0, h - 1);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const int d = std::abs(x - c);
            float v = 150.0f;
            if (d < half) v = 40.0f;
            else if (d < half + haz) v = 95.0f;
            img.at(x, y) = static_cast<float>(v + noise(gen));
        }
    for (int i = 0; i < 40; ++i) {  // recast/spatter specks on the substrate
        int x = ux(gen), y = uy(gen);
        if (std::abs(x - c) >= half + haz + 3) img.at(x, y) = 235.0f;
    }
    return img;
}

// Cross-section IR image: evenly-pitched bright SD layers on a dark wafer.
Image make_sd(int w, int h, unsigned seed) {
    Image img(w, h, 30.0f);
    const int rows[3] = {40, 60, 80};
    std::mt19937 gen(seed);
    std::normal_distribution<double> noise(0.0, 4.0);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float v = 30.0f;
            for (int r : rows)
                if (y >= r - 2 && y <= r + 2) v = 220.0f;
            img.at(x, y) = static_cast<float>(v + noise(gen));
        }
    return img;
}

// Grayscale base + green horizontal SD-layer lines (annotated cross-section).
ColorImage annotate_sd(const Image& img, const laser::SdQuality& q,
                       const stealth::SdResult& sd) {
    const int w = img.width(), h = img.height();
    ColorImage out(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const int v = static_cast<int>(std::lround(std::clamp(img.at(x, y), 0.0f, 255.0f)));
            out.set(x, y, Rgb{static_cast<std::uint8_t>(v), static_cast<std::uint8_t>(v),
                              static_cast<std::uint8_t>(v)});
        }
    for (double dep : sd.layer_depths_px) {
        const int y = static_cast<int>(std::lround(dep));
        for (int x = 0; x < w; ++x) out.set(x, y, Rgb{40, 200, 90});
    }
    (void)q;
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string groove_out = wproc_cli::arg_str(argc, argv, "--groove", "");
    const std::string sd_out = wproc_cli::arg_str(argc, argv, "--sd", "");
    const unsigned seed =
        static_cast<unsigned>(arg_double(argc, argv, "--seed", 2025));

    // 1) Ablation-groove inspection.
    const Image gimg = make_groove(240, 140, seed);
    const laser::GrooveResult gr = laser::inspect_groove(gimg);
    if (!groove_out.empty())
        write_ppm(groove_out, laser::annotate_groove(gimg, gr));

    // 2) Stealth SD-layer / HAZ analysis.
    const Image simg = make_sd(96, 130, seed + 1);
    const laser::SdQuality sq = laser::inspect_sd_haz(simg);
    if (!sd_out.empty()) {
        const stealth::SdResult sd = stealth::inspect_stealth(simg);
        write_ppm(sd_out, annotate_sd(simg, sq, sd));
    }

    // 3) Pulse-energy APC vs open loop over an optics-degradation run.
    laser::AblationPlant plant;
    plant.depth_gain = 1.0;
    plant.drift_um_per_shot = 0.02;
    plant.power_gain = 1.0;
    plant.power_drop_per_shot = 0.01;
    plant.noise_sigma = 0.15;

    const double target = 20.0;
    const double nominal = target;
    const int N = 300;
    std::mt19937 gen(seed + 2);
    std::normal_distribution<double> pn(0.0, plant.noise_sigma);

    laser::PulseParams pp;
    pp.target_um = target;
    pp.gain = 1.0;
    pp.lambda = 0.3;
    pp.energy0 = nominal;
    laser::PulseEnergyController ctrl(pp);
    laser::OpticsMonitor optics;

    std::vector<double> open_d, cl_d, power_s;
    double energy = ctrl.current_energy();
    double sse_open = 0.0, sse_cl = 0.0;
    for (long k = 1; k <= N; ++k) {
        const double od = plant.depth(nominal, k, pn(gen));
        const double cd = plant.depth(energy, k, pn(gen));
        open_d.push_back(od);
        cl_d.push_back(cd);
        sse_open += (od - target) * (od - target);
        sse_cl += (cd - target) * (cd - target);
        // Optics health is gauged from a fixed reference "power-check" pulse each
        // run, so it reflects true beam-delivery decay independent of the APC.
        const double pw = plant.power(nominal, k);
        power_s.push_back(pw);
        optics.add(k, pw);
        energy = ctrl.update(cd);
    }
    const double open_rms = std::sqrt(sse_open / N);
    const double cl_rms = std::sqrt(sse_cl / N);
    const laser::OpticsHealth oh = optics.health(/*power_floor=*/15.0);

    // 4) KABRA SiC-ingot slicing economics + separation-layer QC.
    const laser::SliceComparison cmp =
        laser::compare_slice(/*ingot_mm=*/30.0, /*wafer_um=*/350.0,
                             /*laser_loss_um=*/100.0, /*saw_kerf_um=*/250.0);
    const laser::SliceEcon se =
        laser::slice_economics(cmp, /*ingot_cost=*/5000.0, 8.0, 5.0);
    std::vector<double> sep;
    std::mt19937 sg(seed + 3);
    std::normal_distribution<double> sn(0.0, 0.5);
    for (int i = 0; i < 200; ++i)
        sep.push_back(120.0 + 0.6 * std::sin(i * 0.05) + sn(sg));
    const laser::SepLayerStats ss =
        laser::analyze_separation(sep, /*target=*/120.0, /*tol=*/3.0);

    std::cout.setf(std::ios::fixed);
    std::cout.precision(4);
    std::cout << "{\n"
              << "  \"groove\": {\"width_um\": " << gr.width_um
              << ", \"depth_index\": " << gr.depth_index
              << ", \"haz_um\": " << gr.haz_um
              << ", \"debris_index\": " << gr.debris_index
              << ", \"ok\": " << (gr.ok ? "true" : "false") << "},\n"
              << "  \"sd\": {\"layer_count\": " << sq.layer_count
              << ", \"mean_pitch_px\": " << sq.mean_pitch_px
              << ", \"pitch_uniformity_pct\": " << sq.pitch_uniformity_pct
              << ", \"haz_thickness_px\": " << sq.haz_thickness_px
              << ", \"surface_risk\": " << sq.surface_risk
              << ", \"ok\": " << (sq.ok ? "true" : "false") << "},\n"
              << "  \"apc\": {\"target_um\": " << target
              << ", \"open_rms_um\": " << open_rms
              << ", \"closed_rms_um\": " << cl_rms
              << ", \"final_energy\": " << ctrl.current_energy() << "},\n"
              << "  \"optics\": {\"current_power_w\": " << oh.current_power
              << ", \"power_slope_per_shot\": " << oh.power_slope_per_shot
              << ", \"rul_shots\": " << oh.rul_shots << "},\n"
              << "  \"kabra\": {\"laser_wafers\": " << cmp.laser.wafers
              << ", \"saw_wafers\": " << cmp.saw.wafers
              << ", \"extra_wafers\": " << cmp.extra_wafers
              << ", \"extra_wafers_pct\": " << cmp.extra_wafers_pct
              << ", \"laser_util_pct\": " << cmp.laser.utilization_pct
              << ", \"saw_util_pct\": " << cmp.saw.utilization_pct
              << ", \"cost_per_wafer_laser\": " << se.cost_per_wafer_laser
              << ", \"cost_per_wafer_saw\": " << se.cost_per_wafer_saw
              << ", \"savings_pct\": " << se.savings_pct
              << ", \"sep_uniformity_pct\": " << ss.uniformity_pct
              << ", \"sep_ok\": " << (ss.ok ? "true" : "false") << "},\n";

    auto emit = [](const char* name, const std::vector<double>& v, int prec) {
        std::cout << "  \"" << name << "\": [";
        std::cout.precision(prec);
        for (std::size_t i = 0; i < v.size(); ++i)
            std::cout << (i ? "," : "") << v[i];
        std::cout << "]";
    };
    std::cout << "  \"series\": {";
    emit("open_depth_um", open_d, 3);
    std::cout << ",";
    emit("closed_depth_um", cl_d, 3);
    std::cout << ",";
    emit("power_w", power_s, 3);
    std::cout << "}\n}\n";
    return 0;
}
