// smart_demo.cpp — smart-manufacturing demo: CMP (polish) APC + planarity +
// endpoint, spindle FFT chatter detection, and street auto-alignment (NCC).
//
// Runs the wproc::cmp / wproc::spindle / wproc::align stacks on repeatable
// synthetic data, writes an annotated alignment figure, and prints all metrics
// plus chart series as JSON. (The defect-CNN accuracy/montage come from
// ml/defect_cnn_train.py; its C++ inference is exercised by test_cnn.)
//
// Usage:
//   smart_demo [--align out.ppm] [--seed N]
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cli_args.hpp"
#include "wproc/align.hpp"
#include "wproc/cmp.hpp"
#include "wproc/image.hpp"
#include "wproc/io.hpp"
#include "wproc/spindle.hpp"

using namespace wproc;
using wproc_cli::arg_double;

namespace {
constexpr double kPi = 3.14159265358979323846;

std::vector<double> vib_signal(double fs, int n, double f0, unsigned seed,
                               double chatter_amp, double chatter_f) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> noise(0.0, 0.05);
    std::vector<double> s(n);
    for (int i = 0; i < n; ++i) {
        const double t = i / fs;
        double v = 1.0 * std::sin(2 * kPi * f0 * t) +
                   0.5 * std::sin(2 * kPi * 2 * f0 * t) +
                   0.3 * std::sin(2 * kPi * 3 * f0 * t) + noise(gen);
        if (chatter_amp > 0.0) v += chatter_amp * std::sin(2 * kPi * chatter_f * t);
        s[i] = v;
    }
    return s;
}

const int TW = 48, TH = 48;
const int BLOBS[][2] = {{6, 8}, {30, 6}, {12, 34}, {38, 28}, {22, 20}};
void stamp(Image& img, int px, int py) {
    const int nb = static_cast<int>(sizeof(BLOBS) / sizeof(BLOBS[0]));
    for (int b = 0; b < nb; ++b)
        for (int dy = 0; dy < 5; ++dy)
            for (int dx = 0; dx < 5; ++dx) {
                const int x = px + BLOBS[b][0] + dx, y = py + BLOBS[b][1] + dy;
                if (x >= 0 && x < img.width() && y >= 0 && y < img.height())
                    img.at(x, y) = 220.0f;
            }
}

void emit_series(const char* name, const std::vector<double>& v, int prec) {
    std::cout << "  \"" << name << "\": [";
    std::cout.precision(prec);
    for (std::size_t i = 0; i < v.size(); ++i)
        std::cout << (i ? "," : "") << v[i];
    std::cout << "]";
}
}  // namespace

int main(int argc, char** argv) {
    const std::string align_out = wproc_cli::arg_str(argc, argv, "--align", "");
    const unsigned seed =
        static_cast<unsigned>(arg_double(argc, argv, "--seed", 2026));

    // ── Spindle vibration / chatter ─────────────────────────────────────────
    const double fs = 20000.0, rpm = 30000.0;
    const int nsig = 4096;
    auto healthy = vib_signal(fs, nsig, 500.0, seed, 0.0, 0.0);
    auto chatter = vib_signal(fs, nsig, 500.0, seed, 1.2, 3200.0);
    auto sh = spindle::analyze_vibration(healthy, fs, rpm);
    auto sc = spindle::analyze_vibration(chatter, fs, rpm);
    auto psh = spindle::power_spectrum(healthy, fs);
    auto psc = spindle::power_spectrum(chatter, fs);
    // Downsample the spectra to <=5 kHz for the chart.
    std::vector<double> f_axis, p_healthy, p_chatter;
    for (std::size_t k = 0; k < psh.freq.size(); k += 2) {
        if (psh.freq[k] > 5000.0) break;
        f_axis.push_back(psh.freq[k]);
        p_healthy.push_back(psh.power[k]);
        p_chatter.push_back(psc.power[k]);
    }

    // ── CMP polish: APC + planarity + endpoint ──────────────────────────────
    cmp::PolishPlant plant;
    plant.kp0 = 1.0e-3; plant.pressure = 30.0; plant.velocity = 0.5;
    plant.kp_decay_per_run = 1.0e-6; plant.noise_sigma = 2.0;
    const double target = 300.0;
    const double rate0 = plant.kp0 * plant.pressure * plant.velocity;
    const double nominal = target / rate0;
    const int N = 200;
    std::mt19937 gen(seed + 1);
    std::normal_distribution<double> pn(0.0, plant.noise_sigma);
    cmp::PolishParams pp;
    pp.target_removal_nm = target; pp.kp = plant.kp0; pp.pressure = plant.pressure;
    pp.velocity = plant.velocity; pp.lambda = 0.3; pp.time0 = nominal; pp.time_max = 1e6;
    cmp::PolishController ctrl(pp);
    std::vector<double> open_r, cl_r;
    double t = ctrl.current_time();
    double sse_o = 0, sse_c = 0;
    for (long k = 1; k <= N; ++k) {
        const double o = plant.removed(nominal, k, pn(gen));
        const double c = plant.removed(t, k, pn(gen));
        open_r.push_back(o); cl_r.push_back(c);
        sse_o += (o - target) * (o - target); sse_c += (c - target) * (c - target);
        t = ctrl.update(c);
    }
    // Planarity: field at 500 nm, Cu features recessed 25 nm.
    std::vector<double> profile; std::vector<char> feat;
    std::mt19937 g2(seed + 2);
    std::normal_distribution<double> nz(0.0, 1.0);
    for (int i = 0; i < 400; ++i) {
        const bool f = (i % 40) >= 20;
        profile.push_back((f ? 475.0 : 500.0) + nz(g2));
        feat.push_back(f ? 1 : 0);
    }
    cmp::PlanaritySpec plspec; plspec.nominal_nm = 500.0;
    auto pl = cmp::analyze_planarity(profile, feat, plspec);
    // Endpoint from a friction trace: plateau then drop at the clear point.
    std::vector<double> friction; std::mt19937 g3(seed + 3);
    std::normal_distribution<double> fn(0.0, 0.03);
    const int clear_i = 180;
    for (int i = 0; i < 300; ++i)
        friction.push_back(((i < clear_i) ? 1.0 : 0.3) + fn(g3));
    auto ep = cmp::detect_endpoint(friction, 10.0);

    // ── Street auto-alignment (NCC) ─────────────────────────────────────────
    Image templ(TW, TH, 40.0f); stamp(templ, 0, 0);
    Image scene(220, 180, 40.0f); stamp(scene, 95, 52);   // shifted +15,-8
    auto reg = align::register_pattern(scene, templ, 80, 60, 30);
    auto m = align::match_ncc(scene, templ, 0, 0, scene.width(), scene.height());
    if (!align_out.empty())
        write_ppm(align_out, align::annotate_match(scene, m, TW, TH));

    // ── JSON ────────────────────────────────────────────────────────────────
    std::cout.setf(std::ios::fixed);
    std::cout.precision(4);
    std::cout << "{\n"
              << "  \"spindle\": {\"rpm\": " << rpm << ", \"spindle_hz\": " << sh.spindle_freq
              << ", \"healthy_dom_hz\": " << sh.dominant_freq
              << ", \"healthy_chatter\": " << (sh.chatter ? "true" : "false")
              << ", \"chatter_freq_hz\": " << sc.chatter_freq
              << ", \"chatter_ratio\": " << sc.chatter_ratio
              << ", \"chatter_detected\": " << (sc.chatter ? "true" : "false") << "},\n"
              << "  \"cmp\": {\"target_nm\": " << target
              << ", \"open_rms_nm\": " << std::sqrt(sse_o / N)
              << ", \"closed_rms_nm\": " << std::sqrt(sse_c / N)
              << ", \"dishing_nm\": " << pl.dishing_nm
              << ", \"erosion_nm\": " << pl.erosion_nm
              << ", \"planarity_ok\": " << (pl.ok ? "true" : "false")
              << ", \"endpoint_s\": " << ep.time_s
              << ", \"endpoint_detected\": " << (ep.detected ? "true" : "false") << "},\n"
              << "  \"align\": {\"dx\": " << reg.dx << ", \"dy\": " << reg.dy
              << ", \"angle_deg\": " << reg.angle_deg << ", \"score\": " << reg.score
              << ", \"ok\": " << (reg.ok ? "true" : "false") << "},\n";
    std::cout << "  \"series\": {";
    emit_series("freq_hz", f_axis, 1); std::cout << ",";
    emit_series("psd_healthy", p_healthy, 6); std::cout << ",";
    emit_series("psd_chatter", p_chatter, 6); std::cout << ",";
    emit_series("cmp_open_nm", open_r, 2); std::cout << ",";
    emit_series("cmp_closed_nm", cl_r, 2); std::cout << ",";
    emit_series("friction", friction, 3);
    std::cout << "}\n}\n";
    return 0;
}
