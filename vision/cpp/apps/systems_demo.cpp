// systems_demo.cpp — machine/line systems demo: S-curve motion + servo tracking,
// digital-twin line OEE, bump-array inspection, and the ADC defect pipeline.
//
// Writes two color figures (annotated bump array, ADC classified wafer map) and
// prints all metrics + chart series as JSON. The ADC stage loads the defect-CNN
// weights/testset from --results (skipped if absent).
//
// Usage:
//   systems_demo [--bumps out1.ppm] [--wafer out2.ppm] [--results DIR]
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli_args.hpp"
#include "wproc/adc.hpp"
#include "wproc/cnn.hpp"
#include "wproc/fab.hpp"
#include "wproc/image.hpp"
#include "wproc/io.hpp"
#include "wproc/motion.hpp"
#include "wproc/pkg.hpp"
#include "wproc/wafer_map.hpp"

using namespace wproc;

namespace {

// Tracking loop that records the servo's position (track() returns only stats).
std::vector<double> track_positions(const motion::Trajectory& ref,
                                    const motion::StagePlant& plant,
                                    const motion::ServoGains& g) {
    std::vector<double> out;
    out.reserve(ref.pos.size());
    double p = 0.0, v = 0.0;
    const double dt = ref.dt;
    for (std::size_t i = 0; i < ref.pos.size(); ++i) {
        const double e = ref.pos[i] - p, ed = ref.vel[i] - v;
        double force = g.kp * e + g.kd * ed;
        if (g.feedforward)
            force += plant.mass * ref.acc[i] + plant.damping * ref.vel[i];
        const double a = (force - plant.damping * v) / plant.mass;
        v += a * dt; p += v * dt;
        out.push_back(p);
    }
    return out;
}

Image make_bump_array(int px_pitch, unsigned seed) {
    Image img(170, 170, 40.0f);
    const int r = 5, x0 = 25, y0 = 25;
    std::mt19937 gen(seed);
    std::normal_distribution<double> nz(0.0, 3.0);
    for (int j = 0; j < 6; ++j)
        for (int i = 0; i < 6; ++i) {
            if ((i == 2 && j == 3) || (i == 3 && j == 2)) continue;  // missing
            float val = ((i + j) % 5 == 0) ? 155.0f : 220.0f;        // short bumps
            const int cx = x0 + i * px_pitch, cy = y0 + j * px_pitch;
            for (int dy = -r; dy <= r; ++dy)
                for (int dx = -r; dx <= r; ++dx)
                    if (dx * dx + dy * dy <= r * r)
                        img.at(cx + dx, cy + dy) = static_cast<float>(val + nz(gen));
        }
    return img;
}

void emit(const char* name, const std::vector<double>& v, int prec, int step) {
    std::cout << "  \"" << name << "\": [";
    std::cout.precision(prec);
    bool first = true;
    for (std::size_t i = 0; i < v.size(); i += step) {
        std::cout << (first ? "" : ",") << v[i];
        first = false;
    }
    std::cout << "]";
}

}  // namespace

int main(int argc, char** argv) {
    const std::string bumps_out = wproc_cli::arg_str(argc, argv, "--bumps", "");
    const std::string wafer_out = wproc_cli::arg_str(argc, argv, "--wafer", "");
    const std::string results = wproc_cli::arg_str(argc, argv, "--results", "results");
    // Clamp before the double->unsigned cast: a negative seed would be UB.
    const double seed_arg = wproc_cli::arg_double(argc, argv, "--seed", 2027);
    const unsigned seed = static_cast<unsigned>(seed_arg < 0.0 ? 0.0 : seed_arg);

    // ── Motion: S-curve + servo tracking (FF vs no-FF) ──────────────────────
    motion::Limits lim; lim.vmax = 40.0; lim.amax = 400.0; lim.jmax = 4000.0;
    motion::Trajectory ref = motion::scurve(10.0, lim, 1e-4);
    for (int i = 0; i < 2000; ++i) {   // dwell so the servo settles
        ref.pos.push_back(10.0); ref.vel.push_back(0.0); ref.acc.push_back(0.0);
    }
    motion::StagePlant plant; plant.mass = 1.0; plant.damping = 5.0;
    motion::ServoGains g; g.kp = 400.0; g.kd = 40.0;
    g.feedforward = true;
    const auto pos_ff = track_positions(ref, plant, g);
    const auto tr_ff = motion::track(ref, plant, g);
    g.feedforward = false;
    const auto pos_noff = track_positions(ref, plant, g);
    const auto tr_noff = motion::track(ref, plant, g);

    // ── Fab digital twin ────────────────────────────────────────────────────
    std::vector<fab::Station> line = {
        {"dice", 30.0, 28.0, 0.95, 0.99}, {"grind", 45.0, 42.0, 0.92, 0.98},
        {"polish", 40.0, 38.0, 0.90, 0.97}, {"inspect", 20.0, 20.0, 0.99, 0.995}};
    const fab::LineResult fr = fab::analyze_line(line);

    // ── Package bump inspection ─────────────────────────────────────────────
    const Image barr = make_bump_array(20, seed);
    const pkg::BumpResult pr = pkg::inspect_bumps(barr);
    if (!bumps_out.empty()) write_ppm(bumps_out, pkg::annotate_bumps(barr, pr));

    // ── ADC pipeline (optional; needs the trained CNN) ──────────────────────
    bool adc_ok = false;
    adc::Result ar;
    cnn::DefectNet net;
    try {
        net.load(results + "/defect_cnn_weights.bin");
        std::ifstream tf(results + "/defect_cnn_testset.bin", std::ios::binary);
        if (tf) {
            std::int32_t n = 0, h = 0, w = 0;
            tf.read(reinterpret_cast<char*>(&n), 4);
            tf.read(reinterpret_cast<char*>(&h), 4);
            tf.read(reinterpret_cast<char*>(&w), 4);
            if (!tf || n <= 0 || h <= 0 || w <= 0)
                throw std::runtime_error("invalid or truncated testset header");
            // Only the first M patches are used, so size the buffer to M — not to
            // the full (untrusted) n — and verify the payload read succeeded.
            const int M = std::min<int>(n, 240);
            std::vector<float> pix(static_cast<std::size_t>(M) * h * w);
            tf.read(reinterpret_cast<char*>(pix.data()),
                    static_cast<std::streamsize>(pix.size() * sizeof(float)));
            if (!tf) throw std::runtime_error("truncated testset payload");
            adc::Params ap; ap.cols = 16; ap.rows = 16;
            std::vector<adc::Observation> obs;
            std::mt19937 gen(seed + 5);
            std::uniform_int_distribution<int> uc(0, ap.cols - 1), urr(0, ap.rows - 1);
            for (int i = 0; i < M; ++i) {
                Image p(w, h);
                const float* q = &pix[static_cast<std::size_t>(i) * h * w];
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x) p.at(x, y) = q[y * w + x];
                obs.push_back({uc(gen), urr(gen), std::move(p)});
            }
            ar = adc::run(net, obs, ap);
            adc_ok = true;
            if (!wafer_out.empty()) write_ppm(wafer_out, wafer::render_map(ar.wafer, 14));
        }
    } catch (const std::exception&) { adc_ok = false; }

    // ── JSON ────────────────────────────────────────────────────────────────
    std::cout.setf(std::ios::fixed);
    std::cout.precision(4);
    std::cout << "{\n"
              << "  \"motion\": {\"distance_mm\": 10, \"move_time_s\": " << ref.total_time
              << ", \"ff_max_err\": " << tr_ff.max_error
              << ", \"noff_max_err\": " << tr_noff.max_error << "},\n"
              << "  \"fab\": {\"bottleneck\": \"" << fr.stations[fr.bottleneck].name
              << "\", \"throughput_wph\": " << fr.throughput_wph
              << ", \"good_wph\": " << fr.good_wph
              << ", \"line_yield\": " << fr.line_yield
              << ", \"line_oee\": " << fr.line_oee << ", \"stations\": [";
    for (std::size_t i = 0; i < fr.stations.size(); ++i) {
        const auto& s = fr.stations[i];
        std::cout << (i ? "," : "") << "{\"name\":\"" << s.name << "\",\"oee\":" << s.oee
                  << ",\"util\":" << s.utilization << ",\"bottleneck\":"
                  << (s.bottleneck ? "true" : "false") << "}";
    }
    std::cout << "]},\n"
              << "  \"pkg\": {\"count\": " << pr.count << ", \"pitch_um\": " << pr.pitch_um
              << ", \"mean_diameter_um\": " << pr.mean_diameter_um
              << ", \"diameter_cv\": " << pr.diameter_cv
              << ", \"coplanarity\": " << pr.coplanarity
              << ", \"missing\": " << pr.missing
              << ", \"ok\": " << (pr.ok ? "true" : "false") << "},\n";
    std::cout << "  \"adc\": {\"available\": " << (adc_ok ? "true" : "false");
    if (adc_ok) {
        std::cout << ", \"n\": " << ar.n_defects << ", \"good\": " << ar.class_counts[0]
                  << ", \"chipping\": " << ar.class_counts[1] << ", \"crack\": "
                  << ar.class_counts[2] << ", \"contamination\": " << ar.class_counts[3]
                  << ", \"top\": \"" << cnn::defect_name(ar.top_defect_class)
                  << "\", \"yield_pct\": " << ar.yield_pct
                  << ", \"center_yield_pct\": " << ar.wafer.center_yield_pct
                  << ", \"edge_yield_pct\": " << ar.wafer.edge_yield_pct;
    }
    std::cout << "},\n";
    // Motion chart series (downsampled).
    std::vector<double> t; t.reserve(ref.pos.size());
    for (std::size_t i = 0; i < ref.pos.size(); ++i) t.push_back(i * ref.dt);
    std::cout << "  \"series\": {";
    emit("t", t, 4, 40); std::cout << ",";
    emit("ref", ref.pos, 4, 40); std::cout << ",";
    emit("ff", pos_ff, 4, 40); std::cout << ",";
    emit("noff", pos_noff, 4, 40);
    std::cout << "}\n}\n";
    return 0;
}
