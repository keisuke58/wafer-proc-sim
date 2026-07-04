// optm_demo.cpp — optomechanical lens-barrel demo. Runs all four modules,
// writes three figures (tolerance histogram, AF step response, OIS trace) and
// prints every metric + chart series as JSON.
//
// Usage: optm_demo [--tol a.ppm] [--cam b.ppm] [--ois c.ppm]
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "optomech/cam.hpp"
#include "optomech/fea.hpp"
#include "optomech/ois.hpp"
#include "optomech/ppm.hpp"
#include "optomech/tolerance.hpp"

using namespace optm;

namespace {

const char* arg(int c, char** v, const char* f, const char* d) {
    for (int i = 1; i < c - 1; ++i) if (std::strcmp(v[i], f) == 0) return v[i + 1];
    return d;
}

// Map a data range to canvas pixels for simple line plots.
struct Plot {
    int W, H, ml = 52, mr = 14, mt = 14, mb = 30;
    double xmin = 0, xmax = 1, ymin = 0, ymax = 1;
    int px(double x) const { return ml + static_cast<int>((x - xmin) / (xmax - xmin) * (W - ml - mr)); }
    int py(double y) const { return mt + static_cast<int>((ymax - y) / (ymax - ymin) * (H - mt - mb)); }
};

void axes(Canvas& c, const Plot& p) {
    const Rgb ax{150, 158, 168};
    c.line(p.ml, p.mt, p.ml, p.H - p.mb, ax);
    c.line(p.ml, p.H - p.mb, p.W - p.mr, p.H - p.mb, ax);
}

void polyline(Canvas& c, const Plot& p, const std::vector<double>& xs,
              const std::vector<double>& ys, Rgb col, int step = 1) {
    for (std::size_t i = step; i < xs.size(); i += step)
        c.line(p.px(xs[i - step]), p.py(ys[i - step]), p.px(xs[i]), p.py(ys[i]), col);
}

LensSystem make_lens() {
    LensSystem s;
    s.elements = {{60.0, 6.0, 12.0}, {-40.0, 4.0, 10.0}, {35.0, 8.0, 9.0}, {80.0, 18.0, 8.0}};
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string tol_out = arg(argc, argv, "--tol", "");
    const std::string cam_out = arg(argc, argv, "--cam", "");
    const std::string ois_out = arg(argc, argv, "--ois", "");

    // ── Tolerance stack-up ──────────────────────────────────────────────────
    const LensSystem sys = make_lens();
    // Image-plane boresight budget ~5 sensor pixels (3.9 µm pixel).
    const double budget_um = 20.0;
    std::vector<tol::Tolerance> tols = {
        {1.5, 0.10, 5.0}, {2.0, 0.12, 5.0}, {1.5, 0.08, 4.0}, {1.2, 0.07, 4.0}};
    const tol::Result tr = tol::analyze(sys, tols, budget_um, 30000);

    if (!tol_out.empty()) {
        Canvas c(560, 300, {250, 251, 253});
        Plot p{560, 300}; p.xmin = 0; p.xmax = std::max(budget_um * 1.4, tr.lateral_mc_p99_um * 1.1);
        // Histogram of the Monte Carlo lateral image error.
        const int nb = 40;
        std::vector<int> hist(nb, 0);
        for (double v : tr.mc_lateral_um) {
            int b = static_cast<int>(v / p.xmax * nb);
            if (b >= 0 && b < nb) hist[b]++;
        }
        p.ymin = 0; p.ymax = *std::max_element(hist.begin(), hist.end()) * 1.05;
        axes(c, p);
        for (int b = 0; b < nb; ++b) {
            const int x0 = p.px(b * p.xmax / nb), x1 = p.px((b + 1) * p.xmax / nb);
            const int y0 = p.py(hist[b]), y1 = p.py(0);
            for (int x = x0; x < x1 - 1; ++x) c.line(x, y0, x, y1, {90, 140, 210});
        }
        // Budget line (pixel size) in red.
        const int bx = p.px(budget_um);
        c.line(bx, p.mt, bx, p.H - p.mb, {220, 70, 70});
        write_ppm(tol_out, c);
    }

    // ── Autofocus: cam curve + VCM step (FF vs no-FF) ───────────────────────
    cam::Vcm vcm;
    cam::Pid ff; ff.kp = 0.02; ff.kd = 2e-5; ff.feedforward = true;
    cam::Pid noff = ff; noff.feedforward = false;
    const cam::StepResult s_ff = cam::step_response(vcm, ff, 300.0, 18.0);
    const cam::StepResult s_no = cam::step_response(vcm, noff, 300.0, 18.0);
    const double cam_peak_acc = cam::peak_accel(cam::CamProfile{5.0, 90.0});

    if (!cam_out.empty()) {
        Canvas c(560, 300, {250, 251, 253});
        Plot p{560, 300}; p.xmin = 0; p.xmax = s_ff.t_ms.back();
        p.ymin = 0; p.ymax = 360;
        axes(c, p);
        polyline(c, p, s_ff.t_ms, s_ff.target_um, {150, 158, 168}, 4);
        polyline(c, p, s_no.t_ms, s_no.pos_um, {225, 120, 60}, 2);
        polyline(c, p, s_ff.t_ms, s_ff.pos_um, {45, 120, 210}, 2);
        write_ppm(cam_out, c);
    }

    // ── Barrel FEA ──────────────────────────────────────────────────────────
    const fea::Result fr = fea::analyze(fea::Barrel{}, 1.0, 0.5);

    // ── OIS ─────────────────────────────────────────────────────────────────
    const ois::Result orr = ois::simulate(35.0);
    if (!ois_out.empty()) {
        Canvas c(560, 300, {250, 251, 253});
        Plot p{560, 300}; p.xmin = 0; p.xmax = orr.t_s.back();
        const double amax = std::accumulate(orr.demand_um.begin(), orr.demand_um.end(),
            0.0, [](double m, double v) { return std::max(m, std::fabs(v)); });
        p.ymin = -amax * 1.1; p.ymax = amax * 1.1;
        axes(c, p);
        c.line(p.ml, p.py(0), p.W - p.mr, p.py(0), {200, 206, 214});
        polyline(c, p, orr.t_s, orr.demand_um, {225, 120, 60}, 1);
        polyline(c, p, orr.t_s, orr.residual_um, {45, 120, 210}, 1);
        write_ppm(ois_out, c);
    }

    // ── JSON ─────────────────────────────────────────────────────────────────
    std::cout.setf(std::ios::fixed); std::cout.precision(4);
    std::cout << "{\n"
              << "  \"tolerance\": {\"efl_mm\": " << tr.efl_mm
              << ", \"lateral_rss_um\": " << tr.lateral_rss_um
              << ", \"lateral_mc_rms_um\": " << tr.lateral_mc_rms_um
              << ", \"lateral_p99_um\": " << tr.lateral_mc_p99_um
              << ", \"focus_rss_um\": " << tr.focus_rss_um
              << ", \"budget_um\": " << tr.budget_um
              << ", \"ok\": " << (tr.ok ? "true" : "false") << ", \"ranking\": [";
    for (std::size_t i = 0; i < std::min<std::size_t>(4, tr.ranking.size()); ++i) {
        const auto& c = tr.ranking[i];
        std::cout << (i ? "," : "") << "{\"element\":" << c.element << ",\"source\":\""
                  << c.source << "\",\"sigma_um\":" << c.sigma_um << ",\"pct\":"
                  << c.variance_pct << "}";
    }
    std::cout << "]},\n"
              << "  \"autofocus\": {\"cam_peak_accel\": " << cam_peak_acc
              << ", \"ff_settle_ms\": " << s_ff.settling_ms
              << ", \"ff_ss_um\": " << s_ff.ss_error_um
              << ", \"noff_ss_um\": " << s_no.ss_error_um
              << ", \"ff_overshoot_pct\": " << s_ff.overshoot_pct << "},\n"
              << "  \"fea\": {\"thermal_um_per_C\": " << fr.thermal_focus_drift_um_per_C
              << ", \"first_mode_hz\": " << fr.first_mode_hz
              << ", \"axial_stiffness_N_per_um\": " << fr.axial_stiffness_N_per_um
              << ", \"shock_g\": " << fr.shock_g_peak << ", \"daf\": " << fr.daf
              << ", \"shock_stress_MPa\": " << fr.drop_shock_stress_MPa
              << ", \"safety_factor\": " << fr.safety_factor
              << ", \"ok\": " << (fr.ok ? "true" : "false") << "},\n"
              << "  \"ois\": {\"focal_mm\": " << orr.focal_mm
              << ", \"off_rms_um\": " << orr.off_rms_um
              << ", \"on_rms_um\": " << orr.on_rms_um
              << ", \"ratio\": " << orr.ratio << ", \"stops\": " << orr.stops << "},\n";
    auto series = [](const char* n, const std::vector<double>& v, int step) {
        std::cout << "\"" << n << "\":[";
        for (std::size_t i = 0; i < v.size(); i += step)
            std::cout << (i ? "," : "") << v[i];
        std::cout << "]";
    };
    std::cout << "  \"series\": {";
    series("cam_t", s_ff.t_ms, 40); std::cout << ",";
    series("cam_ff", s_ff.pos_um, 40); std::cout << ",";
    series("cam_noff", s_no.pos_um, 40); std::cout << ",";
    series("ois_t", orr.t_s, 8); std::cout << ",";
    series("ois_demand", orr.demand_um, 8); std::cout << ",";
    series("ois_resid", orr.residual_um, 8);
    std::cout << "}\n}\n";
    return 0;
}
