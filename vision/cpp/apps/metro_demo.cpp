// metro_demo.cpp — metrology & optimization demo: vision calibration + Gage R&R,
// die-to-die registration/diff, GP Bayesian recipe search, and transport routing.
//
// Writes three color figures (calibration residuals, die-to-die diff, A* route)
// and prints every metric + chart series as JSON.
//
// Usage:
//   metro_demo [--resid a.ppm] [--diff b.ppm] [--route c.ppm] [--seed N]
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cli_args.hpp"
#include "wproc/bopt.hpp"
#include "wproc/image.hpp"
#include "wproc/io.hpp"
#include "wproc/regi.hpp"
#include "wproc/route.hpp"
#include "wproc/vcal.hpp"

using namespace wproc;

namespace {

Image make_die(int w, int h) {
    Image d(w, h, 60.0f);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            if ((x % 16 < 6) && (y % 16 < 6)) d.at(x, y) = 210.0f;
            if (((x + y) % 24) < 3) d.at(x, y) += 30.0f;
        }
    return d;
}

void stamp_defect(Image& img, int cx, int cy, int r, float val) {
    for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx)
            if (dx * dx + dy * dy <= r * r) {
                const int x = cx + dx, y = cy + dy;
                if (x >= 0 && x < img.width() && y >= 0 && y < img.height())
                    img.at(x, y) = val;
            }
}

// Chipping-vs-recipe objective: minimum near (feed=0.30, load=0.70).
double recipe(const bopt::Point& x) {
    const double a = x[0] - 0.30, b = x[1] - 0.70;
    return 2.0 * a * a + 3.0 * b * b + 0.03 * std::sin(6.0 * x[0]);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string resid_out = wproc_cli::arg_str(argc, argv, "--resid", "");
    const std::string diff_out = wproc_cli::arg_str(argc, argv, "--diff", "");
    const std::string route_out = wproc_cli::arg_str(argc, argv, "--route", "");
    const unsigned seed =
        static_cast<unsigned>(wproc_cli::arg_double(argc, argv, "--seed", 4040));
    std::mt19937 gen(seed);

    // ── Vision calibration + Gage R&R ───────────────────────────────────────
    const double pitch = 0.005;                 // 5 µm/px
    const double th = 1.5 * M_PI / 180.0;       // 1.5° camera rotation
    const double a00 = pitch * std::cos(th), a01 = -pitch * std::sin(th);
    const double a10 = pitch * std::sin(th), a11 = pitch * std::cos(th);
    std::normal_distribution<double> pxn(0.0, 0.15);  // sub-pixel fiducial noise
    std::vector<vcal::Correspondence> fids;
    for (int gy = 0; gy < 5; ++gy)
        for (int gx = 0; gx < 5; ++gx) {
            const double px = gx * 250.0 + 50.0, py = gy * 250.0 + 50.0;
            vcal::Correspondence c;
            c.px = px + pxn(gen); c.py = py + pxn(gen);
            c.sx = a00 * px + a01 * py + 12.0;
            c.sy = a10 * px + a11 * py + 8.0;
            fids.push_back(c);
        }
    const vcal::CalibResult cal = vcal::calibrate(fids);
    if (!resid_out.empty())
        write_ppm(resid_out, vcal::render_residuals(fids, cal, 360, 360, 60.0));

    // Alignment Gage R&R: 5 fiducials, 3 operators, 3 trials; small equipment
    // noise, tiny operator bias, well-separated parts → a capable system.
    vcal::GageStudy study;
    study.parts = 5; study.operators = 3; study.trials = 3;
    const double part_true[5] = {-2.0, -1.0, 0.0, 1.0, 2.0};  // placement error [µm]
    const double op_bias[3] = {0.0, 0.10, -0.08};
    std::normal_distribution<double> evn(0.0, 0.12);
    study.values.resize(study.operators);
    for (int o = 0; o < study.operators; ++o) {
        study.values[o].resize(study.parts);
        for (int p = 0; p < study.parts; ++p)
            for (int t = 0; t < study.trials; ++t)
                study.values[o][p].push_back(part_true[p] + op_bias[o] + evn(gen));
    }
    const vcal::GageResult gage = vcal::gage_rr(study);

    // ── Die-to-die registration + difference inspection ─────────────────────
    const Image ref = make_die(128, 128);
    Image test = regi::shift_image(ref, 3.5, -2.0);   // stage/vibration offset
    stamp_defect(test, 78, 52, 4, 255.0f);            // extra particle
    stamp_defect(test, 40, 96, 3, 5.0f);              // missing/void
    const regi::DiffResult diff = regi::diff_inspect(ref, test);
    if (!diff_out.empty()) write_ppm(diff_out, regi::annotate_diff(test, diff));

    // ── Bayesian-optimization recipe search ─────────────────────────────────
    bopt::Bounds bnd; bnd.lo = {0.0, 0.0}; bnd.hi = {1.0, 1.0};
    bopt::Config cfg;
    cfg.n_init = 5; cfg.n_iter = 25; cfg.candidates = 500; cfg.seed = seed;
    cfg.kernel.lengthscale = 0.22; cfg.kernel.signal_var = 1.0; cfg.kernel.noise = 1e-6;
    const bopt::Result bo = bopt::minimize(recipe, bnd, cfg);
    // For reference: a grid matching this accuracy would need ~this many trials.
    const int grid_equiv = 21 * 21;  // 0.05 resolution per axis

    // ── Transport-job routing ───────────────────────────────────────────────
    std::vector<route::Station> st = {
        {"LoadPort", 0, 0},   {"Aligner", 120, 20}, {"Chamber-A", 240, 40},
        {"Chamber-B", 240, 160}, {"Chamber-C", 120, 200}, {"Metrology", 20, 180},
        {"Cooldown", 200, 120}};
    std::vector<route::Job> jobs = {
        {0, 2}, {0, 4}, {0, 6}, {0, 1}, {0, 3}, {0, 5}, {6, 2}, {1, 4}};
    route::Station home{"home", 0, 0};
    const route::Schedule fifo = route::plan_fifo(st, jobs, home);
    const route::Schedule opt = route::optimize(st, jobs, home);
    const double improve = fifo.travel > 0
                               ? 100.0 * (fifo.travel - opt.travel) / fifo.travel
                               : 0.0;

    // A* collision-free path through a cluster-tool floor with chamber obstacles.
    route::Grid grid(40, 30);
    for (int y = 5; y < 22; ++y) { grid.block(14, y); grid.block(26, y); }
    for (int x = 14; x <= 26; ++x) grid.block(x, 5);
    const route::Path path = route::astar(grid, {2, 15}, {37, 15});
    if (!route_out.empty()) write_ppm(route_out, route::render_path(grid, path, 10));

    // ── JSON ─────────────────────────────────────────────────────────────────
    std::cout.setf(std::ios::fixed);
    std::cout.precision(4);
    std::cout << "{\n"
              << "  \"vcal\": {\"rms_um\": " << cal.rms_um
              << ", \"max_um\": " << cal.max_um
              << ", \"um_per_px\": " << cal.um_per_px
              << ", \"rotation_deg\": " << cal.rotation_deg
              << ", \"gage\": {\"pct_grr\": " << gage.pct_grr
              << ", \"pct_ev\": " << gage.pct_ev << ", \"pct_av\": " << gage.pct_av
              << ", \"pct_pv\": " << gage.pct_pv << ", \"ndc\": " << gage.ndc
              << ", \"acceptable\": " << (gage.acceptable ? "true" : "false")
              << "}},\n"
              << "  \"regi\": {\"dx\": " << diff.shift.dx << ", \"dy\": " << diff.shift.dy
              << ", \"response\": " << diff.shift.response
              << ", \"threshold\": " << diff.threshold
              << ", \"residual_rms\": " << diff.residual_rms
              << ", \"defect_count\": " << diff.defect_count
              << ", \"defect_area\": " << diff.defect_area << "},\n"
              << "  \"bopt\": {\"best_feed\": " << bo.best_x[0]
              << ", \"best_load\": " << bo.best_x[1]
              << ", \"best_chipping\": " << bo.best_y
              << ", \"evaluations\": " << bo.evaluations
              << ", \"grid_equiv\": " << grid_equiv << ", \"series\": {\"eval\": [";
    for (std::size_t i = 0; i < bo.history.size(); ++i)
        std::cout << (i ? "," : "") << (i + 1);
    std::cout << "], \"best\": [";
    for (std::size_t i = 0; i < bo.history.size(); ++i)
        std::cout << (i ? "," : "") << bo.history[i].best_so_far;
    std::cout << "]}},\n"
              << "  \"route\": {\"fifo_travel\": " << fifo.travel
              << ", \"opt_travel\": " << opt.travel
              << ", \"improve_pct\": " << improve
              << ", \"fifo_makespan_s\": " << fifo.makespan_s
              << ", \"opt_makespan_s\": " << opt.makespan_s
              << ", \"jobs\": " << jobs.size()
              << ", \"astar_length\": " << path.length
              << ", \"astar_expanded\": " << path.expanded
              << ", \"astar_found\": " << (path.found ? "true" : "false") << "}\n"
              << "}\n";
    return 0;
}
