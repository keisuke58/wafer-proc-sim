/*
 * _5axis_interpolation.cpp
 *
 * 5-axis trajectory interpolation for DISCO dicing saw:
 *   X  — wafer table feed (dicing direction) [mm]
 *   Y  — index between streets [mm]
 *   Z  — blade plunge depth [mm]
 *   TH — chuck θ rotation [deg]
 *   S  — spindle speed command [rpm]
 *
 * Functions:
 *   linear_interp(start, end, n)          -> list of 5-tuples
 *   arc_interp(cx,cy,r,a0,a1,z0,z1,n)    -> list of 5-tuples (helix)
 *   dice_lane(x0,x1,y,z_cut,feed,dt)     -> trajectory dict
 *   full_wafer_grid(streets_x,streets_y,wafer_r,z_cut,feed_mms,spindle_rpm,dt_s)
 *                                          -> trajectory dict + summary
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <stdexcept>

namespace py = pybind11;

using Pose5 = std::array<double, 5>;   // {x_mm, y_mm, z_mm, theta_deg, spindle_rpm}

// ── linear interpolation between two 5D poses ────────────────────────────────

std::vector<Pose5> linear_interp(const Pose5& start, const Pose5& end, int n) {
    if (n < 2) throw std::invalid_argument("n must be >= 2");
    std::vector<Pose5> traj;
    traj.reserve(n);
    for (int i = 0; i < n; ++i) {
        double t = (double)i / (n - 1);
        Pose5 p;
        for (int j = 0; j < 5; ++j)
            p[j] = start[j] + t * (end[j] - start[j]);
        traj.push_back(p);
    }
    return traj;
}

// ── helical arc interpolation (XY-plane arc + Z ramp) ───────────────────────

// cx,cy: arc center [mm], r: radius [mm]
// a0,a1: start/end angle [deg], z0,z1: Z start/end [mm]
// theta_deg: chuck rotation (held constant), spindle_rpm: held constant
std::vector<Pose5> arc_interp(double cx, double cy, double r,
                               double a0_deg, double a1_deg,
                               double z0, double z1,
                               double theta_deg, double spindle_rpm,
                               int n) {
    if (n < 2) throw std::invalid_argument("n must be >= 2");
    std::vector<Pose5> traj;
    traj.reserve(n);
    const double deg2rad = M_PI / 180.0;
    for (int i = 0; i < n; ++i) {
        double t = (double)i / (n - 1);
        double a = (a0_deg + t * (a1_deg - a0_deg)) * deg2rad;
        double z = z0 + t * (z1 - z0);
        Pose5 p = {cx + r * std::cos(a), cy + r * std::sin(a), z, theta_deg, spindle_rpm};
        traj.push_back(p);
    }
    return traj;
}

// ── single dicing lane ────────────────────────────────────────────────────────

// One street: rapid-to-start → plunge → cut → retract
// Returns dict with x/y/z/theta/spindle/phase arrays and distance_mm
py::dict dice_lane(double x0, double x1, double y_mm,
                   double z_air,           // Z clearance height [mm]
                   double z_cut,           // Z cut depth [mm]
                   double feed_mms,        // feed rate [mm/s]
                   double spindle_rpm,
                   double theta_deg,
                   double dt_s) {
    if (dt_s <= 0.0) throw std::invalid_argument("dt_s must be > 0");
    if (feed_mms <= 0.0) throw std::invalid_argument("feed_mms must be > 0");

    // Phases: PLUNGE, CUT, RETRACT
    // Rapid moves (approach/index) are omitted — not time-critical for stiffness model
    double cut_len = std::abs(x1 - x0);
    int n_cut = std::max(2, (int)std::ceil(cut_len / (feed_mms * dt_s)) + 1);

    // Plunge: 10 steps at plunge feed (half cut feed)
    int n_plunge = 10;

    std::vector<double> ax, ay, az, atheta, aspindle;
    std::vector<int> aphase;  // 0=plunge 1=cut 2=retract

    auto push = [&](double x, double y, double z, double th, double s, int ph) {
        ax.push_back(x); ay.push_back(y); az.push_back(z);
        atheta.push_back(th); aspindle.push_back(s); aphase.push_back(ph);
    };

    // Plunge
    for (int i = 0; i < n_plunge; ++i) {
        double t = (double)i / (n_plunge - 1);
        push(x0, y_mm, z_air + t * (z_cut - z_air), theta_deg, spindle_rpm, 0);
    }

    // Cut
    for (int i = 0; i < n_cut; ++i) {
        double t = (double)i / (n_cut - 1);
        push(x0 + t * (x1 - x0), y_mm, z_cut, theta_deg, spindle_rpm, 1);
    }

    // Retract
    for (int i = 0; i < n_plunge; ++i) {
        double t = (double)i / (n_plunge - 1);
        push(x1, y_mm, z_cut + t * (z_air - z_cut), theta_deg, spindle_rpm, 2);
    }

    py::dict result;
    result["x_mm"]       = ax;
    result["y_mm"]       = ay;
    result["z_mm"]       = az;
    result["theta_deg"]  = atheta;
    result["spindle_rpm"]= aspindle;
    result["phase"]      = aphase;   // 0=plunge 1=cut 2=retract
    result["cut_len_mm"] = cut_len;
    result["n_cut_pts"]  = n_cut;
    return result;
}

// ── full wafer dicing grid ────────────────────────────────────────────────────

// streets_x: X-direction street positions [mm] (absolute from wafer center)
// streets_y: Y-direction street positions [mm]
// wafer_r:   wafer radius [mm] (clips street length)
// z_cut:     blade cut depth (negative = below surface) [mm]
// Returns merged trajectory + per-lane summary
py::dict full_wafer_grid(
    const std::vector<double>& streets_x,
    const std::vector<double>& streets_y,
    double wafer_r,
    double z_air,
    double z_cut,
    double feed_mms,
    double spindle_rpm,
    double dt_s
) {
    std::vector<double> ax, ay, az, atheta, aspindle;
    std::vector<int> aphase;
    int total_lanes = 0;
    double total_cut_mm = 0.0;

    // Helper: clip street at wafer boundary
    auto clip = [&](double pos, bool is_y) -> std::pair<double,double> {
        double half = std::sqrt(std::max(0.0, wafer_r * wafer_r - pos * pos));
        return {-half, half};
    };

    auto append_lane = [&](const py::dict& lane) {
        auto vx = lane["x_mm"].cast<std::vector<double>>();
        auto vy = lane["y_mm"].cast<std::vector<double>>();
        auto vz = lane["z_mm"].cast<std::vector<double>>();
        auto vt = lane["theta_deg"].cast<std::vector<double>>();
        auto vs = lane["spindle_rpm"].cast<std::vector<double>>();
        auto vp = lane["phase"].cast<std::vector<int>>();
        for (size_t i = 0; i < vx.size(); ++i) {
            ax.push_back(vx[i]); ay.push_back(vy[i]); az.push_back(vz[i]);
            atheta.push_back(vt[i]); aspindle.push_back(vs[i]); aphase.push_back(vp[i]);
        }
        total_cut_mm += lane["cut_len_mm"].cast<double>();
        ++total_lanes;
    };

    // X-direction streets (θ=0°): blade feeds in +X
    for (double sy : streets_y) {
        if (std::abs(sy) > wafer_r) continue;
        std::pair<double,double> cx = clip(sy, true);
        auto lane = dice_lane(cx.first, cx.second, sy, z_air, z_cut, feed_mms, spindle_rpm, 0.0, dt_s);
        append_lane(lane);
    }

    // Y-direction streets (θ=90°): rotate chuck, blade feeds in +X (= wafer +Y)
    for (double sx : streets_x) {
        if (std::abs(sx) > wafer_r) continue;
        std::pair<double,double> cy = clip(sx, false);
        auto lane = dice_lane(cy.first, cy.second, sx, z_air, z_cut, feed_mms, spindle_rpm, 90.0, dt_s);
        append_lane(lane);
    }

    py::dict result;
    result["x_mm"]        = ax;
    result["y_mm"]        = ay;
    result["z_mm"]        = az;
    result["theta_deg"]   = atheta;
    result["spindle_rpm"] = aspindle;
    result["phase"]       = aphase;
    result["total_lanes"]    = total_lanes;
    result["total_cut_mm"]   = total_cut_mm;
    result["total_points"]   = (int)ax.size();
    return result;
}

// ── street grid generator (helper) ──────────────────────────────────────────

// Given die pitch and wafer diameter, return street center positions
std::vector<double> street_positions(double pitch_mm, double wafer_r_mm,
                                     double kerf_mm = 0.0) {
    std::vector<double> pos;
    double p = pitch_mm;
    for (double x = 0.0; x <= wafer_r_mm; x += p) {
        if (x < wafer_r_mm) { pos.push_back(x);  if (x > 0.0) pos.push_back(-x); }
    }
    return pos;
}

// ── module ────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(_5axis_interpolation, m) {
    m.doc() = "5-axis dicing saw trajectory interpolation (X/Y/Z/theta/spindle)";

    m.def("linear_interp", &linear_interp,
          py::arg("start"), py::arg("end"), py::arg("n"),
          "Linear interpolation between two 5D poses [{x,y,z,theta_deg,spindle_rpm}]");

    m.def("arc_interp", &arc_interp,
          py::arg("cx"), py::arg("cy"), py::arg("r"),
          py::arg("a0_deg"), py::arg("a1_deg"),
          py::arg("z0"), py::arg("z1"),
          py::arg("theta_deg"), py::arg("spindle_rpm"),
          py::arg("n"),
          "Helical arc interpolation in XY plane with Z ramp");

    m.def("dice_lane", &dice_lane,
          py::arg("x0"), py::arg("x1"), py::arg("y_mm"),
          py::arg("z_air"), py::arg("z_cut"),
          py::arg("feed_mms"), py::arg("spindle_rpm"),
          py::arg("theta_deg") = 0.0,
          py::arg("dt_s") = 1e-4,
          "Single dicing lane: plunge -> cut -> retract trajectory");

    m.def("full_wafer_grid", &full_wafer_grid,
          py::arg("streets_x"), py::arg("streets_y"),
          py::arg("wafer_r"),
          py::arg("z_air") = 1.0,
          py::arg("z_cut") = -0.3,
          py::arg("feed_mms") = 60.0,
          py::arg("spindle_rpm") = 30000.0,
          py::arg("dt_s") = 1e-4,
          "Full wafer dicing grid trajectory (X+Y streets, 90deg chuck rotation)");

    m.def("street_positions", &street_positions,
          py::arg("pitch_mm"), py::arg("wafer_r_mm"), py::arg("kerf_mm") = 0.0,
          "Generate symmetric street center positions for given die pitch");
}
