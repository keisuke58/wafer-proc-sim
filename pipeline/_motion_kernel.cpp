// Real-time P-controller motion kernel (pybind11).
// Mirrors RTMotionLoop.move_to() in pipeline/disco_sw_stack.py.
//
// WHY C++: DISCO's actual servo controller runs at 1 kHz inside a hard-real-time
// C/C++ task (PLCopen MC_MoveAbsolute pattern). This kernel reproduces the same
// numeric cycle — proportional control, velocity saturation, position integration,
// settle check — at C speed. The Python caller stays the orchestration layer;
// the tight loop crosses the FFI boundary exactly the way the embedded RT task
// crosses the OS/RT boundary.
//
// The kernel also records full per-cycle trajectory data as NumPy arrays so that
// the existing Python logging / visualization code requires no changes.
//
// Build: bash build_all_kernels.sh

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;

// Run the P-controller loop from start_mm to target_mm.
// Returns a dict: pos_mm, error_um, vel_mms (all 1-D arrays), n_cycles (int).
static py::dict move_to(
    double start_mm,
    double target_mm,
    double Kp,
    double max_vel_mms,
    double dt_s,
    double settle_um,
    int    max_cycles)
{
    if (max_cycles <= 0)
        throw std::runtime_error("move_to: max_cycles must be positive");
    if (dt_s <= 0.0)
        throw std::runtime_error("move_to: dt_s must be positive");

    std::vector<double> pos_log, err_log, vel_log;
    pos_log.reserve(max_cycles);
    err_log.reserve(max_cycles);
    vel_log.reserve(max_cycles);

    double pos    = start_mm;
    int    n_used = max_cycles;
    const  double settle_mm = settle_um * 1e-3;

    for (int i = 0; i < max_cycles; ++i) {
        const double error   = target_mm - pos;
        double cmd_vel = Kp * error;
        cmd_vel = std::max(-max_vel_mms, std::min(max_vel_mms, cmd_vel));
        pos += cmd_vel * dt_s;

        pos_log.push_back(pos);
        err_log.push_back(error * 1000.0);  // convert to µm
        vel_log.push_back(cmd_vel);

        if (std::abs(error) < settle_mm) {
            n_used = i + 1;
            break;
        }
    }

    const int n = (int)pos_log.size();
    auto out_pos = py::array_t<double>(n);
    auto out_err = py::array_t<double>(n);
    auto out_vel = py::array_t<double>(n);
    std::copy(pos_log.begin(), pos_log.end(), out_pos.mutable_data());
    std::copy(err_log.begin(), err_log.end(), out_err.mutable_data());
    std::copy(vel_log.begin(), vel_log.end(), out_vel.mutable_data());

    py::dict result;
    result["pos_mm"]   = out_pos;
    result["error_um"] = out_err;
    result["vel_mms"]  = out_vel;
    result["n_cycles"] = n_used;
    return result;
}

// Simulate a multi-segment move sequence (list of targets).
// Returns concatenated trajectory arrays across all segments.
static py::dict move_sequence(
    double              start_mm,
    py::list            targets_mm,
    double              Kp,
    double              max_vel_mms,
    double              dt_s,
    double              settle_um,
    int                 max_cycles_per_segment)
{
    std::vector<double> all_pos, all_err, all_vel;
    std::vector<int>    seg_ends;

    double pos = start_mm;
    for (auto t : targets_mm) {
        double target = py::cast<double>(t);
        auto seg = move_to(pos, target, Kp, max_vel_mms, dt_s, settle_um,
                           max_cycles_per_segment);

        auto s_pos = seg["pos_mm"].cast<py::array_t<double>>();
        auto s_err = seg["error_um"].cast<py::array_t<double>>();
        auto s_vel = seg["vel_mms"].cast<py::array_t<double>>();
        const int ns = (int)s_pos.size();

        all_pos.insert(all_pos.end(), s_pos.data(), s_pos.data() + ns);
        all_err.insert(all_err.end(), s_err.data(), s_err.data() + ns);
        all_vel.insert(all_vel.end(), s_vel.data(), s_vel.data() + ns);
        seg_ends.push_back((int)all_pos.size());

        // Start next segment from settled position.
        pos = all_pos.back();
    }

    const int n = (int)all_pos.size();
    auto out_pos = py::array_t<double>(n);
    auto out_err = py::array_t<double>(n);
    auto out_vel = py::array_t<double>(n);
    std::copy(all_pos.begin(), all_pos.end(), out_pos.mutable_data());
    std::copy(all_err.begin(), all_err.end(), out_err.mutable_data());
    std::copy(all_vel.begin(), all_vel.end(), out_vel.mutable_data());

    py::list seg_list;
    for (int v : seg_ends) seg_list.append(v);

    py::dict result;
    result["pos_mm"]   = out_pos;
    result["error_um"] = out_err;
    result["vel_mms"]  = out_vel;
    result["seg_ends"] = seg_list;
    return result;
}

PYBIND11_MODULE(_motion_kernel, m) {
    m.doc() = "C++ RT P-controller motion kernel (pybind11)";
    m.def("move_to", &move_to,
          "Run P-controller from start_mm to target_mm, return trajectory dict "
          "(pos_mm, error_um, vel_mms arrays + n_cycles int)",
          py::arg("start_mm")    = 0.0,
          py::arg("target_mm")   = 10.0,
          py::arg("Kp")          = 50.0,
          py::arg("max_vel_mms") = 300.0,
          py::arg("dt_s")        = 0.001,
          py::arg("settle_um")   = 1.0,
          py::arg("max_cycles")  = 500);
    m.def("move_sequence", &move_sequence,
          "Run multi-segment move: each target in targets_mm is a separate "
          "move_to call; returns concatenated trajectory + seg_ends index list",
          py::arg("start_mm"),
          py::arg("targets_mm"),
          py::arg("Kp")                       = 50.0,
          py::arg("max_vel_mms")              = 300.0,
          py::arg("dt_s")                     = 0.001,
          py::arg("settle_um")                = 1.0,
          py::arg("max_cycles_per_segment")   = 500);
}
