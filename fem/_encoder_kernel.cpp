/*
 * _encoder_kernel.cpp
 *
 * ABZ quadrature encoder: 4x position decode + velocity estimation.
 *
 * DISCO context:
 *   - XY stage linear encoder: 0.1 µm resolution (≈ 2500 PPR glass scale)
 *   - Spindle encoder: 2500 PPR → 10 000 counts/rev with 4x decode
 *   - Velocity from M/T method (count-over-time)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <cstdint>

namespace py = pybind11;

static const double TWO_PI = 2.0 * M_PI;

// ── 4x quadrature decode (state machine) ─────────────────────────────────────

// a_sig, b_sig: integer arrays of 0/1 samples
// Returns cumulative count array (int64) of same length.
py::array_t<int64_t> decode_quadrature(
    const py::array_t<int8_t>& a_arr,
    const py::array_t<int8_t>& b_arr)
{
    auto a = a_arr.unchecked<1>();
    auto b = b_arr.unchecked<1>();
    ssize_t N = a.shape(0);
    if (b.shape(0) != N)
        throw std::invalid_argument("a and b must have same length");

    auto counts_arr = py::array_t<int64_t>(N);
    auto counts = counts_arr.mutable_unchecked<1>();

    // 4x state machine: state = (A<<1)|B, direction from XOR table
    // Transitions: 00→01=+1, 01→11=+1, 11→10=+1, 10→00=+1 (CW)
    //              00→10=-1, 10→11=-1, 11→01=-1, 01→00=-1 (CCW)
    static const int8_t lut[4][4] = {
        // to: 00  01  10  11
        {  0, +1, -1,  0 },  // from 00
        { -1,  0,  0, +1 },  // from 01
        { +1,  0,  0, -1 },  // from 10
        {  0, -1, +1,  0 },  // from 11
    };

    int64_t cnt = 0;
    int prev = ((int)a(0) << 1) | (int)b(0);
    counts(0) = 0;

    for (ssize_t i = 1; i < N; ++i) {
        int cur = ((int)a(i) << 1) | (int)b(i);
        cnt += lut[prev][cur];
        counts(i) = cnt;
        prev = cur;
    }
    return counts_arr;
}

// ── Velocity from count array (M/T method, simple finite difference) ─────────

// counts: cumulative count array, dt_s: sample interval [s], ppr: pulses/rev
// Returns velocity in rpm (same length as counts, first element = 0)
py::array_t<double> velocity_from_counts(
    const py::array_t<int64_t>& counts_arr,
    double dt_s, int ppr, int avg_window = 1)
{
    if (dt_s <= 0.0) throw std::invalid_argument("dt_s must be > 0");
    if (ppr <= 0)    throw std::invalid_argument("ppr must be > 0");
    auto c = counts_arr.unchecked<1>();
    ssize_t N = c.shape(0);
    auto vel_arr = py::array_t<double>(N);
    auto vel = vel_arr.mutable_unchecked<1>();
    vel(0) = 0.0;

    double cpr = (double)ppr * 4.0;   // counts per revolution (4x decode)
    for (ssize_t i = 1; i < N; ++i) {
        ssize_t i0 = (i > avg_window) ? (i - avg_window) : 0;
        double delta_count = (double)(c(i) - c(i0));
        double delta_t     = (double)(i - i0) * dt_s;
        vel(i) = delta_count / (cpr * delta_t) * 60.0;
    }
    return vel_arr;
}

// ── EncoderSimulator class ────────────────────────────────────────────────────

class EncoderSimulator {
public:
    int ppr;           // pulses per revolution
    double jitter_rad; // timing jitter [rad] for realism

    int64_t count;
    int     prev_state;

    explicit EncoderSimulator(int ppr_=2500, double jitter_rad_=0.0)
        : ppr(ppr_), jitter_rad(jitter_rad_), count(0), prev_state(0) {}

    // Encode true angular position to A/B signals (with optional quantisation)
    // Returns {A, B} as 0/1 integers
    std::array<int,2> encode(double theta_rad) const {
        double cpr = (double)ppr * 4.0;
        double phase = std::fmod(theta_rad * cpr / TWO_PI, 4.0);
        if (phase < 0.0) phase += 4.0;
        int quad = (int)phase;  // 0,1,2,3
        // Gray code: 00,01,11,10
        static const int A_lut[4] = {0,0,1,1};
        static const int B_lut[4] = {0,1,1,0};
        return {A_lut[quad], B_lut[quad]};
    }

    // Simulate over a position profile [rad]; returns decoded counts + rpm
    py::dict simulate(const std::vector<double>& theta_rad_profile, double dt_s) {
        int N = (int)theta_rad_profile.size();
        if (N == 0) throw std::invalid_argument("profile must not be empty");

        std::vector<int8_t> a_sig(N), b_sig(N);
        for (int i = 0; i < N; ++i) {
            auto ab = encode(theta_rad_profile[i]);
            a_sig[i] = (int8_t)ab[0];
            b_sig[i] = (int8_t)ab[1];
        }

        // Decode
        py::array_t<int8_t> a_arr(N), b_arr(N);
        std::copy(a_sig.begin(), a_sig.end(),
                  a_arr.mutable_unchecked<1>().mutable_data(0));
        std::copy(b_sig.begin(), b_sig.end(),
                  b_arr.mutable_unchecked<1>().mutable_data(0));
        auto cnt_arr = decode_quadrature(a_arr, b_arr);
        auto vel_arr = velocity_from_counts(cnt_arr, dt_s, ppr, 4);

        // Position from counts [rad]
        double cpr = (double)ppr * 4.0;
        std::vector<double> pos_rad(N);
        auto cnt_acc = cnt_arr.unchecked<1>();
        for (int i = 0; i < N; ++i)
            pos_rad[i] = (double)cnt_acc(i) / cpr * TWO_PI;

        // Quantisation error
        std::vector<double> err_rad(N);
        for (int i = 0; i < N; ++i)
            err_rad[i] = theta_rad_profile[i] - pos_rad[i];

        py::dict result;
        result["counts"]  = cnt_arr;
        result["pos_rad"] = pos_rad;
        result["vel_rpm"] = vel_arr;
        result["err_rad"] = err_rad;
        result["a_sig"]   = a_sig;
        result["b_sig"]   = b_sig;
        return result;
    }

    double position_resolution_rad() const {
        return TWO_PI / ((double)ppr * 4.0);
    }

    void reset() { count = 0; prev_state = 0; }
};

PYBIND11_MODULE(_encoder_kernel, m) {
    m.doc() = "ABZ quadrature encoder: 4x decode and velocity estimation";

    m.def("decode_quadrature", &decode_quadrature,
          py::arg("a_sig"), py::arg("b_sig"),
          "4x quadrature decode of A/B signal arrays -> cumulative count array");

    m.def("velocity_from_counts", &velocity_from_counts,
          py::arg("counts"), py::arg("dt_s"), py::arg("ppr"),
          py::arg("avg_window") = 1,
          "Velocity [rpm] from count array using finite difference");

    py::class_<EncoderSimulator>(m, "EncoderSimulator")
        .def(py::init<int, double>(),
             py::arg("ppr") = 2500, py::arg("jitter_rad") = 0.0)
        .def("encode",   &EncoderSimulator::encode,
             py::arg("theta_rad"),
             "True position -> {A, B} quadrature signals")
        .def("simulate", &EncoderSimulator::simulate,
             py::arg("theta_rad_profile"), py::arg("dt_s"),
             "Simulate encoder over position profile -> counts/pos/vel/err")
        .def("position_resolution_rad", &EncoderSimulator::position_resolution_rad)
        .def("reset",    &EncoderSimulator::reset)
        .def_readwrite("ppr", &EncoderSimulator::ppr);
}
