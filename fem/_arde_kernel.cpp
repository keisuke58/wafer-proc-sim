// Vectorized ARDE (Aspect Ratio Dependent Etching) plasma model.
// Mirrors plasma_aspect_ratio_limit() in sic/physical_limits.py.
//
// WHY C++: The ARDE model is called in parameter sweeps (width × thickness ×
// pressure grids) inside the Bayesian optimizer and in validation sweeps.
// Moving the inner arithmetic from Python scalars to a C++ array kernel
// eliminates per-call Python overhead — same "orchestration in Python, hot
// loop in C++" pattern used for the stealth-dicing kernel.
//
// Build: bash build_all_kernels.sh
// Fallback: sic/physical_limits_vectorized.py falls back to NumPy if absent.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;
using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Element-wise ARDE sweep over (trench_width, wafer_thickness, pressure) arrays.
// All three input arrays must have the same length (broadcast done by caller).
static py::dict arde_sweep(
    Arr trench_width_um,
    Arr wafer_thickness_um,
    Arr pressure_mTorr,
    double AR_crit_ref,
    double arde_exp,
    double min_rate_frac)
{
    const double P_ref = 10.0;
    const py::ssize_t n = trench_width_um.shape(0);

    if (wafer_thickness_um.shape(0) != n || pressure_mTorr.shape(0) != n)
        throw std::runtime_error("arde_sweep: all input arrays must have the same length");

    auto bw = trench_width_um.unchecked<1>();
    auto bt = wafer_thickness_um.unchecked<1>();
    auto bp = pressure_mTorr.unchecked<1>();

    auto AR_crit_out  = py::array_t<double>(n);
    auto AR_max_out   = py::array_t<double>(n);
    auto cur_AR_out   = py::array_t<double>(n);
    auto etch_rate_out = py::array_t<double>(n);
    auto feasible_out = py::array_t<bool>(n);

    auto bARc  = AR_crit_out.mutable_unchecked<1>();
    auto bARm  = AR_max_out.mutable_unchecked<1>();
    auto bCAR  = cur_AR_out.mutable_unchecked<1>();
    auto bER   = etch_rate_out.mutable_unchecked<1>();
    auto bFeas = feasible_out.mutable_unchecked<1>();

    const double log_inv_mrf = std::log(1.0 / min_rate_frac - 1.0);

    for (py::ssize_t i = 0; i < n; ++i) {
        const double p = bp(i) < 1.0 ? 1.0 : bp(i);
        const double AR_crit = AR_crit_ref * std::sqrt(p / P_ref);
        // AR_max = AR_crit * (1/min_rate_frac - 1)^(1/n)
        const double AR_max  = AR_crit * std::exp(log_inv_mrf / arde_exp);
        const double cur_AR  = bt(i) / bw(i);
        const double ratio   = cur_AR / AR_crit;
        const double etch_rate = 1.0 / (1.0 + std::pow(ratio, arde_exp));

        bARc(i)  = AR_crit;
        bARm(i)  = AR_max;
        bCAR(i)  = cur_AR;
        bER(i)   = etch_rate;
        bFeas(i) = cur_AR < AR_max;
    }

    py::dict result;
    result["AR_crit"]    = AR_crit_out;
    result["AR_max"]     = AR_max_out;
    result["current_AR"] = cur_AR_out;
    result["etch_rate"]  = etch_rate_out;
    result["feasible"]   = feasible_out;
    return result;
}

PYBIND11_MODULE(_arde_kernel, m) {
    m.doc() = "C++ ARDE plasma model kernel (pybind11)";
    m.def("arde_sweep", &arde_sweep,
          "Vectorized ARDE model: (trench_width, wafer_thickness, pressure) arrays "
          "-> dict of AR_crit, AR_max, current_AR, etch_rate, feasible arrays",
          py::arg("trench_width_um"),
          py::arg("wafer_thickness_um"),
          py::arg("pressure_mTorr"),
          py::arg("AR_crit_ref")   = 8.0,
          py::arg("arde_exp")      = 2.0,
          py::arg("min_rate_frac") = 0.05);
}
