// Real C++ compute kernel for the stealth-dicing hot loop, exposed to Python
// via pybind11.
//
// WHY C++ (not just numba): DISCO's dicer/grinder tools run C/C++ on the
// embedded real-time machine-control side (see pipeline/disco_sw_stack.py —
// real-time motion loop + IEC 61131-3 state machine). This extension mirrors
// that architecture in miniature: Python orchestration on top, a compiled C++
// kernel underneath, crossing the FFI boundary exactly the way a production
// tool stack is layered.
//
// It is numerically identical to the scalar stealth_dicing() reference and to
// the NumPy / numba kernels (see tests/test_laser_vectorized.py).
//
// Build:  bash fem/build_cpp_kernel.sh

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>

namespace py = pybind11;

// Match NumPy's double-precision pi so results are bit-comparable.
static const double PI = 3.141592653589793;
static const double WL = 1.064;     // Nd:YAG IR wavelength [um]
static const double N_SIC = 2.65;   // SiC refractive index @ 1064 nm

using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Element-wise surface-chipping kernel. All input arrays are broadcast to a
// common 1-D shape by the Python caller before reaching here.
static py::array_t<double> stealth_chipping(
    Arr P, Arr v, Arr f_hz, Arr n_p, Arr focal_z, Arr NA, Arr n_lay,
    double F_th_MPI)
{
    auto bP = P.unchecked<1>();
    auto bv = v.unchecked<1>();
    auto bf = f_hz.unchecked<1>();
    auto bnp = n_p.unchecked<1>();
    auto bfocal = focal_z.unchecked<1>();
    auto bNA = NA.unchecked<1>();
    auto bnlay = n_lay.unchecked<1>();

    const py::ssize_t n = P.shape(0);
    auto out = py::array_t<double>(n);
    auto bout = out.mutable_unchecked<1>();

    for (py::ssize_t i = 0; i < n; ++i) {
        double r_focal = 0.61 * WL / (bNA(i) * N_SIC);
        if (r_focal < 0.15) r_focal = 0.15;

        const double z_R = PI * r_focal * r_focal / WL * N_SIC;
        const double E_pulse = bP(i) / bf(i);
        const double r_cm = r_focal * 1e-4;
        const double F_peak = E_pulse / (PI * r_cm * r_cm);

        if (F_peak <= F_th_MPI) { bout(i) = 0.0; continue; }

        const double excess = std::log(F_peak / F_th_MPI);
        const double pulse_pitch = bv(i) / bf(i) * 1e3;
        double overlap = 1.0 - pulse_pitch / (2.0 * r_focal);
        if (overlap < 0.05) overlap = 0.05;

        double mod_h = z_R * 2.0 * std::sqrt(excess) * overlap * bnlay(i) * bnp(i);
        const double cap = bfocal(i) * 0.6;
        if (mod_h > cap) mod_h = cap;
        if (mod_h <= 0.0) { bout(i) = 0.0; continue; }

        const double mod_w = 2.0 * r_focal * std::sqrt(excess)
                           * std::pow(bnlay(i), 0.25) * std::pow(bnp(i), 0.2);

        double mm = bfocal(i) / 100.0;
        if (mm > 1.0) mm = 1.0;
        double crack = 75.0 * mm + 45.0 * (1.0 - mm);
        if (crack < 30.0) crack = 30.0;

        double chip = mod_w * 0.25 / std::tan(crack * PI / 180.0);
        if (chip > mod_w) chip = mod_w;
        bout(i) = chip;
    }
    return out;
}

PYBIND11_MODULE(_stealth_kernel, m) {
    m.doc() = "C++ stealth-dicing chipping kernel (pybind11)";
    m.def("stealth_chipping", &stealth_chipping,
          "Element-wise surface chipping [um] for the stealth-dicing model",
          py::arg("P"), py::arg("v"), py::arg("f_hz"), py::arg("n_p"),
          py::arg("focal_z"), py::arg("NA"), py::arg("n_lay"),
          py::arg("F_th_MPI"));
}
