/*
 * _servo_inverter_kernel.cpp
 *
 * 3-phase servo drive inverter: Space Vector PWM (SVPWM) with dead-time
 * compensation and switching loss estimation.
 *
 * DISCO context: spindle servo 30 000 rpm PMSM, 600 V DC bus, 15 kHz switching.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;

// ── SVPWM ─────────────────────────────────────────────────────────────────────

py::dict svpwm(double v_alpha, double v_beta, double v_dc) {
    if (v_dc <= 0.0) throw std::invalid_argument("v_dc must be > 0");

    double V_a =  v_alpha;
    double V_b = (-v_alpha + std::sqrt(3.0) * v_beta) / 2.0;
    double V_c = (-v_alpha - std::sqrt(3.0) * v_beta) / 2.0;

    // Sector from sign pattern
    int sector;
    if      (V_a >= 0 && V_b <  0 && V_c <  0) sector = 1;
    else if (V_a >= 0 && V_b >= 0 && V_c <  0) sector = 2;
    else if (V_a <  0 && V_b >= 0 && V_c <  0) sector = 3;
    else if (V_a <  0 && V_b >= 0 && V_c >= 0) sector = 4;
    else if (V_a <  0 && V_b <  0 && V_c >= 0) sector = 5;
    else                                         sector = 6;

    // SVPWM common-mode injection: centres triangle to extend linear range
    double v_max = std::max(V_a, std::max(V_b, V_c));
    double v_min = std::min(V_a, std::min(V_b, V_c));
    double v_off = -(v_max + v_min) / 2.0;

    auto clamp01 = [](double x){ return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); };
    double da = clamp01((V_a + v_off) / v_dc + 0.5);
    double db = clamp01((V_b + v_off) / v_dc + 0.5);
    double dc = clamp01((V_c + v_off) / v_dc + 0.5);

    double v_mag = std::sqrt(v_alpha*v_alpha + v_beta*v_beta);

    py::dict r;
    r["duty_a"] = da; r["duty_b"] = db; r["duty_c"] = dc;
    r["sector"] = sector;
    r["modulation_index"] = v_mag / (v_dc / std::sqrt(3.0));
    return r;
}

// ── Dead-time compensation ─────────────────────────────────────────────────────

std::array<double,3> dead_time_compensate(
    const std::array<double,3>& duties,
    const std::array<double,3>& currents,
    double t_dead_us, double T_sw_us)
{
    if (T_sw_us <= 0.0) throw std::invalid_argument("T_sw_us must be > 0");
    double ratio = t_dead_us / T_sw_us;
    std::array<double,3> comp;
    for (int i = 0; i < 3; ++i) {
        double sign = (currents[i] >= 0.0) ? -1.0 : 1.0;
        double d = duties[i] + sign * ratio;
        comp[i] = d < 0.0 ? 0.0 : (d > 1.0 ? 1.0 : d);
    }
    return comp;
}

// ── Switching loss (per-bridge, simplified datasheet model) ──────────────────

double switching_loss(double i_rms_A, double v_dc_V,
                      double E_on_mJ, double E_off_mJ, double f_sw_kHz,
                      double i_rated_A = 20.0, double v_rated_V = 600.0) {
    double E_J = (E_on_mJ + E_off_mJ) * 1e-3;
    return 6.0 * f_sw_kHz * 1e3 * E_J
           * (std::abs(i_rms_A) / i_rated_A)
           * (v_dc_V / v_rated_V);
}

// ── InverterSimulator ──────────────────────────────────────────────────────────

class InverterSimulator {
public:
    double v_dc, t_dead_us, T_sw_us, E_on_mJ, E_off_mJ;
    double i_a, i_b, i_c, energy_loss_J;

    InverterSimulator(double v_dc_=600.0, double t_dead_us_=1.5,
                      double f_sw_kHz=15.0,
                      double E_on_mJ_=0.5, double E_off_mJ_=0.3)
        : v_dc(v_dc_), t_dead_us(t_dead_us_), T_sw_us(1e3/f_sw_kHz),
          E_on_mJ(E_on_mJ_), E_off_mJ(E_off_mJ_),
          i_a(0), i_b(0), i_c(0), energy_loss_J(0) {}

    void set_currents(double ia, double ib, double ic)
        { i_a=ia; i_b=ib; i_c=ic; }

    py::dict step(double v_alpha, double v_beta) {
        auto sv = svpwm(v_alpha, v_beta, v_dc);
        std::array<double,3> d = {sv["duty_a"].cast<double>(),
                                   sv["duty_b"].cast<double>(),
                                   sv["duty_c"].cast<double>()};
        std::array<double,3> curr = {i_a, i_b, i_c};
        auto cd = dead_time_compensate(d, curr, t_dead_us, T_sw_us);

        double i_rms = std::sqrt((i_a*i_a + i_b*i_b + i_c*i_c) / 3.0);
        double p_sw  = switching_loss(i_rms, v_dc, E_on_mJ, E_off_mJ,
                                      1e3/T_sw_us);
        energy_loss_J += p_sw * T_sw_us * 1e-6;

        py::dict r;
        r["v_a_V"]  = (cd[0]-0.5)*v_dc;
        r["v_b_V"]  = (cd[1]-0.5)*v_dc;
        r["v_c_V"]  = (cd[2]-0.5)*v_dc;
        r["duty_a"] = cd[0]; r["duty_b"] = cd[1]; r["duty_c"] = cd[2];
        r["P_sw_W"] = p_sw;
        r["sector"] = sv["sector"];
        return r;
    }

    py::dict simulate(int n, double v_alpha, double v_beta) {
        std::vector<double> va,vb,vc,da,db,dc,psw;
        std::vector<int> sec;
        va.reserve(n); vb.reserve(n); vc.reserve(n);
        for (int k=0;k<n;++k) {
            auto r = step(v_alpha, v_beta);
            va.push_back(r["v_a_V"].cast<double>());
            vb.push_back(r["v_b_V"].cast<double>());
            vc.push_back(r["v_c_V"].cast<double>());
            da.push_back(r["duty_a"].cast<double>());
            db.push_back(r["duty_b"].cast<double>());
            dc.push_back(r["duty_c"].cast<double>());
            psw.push_back(r["P_sw_W"].cast<double>());
            sec.push_back(r["sector"].cast<int>());
        }
        py::dict out;
        out["v_a_V"]=va; out["v_b_V"]=vb; out["v_c_V"]=vc;
        out["duty_a"]=da; out["duty_b"]=db; out["duty_c"]=dc;
        out["P_sw_W"]=psw; out["sector"]=sec;
        out["total_energy_loss_J"]=energy_loss_J;
        return out;
    }

    void reset() { i_a=i_b=i_c=energy_loss_J=0.0; }
};

PYBIND11_MODULE(_servo_inverter_kernel, m) {
    m.doc() = "3-phase SVPWM inverter with dead-time compensation and switching loss";

    m.def("svpwm", &svpwm,
          py::arg("v_alpha"), py::arg("v_beta"), py::arg("v_dc"));
    m.def("dead_time_compensate", &dead_time_compensate,
          py::arg("duties"), py::arg("currents"),
          py::arg("t_dead_us"), py::arg("T_sw_us"));
    m.def("switching_loss", &switching_loss,
          py::arg("i_rms_A"), py::arg("v_dc_V"),
          py::arg("E_on_mJ"), py::arg("E_off_mJ"), py::arg("f_sw_kHz"),
          py::arg("i_rated_A")=20.0, py::arg("v_rated_V")=600.0);

    py::class_<InverterSimulator>(m, "InverterSimulator")
        .def(py::init<double,double,double,double,double>(),
             py::arg("v_dc")=600.0, py::arg("t_dead_us")=1.5,
             py::arg("f_sw_kHz")=15.0,
             py::arg("E_on_mJ")=0.5, py::arg("E_off_mJ")=0.3)
        .def("set_currents", &InverterSimulator::set_currents)
        .def("step",         &InverterSimulator::step)
        .def("simulate",     &InverterSimulator::simulate)
        .def("reset",        &InverterSimulator::reset)
        .def_readwrite("v_dc",          &InverterSimulator::v_dc)
        .def_readwrite("t_dead_us",     &InverterSimulator::t_dead_us)
        .def_readwrite("energy_loss_J", &InverterSimulator::energy_loss_J);
}
