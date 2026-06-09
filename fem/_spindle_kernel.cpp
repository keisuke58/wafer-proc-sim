// PMSM + FOC spindle motor simulation kernel (pybind11).
// Mirrors the Euler integration loop in fem/spindle_motor_control.py.
//
// WHY C++: DISCO's dicing saw spindle (DFD6361, 60 krpm, 400 W) is driven by
// a PMSM with Field-Oriented Control. The embedded drive firmware runs the
// identical dq-model + PI-controller loop in a hard-real-time ISR at 10–50 µs
// intervals (same timestep as this simulation). This kernel is the exact same
// structure moved to C++ — each call to step() or simulate() mirrors one ISR
// execution of the real drive.
//
// Two usage modes:
//   simulate()  — batch: run N steps, return full trajectory arrays.
//   step()      — real-time: advance one ISR tick, caller drives the loop.
//
// Physics (dq-axis model, Vas 1998):
//   v_d = R·i_d + L_d·di_d/dt  − ω·L_q·i_q
//   v_q = R·i_q + L_q·di_q/dt  + ω·(L_d·i_d + ψ_f)
//   T_e = 1.5·p·(ψ_f·i_q + (L_d−L_q)·i_d·i_q)
//   J·dω/dt = T_e − T_load − B·ω
//
// Build: bash build_all_kernels.sh

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace py = pybind11;

// ── PMSMSimulator class ───────────────────────────────────────────────────────

class PMSMSimulator {
public:
    // Motor & controller parameters (all SI units).
    double R;             // stator resistance [Ω]
    double Ld;            // d-axis inductance [H]
    double Lq;            // q-axis inductance [H]
    double psi_f;         // PM flux linkage [Wb]
    int    p;             // pole pairs
    double J;             // rotor inertia [kg·m²]
    double B;             // viscous damping [N·m·s]
    double r_blade;       // blade radius [m]
    double F_cut;         // tangential cutting force [N]
    double Kp_i;          // PI proportional gain
    double Ki_i;          // PI integral gain
    double dt;            // timestep [s]
    double omega_ref;     // reference speed [rad/s]
    double omega_ref_rpm; // reference speed [rpm] (for contact calculation)
    double contact_duty;  // fraction of revolution with blade contact

    // Integrator state.
    double   omega;
    double   i_d;
    double   i_q;
    double   int_iq;
    uint64_t _step;       // step counter — used for contact check to avoid FP drift

    PMSMSimulator(
        double omega_ref_rpm_,
        double R_, double L_d_mH, double L_q_mH,
        double psi_f_, int p_pairs,
        double J_, double B_,
        double r_blade_mm, double F_cut_N,
        double Kp_i_, double Ki_i_,
        double dt_us,
        double contact_duty_ = 0.3)
    :   R(R_), Ld(L_d_mH * 1e-3), Lq(L_q_mH * 1e-3),
        psi_f(psi_f_), p(p_pairs), J(J_), B(B_),
        r_blade(r_blade_mm * 1e-3), F_cut(F_cut_N),
        Kp_i(Kp_i_), Ki_i(Ki_i_),
        dt(dt_us * 1e-6),
        omega_ref(omega_ref_rpm_ * 2.0 * M_PI / 60.0),
        omega_ref_rpm(omega_ref_rpm_),
        contact_duty(contact_duty_),
        omega(omega_ref_rpm_ * 2.0 * M_PI / 60.0 * 0.9),
        i_d(0.0), i_q(0.0), int_iq(0.0), _step(0)
    {}

    void reset() {
        omega  = omega_ref * 0.9;
        i_d = i_q = int_iq = 0.0;
        _step = 0;
    }

    // Advance one ISR tick.
    // Returns {omega_rad_s, i_d_A, i_q_A, T_e_Nm, P_elec_W, T_load_Nm}.
    std::array<double, 6> step() {
        const double T_load_base = F_cut * r_blade;

        // Blade-contact load profile.
        // Use step counter * dt to match Python's `t = k * dt` exactly —
        // accumulated t += dt accumulates FP rounding error over 5000+ steps,
        // causing contact-check divergence from the Python reference.
        const double t_k = (double)_step * dt;
        const bool in_contact =
            std::fmod(t_k * omega_ref_rpm / 60.0, 1.0) < contact_duty;
        const double T_l = in_contact ? T_load_base : T_load_base * 0.1;

        // Outer speed loop → iq reference (simplified PI).
        const double err_omega = omega_ref - omega;
        const double i_q_ref   = Kp_i * err_omega * 0.01;

        // Inner current PI (q-axis only; i_d* = 0 for MTPA on non-salient PMSM).
        const double err_iq = i_q_ref - i_q;
        int_iq += err_iq * dt;
        const double v_q = Kp_i * err_iq + Ki_i * int_iq;
        const double v_d = 0.0;  // i_d* = 0

        // dq-axis dynamics (Euler forward).
        const double di_d = (v_d - R * i_d + omega * p * Lq * i_q) / Ld;
        const double di_q = (v_q - R * i_q - omega * p * (Ld * i_d + psi_f)) / Lq;
        i_d += di_d * dt;
        i_q += di_q * dt;

        // Electromagnetic torque.
        const double T_e = 1.5 * p * (psi_f * i_q + (Ld - Lq) * i_d * i_q);

        // Mechanical dynamics.
        const double domega = (T_e - T_l - B * omega) / J;
        omega += domega * dt;

        // Electrical power (1.5 factor for 3-phase power from dq).
        const double P_elec = 1.5 * (v_d * i_d + v_q * i_q);

        ++_step;
        return {omega, i_d, i_q, T_e, P_elec, T_l};
    }

    // Batch simulation: run `n_steps` steps, return trajectory dict.
    py::dict simulate(int n_steps) {
        if (n_steps <= 0)
            throw std::runtime_error("simulate: n_steps must be positive");

        std::vector<double> v_omega(n_steps), v_id(n_steps), v_iq(n_steps);
        std::vector<double> v_Te(n_steps), v_P(n_steps), v_Tl(n_steps);

        for (int k = 0; k < n_steps; ++k) {
            auto s = step();
            v_omega[k] = s[0];
            v_id[k]    = s[1];
            v_iq[k]    = s[2];
            v_Te[k]    = s[3];
            v_P[k]     = s[4];
            v_Tl[k]    = s[5];
        }

        auto mk = [&](std::vector<double>& v) {
            auto arr = py::array_t<double>(n_steps);
            std::copy(v.begin(), v.end(), arr.mutable_data());
            return arr;
        };

        py::dict result;
        result["omega_rad_s"]  = mk(v_omega);
        result["omega_rpm"]    = [&]{
            auto arr = py::array_t<double>(n_steps);
            auto d   = arr.mutable_data();
            for (int k = 0; k < n_steps; ++k)
                d[k] = v_omega[k] * 60.0 / (2.0 * M_PI);
            return arr;
        }();
        result["i_d_A"]        = mk(v_id);
        result["i_q_A"]        = mk(v_iq);
        result["T_e_Nm"]       = mk(v_Te);
        result["P_elec_W"]     = mk(v_P);
        result["T_load_Nm"]    = mk(v_Tl);
        return result;
    }

    // Convenience: run until settled (|omega - omega_ref| / omega_ref < tol).
    py::dict simulate_until_settled(double tol = 0.001, int max_steps = 100000) {
        std::vector<double> v_omega, v_id, v_iq, v_Te, v_P;
        v_omega.reserve(max_steps);
        v_id.reserve(max_steps);
        v_iq.reserve(max_steps);
        v_Te.reserve(max_steps);
        v_P.reserve(max_steps);

        int n_used = max_steps;
        for (int k = 0; k < max_steps; ++k) {
            auto s = step();
            v_omega.push_back(s[0]);
            v_id.push_back(s[1]);
            v_iq.push_back(s[2]);
            v_Te.push_back(s[3]);
            v_P.push_back(s[4]);
            if (std::abs(s[0] - omega_ref) / omega_ref < tol) {
                n_used = k + 1;
                break;
            }
        }

        const int n = (int)v_omega.size();
        auto mk = [&](std::vector<double>& v) {
            auto arr = py::array_t<double>(n);
            std::copy(v.begin(), v.end(), arr.mutable_data());
            return arr;
        };

        py::dict result;
        result["omega_rad_s"] = mk(v_omega);
        result["omega_rpm"]   = [&]{
            auto arr = py::array_t<double>(n);
            auto d   = arr.mutable_data();
            for (int k = 0; k < n; ++k)
                d[k] = v_omega[k] * 60.0 / (2.0 * M_PI);
            return arr;
        }();
        result["i_d_A"]      = mk(v_id);
        result["i_q_A"]      = mk(v_iq);
        result["T_e_Nm"]     = mk(v_Te);
        result["P_elec_W"]   = mk(v_P);
        result["n_steps"]    = n_used;
        result["settled"]    = (n_used < max_steps);
        return result;
    }

    // Speed ripple: (max - min) / omega_ref over a window.
    double speed_ripple_pct(const py::array_t<double>& omega_arr) const {
        const int n = (int)omega_arr.size();
        if (n == 0) return 0.0;
        const double* d = omega_arr.data();
        double mn = d[0], mx = d[0];
        for (int i = 1; i < n; ++i) {
            if (d[i] < mn) mn = d[i];
            if (d[i] > mx) mx = d[i];
        }
        return (mx - mn) / omega_ref * 100.0;
    }

    double get_omega()     const { return omega; }
    double get_omega_rpm() const { return omega * 60.0 / (2.0 * M_PI); }
    double get_i_d()       const { return i_d; }
    double get_i_q()       const { return i_q; }
    double get_t()         const { return (double)_step * dt; }
};


// ── Standalone batch function (no class state) ────────────────────────────────

static py::dict simulate_foc(
    double omega_ref_rpm,
    double R, double L_d_mH, double L_q_mH,
    double psi_f, int p_pairs,
    double J, double B,
    double r_blade_mm, double F_cut_N,
    double Kp_i, double Ki_i,
    double dt_us, int n_steps,
    double contact_duty)
{
    PMSMSimulator sim(omega_ref_rpm, R, L_d_mH, L_q_mH, psi_f, p_pairs,
                      J, B, r_blade_mm, F_cut_N, Kp_i, Ki_i, dt_us,
                      contact_duty);
    return sim.simulate(n_steps);
}


PYBIND11_MODULE(_spindle_kernel, m) {
    m.doc() = "PMSM + FOC spindle motor simulation kernel (pybind11)\n"
              "Mirrors the Euler integration loop in fem/spindle_motor_control.py.\n"
              "Each step() call is one ISR tick of the embedded drive firmware.";

    py::class_<PMSMSimulator>(m, "PMSMSimulator",
        "PMSM + FOC simulator.\n\n"
        "Constructor args (all SI unless noted):\n"
        "  omega_ref_rpm, R_ohm, L_d_mH, L_q_mH, psi_f_Wb, p_pairs,\n"
        "  J_kgm2, B_Nms, r_blade_mm, F_cut_N, Kp_i, Ki_i, dt_us,\n"
        "  contact_duty (default 0.3).\n\n"
        "Use simulate(n) for batch mode or step() for real-time.")
        .def(py::init<double,double,double,double,double,int,
                      double,double,double,double,double,double,double,double>(),
             py::arg("omega_ref_rpm") = 30000.0,
             py::arg("R_ohm")        = 0.5,
             py::arg("L_d_mH")       = 1.2,
             py::arg("L_q_mH")       = 1.4,
             py::arg("psi_f_Wb")     = 0.08,
             py::arg("p_pairs")      = 2,
             py::arg("J_kgm2")       = 5e-5,
             py::arg("B_Nms")        = 1e-4,
             py::arg("r_blade_mm")   = 55.0,
             py::arg("F_cut_N")      = 15.0,
             py::arg("Kp_i")         = 5.0,
             py::arg("Ki_i")         = 200.0,
             py::arg("dt_us")        = 10.0,
             py::arg("contact_duty") = 0.3)
        .def("reset",    &PMSMSimulator::reset,
             "Reset integrator state to initial conditions.")
        .def("step",     &PMSMSimulator::step,
             "Advance one ISR tick. Returns [omega, i_d, i_q, T_e, P_elec, T_load].")
        .def("simulate", &PMSMSimulator::simulate,
             "Run n_steps steps from current state. Returns trajectory dict.",
             py::arg("n_steps"))
        .def("simulate_until_settled", &PMSMSimulator::simulate_until_settled,
             "Run until |omega - omega_ref| / omega_ref < tol.",
             py::arg("tol") = 0.001, py::arg("max_steps") = 100000)
        .def("speed_ripple_pct", &PMSMSimulator::speed_ripple_pct,
             "Compute (omega_max - omega_min) / omega_ref * 100 over omega array.",
             py::arg("omega_rad_s"))
        .def_property_readonly("omega",      &PMSMSimulator::get_omega)
        .def_property_readonly("omega_rpm",  &PMSMSimulator::get_omega_rpm)
        .def_property_readonly("i_d",        &PMSMSimulator::get_i_d)
        .def_property_readonly("i_q",        &PMSMSimulator::get_i_q)
        .def_property_readonly("t",          &PMSMSimulator::get_t)
        .def_readwrite("omega_ref",    &PMSMSimulator::omega_ref)
        .def_readwrite("F_cut",        &PMSMSimulator::F_cut)
        .def_readwrite("contact_duty", &PMSMSimulator::contact_duty)
        .def("__repr__", [](const PMSMSimulator& s){
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "<PMSMSimulator omega=%.0f rpm, t=%.3f ms>",
                s.get_omega_rpm(), s.get_t() * 1e3);
            return std::string(buf);
        });

    m.def("simulate_foc", &simulate_foc,
          "Standalone batch FOC simulation (no persistent state).",
          py::arg("omega_ref_rpm") = 30000.0,
          py::arg("R_ohm")        = 0.5,
          py::arg("L_d_mH")       = 1.2,
          py::arg("L_q_mH")       = 1.4,
          py::arg("psi_f_Wb")     = 0.08,
          py::arg("p_pairs")      = 2,
          py::arg("J_kgm2")       = 5e-5,
          py::arg("B_Nms")        = 1e-4,
          py::arg("r_blade_mm")   = 55.0,
          py::arg("F_cut_N")      = 15.0,
          py::arg("Kp_i")         = 5.0,
          py::arg("Ki_i")         = 200.0,
          py::arg("dt_us")        = 10.0,
          py::arg("n_steps")      = 5000,
          py::arg("contact_duty") = 0.3);
}
