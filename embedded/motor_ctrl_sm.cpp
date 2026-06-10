// motor_ctrl_sm.cpp — Bare-metal style PMSM FOC state machine (pybind11)
//
// Simulates a 10 kHz interrupt-driven servo control loop as it would run on
// a DSP or FPGA in a DISCO spindle / linear stage drive.
//
// State machine:
//   IDLE → POWERON → INIT → RAMPING → STEADY → DECEL → IDLE
//                                         ↓
//                                       FAULT → (clear_fault) → IDLE
//                              (any) → ESTOP
//
// Inner loop: FOC (field-oriented control) with PI current controllers (d-q frame)
// Outer loop: PI speed controller

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace py = pybind11;

// ── State enum ───────────────────────────────────────────────────────────────

enum class MotorState : uint8_t {
    IDLE     = 0,
    POWERON  = 1,
    INIT     = 2,
    RAMPING  = 3,
    STEADY   = 4,
    DECEL    = 5,
    FAULT    = 6,
    ESTOP    = 7,
};

static const char* state_name(MotorState s) {
    switch (s) {
        case MotorState::IDLE:    return "IDLE";
        case MotorState::POWERON: return "POWERON";
        case MotorState::INIT:    return "INIT";
        case MotorState::RAMPING: return "RAMPING";
        case MotorState::STEADY:  return "STEADY";
        case MotorState::DECEL:   return "DECEL";
        case MotorState::FAULT:   return "FAULT";
        case MotorState::ESTOP:   return "ESTOP";
        default:                  return "UNKNOWN";
    }
}

// ── PI controllers (d-q frame FOC + outer speed loop) ────────────────────────

struct PIController {
    float kp, ki;
    float integrator = 0.0f;
    float i_limit;

    PIController(float kp_, float ki_, float i_limit_ = 10.0f)
        : kp(kp_), ki(ki_), i_limit(i_limit_) {}

    void reset() { integrator = 0.0f; }

    float update(float ref, float meas, float dt) {
        float err = ref - meas;
        integrator += err * dt;
        integrator = std::max(-i_limit, std::min(i_limit, integrator));
        return kp * err + ki * integrator;
    }
};

// ── Plant (simplified PMSM + load) ───────────────────────────────────────────

struct PMSMPlant {
    float J      = 1e-4f;   // rotor inertia [kg·m²]
    float Kt     = 0.05f;   // torque constant [N·m/A]
    float B      = 0.001f;  // viscous damping [N·m·s/rad]
    float tau_el = 0.001f;  // electrical time constant [s] (L/R ≈ 1 ms)
    int   n_pp   = 4;       // pole pairs

    float omega   = 0.0f;   // mechanical angular velocity [rad/s]
    float theta_e = 0.0f;   // electrical angle [rad]
    float i_d     = 0.0f;
    float i_q     = 0.0f;

    void step(float vd_cmd, float vq_cmd, float dt) {
        // First-order electrical dynamics (I·τ = V·dt simplification)
        (void)vd_cmd;  // id reference = 0 for MTPA
        (void)vq_cmd;
        // In real FOC: i_d/q lag behind v_d/q commands via L/R time constant
        // Here we track the commanded currents with the electrical tau
        // (caller drives i_q_cmd via speed PI, we just model the lag)
    }

    // Update with explicit i_q command (from speed PI → current PI bypassed for simplicity)
    void update(float iq_cmd, float dt) {
        i_q += (iq_cmd - i_q) * dt / tau_el;
        float T_e    = Kt * i_q;
        float T_load = B  * omega;
        omega += (T_e - T_load) / J * dt;
        omega  = std::max(0.0f, omega);
        theta_e += omega * dt * static_cast<float>(n_pp);
    }
};

// ── Main state machine class ──────────────────────────────────────────────────

class MotorControlSM {
public:
    MotorState state_ = MotorState::IDLE;

    PIController speed_pi_{0.05f, 2.0f,  20.0f};
    PIController id_pi_   {0.5f,  20.0f, 10.0f};
    PIController iq_pi_   {0.5f,  20.0f, 10.0f};
    PMSMPlant    plant_;

    float omega_ref_  = 0.0f;
    float omega_ramp_ = 0.0f;
    float ramp_rate_  = 100.0f;  // [rad/s²]
    float iq_cmd_     = 0.0f;
    float i_max_      = 15.0f;
    float t_          = 0.0f;
    float dt_;

    std::vector<float> omega_log_, iq_log_, state_log_, t_log_;

    explicit MotorControlSM(float dt = 1e-4f, float ramp_rate = 100.0f)
        : ramp_rate_(ramp_rate), dt_(dt) {}

    void power_on() {
        if (state_ == MotorState::IDLE) state_ = MotorState::POWERON;
    }

    void set_target_omega(float omega_rad_s) {
        omega_ref_ = omega_rad_s;
        if (state_ == MotorState::IDLE    || state_ == MotorState::STEADY ||
            state_ == MotorState::POWERON || state_ == MotorState::INIT)
            state_ = MotorState::RAMPING;
    }

    void inject_fault() {
        state_ = MotorState::FAULT;
        speed_pi_.reset();
    }

    void decelerate() {
        if (state_ == MotorState::STEADY || state_ == MotorState::RAMPING) {
            omega_ref_  = 0.0f;
            state_      = MotorState::DECEL;
        }
    }

    void emergency_stop() {
        state_    = MotorState::ESTOP;
        iq_cmd_   = 0.0f;
        speed_pi_.reset(); id_pi_.reset(); iq_pi_.reset();
    }

    void clear_fault() {
        if (state_ == MotorState::FAULT) {
            speed_pi_.reset(); id_pi_.reset(); iq_pi_.reset();
            omega_ramp_ = 0.0f;
            state_ = MotorState::IDLE;
        }
    }

    // One 10 kHz ISR tick
    void isr_tick() {
        t_ += dt_;

        // Hardware over-current protection (simulated)
        if (state_ != MotorState::FAULT && state_ != MotorState::ESTOP) {
            if (std::abs(plant_.i_q) > i_max_) {
                state_ = MotorState::FAULT;
                iq_cmd_ = 0.0f;
                speed_pi_.reset();
                goto log;
            }
        }

        switch (state_) {
            case MotorState::POWERON:
                if (t_ > 0.05f) state_ = MotorState::INIT;
                break;

            case MotorState::INIT:
                speed_pi_.reset(); id_pi_.reset(); iq_pi_.reset();
                plant_.omega = 0.0f; plant_.i_q = 0.0f;
                state_ = MotorState::IDLE;
                break;

            case MotorState::RAMPING: {
                float delta = ramp_rate_ * dt_;
                if (std::abs(omega_ref_ - omega_ramp_) < delta)
                    omega_ramp_ = omega_ref_;
                else
                    omega_ramp_ += (omega_ref_ > omega_ramp_) ? delta : -delta;

                iq_cmd_ = speed_pi_.update(omega_ramp_, plant_.omega, dt_);
                iq_cmd_ = std::max(-i_max_, std::min(i_max_, iq_cmd_));
                plant_.update(iq_cmd_, dt_);
                if (std::abs(omega_ramp_ - omega_ref_) < 1.0f) state_ = MotorState::STEADY;
                break;
            }

            case MotorState::STEADY:
                iq_cmd_ = speed_pi_.update(omega_ref_, plant_.omega, dt_);
                iq_cmd_ = std::max(-i_max_, std::min(i_max_, iq_cmd_));
                plant_.update(iq_cmd_, dt_);
                break;

            case MotorState::DECEL: {
                float delta = ramp_rate_ * dt_;
                omega_ramp_ = std::max(0.0f, omega_ramp_ - delta);
                iq_cmd_ = speed_pi_.update(omega_ramp_, plant_.omega, dt_);
                iq_cmd_ = std::max(-i_max_, std::min(i_max_, iq_cmd_));
                plant_.update(iq_cmd_, dt_);
                if (plant_.omega < 1.0f) { state_ = MotorState::IDLE; omega_ramp_ = 0.0f; }
                break;
            }

            default:
                break;
        }

        log:
        omega_log_.push_back(plant_.omega);
        iq_log_.push_back(plant_.i_q);
        state_log_.push_back(static_cast<float>(static_cast<uint8_t>(state_)));
        t_log_.push_back(t_);
    }

    py::dict run(int n_steps) {
        for (int i = 0; i < n_steps; ++i) isr_tick();
        py::dict d;
        d["omega_rad_s"]       = omega_log_;
        d["iq_A"]              = iq_log_;
        d["state_id"]          = state_log_;
        d["t_s"]               = t_log_;
        d["final_omega_rad_s"] = plant_.omega;
        d["final_state"]       = std::string(state_name(state_));
        return d;
    }

    std::string get_state() const { return std::string(state_name(state_)); }
    float get_omega()       const { return plant_.omega; }
};

PYBIND11_MODULE(_motor_ctrl_sm_kernel, m) {
    m.doc() = "Bare-metal PMSM FOC state machine — 10 kHz ISR simulation (C++14)";

    py::class_<MotorControlSM>(m, "MotorControlSM")
        .def(py::init<float, float>(),
             py::arg("dt") = 1e-4f, py::arg("ramp_rate") = 100.0f,
             "dt [s]: ISR period (default 1e-4 = 10 kHz). "
             "ramp_rate [rad/s²]: speed ramp acceleration.")
        .def("power_on",         &MotorControlSM::power_on,
             "Transition IDLE → POWERON (DC bus charge phase)")
        .def("set_target_omega", &MotorControlSM::set_target_omega,
             py::arg("omega_rad_s"), "Set speed reference and enter RAMPING state")
        .def("decelerate",       &MotorControlSM::decelerate,
             "Ramp down to zero and enter IDLE")
        .def("emergency_stop",   &MotorControlSM::emergency_stop,
             "Immediate zero-torque command → ESTOP")
        .def("clear_fault",      &MotorControlSM::clear_fault,
             "Reset from FAULT state back to IDLE")
        .def("isr_tick",         &MotorControlSM::isr_tick,
             "Execute one ISR tick (speed PI + plant model)")
        .def("inject_fault",     &MotorControlSM::inject_fault,
             "Force FAULT state (for testing)")
        .def("run",              &MotorControlSM::run,  py::arg("n_steps"),
             "Run n_steps ISR ticks; return telemetry dict "
             "{omega_rad_s, iq_A, state_id, t_s, final_omega_rad_s, final_state}")
        .def("get_state",        &MotorControlSM::get_state)
        .def("get_omega",        &MotorControlSM::get_omega);
}
