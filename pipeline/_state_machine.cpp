// IEC 61131-3–inspired dicing machine state machine in C++ (pybind11).
//
// WHY C++: DISCO's dicing saws, grinders, and laser slicers run their main
// state machine in a hard-real-time C/C++ task on the machine controller
// (VxWorks / RTOS layer, IEC 61131-3 Structured Text compiled to native code).
// The Python class `StateMachine` in pipeline/disco_sw_stack.py models the
// same state/transition logic in the orchestration layer. This C++ class
// mirrors that architecture one layer lower: compiled state, validated
// transitions, and callback hooks — the same interface, but executing at the
// speed the embedded OS expects.
//
// States: IDLE → SETUP → DICING → INSPECT → DONE (+ FAULT, E_STOP)
// Transitions are whitelisted; invalid transitions return false without
// mutating state (matches IEC 61131-3 safe-state behaviour).
//
// Build: bash build_all_kernels.sh

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <stdexcept>

namespace py = pybind11;

// ── State enum ────────────────────────────────────────────────────────────────

enum class State : int {
    IDLE    = 0,
    SETUP   = 1,
    DICING  = 2,
    INSPECT = 3,
    DONE    = 4,
    FAULT   = 5,
    E_STOP  = 6,
};

static const char* state_name(State s) {
    switch (s) {
        case State::IDLE:    return "IDLE";
        case State::SETUP:   return "SETUP";
        case State::DICING:  return "DICING";
        case State::INSPECT: return "INSPECT";
        case State::DONE:    return "DONE";
        case State::FAULT:   return "FAULT";
        case State::E_STOP:  return "E_STOP";
    }
    return "UNKNOWN";
}

// ── Transition table (mirrors VALID_TRANSITIONS in disco_sw_stack.py) ─────────

static const std::unordered_map<int, std::vector<int>> TRANSITIONS = {
    {(int)State::IDLE,    {(int)State::SETUP,   (int)State::E_STOP}},
    {(int)State::SETUP,   {(int)State::DICING,  (int)State::FAULT, (int)State::E_STOP}},
    {(int)State::DICING,  {(int)State::INSPECT, (int)State::FAULT, (int)State::E_STOP}},
    {(int)State::INSPECT, {(int)State::DICING,  (int)State::DONE,  (int)State::E_STOP}},
    {(int)State::DONE,    {(int)State::IDLE}},
    {(int)State::FAULT,   {(int)State::IDLE}},
    {(int)State::E_STOP,  {(int)State::IDLE}},
};

// ── StateMachineCpp ───────────────────────────────────────────────────────────

struct TransitionRecord {
    std::string from_state;
    std::string to_state;
    std::string reason;
};

class StateMachineCpp {
public:
    StateMachineCpp() : _state(State::IDLE) {}

    // Attempt transition to `target`. Returns true on success.
    // Matches IEC 61131-3 safe-state: no side effect on invalid transition.
    bool transition(State target, const std::string& reason = "") {
        auto it = TRANSITIONS.find((int)_state);
        if (it == TRANSITIONS.end()) return false;
        const auto& allowed = it->second;
        bool ok = false;
        for (int s : allowed) {
            if (s == (int)target) { ok = true; break; }
        }
        if (!ok) return false;

        _history.push_back({state_name(_state), state_name(target), reason});

        // Fire exit callbacks for old state.
        auto exit_it = _exit_cbs.find((int)_state);
        if (exit_it != _exit_cbs.end())
            for (auto& cb : exit_it->second) cb();

        _state = target;

        // Fire entry callbacks for new state.
        auto enter_it = _enter_cbs.find((int)_state);
        if (enter_it != _enter_cbs.end())
            for (auto& cb : enter_it->second) cb();

        return true;
    }

    State state() const { return _state; }
    std::string state_name_str() const { return state_name(_state); }

    // Register a callback fired when entering `state`.
    void on_enter(State s, std::function<void()> cb) {
        _enter_cbs[(int)s].push_back(cb);
    }
    // Register a callback fired when leaving `state`.
    void on_exit(State s, std::function<void()> cb) {
        _exit_cbs[(int)s].push_back(cb);
    }

    const std::vector<TransitionRecord>& history() const { return _history; }
    void clear_history() { _history.clear(); }

    // IEC 61131-3 – safe reset: always permitted back to IDLE.
    void reset() { _state = State::IDLE; }

    // Returns list of valid targets from current state.
    std::vector<std::string> valid_next() const {
        std::vector<std::string> out;
        auto it = TRANSITIONS.find((int)_state);
        if (it != TRANSITIONS.end())
            for (int s : it->second) out.emplace_back(state_name((State)s));
        return out;
    }

    // Convenience: are we in a "running" state (not parked)?
    bool is_running() const {
        return _state == State::DICING || _state == State::INSPECT;
    }
    bool is_faulted() const {
        return _state == State::FAULT || _state == State::E_STOP;
    }

private:
    State _state;
    std::vector<TransitionRecord> _history;
    std::unordered_map<int, std::vector<std::function<void()>>> _enter_cbs;
    std::unordered_map<int, std::vector<std::function<void()>>> _exit_cbs;
};


PYBIND11_MODULE(_state_machine, m) {
    m.doc() = "IEC 61131-3–inspired dicing machine state machine (C++ / pybind11)";

    // Expose the enum so Python can write `_state_machine.State.DICING`.
    py::enum_<State>(m, "State")
        .value("IDLE",    State::IDLE)
        .value("SETUP",   State::SETUP)
        .value("DICING",  State::DICING)
        .value("INSPECT", State::INSPECT)
        .value("DONE",    State::DONE)
        .value("FAULT",   State::FAULT)
        .value("E_STOP",  State::E_STOP)
        .export_values();

    py::class_<TransitionRecord>(m, "TransitionRecord")
        .def_readonly("from_state", &TransitionRecord::from_state)
        .def_readonly("to_state",   &TransitionRecord::to_state)
        .def_readonly("reason",     &TransitionRecord::reason)
        .def("__repr__", [](const TransitionRecord& r){
            return r.from_state + " → " + r.to_state
                   + (r.reason.empty() ? "" : " (" + r.reason + ")");
        });

    py::class_<StateMachineCpp>(m, "StateMachine",
        "IEC 61131-3–inspired state machine for DISCO dicing machines.\n"
        "States: IDLE → SETUP → DICING → INSPECT → DONE (+ FAULT, E_STOP).\n"
        "Invalid transitions return False without mutating state.")
        .def(py::init<>())
        .def("transition", &StateMachineCpp::transition,
             "Attempt transition to `target` (State enum). Returns bool.",
             py::arg("target"), py::arg("reason") = "")
        .def_property_readonly("state",      &StateMachineCpp::state)
        .def_property_readonly("state_name", &StateMachineCpp::state_name_str)
        .def("on_enter",     &StateMachineCpp::on_enter,
             "Register callback fired on entering state s.",
             py::arg("state"), py::arg("callback"))
        .def("on_exit",      &StateMachineCpp::on_exit,
             "Register callback fired on leaving state s.",
             py::arg("state"), py::arg("callback"))
        .def_property_readonly("history",    &StateMachineCpp::history)
        .def("clear_history",&StateMachineCpp::clear_history)
        .def("reset",        &StateMachineCpp::reset,
             "Force state to IDLE (emergency reset, always permitted).")
        .def("valid_next",   &StateMachineCpp::valid_next,
             "List valid next states from current state.")
        .def("is_running",   &StateMachineCpp::is_running,
             "True if in DICING or INSPECT.")
        .def("is_faulted",   &StateMachineCpp::is_faulted,
             "True if in FAULT or E_STOP.")
        .def("__repr__", [](const StateMachineCpp& sm){
            return std::string("<StateMachine state=") + sm.state_name_str() + ">";
        });
}
