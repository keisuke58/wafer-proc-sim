/*
 * _recipe_sequencer.cpp
 *
 * Process recipe execution engine for DISCO dicing saw.
 * Mirrors the recipe/sequence layer of real equipment control software.
 *
 * A recipe is an ordered list of named steps with expected durations and
 * timeout limits.  Python drives the clock by calling tick(dt_ms); it signals
 * step completion with complete_step(success).
 *
 * States: IDLE → RUNNING → DONE | FAULT
 *         RUNNING can be PAUSED / RESUMED
 *
 * Default dicing recipe:
 *   LOAD_WAFER → ALIGN → PRIME_SPINDLE → CUT_X → ROTATE_CHUCK →
 *   CUT_Y → RETRACT → UNLOAD_WAFER
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

// ── RecipeStep ────────────────────────────────────────────────────────────────

struct RecipeStep {
    std::string name;
    double expected_ms;   // nominal duration
    double timeout_ms;    // fault if exceeded
    std::string description;
};

// ── StepRecord (history entry) ────────────────────────────────────────────────

struct StepRecord {
    std::string name;
    double elapsed_ms;
    bool success;
};

// ── Sequencer states ──────────────────────────────────────────────────────────

enum class SeqState : int {
    IDLE    = 0,
    RUNNING = 1,
    PAUSED  = 2,
    DONE    = 3,
    FAULT   = 4,
};

static const char* seq_state_name(SeqState s) {
    switch(s) {
        case SeqState::IDLE:    return "IDLE";
        case SeqState::RUNNING: return "RUNNING";
        case SeqState::PAUSED:  return "PAUSED";
        case SeqState::DONE:    return "DONE";
        case SeqState::FAULT:   return "FAULT";
        default:                return "UNKNOWN";
    }
}

// ── RecipeSequencer ───────────────────────────────────────────────────────────

class RecipeSequencer {
public:
    std::vector<RecipeStep> recipe;
    std::vector<StepRecord> history;

    SeqState state;
    int      step_idx;     // index into recipe
    double   elapsed_ms;   // time in current step
    double   total_ms;     // time since start()
    std::string fault_reason;

    RecipeSequencer() : state(SeqState::IDLE), step_idx(0),
                        elapsed_ms(0), total_ms(0) {}

    // ── recipe building ───────────────────────────────────────────────────────

    void add_step(const std::string& name, double expected_ms,
                  double timeout_ms, const std::string& desc = "") {
        recipe.push_back({name, expected_ms, timeout_ms, desc});
    }

    void clear_recipe() { recipe.clear(); reset(); }

    // ── control ───────────────────────────────────────────────────────────────

    void start() {
        if (recipe.empty()) throw std::runtime_error("recipe is empty");
        if (state != SeqState::IDLE && state != SeqState::DONE)
            throw std::runtime_error("can only start from IDLE or DONE");
        step_idx   = 0;
        elapsed_ms = 0;
        total_ms   = 0;
        history.clear();
        fault_reason.clear();
        state = SeqState::RUNNING;
    }

    void pause() {
        if (state == SeqState::RUNNING) state = SeqState::PAUSED;
    }

    void resume() {
        if (state == SeqState::PAUSED) state = SeqState::RUNNING;
    }

    void abort(const std::string& reason = "operator abort") {
        fault_reason = reason;
        state = SeqState::FAULT;
    }

    void reset() {
        state      = SeqState::IDLE;
        step_idx   = 0;
        elapsed_ms = 0;
        total_ms   = 0;
        history.clear();
        fault_reason.clear();
    }

    // ── tick — advance time ────────────────────────────────────────────────────

    // Call every control cycle from Python.  Returns state name.
    std::string tick(double dt_ms) {
        if (state != SeqState::RUNNING) return seq_state_name(state);
        if (step_idx >= (int)recipe.size()) {
            state = SeqState::DONE;
            return seq_state_name(state);
        }

        elapsed_ms += dt_ms;
        total_ms   += dt_ms;

        const auto& step = recipe[(size_t)step_idx];
        if (elapsed_ms > step.timeout_ms) {
            fault_reason = "timeout in step '" + step.name + "'";
            state = SeqState::FAULT;
        }
        return seq_state_name(state);
    }

    // ── complete_step — signal from Python that step finished ─────────────────

    // success=true → advance; success=false → FAULT
    bool complete_step(bool success = true) {
        if (state != SeqState::RUNNING) return false;
        if (step_idx >= (int)recipe.size()) return false;

        const auto& step = recipe[(size_t)step_idx];
        history.push_back({step.name, elapsed_ms, success});

        if (!success) {
            fault_reason = "step '" + step.name + "' reported failure";
            state = SeqState::FAULT;
            return false;
        }

        ++step_idx;
        elapsed_ms = 0;
        if (step_idx >= (int)recipe.size())
            state = SeqState::DONE;
        return true;
    }

    // ── accessors ─────────────────────────────────────────────────────────────

    std::string current_step_name() const {
        if (step_idx < (int)recipe.size())
            return recipe[(size_t)step_idx].name;
        return "";
    }

    double current_step_expected_ms() const {
        if (step_idx < (int)recipe.size())
            return recipe[(size_t)step_idx].expected_ms;
        return 0.0;
    }

    std::string state_name() const { return seq_state_name(state); }

    int steps_done()  const { return (int)history.size(); }
    int steps_total() const { return (int)recipe.size(); }

    double progress_pct() const {
        if (recipe.empty()) return 0.0;
        return 100.0 * (double)step_idx / (double)recipe.size();
    }

    py::list get_history() const {
        py::list lst;
        for (const auto& r : history) {
            py::dict d;
            d["name"]       = r.name;
            d["elapsed_ms"] = r.elapsed_ms;
            d["success"]    = r.success;
            lst.append(d);
        }
        return lst;
    }

    // Factory: default DISCO dicing recipe
    static RecipeSequencer disco_dicing_recipe() {
        RecipeSequencer seq;
        seq.add_step("LOAD_WAFER",    5000.0,  15000.0, "Robot loads wafer onto chuck");
        seq.add_step("VACUUM_ON",      500.0,   2000.0, "Chuck vacuum engage");
        seq.add_step("ALIGN_WAFER",   3000.0,  10000.0, "Camera vision alignment");
        seq.add_step("PRIME_SPINDLE", 2000.0,   8000.0, "Spin up to 30000 rpm");
        seq.add_step("COOLANT_ON",     500.0,   2000.0, "Coolant flow engage");
        seq.add_step("CUT_X_STREETS",60000.0, 120000.0, "Dice all X-direction streets");
        seq.add_step("ROTATE_CHUCK",  1000.0,   5000.0, "Rotate chuck 90 degrees");
        seq.add_step("CUT_Y_STREETS",60000.0, 120000.0, "Dice all Y-direction streets");
        seq.add_step("RETRACT_BLADE",  500.0,   2000.0, "Z-axis retract to safe height");
        seq.add_step("COOLANT_OFF",    500.0,   2000.0, "Coolant flow off");
        seq.add_step("SPINDLE_OFF",   1000.0,   5000.0, "Spindle coast to stop");
        seq.add_step("UNLOAD_WAFER",  5000.0,  15000.0, "Robot unloads wafer to cassette");
        return seq;
    }
};

PYBIND11_MODULE(_recipe_sequencer, m) {
    m.doc() = "Process recipe execution engine for DISCO dicing saw";

    py::enum_<SeqState>(m, "SeqState")
        .value("IDLE",    SeqState::IDLE)
        .value("RUNNING", SeqState::RUNNING)
        .value("PAUSED",  SeqState::PAUSED)
        .value("DONE",    SeqState::DONE)
        .value("FAULT",   SeqState::FAULT);

    py::class_<RecipeStep>(m, "RecipeStep")
        .def(py::init<>())
        .def_readwrite("name",         &RecipeStep::name)
        .def_readwrite("expected_ms",  &RecipeStep::expected_ms)
        .def_readwrite("timeout_ms",   &RecipeStep::timeout_ms)
        .def_readwrite("description",  &RecipeStep::description);

    py::class_<RecipeSequencer>(m, "RecipeSequencer")
        .def(py::init<>())
        .def("add_step",    &RecipeSequencer::add_step,
             py::arg("name"), py::arg("expected_ms"), py::arg("timeout_ms"),
             py::arg("desc") = "")
        .def("clear_recipe", &RecipeSequencer::clear_recipe)
        .def("start",        &RecipeSequencer::start)
        .def("pause",        &RecipeSequencer::pause)
        .def("resume",       &RecipeSequencer::resume)
        .def("abort",        &RecipeSequencer::abort, py::arg("reason") = "operator abort")
        .def("reset",        &RecipeSequencer::reset)
        .def("tick",         &RecipeSequencer::tick,  py::arg("dt_ms"))
        .def("complete_step",&RecipeSequencer::complete_step, py::arg("success") = true)
        .def("current_step_name",        &RecipeSequencer::current_step_name)
        .def("current_step_expected_ms", &RecipeSequencer::current_step_expected_ms)
        .def("state_name",   &RecipeSequencer::state_name)
        .def("steps_done",   &RecipeSequencer::steps_done)
        .def("steps_total",  &RecipeSequencer::steps_total)
        .def("progress_pct", &RecipeSequencer::progress_pct)
        .def("get_history",  &RecipeSequencer::get_history)
        .def_static("disco_dicing_recipe", &RecipeSequencer::disco_dicing_recipe,
             "Factory: 12-step DISCO wafer dicing recipe")
        .def_readonly("state",        &RecipeSequencer::state)
        .def_readonly("step_idx",     &RecipeSequencer::step_idx)
        .def_readonly("elapsed_ms",   &RecipeSequencer::elapsed_ms)
        .def_readonly("total_ms",     &RecipeSequencer::total_ms)
        .def_readonly("fault_reason", &RecipeSequencer::fault_reason);
}
