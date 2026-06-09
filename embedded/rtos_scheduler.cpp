// rtos_scheduler.cpp — Fixed-priority preemptive RTOS task scheduler simulation
//
// AP試験 concepts demonstrated:
//   - 固定優先度プリエンプティブスケジューリング (Rate Monotonic ordering)
//   - 割り込み駆動タスク起動: periodic interrupt → READY → RUNNING → IDLE
//   - タスク状態: IDLE / READY / RUNNING (3-state transition)
//   - RM 利用率上限: U = Σ(Ci/Ti) ≤ n(2^(1/n)−1)  [Liu & Layland 1973]
//   - コンテキストスイッチ / プリエンプション数 / 応答時間ジッタ計測
//
// Application (DISCO DFL7160):
//   Tick = 10 µs.  simulate_disco_tasks() models the actual 4-task control hierarchy.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <string>
#include <vector>

namespace py = pybind11;

struct TaskConfig {
    std::string name;
    uint32_t    period_ticks;   // activation period [ticks]
    uint32_t    wcet_ticks;     // worst-case execution time [ticks]
    uint8_t     priority;       // lower number = higher priority (RM: assign by period)
};

// ── Simulator ────────────────────────────────────────────────────────────────

static py::dict run_scheduler(
    const std::vector<TaskConfig>& tasks,
    uint64_t n_ticks
) {
    int N = static_cast<int>(tasks.size());

    struct State {
        uint32_t next_act       = 0;
        uint32_t remaining      = 0;
        bool     active         = false;
        uint32_t act_count      = 0;
        uint32_t dl_misses      = 0;
        uint32_t preempts       = 0;
        uint64_t resp_sum       = 0;
        uint64_t resp_min       = UINT64_MAX;
        uint64_t resp_max       = 0;
        uint64_t act_start_tick = 0;
    };
    std::vector<State> st(N);
    // First activation = one full period in (avoids t=0 startup spike)
    for (int i = 0; i < N; ++i)
        st[i].next_act = tasks[i].period_ticks;

    uint32_t context_switches = 0;
    int running = -1;

    for (uint64_t t = 0; t < n_ticks; ++t) {
        // Interrupt: activate tasks whose deadline has arrived
        for (int i = 0; i < N; ++i) {
            if (t == st[i].next_act) {
                if (st[i].active)
                    ++st[i].dl_misses;     // previous job not finished = miss
                st[i].active         = true;
                st[i].remaining      = tasks[i].wcet_ticks;
                st[i].next_act      += tasks[i].period_ticks;
                ++st[i].act_count;
                st[i].act_start_tick = t;
            }
        }

        // Pick highest-priority ready task
        int next_running = -1;
        for (int i = 0; i < N; ++i) {
            if (!st[i].active) continue;
            if (next_running < 0 ||
                tasks[i].priority < tasks[next_running].priority)
                next_running = i;
        }

        // Detect preemption / context switch
        if (next_running != running) {
            if (next_running >= 0) ++context_switches;
            if (running >= 0 && next_running >= 0 &&
                tasks[next_running].priority < tasks[running].priority)
                ++st[running].preempts;
            running = next_running;
        }

        // Execute running task 1 tick
        if (running >= 0) {
            --st[running].remaining;
            if (st[running].remaining == 0) {
                uint64_t resp = t + 1 - st[running].act_start_tick;
                st[running].resp_sum += resp;
                if (resp < st[running].resp_min) st[running].resp_min = resp;
                if (resp > st[running].resp_max) st[running].resp_max = resp;
                st[running].active = false;
                running = -1;
            }
        }
    }

    // RM utilization bound  U ≤ n·(2^(1/n)−1)
    double util_total = 0.0;
    for (int i = 0; i < N; ++i)
        util_total += static_cast<double>(tasks[i].wcet_ticks) /
                      static_cast<double>(tasks[i].period_ticks);
    double rm_bound = N > 0 ? N * (std::pow(2.0, 1.0 / N) - 1.0) : 1.0;

    py::list task_stats;
    for (int i = 0; i < N; ++i) {
        double util = static_cast<double>(tasks[i].wcet_ticks) /
                      static_cast<double>(tasks[i].period_ticks);
        double resp_avg = st[i].act_count > 0
            ? static_cast<double>(st[i].resp_sum) / st[i].act_count : 0.0;
        uint64_t resp_min = (st[i].resp_min == UINT64_MAX) ? 0 : st[i].resp_min;

        py::dict d;
        d["name"]               = tasks[i].name;
        d["period_ticks"]       = tasks[i].period_ticks;
        d["wcet_ticks"]         = tasks[i].wcet_ticks;
        d["priority"]           = tasks[i].priority;
        d["activations"]        = st[i].act_count;
        d["deadline_misses"]    = st[i].dl_misses;
        d["preemptions"]        = st[i].preempts;
        d["utilization"]        = util;
        d["response_time_min"]  = static_cast<double>(resp_min);
        d["response_time_max"]  = static_cast<double>(st[i].resp_max);
        d["response_time_avg"]  = resp_avg;
        task_stats.append(d);
    }

    py::dict res;
    res["tasks"]              = task_stats;
    res["context_switches"]   = context_switches;
    res["total_utilization"]  = util_total;
    res["rm_bound"]           = rm_bound;
    res["rm_schedulable"]     = util_total <= rm_bound;
    res["n_ticks_simulated"]  = n_ticks;
    return res;
}

// Pre-configured DISCO DFL7160-class task set (tick = 10 µs):
//
//   CTRL   : 10 kHz (T=100)     WCET=8   ticks (80 µs)  → U=0.080  prio 0 (RM: shortest period first)
//   COMM   :  1 kHz (T=1000)    WCET=50  ticks (0.5 ms) → U=0.050  prio 1
//   VISION : 100 Hz (T=10000)   WCET=200 ticks (2 ms)   → U=0.020  prio 2
//   LOGGER :  10 Hz (T=100000)  WCET=100 ticks (1 ms)   → U=0.001  prio 3
//
//   Total U = 0.151.  RM bound (n=4) = 0.757.  Ample margin → schedulable.
static py::dict simulate_disco_tasks(uint64_t n_ticks) {
    std::vector<TaskConfig> tasks = {
        {"CTRL",   100,   8, 0},
        {"COMM",  1000,  50, 1},
        {"VISION",10000,200, 2},
        {"LOGGER",100000,100,3},
    };
    return run_scheduler(tasks, n_ticks);
}

// ── pybind11 module ──────────────────────────────────────────────────────────

PYBIND11_MODULE(_rtos_scheduler_kernel, m) {
    m.doc() =
        "Fixed-priority preemptive RTOS task scheduler simulation.\n"
        "\n"
        "AP試験 topics: 割り込み, RTOS, タスク管理, スケジューリング, 応答時間解析\n"
        "\n"
        "Tick unit is user-defined; DISCO pre-set uses 10 µs/tick.\n"
        "Rate Monotonic: assign priority by period (shorter = higher).";

    py::class_<TaskConfig>(m, "TaskConfig")
        .def(py::init<std::string, uint32_t, uint32_t, uint8_t>(),
             py::arg("name"), py::arg("period_ticks"),
             py::arg("wcet_ticks"), py::arg("priority"))
        .def_readwrite("name",          &TaskConfig::name)
        .def_readwrite("period_ticks",  &TaskConfig::period_ticks)
        .def_readwrite("wcet_ticks",    &TaskConfig::wcet_ticks)
        .def_readwrite("priority",      &TaskConfig::priority)
        .def("__repr__", [](const TaskConfig& t) {
            return "TaskConfig(" + t.name +
                   ", T=" + std::to_string(t.period_ticks) +
                   ", C=" + std::to_string(t.wcet_ticks) +
                   ", prio=" + std::to_string(static_cast<int>(t.priority)) + ")";
        });

    m.def("run_scheduler", &run_scheduler,
          py::arg("tasks"), py::arg("n_ticks"),
          "Run fixed-priority preemptive scheduler.\n"
          "\n"
          "Returns dict: tasks (list of per-task stats), context_switches,\n"
          "  total_utilization, rm_bound, rm_schedulable, n_ticks_simulated.\n"
          "\n"
          "RM ordering rule: assign priority = rank by period (0 = shortest).\n"
          "rm_schedulable: True iff Σ(Ci/Ti) ≤ n(2^(1/n)−1).");

    m.def("simulate_disco_tasks", &simulate_disco_tasks,
          py::arg("n_ticks") = 1000000,
          "DISCO DFL7160 task set: CTRL/COMM/VISION/LOGGER.\n"
          "Tick=10µs.  n_ticks=1000000 → 10 s simulated time.\n"
          "Total utilization 0.151; RM bound 0.757 → safely schedulable.");
}
