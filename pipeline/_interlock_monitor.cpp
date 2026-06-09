/*
 * _interlock_monitor.cpp
 *
 * Safety interlock monitor for DISCO dicing saw.
 * Inspired by IEC 61508 SIL 2 functional safety requirements.
 *
 * Sensors report continuous float values; thresholds trigger:
 *   WARN  — alert operator, log event
 *   STOP  — controlled stop (finish current cut, then halt)
 *   FAULT — immediate halt + E-STOP output
 *
 * DISCO-specific sensors:
 *   spindle_rpm      — must be in [28000, 32000] during cutting
 *   coolant_flow_lpm — must be > 2.0 during cutting
 *   door_closed      — must be 1 during operation
 *   blade_depth_mm   — must not exceed limit
 *   chuck_vacuum_kPa — must stay above threshold
 *   blade_wear_um    — WARN when high, STOP when critical
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

namespace py = pybind11;

// ── Alarm severity ────────────────────────────────────────────────────────────

enum class AlarmLevel : int {
    OK    = 0,
    WARN  = 1,
    STOP  = 2,
    FAULT = 3,
};

static const char* alarm_level_name(AlarmLevel a) {
    switch(a) {
        case AlarmLevel::OK:    return "OK";
        case AlarmLevel::WARN:  return "WARN";
        case AlarmLevel::STOP:  return "STOP";
        case AlarmLevel::FAULT: return "FAULT";
        default:                return "UNKNOWN";
    }
}

// ── Sensor definition ─────────────────────────────────────────────────────────

struct Sensor {
    std::string name;
    double value;
    // Thresholds (NaN = disabled)
    double lo_warn, hi_warn;   // WARN band
    double lo_stop, hi_stop;   // STOP band
    double lo_fault, hi_fault; // FAULT band
    std::string unit;

    AlarmLevel level() const {
        if (!std::isnan(lo_fault) && value < lo_fault) return AlarmLevel::FAULT;
        if (!std::isnan(hi_fault) && value > hi_fault) return AlarmLevel::FAULT;
        if (!std::isnan(lo_stop)  && value < lo_stop)  return AlarmLevel::STOP;
        if (!std::isnan(hi_stop)  && value > hi_stop)  return AlarmLevel::STOP;
        if (!std::isnan(lo_warn)  && value < lo_warn)  return AlarmLevel::WARN;
        if (!std::isnan(hi_warn)  && value > hi_warn)  return AlarmLevel::WARN;
        return AlarmLevel::OK;
    }
};

static const double NaN = std::numeric_limits<double>::quiet_NaN();

// ── AlarmRecord ───────────────────────────────────────────────────────────────

struct AlarmRecord {
    std::string sensor;
    AlarmLevel  level;
    double      value;
    double      timestamp_ms;
};

// ── InterlockMonitor ──────────────────────────────────────────────────────────

class InterlockMonitor {
public:
    std::unordered_map<std::string, Sensor> sensors;
    std::vector<AlarmRecord> alarm_log;
    std::vector<AlarmRecord> active_alarms;
    bool e_stop_active;
    double clock_ms;

    InterlockMonitor() : e_stop_active(false), clock_ms(0.0) {}

    // ── sensor management ─────────────────────────────────────────────────────

    void add_sensor(const std::string& name,
                    double lo_warn=NaN, double hi_warn=NaN,
                    double lo_stop=NaN, double hi_stop=NaN,
                    double lo_fault=NaN, double hi_fault=NaN,
                    const std::string& unit="") {
        sensors[name] = {name, 0.0,
                         lo_warn, hi_warn,
                         lo_stop,  hi_stop,
                         lo_fault, hi_fault, unit};
    }

    // Update sensor value and check thresholds. Returns new alarm level.
    AlarmLevel update(const std::string& name, double value, double dt_ms=0.0) {
        auto it = sensors.find(name);
        if (it == sensors.end())
            throw std::invalid_argument("unknown sensor: " + name);

        clock_ms += dt_ms;
        auto& s = it->second;
        s.value = value;
        AlarmLevel lv = s.level();

        if (lv != AlarmLevel::OK) {
            AlarmRecord rec{name, lv, value, clock_ms};
            alarm_log.push_back(rec);
            // Update or insert active alarm for this sensor
            bool found = false;
            for (auto& a : active_alarms) {
                if (a.sensor == name) { a = rec; found = true; break; }
            }
            if (!found) active_alarms.push_back(rec);
            if (lv == AlarmLevel::FAULT) e_stop_active = true;
        } else {
            // Clear active alarm for this sensor if it went back to OK
            active_alarms.erase(
                std::remove_if(active_alarms.begin(), active_alarms.end(),
                    [&](const AlarmRecord& a){ return a.sensor == name; }),
                active_alarms.end());
        }
        return lv;
    }

    // ── bulk update ───────────────────────────────────────────────────────────

    // values: dict {sensor_name -> float}
    AlarmLevel update_all(const py::dict& values, double dt_ms=0.0) {
        AlarmLevel worst = AlarmLevel::OK;
        clock_ms += dt_ms;
        for (auto item : values) {
            std::string name = item.first.cast<std::string>();
            double val       = item.second.cast<double>();
            auto lv = update(name, val, 0.0);  // clock already advanced above
            if ((int)lv > (int)worst) worst = lv;
        }
        return worst;
    }

    // ── state ─────────────────────────────────────────────────────────────────

    AlarmLevel system_level() const {
        AlarmLevel worst = AlarmLevel::OK;
        for (const auto& a : active_alarms)
            if ((int)a.level > (int)worst) worst = a.level;
        return worst;
    }

    std::string system_level_name() const {
        return alarm_level_name(system_level());
    }

    bool is_safe_to_run() const {
        return !e_stop_active &&
               (int)system_level() < (int)AlarmLevel::STOP;
    }

    void reset_estop() {
        if ((int)system_level() < (int)AlarmLevel::FAULT)
            e_stop_active = false;
    }

    void acknowledge_alarm(const std::string& sensor_name) {
        active_alarms.erase(
            std::remove_if(active_alarms.begin(), active_alarms.end(),
                [&](const AlarmRecord& a){ return a.sensor == sensor_name; }),
            active_alarms.end());
        if (active_alarms.empty()) e_stop_active = false;
    }

    void reset() {
        active_alarms.clear();
        alarm_log.clear();
        e_stop_active = false;
        clock_ms = 0.0;
    }

    // ── Python-friendly accessors ─────────────────────────────────────────────

    py::list get_active_alarms() const {
        py::list lst;
        for (const auto& a : active_alarms) {
            py::dict d;
            d["sensor"]       = a.sensor;
            d["level"]        = alarm_level_name(a.level);
            d["value"]        = a.value;
            d["timestamp_ms"] = a.timestamp_ms;
            lst.append(d);
        }
        return lst;
    }

    py::list get_alarm_log(int last_n = -1) const {
        py::list lst;
        int start = (last_n > 0 && (int)alarm_log.size() > last_n)
                    ? (int)alarm_log.size() - last_n : 0;
        for (int i = start; i < (int)alarm_log.size(); ++i) {
            const auto& a = alarm_log[(size_t)i];
            py::dict d;
            d["sensor"]       = a.sensor;
            d["level"]        = alarm_level_name(a.level);
            d["value"]        = a.value;
            d["timestamp_ms"] = a.timestamp_ms;
            lst.append(d);
        }
        return lst;
    }

    double get_sensor_value(const std::string& name) const {
        auto it = sensors.find(name);
        if (it == sensors.end())
            throw std::invalid_argument("unknown sensor: " + name);
        return it->second.value;
    }

    std::string get_sensor_level(const std::string& name) const {
        auto it = sensors.find(name);
        if (it == sensors.end())
            throw std::invalid_argument("unknown sensor: " + name);
        return alarm_level_name(it->second.level());
    }

    // Factory: DISCO dicing saw interlock set
    static InterlockMonitor disco_dicing_interlocks() {
        InterlockMonitor m;
        // spindle_rpm: warn < 29000 | > 31000, stop < 27000 | > 33000, fault < 0 | > 35000
        m.add_sensor("spindle_rpm",
                     29000.0, 31000.0,
                     27000.0, 33000.0,
                       0.0,  35000.0, "rpm");
        // coolant_flow_lpm: fault < 0.5, stop < 1.0, warn < 2.0
        m.add_sensor("coolant_flow_lpm",
                     2.0, NaN,
                     1.0, NaN,
                     0.5, NaN, "L/min");
        // door_closed: fault < 0.5 (0=open, 1=closed)
        m.add_sensor("door_closed",
                     NaN, NaN,
                     NaN, NaN,
                     0.5, NaN, "bool");
        // blade_depth_mm: more negative = deeper; fault when value < -0.50
        m.add_sensor("blade_depth_mm",
                     -0.35, NaN,
                     -0.45, NaN,
                     -0.50, NaN, "mm");
        // chuck_vacuum_kPa: warn < 60, stop < 40, fault < 20
        m.add_sensor("chuck_vacuum_kPa",
                     60.0, NaN,
                     40.0, NaN,
                     20.0, NaN, "kPa");
        // blade_wear_um: warn > 30, stop > 50, fault > 80
        m.add_sensor("blade_wear_um",
                     NaN, 30.0,
                     NaN, 50.0,
                     NaN, 80.0, "um");
        return m;
    }
};

PYBIND11_MODULE(_interlock_monitor, m) {
    m.doc() = "Safety interlock monitor for DISCO dicing saw (IEC 61508-inspired)";

    py::enum_<AlarmLevel>(m, "AlarmLevel")
        .value("OK",    AlarmLevel::OK)
        .value("WARN",  AlarmLevel::WARN)
        .value("STOP",  AlarmLevel::STOP)
        .value("FAULT", AlarmLevel::FAULT);

    py::class_<InterlockMonitor>(m, "InterlockMonitor")
        .def(py::init<>())
        .def("add_sensor",   &InterlockMonitor::add_sensor,
             py::arg("name"),
             py::arg("lo_warn")=NaN, py::arg("hi_warn")=NaN,
             py::arg("lo_stop")=NaN, py::arg("hi_stop")=NaN,
             py::arg("lo_fault")=NaN, py::arg("hi_fault")=NaN,
             py::arg("unit")="")
        .def("update",       &InterlockMonitor::update,
             py::arg("name"), py::arg("value"), py::arg("dt_ms")=0.0)
        .def("update_all",   &InterlockMonitor::update_all,
             py::arg("values"), py::arg("dt_ms")=0.0)
        .def("system_level",      &InterlockMonitor::system_level)
        .def("system_level_name", &InterlockMonitor::system_level_name)
        .def("is_safe_to_run",    &InterlockMonitor::is_safe_to_run)
        .def("reset_estop",       &InterlockMonitor::reset_estop)
        .def("acknowledge_alarm", &InterlockMonitor::acknowledge_alarm)
        .def("reset",             &InterlockMonitor::reset)
        .def("get_active_alarms", &InterlockMonitor::get_active_alarms)
        .def("get_alarm_log",     &InterlockMonitor::get_alarm_log,
             py::arg("last_n")=-1)
        .def("get_sensor_value",  &InterlockMonitor::get_sensor_value)
        .def("get_sensor_level",  &InterlockMonitor::get_sensor_level)
        .def_static("disco_dicing_interlocks",
                    &InterlockMonitor::disco_dicing_interlocks,
                    "Factory: DISCO dicing saw interlock set (6 sensors)")
        .def_readonly("e_stop_active", &InterlockMonitor::e_stop_active)
        .def_readonly("clock_ms",      &InterlockMonitor::clock_ms);
}
