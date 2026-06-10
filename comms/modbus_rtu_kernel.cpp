// modbus_rtu_kernel.cpp — Modbus RTU frame parser + register slave (pybind11)
//
// Implements:
//   crc16_modbus(data)                              — CRC-16/IBM
//   encode_read_holding(slave_id, start, count)     — FC03 request
//   encode_write_single(slave_id, reg, value)       — FC06 request
//   parse_frame(bytes)                              — decode raw frame → dict
//   ModbusRTUSlave                                  — holding register server
//
// Used to interface with spindle drive (e.g. Yaskawa Sigma-7) or stage
// controller over RS-485 / EtherNet gateway.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

// ── CRC-16/IBM (Modbus standard, polynomial 0x8005 reflected) ────────────────

static uint16_t crc16_modbus(const uint8_t* data, size_t len) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; ++i) {
        crc ^= static_cast<uint16_t>(data[i]);
        for (int b = 0; b < 8; ++b)
            crc = (crc & 1u) ? static_cast<uint16_t>((crc >> 1) ^ 0xA001u)
                              : static_cast<uint16_t>(crc >> 1);
    }
    return crc;
}

constexpr uint8_t FC_READ_HOLDING = 0x03;
constexpr uint8_t FC_READ_INPUT   = 0x04;
constexpr uint8_t FC_WRITE_SINGLE = 0x06;
constexpr uint8_t FC_ERROR_FLAG   = 0x80;

// ── Frame encode / decode ────────────────────────────────────────────────────

static std::vector<uint8_t> encode_read_holding(
    uint8_t slave_id, uint16_t start_reg, uint16_t count
) {
    std::vector<uint8_t> f = {
        slave_id, FC_READ_HOLDING,
        static_cast<uint8_t>(start_reg >> 8), static_cast<uint8_t>(start_reg & 0xFF),
        static_cast<uint8_t>(count >> 8),     static_cast<uint8_t>(count & 0xFF),
    };
    uint16_t crc = crc16_modbus(f.data(), f.size());
    f.push_back(static_cast<uint8_t>(crc & 0xFF));
    f.push_back(static_cast<uint8_t>(crc >> 8));
    return f;
}

static std::vector<uint8_t> encode_write_single(
    uint8_t slave_id, uint16_t reg, uint16_t value
) {
    std::vector<uint8_t> f = {
        slave_id, FC_WRITE_SINGLE,
        static_cast<uint8_t>(reg >> 8),   static_cast<uint8_t>(reg & 0xFF),
        static_cast<uint8_t>(value >> 8), static_cast<uint8_t>(value & 0xFF),
    };
    uint16_t crc = crc16_modbus(f.data(), f.size());
    f.push_back(static_cast<uint8_t>(crc & 0xFF));
    f.push_back(static_cast<uint8_t>(crc >> 8));
    return f;
}

static py::dict parse_frame(const std::vector<uint8_t>& raw) {
    if (raw.size() < 4)
        throw std::runtime_error("Modbus RTU frame too short (min 4 bytes)");
    uint8_t slave_id  = raw[0];
    uint8_t func_code = raw[1];
    std::vector<uint8_t> pdu(raw.begin() + 2, raw.end() - 2);
    uint16_t crc_recv = static_cast<uint16_t>(raw[raw.size()-2])
                      | static_cast<uint16_t>(raw[raw.size()-1]) << 8;
    uint16_t crc_calc = crc16_modbus(raw.data(), raw.size() - 2);
    py::dict d;
    d["slave_id"]  = slave_id;
    d["func_code"] = func_code;
    d["pdu_data"]  = pdu;
    d["crc"]       = crc_recv;
    d["crc_valid"] = (crc_recv == crc_calc);
    return d;
}

// ── ModbusRTUSlave ───────────────────────────────────────────────────────────

class ModbusRTUSlave {
public:
    explicit ModbusRTUSlave(uint8_t slave_id = 1, size_t n_regs = 256)
        : slave_id_(slave_id), regs_(n_regs, 0) {}

    void write_register(uint16_t addr, uint16_t value) {
        if (addr >= regs_.size()) throw std::out_of_range("Register out of range");
        regs_[addr] = value;
    }
    uint16_t read_register(uint16_t addr) const {
        if (addr >= regs_.size()) throw std::out_of_range("Register out of range");
        return regs_[addr];
    }

    std::vector<uint8_t> process_request(const std::vector<uint8_t>& req) {
        if (req.size() < 4) return build_error(slave_id_, 0, 0x03);
        auto f = parse_frame(req);
        if (!f["crc_valid"].cast<bool>()) return build_error(slave_id_, f["func_code"].cast<uint8_t>(), 0x04);
        if (f["slave_id"].cast<uint8_t>() != slave_id_) return {};

        uint8_t fc  = f["func_code"].cast<uint8_t>();
        auto pdu    = f["pdu_data"].cast<std::vector<uint8_t>>();

        if (fc == FC_READ_HOLDING || fc == FC_READ_INPUT) {
            if (pdu.size() < 4) return build_error(slave_id_, fc, 0x03);
            uint16_t start = (static_cast<uint16_t>(pdu[0]) << 8) | pdu[1];
            uint16_t count = (static_cast<uint16_t>(pdu[2]) << 8) | pdu[3];
            if (start + count > regs_.size()) return build_error(slave_id_, fc, 0x02);
            std::vector<uint8_t> resp = {slave_id_, fc, static_cast<uint8_t>(count * 2)};
            for (uint16_t i = 0; i < count; ++i) {
                resp.push_back(static_cast<uint8_t>(regs_[start+i] >> 8));
                resp.push_back(static_cast<uint8_t>(regs_[start+i] & 0xFF));
            }
            uint16_t crc = crc16_modbus(resp.data(), resp.size());
            resp.push_back(static_cast<uint8_t>(crc & 0xFF));
            resp.push_back(static_cast<uint8_t>(crc >> 8));
            return resp;
        } else if (fc == FC_WRITE_SINGLE) {
            if (pdu.size() < 4) return build_error(slave_id_, fc, 0x03);
            uint16_t addr  = (static_cast<uint16_t>(pdu[0]) << 8) | pdu[1];
            uint16_t value = (static_cast<uint16_t>(pdu[2]) << 8) | pdu[3];
            if (addr >= regs_.size()) return build_error(slave_id_, fc, 0x02);
            regs_[addr] = value;
            return req;  // FC06 response = echo
        }
        return build_error(slave_id_, fc, 0x01);
    }

private:
    uint8_t slave_id_;
    std::vector<uint16_t> regs_;

    static std::vector<uint8_t> build_error(uint8_t sid, uint8_t fc, uint8_t exc) {
        std::vector<uint8_t> resp = {sid, static_cast<uint8_t>(fc | FC_ERROR_FLAG), exc};
        uint16_t crc = crc16_modbus(resp.data(), resp.size());
        resp.push_back(static_cast<uint8_t>(crc & 0xFF));
        resp.push_back(static_cast<uint8_t>(crc >> 8));
        return resp;
    }
};

PYBIND11_MODULE(_modbus_rtu_kernel, m) {
    m.doc() = "Modbus RTU frame codec + holding-register slave simulator (C++14)";

    m.def("crc16_modbus", [](py::bytes data) -> uint16_t {
        std::string s = data;
        return crc16_modbus(reinterpret_cast<const uint8_t*>(s.data()), s.size());
    }, "Compute Modbus CRC-16/IBM");

    m.def("encode_read_holding", &encode_read_holding,
          py::arg("slave_id"), py::arg("start_reg"), py::arg("count"),
          "Encode FC03 read-holding-registers request → bytes list");

    m.def("encode_write_single", &encode_write_single,
          py::arg("slave_id"), py::arg("reg"), py::arg("value"),
          "Encode FC06 write-single-register request → bytes list");

    m.def("parse_frame", [](py::bytes raw) -> py::dict {
        std::string s = raw;
        return parse_frame(std::vector<uint8_t>(s.begin(), s.end()));
    }, "Parse raw Modbus RTU frame bytes → {slave_id, func_code, pdu_data, crc, crc_valid}");

    py::class_<ModbusRTUSlave>(m, "ModbusRTUSlave")
        .def(py::init<uint8_t, size_t>(),
             py::arg("slave_id") = 1, py::arg("n_regs") = 256)
        .def("write_register", &ModbusRTUSlave::write_register)
        .def("read_register",  &ModbusRTUSlave::read_register)
        .def("process_request", [](ModbusRTUSlave& s, py::bytes req) -> py::bytes {
            std::string r = req;
            auto resp = s.process_request(std::vector<uint8_t>(r.begin(), r.end()));
            return py::bytes(reinterpret_cast<const char*>(resp.data()), resp.size());
        }, "Process raw request bytes and return raw response bytes");
}
