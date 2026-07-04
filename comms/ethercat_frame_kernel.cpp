// ethercat_frame_kernel.cpp — EtherCAT frame parser + slave simulator (pybind11)
//
// Implements EtherCAT Ethernet frame parsing per IEC 61158-3 / ETG.1000.
// Supports command codes: NOP, APRD, APWR, FPRD, FPWR, BRD, BWR, LRW.
//
// Typical use in DISCO: EtherCAT master ↔ Beckhoff/Yaskawa servo drives
// for spindle + XY stage synchronised motion.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

// ── Command codes ─────────────────────────────────────────────────────────────

enum EcCmd : uint8_t {
    NOP  = 0x00, APRD = 0x01, APWR = 0x02,
    FPRD = 0x04, FPWR = 0x05,
    BRD  = 0x07, BWR  = 0x08,
    LRW  = 0x0C,
};

static const char* cmd_name(uint8_t c) {
    switch (c) {
        case NOP:  return "NOP";
        case APRD: return "APRD";
        case APWR: return "APWR";
        case FPRD: return "FPRD";
        case FPWR: return "FPWR";
        case BRD:  return "BRD";
        case BWR:  return "BWR";
        case LRW:  return "LRW";
        default:   return "UNKNOWN";
    }
}

// ── Datagram structure ────────────────────────────────────────────────────────

struct EcDatagram {
    uint8_t  cmd  = NOP;
    uint8_t  idx  = 0;       // datagram index (request/response matching)
    uint32_t addr = 0;       // ADP:ADO (16:16) or logical address (32)
    std::vector<uint8_t> data;
    uint16_t wkc  = 0;       // working counter (incremented by each slave that responds)
};

// ── Frame parse (input: bytes after Ethernet type 0x88A4) ────────────────────

static std::vector<EcDatagram> parse_ec_frame(const std::vector<uint8_t>& raw) {
    if (raw.size() < 2)
        throw std::runtime_error("EtherCAT frame too short");

    // Bytes 0-1: frame header  [length:11 | type:4 | reserved:1]
    // Bytes 2+:  datagrams
    std::vector<EcDatagram> out;
    size_t off = 2;

    while (off + 10 <= raw.size()) {
        EcDatagram dg;
        dg.cmd = raw[off++];
        dg.idx = raw[off++];
        dg.addr = static_cast<uint32_t>(raw[off])
                | static_cast<uint32_t>(raw[off+1]) << 8
                | static_cast<uint32_t>(raw[off+2]) << 16
                | static_cast<uint32_t>(raw[off+3]) << 24;
        off += 4;
        uint16_t len_flags = static_cast<uint16_t>(raw[off])
                           | static_cast<uint16_t>(raw[off+1]) << 8;
        off += 2;
        off += 2;  // IRQ (skip)
        uint16_t dlen    = len_flags & 0x07FFu;
        bool     has_next = (len_flags >> 15) & 1u;

        if (off + dlen + 2 > raw.size()) break;
        dg.data.assign(raw.begin() + off, raw.begin() + off + dlen);
        off += dlen;
        dg.wkc = static_cast<uint16_t>(raw[off]) | static_cast<uint16_t>(raw[off+1]) << 8;
        off += 2;
        out.push_back(dg);
        if (!has_next) break;
    }
    return out;
}

// ── Frame builder ─────────────────────────────────────────────────────────────

static std::vector<uint8_t> build_ec_frame(
    uint8_t cmd, uint8_t idx, uint32_t addr,
    const std::vector<uint8_t>& data, bool has_next
) {
    uint16_t dlen      = static_cast<uint16_t>(data.size());
    uint16_t len_flags = dlen | (has_next ? static_cast<uint16_t>(1u << 15) : 0u);
    uint16_t frame_len = static_cast<uint16_t>(10 + dlen + 2);
    uint16_t hdr       = static_cast<uint16_t>((frame_len & 0x07FFu) | (1u << 12));

    std::vector<uint8_t> raw;
    raw.push_back(static_cast<uint8_t>(hdr & 0xFF));
    raw.push_back(static_cast<uint8_t>(hdr >> 8));
    raw.push_back(cmd); raw.push_back(idx);
    raw.push_back(static_cast<uint8_t>(addr & 0xFF));
    raw.push_back(static_cast<uint8_t>((addr >> 8) & 0xFF));
    raw.push_back(static_cast<uint8_t>((addr >> 16) & 0xFF));
    raw.push_back(static_cast<uint8_t>((addr >> 24) & 0xFF));
    raw.push_back(static_cast<uint8_t>(len_flags & 0xFF));
    raw.push_back(static_cast<uint8_t>(len_flags >> 8));
    raw.push_back(0); raw.push_back(0);   // IRQ
    raw.insert(raw.end(), data.begin(), data.end());
    raw.push_back(0); raw.push_back(0);   // WKC placeholder
    return raw;
}

// ── EtherCAT slave simulator ──────────────────────────────────────────────────

class EcSlave {
public:
    uint16_t addr_;
    std::unordered_map<uint32_t, uint8_t> mem_;

    explicit EcSlave(uint16_t configured_addr = 0x0001)
        : addr_(configured_addr) {
        // Minimal ESC register map (ETG.1000.4 Table A.1)
        mem_[0x0000] = 0x02;   // ESC type
        mem_[0x0001] = 0x05;   // ESC revision
        mem_[0x0100] = 0x00;   // AL Control
        mem_[0x0130] = 0x01;   // AL Status: INIT state
        mem_[0x0134] = 0x00;   // AL Status Code: no error
    }

    void write_byte(uint32_t a, uint8_t v) { mem_[a] = v; }
    uint8_t read_byte(uint32_t a) const {
        auto it = mem_.find(a);
        return it != mem_.end() ? it->second : 0x00;
    }

    // Process a datagram dict (from parse_ec_frame) and return updated dict with WKC
    py::dict process_datagram(py::dict in_dg) {
        uint8_t  cmd  = in_dg["cmd"].cast<uint8_t>();
        uint8_t  idx  = in_dg["idx"].cast<uint8_t>();
        uint32_t a    = in_dg["addr"].cast<uint32_t>();
        uint16_t wkc  = in_dg["wkc"].cast<uint16_t>();
        auto data     = in_dg["data"].cast<std::vector<uint8_t>>();

        uint16_t adp = static_cast<uint16_t>(a & 0xFFFF);
        uint16_t ado = static_cast<uint16_t>(a >> 16);
        bool addressed = false;

        if (cmd == FPRD || cmd == FPWR)  addressed = (adp == addr_);
        else if (cmd == BRD || cmd == BWR || cmd == LRW) addressed = true;

        if (addressed) {
            if (cmd == FPRD || cmd == BRD)
                for (size_t i = 0; i < data.size(); ++i)
                    data[i] = read_byte(ado + static_cast<uint32_t>(i));
            else if (cmd == FPWR || cmd == BWR)
                for (size_t i = 0; i < data.size(); ++i)
                    write_byte(ado + static_cast<uint32_t>(i), data[i]);
            else if (cmd == LRW)
                for (size_t i = 0; i < data.size(); ++i)
                    data[i] = read_byte(ado + static_cast<uint32_t>(i));
            ++wkc;
        }
        py::dict out;
        out["cmd"]      = cmd;
        out["cmd_name"] = std::string(cmd_name(cmd));
        out["idx"]      = idx;
        out["addr"]     = a;
        out["data"]     = data;
        out["wkc"]      = wkc;
        return out;
    }
};

PYBIND11_MODULE(_ethercat_frame_kernel, m) {
    m.doc() = "EtherCAT frame parser + slave simulator (C++14, IEC 61158-3 / ETG.1000)";

    m.def("parse_ec_frame", [](py::bytes raw) -> py::list {
        std::string s = raw;
        auto dgs = parse_ec_frame(std::vector<uint8_t>(s.begin(), s.end()));
        py::list lst;
        for (const auto& dg : dgs) {
            py::dict d;
            d["cmd"]      = dg.cmd;
            d["cmd_name"] = std::string(cmd_name(dg.cmd));
            d["idx"]      = dg.idx;
            d["addr"]     = dg.addr;
            d["data"]     = dg.data;
            d["wkc"]      = dg.wkc;
            lst.append(d);
        }
        return lst;
    }, "Parse EtherCAT frame bytes (after Ethernet type field) → list of datagram dicts");

    m.def("build_ec_frame", [](uint8_t cmd, uint8_t idx, uint32_t addr,
                                py::bytes data, bool has_next) -> py::bytes {
        std::string s = data;
        auto raw = build_ec_frame(cmd, idx, addr,
                                  std::vector<uint8_t>(s.begin(), s.end()), has_next);
        return py::bytes(reinterpret_cast<const char*>(raw.data()), raw.size());
    }, py::arg("cmd"), py::arg("idx"), py::arg("addr"),
       py::arg("data"), py::arg("has_next") = false,
       "Build EtherCAT frame with one datagram → bytes");

    m.def("cmd_name", [](uint8_t c){ return std::string(cmd_name(c)); });

    py::class_<EcSlave>(m, "EcSlave")
        .def(py::init<uint16_t>(), py::arg("configured_addr") = 0x0001)
        .def("write_byte",       &EcSlave::write_byte)
        .def("read_byte",        &EcSlave::read_byte)
        .def("process_datagram", &EcSlave::process_datagram,
             "Process a datagram dict and return updated dict with WKC incremented");
}
