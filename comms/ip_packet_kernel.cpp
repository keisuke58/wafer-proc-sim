// ip_packet_kernel.cpp — IPv4 / TCP / UDP packet building + TCP state machine
//
// AP試験 concepts demonstrated (ネットワーク・TCP/IP):
//   - IPv4 ヘッダ構造 (RFC 791): version, IHL, DSCP, total_length,
//       identification, flags(DF/MF), fragment_offset, TTL, protocol,
//       header_checksum, src_addr, dst_addr
//   - インターネットチェックサム (RFC 1071): 16-bit one's complement sum
//   - TCP ヘッダ (RFC 793): src/dst port, seq/ack, flags(SYN/ACK/FIN/RST/PSH)
//   - UDP ヘッダ (RFC 768): src/dst port, length, checksum
//   - TCP 3ウェイハンドシェイク: SYN → SYN+ACK → ACK
//   - TCP 状態遷移: CLOSED→SYN_SENT→ESTABLISHED→FIN_WAIT→TIME_WAIT→CLOSED
//   - OSI / TCP/IP モデル: L2(EtherCAT) L3(IP) L4(TCP/UDP) の対応
//
// Application: DISCO telemetry over Ethernet uses UDP/IP.
//   EtherCAT runs L2 (no IP), Modbus TCP would use TCP/IP.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// ── Internet checksum (RFC 1071) ─────────────────────────────────────────────
// Sum all 16-bit words, add carry into low 16 bits, one's-complement.
// Checksum of header with this field set to result should be 0xffff.
static uint16_t internet_checksum(const uint8_t* data, size_t len) {
    uint32_t sum = 0;
    for (size_t i = 0; i + 1 < len; i += 2)
        sum += (uint32_t(data[i]) << 8) | data[i+1];
    if (len & 1) sum += uint32_t(data[len-1]) << 8;
    while (sum >> 16) sum = (sum & 0xffffu) + (sum >> 16);
    return static_cast<uint16_t>(~sum);
}

// IP address string → 4 bytes (e.g. "192.168.1.1")
static std::array<uint8_t,4> parse_ip(const std::string& s) {
    std::array<uint8_t,4> ip{};
    int oct = 0, val = 0;
    for (char c : s) {
        if (c == '.') { if (oct < 4) ip[oct++] = static_cast<uint8_t>(val); val = 0; }
        else           { val = val * 10 + (c - '0'); }
    }
    if (oct < 4) ip[oct] = static_cast<uint8_t>(val);
    return ip;
}

static std::string ip_to_str(const std::array<uint8_t,4>& ip) {
    std::ostringstream o;
    o << int(ip[0]) << '.' << int(ip[1]) << '.' << int(ip[2]) << '.' << int(ip[3]);
    return o.str();
}

// ── IPv4 header builder ──────────────────────────────────────────────────────

py::bytes build_ipv4_header(
    const std::string& src_ip,
    const std::string& dst_ip,
    uint8_t  protocol,      // 6=TCP, 17=UDP, 1=ICMP
    uint8_t  ttl,
    uint16_t payload_len,
    uint16_t identification,
    bool     df_flag        // Don't Fragment
) {
    uint16_t total_len = 20 + payload_len;
    uint8_t hdr[20] = {};
    hdr[0]  = 0x45u;                      // Version=4, IHL=5 (20 bytes)
    hdr[1]  = 0x00u;                      // DSCP/ECN
    hdr[2]  = (total_len >> 8) & 0xffu;   // Total length high
    hdr[3]  =  total_len       & 0xffu;   // Total length low
    hdr[4]  = (identification >> 8) & 0xffu;
    hdr[5]  =  identification       & 0xffu;
    uint16_t flags_frag = df_flag ? 0x4000u : 0x0000u;
    hdr[6]  = (flags_frag >> 8) & 0xffu;
    hdr[7]  =  flags_frag       & 0xffu;
    hdr[8]  = ttl;
    hdr[9]  = protocol;
    hdr[10] = 0; hdr[11] = 0;             // Checksum placeholder
    auto src = parse_ip(src_ip);
    auto dst = parse_ip(dst_ip);
    std::copy(src.begin(), src.end(), hdr + 12);
    std::copy(dst.begin(), dst.end(), hdr + 16);

    uint16_t cksum = internet_checksum(hdr, 20);
    hdr[10] = (cksum >> 8) & 0xffu;
    hdr[11] =  cksum       & 0xffu;

    return py::bytes(reinterpret_cast<const char*>(hdr), 20);
}

py::dict parse_ipv4_header(const std::string& raw) {
    if (raw.size() < 20)
        throw std::invalid_argument("IPv4 header must be at least 20 bytes");
    const auto* h = reinterpret_cast<const uint8_t*>(raw.data());

    uint8_t  version  = (h[0] >> 4) & 0x0fu;
    uint8_t  ihl      = (h[0]     ) & 0x0fu;
    uint16_t total_len= (uint16_t(h[2]) << 8) | h[3];
    uint16_t id       = (uint16_t(h[4]) << 8) | h[5];
    bool     df       = (h[6] & 0x40u) != 0;
    bool     mf       = (h[6] & 0x20u) != 0;
    uint16_t frag_off = ((uint16_t(h[6] & 0x1fu) << 8) | h[7]) * 8;
    uint8_t  ttl      = h[8];
    uint8_t  proto    = h[9];
    uint16_t cksum    = (uint16_t(h[10]) << 8) | h[11];
    std::array<uint8_t,4> src{h[12],h[13],h[14],h[15]};
    std::array<uint8_t,4> dst{h[16],h[17],h[18],h[19]};

    // Verify checksum: internet_checksum returns ~sum, so a valid header
    // (where raw 1's-complement sum = 0xffff) gives ~0xffff = 0x0000.
    uint16_t check_result = internet_checksum(h, ihl * 4u);

    const char* proto_name = proto == 6 ? "TCP" : proto == 17 ? "UDP" :
                             proto == 1 ? "ICMP" : "OTHER";

    py::dict r;
    r["version"]       = version;
    r["ihl_bytes"]     = ihl * 4;
    r["total_length"]  = total_len;
    r["identification"]= id;
    r["df"]            = df;
    r["mf"]            = mf;
    r["fragment_offset"]= frag_off;
    r["ttl"]           = ttl;
    r["protocol"]      = proto;
    r["protocol_name"] = std::string(proto_name);
    r["checksum"]      = cksum;
    r["checksum_ok"]   = (check_result == 0x0000u);
    r["src"]           = ip_to_str(src);
    r["dst"]           = ip_to_str(dst);
    return r;
}

// ── UDP header builder ────────────────────────────────────────────────────────

py::bytes build_udp_header(
    uint16_t src_port,
    uint16_t dst_port,
    uint16_t payload_len
) {
    uint16_t udp_len = 8 + payload_len;
    uint8_t hdr[8] = {};
    hdr[0] = (src_port >> 8) & 0xffu;
    hdr[1] =  src_port       & 0xffu;
    hdr[2] = (dst_port >> 8) & 0xffu;
    hdr[3] =  dst_port       & 0xffu;
    hdr[4] = (udp_len  >> 8) & 0xffu;
    hdr[5] =  udp_len        & 0xffu;
    hdr[6] = 0; hdr[7] = 0;   // Checksum (optional for UDP over IPv4)
    return py::bytes(reinterpret_cast<const char*>(hdr), 8);
}

// ── TCP header builder ────────────────────────────────────────────────────────
// flags bits: SYN=0x02, ACK=0x10, FIN=0x01, RST=0x04, PSH=0x08

py::bytes build_tcp_header(
    uint16_t src_port,
    uint16_t dst_port,
    uint32_t seq,
    uint32_t ack_seq,
    uint8_t  flags,       // SYN=0x02, ACK=0x10, FIN=0x01, RST=0x04
    uint16_t window_size
) {
    uint8_t hdr[20] = {};
    hdr[0] = (src_port >> 8) & 0xffu;
    hdr[1] =  src_port       & 0xffu;
    hdr[2] = (dst_port >> 8) & 0xffu;
    hdr[3] =  dst_port       & 0xffu;
    for (int i = 0; i < 4; ++i) hdr[4+i] = (seq     >> (24-8*i)) & 0xffu;
    for (int i = 0; i < 4; ++i) hdr[8+i] = (ack_seq >> (24-8*i)) & 0xffu;
    hdr[12] = 0x50u;   // Data offset = 5 (20 bytes), reserved = 0
    hdr[13] = flags;
    hdr[14] = (window_size >> 8) & 0xffu;
    hdr[15] =  window_size       & 0xffu;
    hdr[16] = 0; hdr[17] = 0;   // Checksum (requires pseudo-header; 0 here)
    hdr[18] = 0; hdr[19] = 0;   // Urgent pointer
    return py::bytes(reinterpret_cast<const char*>(hdr), 20);
}

// ── TCP 3-way handshake + teardown simulation ─────────────────────────────────
// Returns step-by-step trace as list of state-transition dicts.
// TCP flags: SYN=2, ACK=16, FIN=1, SYN+ACK=18
py::list tcp_handshake_demo() {
    py::list steps;

    struct Step {
        std::string from, to, c_state, s_state, flags, note;
        uint32_t seq, ack;
    };

    // Assign: Client ISN=1000, Server ISN=5000
    uint32_t c_seq = 1000, s_seq = 5000;

    // AP試験 TCP状態遷移 (RFC 793)
    std::vector<Step> trace = {
        // 3-way handshake (connection establishment)
        {"CLIENT","SERVER","SYN_SENT",    "SYN_RECEIVED","SYN",
         "クライアントが接続要求。ISN=1000",
         c_seq, 0},

        {"SERVER","CLIENT","SYN_SENT",    "SYN_RECEIVED","SYN+ACK",
         "サーバが同期確認。ISN=5000, ACK=client_ISN+1",
         s_seq, c_seq + 1},

        {"CLIENT","SERVER","ESTABLISHED", "ESTABLISHED", "ACK",
         "クライアントがACK送信。3ウェイハンドシェイク完了",
         c_seq + 1, s_seq + 1},

        // Data transfer (abbreviated)
        {"CLIENT","SERVER","ESTABLISHED", "ESTABLISHED", "PSH+ACK",
         "データ送信: RECIPEコマンド (100 bytes)",
         c_seq + 1, s_seq + 1},

        {"SERVER","CLIENT","ESTABLISHED", "ESTABLISHED", "ACK",
         "データ受信確認",
         s_seq + 1, c_seq + 101},

        // Connection teardown (active close by client)
        {"CLIENT","SERVER","FIN_WAIT_1",  "CLOSE_WAIT",  "FIN+ACK",
         "クライアントが接続終了要求",
         c_seq + 101, s_seq + 1},

        {"SERVER","CLIENT","FIN_WAIT_2",  "CLOSE_WAIT",  "ACK",
         "サーバがFINを確認",
         s_seq + 1, c_seq + 102},

        {"SERVER","CLIENT","FIN_WAIT_2",  "LAST_ACK",    "FIN+ACK",
         "サーバも終了要求",
         s_seq + 1, c_seq + 102},

        {"CLIENT","SERVER","TIME_WAIT",   "CLOSED",      "ACK",
         "最終ACK。TIME_WAIT (2MSL) 後 CLOSED",
         c_seq + 102, s_seq + 2},
    };

    for (const auto& s : trace) {
        py::dict d;
        d["from"]          = s.from;
        d["to"]            = s.to;
        d["client_state"]  = s.c_state;
        d["server_state"]  = s.s_state;
        d["flags"]         = s.flags;
        d["seq"]           = s.seq;
        d["ack"]           = s.ack;
        d["note"]          = s.note;
        steps.append(d);
    }
    return steps;
}

// ── Internet checksum Python wrapper ─────────────────────────────────────────
static int py_internet_checksum(const std::string& data) {
    return static_cast<int>(internet_checksum(
        reinterpret_cast<const uint8_t*>(data.data()), data.size()));
}

// ── pybind11 module ──────────────────────────────────────────────────────────

PYBIND11_MODULE(_ip_packet_kernel, m) {
    m.doc() =
        "IPv4 / TCP / UDP packet builder + TCP 3-way handshake simulation.\n"
        "\n"
        "AP試験 topics: IPv4ヘッダ, インターネットチェックサム, TCP/UDP,\n"
        "  3ウェイハンドシェイク, TCP状態遷移, OSI L3/L4対応\n"
        "\n"
        "OSI mapping in this repo:\n"
        "  L2: ethercat_frame_kernel.cpp  (EtherCAT, IEEE 802.3)\n"
        "  L2: modbus_rtu_kernel.cpp      (Modbus RTU over RS-485)\n"
        "  L3: ip_packet_kernel.cpp       (IPv4)          ← this file\n"
        "  L4: ip_packet_kernel.cpp       (TCP / UDP)     ← this file\n"
        "  L7: comms/hmac_auth.cpp        (HMAC authentication)";

    m.def("build_ipv4_header", &build_ipv4_header,
          py::arg("src_ip")       = std::string("192.168.1.10"),
          py::arg("dst_ip")       = std::string("192.168.1.1"),
          py::arg("protocol")     = 17,
          py::arg("ttl")          = 64,
          py::arg("payload_len")  = 0,
          py::arg("identification")= 0x1234,
          py::arg("df_flag")      = true,
          "Build 20-byte IPv4 header with correct Internet checksum.\n"
          "protocol: 6=TCP, 17=UDP, 1=ICMP.  Returns bytes.");

    m.def("parse_ipv4_header", &parse_ipv4_header,
          py::arg("raw"),
          "Parse raw bytes as IPv4 header.\n"
          "Returns dict: version, ihl_bytes, total_length, identification,\n"
          "  df, mf, fragment_offset, ttl, protocol, protocol_name,\n"
          "  checksum, checksum_ok, src, dst.");

    m.def("build_udp_header", &build_udp_header,
          py::arg("src_port"), py::arg("dst_port"), py::arg("payload_len") = 0,
          "Build 8-byte UDP header.  Returns bytes.");

    m.def("build_tcp_header", &build_tcp_header,
          py::arg("src_port"), py::arg("dst_port"),
          py::arg("seq")         = 0,
          py::arg("ack_seq")     = 0,
          py::arg("flags")       = 0x02,  // SYN
          py::arg("window_size") = 65535,
          "Build 20-byte TCP header.\n"
          "flags: SYN=0x02, ACK=0x10, FIN=0x01, RST=0x04, PSH=0x08.\n"
          "Returns bytes.");

    m.def("internet_checksum", &py_internet_checksum,
          py::arg("data"),
          "Compute Internet checksum (RFC 1071): 16-bit one's-complement sum.\n"
          "checksum(header with this field = 0) inserted into header.\n"
          "checksum(header with filled checksum) should equal 0xffff.");

    m.def("tcp_handshake_demo", &tcp_handshake_demo,
          "Simulate TCP 3-way handshake + data transfer + graceful teardown.\n"
          "Returns list of step dicts: from, to, client_state, server_state,\n"
          "  flags, seq, ack, note.");
}
