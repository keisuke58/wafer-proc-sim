// hmac_auth.cpp — SHA-256 + HMAC-SHA256 + OT authentication demo
//
// AP試験 concepts demonstrated (セキュリティ):
//   - SHA-256 ハッシュ関数 (FIPS 180-4): 一方向性, 衝突耐性
//   - HMAC-SHA256 (RFC 2104): HMAC(K,m) = H((K⊕opad) || H((K⊕ipad) || m))
//   - チャレンジ・レスポンス認証: なりすまし / リプレイ攻撃対策
//   - IEC 62443 OT セキュリティ: PLC コマンドの完全性検証
//   - MAC vs 電子署名の違い: 共有秘密鍵 → MAC, 公開鍵 → 署名
//
// Application: DISCO DFL7160 の上位 PC → PLC コマンド認証。
//   プロセスレシピ改ざん検知 / 不正コマンド注入防止。
//
// Pure C++14 — no OpenSSL / Botan dependency.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// ── SHA-256 (FIPS 180-4) ─────────────────────────────────────────────────────

static const uint32_t SHA256_H0[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
};

static const uint32_t SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

static inline uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

static std::array<uint8_t, 32> sha256_raw(const uint8_t* data, size_t len) {
    uint32_t h[8];
    std::copy(SHA256_H0, SHA256_H0 + 8, h);

    // Padding: append 0x80, zeros, 64-bit big-endian length in bits
    size_t padded_len = ((len + 8) / 64 + 1) * 64;
    std::vector<uint8_t> padded(padded_len, 0);
    std::copy(data, data + len, padded.begin());
    padded[len] = 0x80u;
    uint64_t bit_len = static_cast<uint64_t>(len) * 8;
    for (int i = 0; i < 8; ++i)
        padded[padded_len - 8 + i] = static_cast<uint8_t>(bit_len >> (56 - 8 * i));

    // Process 512-bit blocks
    for (size_t blk = 0; blk < padded_len; blk += 64) {
        uint32_t W[64];
        for (int i = 0; i < 16; ++i)
            W[i] = (uint32_t(padded[blk+i*4])   << 24) |
                   (uint32_t(padded[blk+i*4+1]) << 16) |
                   (uint32_t(padded[blk+i*4+2]) <<  8) |
                    uint32_t(padded[blk+i*4+3]);
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr32(W[i-15], 7) ^ rotr32(W[i-15],18) ^ (W[i-15] >> 3);
            uint32_t s1 = rotr32(W[i-2], 17) ^ rotr32(W[i-2], 19) ^ (W[i-2]  >>10);
            W[i] = W[i-16] + s0 + W[i-7] + s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hv=h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1   = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
            uint32_t ch   = (e & f) ^ (~e & g);
            uint32_t t1   = hv + S1 + ch + SHA256_K[i] + W[i];
            uint32_t S0   = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
            uint32_t maj  = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2   = S0 + maj;
            hv=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d;
        h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hv;
    }

    std::array<uint8_t, 32> digest;
    for (int i = 0; i < 8; ++i) {
        digest[i*4]   = (h[i] >> 24) & 0xffu;
        digest[i*4+1] = (h[i] >> 16) & 0xffu;
        digest[i*4+2] = (h[i] >>  8) & 0xffu;
        digest[i*4+3] =  h[i]        & 0xffu;
    }
    return digest;
}

static inline std::string to_hex(const std::array<uint8_t,32>& d) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (uint8_t b : d) oss << std::setw(2) << static_cast<int>(b);
    return oss.str();
}

// ── HMAC-SHA256 (RFC 2104) ────────────────────────────────────────────────────
// HMAC(K, m) = H( (K⊕opad) || H( (K⊕ipad) || m ) )
// ipad = 0x36 × 64,  opad = 0x5c × 64
static std::array<uint8_t, 32> hmac_sha256_raw(
    const uint8_t* key, size_t key_len,
    const uint8_t* msg, size_t msg_len
) {
    constexpr size_t BLOCK = 64;
    uint8_t K[BLOCK] = {};

    if (key_len > BLOCK) {
        // Key too long: hash it first
        auto hk = sha256_raw(key, key_len);
        std::copy(hk.begin(), hk.end(), K);
    } else {
        std::copy(key, key + key_len, K);
    }

    uint8_t ipad[BLOCK], opad[BLOCK];
    for (size_t i = 0; i < BLOCK; ++i) {
        ipad[i] = K[i] ^ 0x36u;
        opad[i] = K[i] ^ 0x5cu;
    }

    // inner = H(ipad || msg)
    std::vector<uint8_t> inner(BLOCK + msg_len);
    std::copy(ipad, ipad + BLOCK, inner.begin());
    std::copy(msg, msg + msg_len, inner.begin() + BLOCK);
    auto inner_hash = sha256_raw(inner.data(), inner.size());

    // outer = H(opad || inner_hash)
    std::vector<uint8_t> outer(BLOCK + 32);
    std::copy(opad, opad + BLOCK, outer.begin());
    std::copy(inner_hash.begin(), inner_hash.end(), outer.begin() + BLOCK);
    return sha256_raw(outer.data(), outer.size());
}

// ── Python-facing helpers ─────────────────────────────────────────────────────

static std::string sha256_hex(const std::string& data) {
    return to_hex(sha256_raw(
        reinterpret_cast<const uint8_t*>(data.data()), data.size()));
}

static std::string hmac_sha256_hex(const std::string& key, const std::string& msg) {
    return to_hex(hmac_sha256_raw(
        reinterpret_cast<const uint8_t*>(key.data()), key.size(),
        reinterpret_cast<const uint8_t*>(msg.data()), msg.size()));
}

static bool verify_hmac(const std::string& key,
                        const std::string& msg,
                        const std::string& expected_hex) {
    std::string computed = hmac_sha256_hex(key, msg);
    // Constant-time comparison (prevent timing attack)
    if (computed.size() != expected_hex.size()) return false;
    uint8_t diff = 0;
    for (size_t i = 0; i < computed.size(); ++i)
        diff |= static_cast<uint8_t>(computed[i] ^ expected_hex[i]);
    return diff == 0;
}

// ── OT Challenge-Response authentication demo ─────────────────────────────────
// Models PLC ↔ Host authentication (IEC 62443 concept):
//   1. Host generates random challenge (nonce)
//   2. PLC computes response = HMAC-SHA256(shared_key, challenge || command)
//   3. Host verifies response — accepts only if MAC matches
//   4. Replay attack: re-using an old (challenge, response) pair fails if
//      nonce is unique per session (demonstrated by this function).
py::dict challenge_response_demo(
    const std::string& shared_key,
    const std::string& command,
    const std::string& challenge
) {
    // PLC signs: HMAC(key, challenge || command)
    std::string payload = challenge + command;
    std::string mac = hmac_sha256_hex(shared_key, payload);

    // Host verifies
    bool auth_ok = verify_hmac(shared_key, payload, mac);

    // Replay attack: attacker replays with a different challenge — fails
    std::string replay_challenge = challenge + "_old";
    std::string replay_payload   = replay_challenge + command;
    bool replay_ok = verify_hmac(shared_key, replay_payload, mac);

    // Tamper attack: attacker flips one byte of command — fails
    std::string tampered_cmd = command;
    if (!tampered_cmd.empty()) tampered_cmd[0] ^= 0x01;
    std::string tamper_payload = challenge + tampered_cmd;
    bool tamper_ok = verify_hmac(shared_key, tamper_payload, mac);

    py::dict r;
    r["challenge"]       = challenge;
    r["command"]         = command;
    r["mac_hex"]         = mac;
    r["auth_ok"]         = auth_ok;      // True: legitimate command accepted
    r["replay_rejected"] = !replay_ok;   // True: replay attack blocked
    r["tamper_rejected"] = !tamper_ok;   // True: tampering detected
    r["hash_bits"]       = 256;
    r["algo"]            = std::string("HMAC-SHA256");
    return r;
}

// ── pybind11 module ──────────────────────────────────────────────────────────

PYBIND11_MODULE(_hmac_auth_kernel, m) {
    m.doc() =
        "SHA-256 (FIPS 180-4) + HMAC-SHA256 (RFC 2104) + OT authentication demo.\n"
        "\n"
        "AP試験 topics: ハッシュ関数, MAC, チャレンジ・レスポンス認証,\n"
        "  IEC 62443 OTセキュリティ, リプレイ攻撃対策, 改ざん検知\n"
        "\n"
        "Pure C++14 — no OpenSSL / external crypto dependency.\n"
        "Test vector: sha256('abc') = ba7816bf...d422a25d  (NIST FIPS 180-4)";

    m.def("sha256", &sha256_hex,
          py::arg("data"),
          "SHA-256 hash of data (str → treats as bytes).\n"
          "Returns 64-char hex string.  One-way; collision-resistant.\n"
          "Test: sha256('abc') = 'ba7816bf...d422a25d'");

    m.def("hmac_sha256", &hmac_sha256_hex,
          py::arg("key"), py::arg("message"),
          "HMAC-SHA256(key, message).  Returns 64-char hex string.\n"
          "Provides authentication + integrity (not just integrity like SHA-256).\n"
          "Requires shared secret key; attacker cannot forge MAC without key.");

    m.def("verify_hmac", &verify_hmac,
          py::arg("key"), py::arg("message"), py::arg("expected_hex"),
          "Constant-time HMAC verification (prevents timing attacks).\n"
          "Returns True if HMAC(key, message) == expected_hex.");

    m.def("challenge_response_demo", &challenge_response_demo,
          py::arg("shared_key"),
          py::arg("command")   = std::string("RECIPE:R01;FEED:80;RPM:30000"),
          py::arg("challenge") = std::string("NONCE_7a3f9b"),
          "Simulate PLC ↔ Host HMAC-based authentication (IEC 62443).\n"
          "Demonstrates: legitimate auth accepted, replay blocked, tamper detected.\n"
          "Returns: challenge, command, mac_hex, auth_ok, replay_rejected, tamper_rejected.");
}
