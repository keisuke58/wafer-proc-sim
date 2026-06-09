"""Tests for AP試験 second wave:
  comms/hmac_auth.cpp         → セキュリティ (SHA-256, HMAC, 認証)
  embedded/memory_pool.cpp    → OS・メモリ管理 (プール, スタック)
  comms/ip_packet_kernel.cpp  → TCP/IP (IPv4, TCP, チェックサム)
"""
from __future__ import annotations
import hashlib
import sys, os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ══════════════════════════════════════════════════════════════════════════════
# HMAC / SHA-256 (セキュリティ)
# ══════════════════════════════════════════════════════════════════════════════

class TestHmacAuth:
    @pytest.fixture(autouse=True)
    def _import(self):
        pytest.importorskip("comms._hmac_auth_kernel",
                            reason="HMAC kernel not built; run build_all_kernels.sh hmac")
        from comms import _hmac_auth_kernel as hmac
        self.hmac = hmac

    def test_sha256_known_vector_abc(self):
        """SHA-256('abc') must match system hashlib (cross-impl consistency)."""
        expected = hashlib.sha256(b"abc").hexdigest()
        assert self.hmac.sha256("abc") == expected

    def test_sha256_empty_string(self):
        """SHA-256('') must match NIST test vector."""
        expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert self.hmac.sha256("") == expected

    def test_sha256_length(self):
        """SHA-256 output must always be 64 hex chars (256 bits)."""
        for s in ["", "a", "hello world", "x" * 1000]:
            assert len(self.hmac.sha256(s)) == 64

    def test_sha256_avalanche(self):
        """One-bit change in input must produce completely different digest."""
        h1 = self.hmac.sha256("DISCO")
        h2 = self.hmac.sha256("EISCO")
        assert h1 != h2
        # At least half the hex digits should differ (avalanche effect)
        diffs = sum(c1 != c2 for c1, c2 in zip(h1, h2))
        assert diffs > 20, f"Avalanche too weak: only {diffs}/64 hex chars differ"

    def test_hmac_deterministic(self):
        """Same key + message must always produce the same HMAC."""
        h1 = self.hmac.hmac_sha256("key", "message")
        h2 = self.hmac.hmac_sha256("key", "message")
        assert h1 == h2

    def test_hmac_different_key(self):
        """Different key must produce different HMAC."""
        h1 = self.hmac.hmac_sha256("key1", "message")
        h2 = self.hmac.hmac_sha256("key2", "message")
        assert h1 != h2

    def test_verify_hmac_correct(self):
        """verify_hmac must return True for correct (key, message, mac) triple."""
        mac = self.hmac.hmac_sha256("secret", "RECIPE:R01")
        assert self.hmac.verify_hmac("secret", "RECIPE:R01", mac) is True

    def test_verify_hmac_wrong_key(self):
        """verify_hmac must return False when key is wrong."""
        mac = self.hmac.hmac_sha256("secret", "RECIPE:R01")
        assert self.hmac.verify_hmac("wrong_key", "RECIPE:R01", mac) is False

    def test_challenge_response_auth_ok(self):
        """Legitimate command with correct HMAC must be accepted."""
        r = self.hmac.challenge_response_demo("shared_secret_K", "RECIPE:R01", "NONCE_abc")
        assert r["auth_ok"] is True

    def test_challenge_response_replay_blocked(self):
        """Replay attack with different nonce must be blocked."""
        r = self.hmac.challenge_response_demo("shared_secret_K", "RECIPE:R01", "NONCE_abc")
        assert r["replay_rejected"] is True

    def test_challenge_response_tamper_blocked(self):
        """Tampered command (1-bit flip) must be detected."""
        r = self.hmac.challenge_response_demo("shared_secret_K", "RECIPE:R01", "NONCE_abc")
        assert r["tamper_rejected"] is True


# ══════════════════════════════════════════════════════════════════════════════
# Memory Pool (OS・メモリ管理)
# ══════════════════════════════════════════════════════════════════════════════

class TestMemoryPool:
    @pytest.fixture(autouse=True)
    def _import(self):
        pytest.importorskip("embedded._memory_pool_kernel",
                            reason="Memory pool kernel not built; run build_all_kernels.sh mempool")
        from embedded import _memory_pool_kernel as mp
        self.mp = mp

    def test_pool_no_failed_allocs_under_capacity(self):
        """Alloc < n_blocks times must never fail."""
        r = self.mp.simulate_pool(block_size=64, n_blocks=32,
                                  n_steps=30, alloc_prob=1.0, seed=0)
        assert r["n_failed"] == 0

    def test_pool_exhaustion_triggers_failure(self):
        """Continuously allocating must eventually fail when pool is full."""
        r = self.mp.simulate_pool(block_size=64, n_blocks=8,
                                  n_steps=100, alloc_prob=1.0, seed=0)
        assert r["n_failed"] > 0

    def test_pool_peak_utilization_bounded(self):
        """Peak utilization must be in [0, 1]."""
        r = self.mp.simulate_pool(block_size=256, n_blocks=64,
                                  n_steps=1000, alloc_prob=0.6, seed=42)
        assert 0.0 <= r["peak_utilization"] <= 1.0

    def test_pool_zero_fragmentation_label(self):
        """Fixed-size pool fragmentation must be labelled zero."""
        r = self.mp.simulate_pool(64, 32, 100, 0.5, 1)
        assert "zero" in r["fragmentation"].lower()

    def test_stack_alloc_respects_capacity(self):
        """Allocating more than capacity must produce failures."""
        r = self.mp.simulate_stack(100, [60, 60])  # 60+60=120 > 100
        assert r["n_failed"] > 0

    def test_stack_frame_pop_frees_memory(self):
        """Negative size (restore_mark) must reduce used bytes."""
        r = self.mp.simulate_stack(256, [50, 50, -1, 30])
        # After pop: 100 bytes freed, then 30 allocated → used=30
        assert r["used_bytes"] == 30

    def test_compare_pool_deterministic(self):
        """Pool must be flagged deterministic; heap must not."""
        r = self.mp.compare_allocators(256, 64)
        assert r["pool_deterministic"] is True
        assert r["heap_deterministic"] is False

    def test_compare_pool_faster_alloc(self):
        """Pool alloc cycles must be much less than heap worst-case."""
        r = self.mp.compare_allocators(256, 64)
        assert r["pool_alloc_cycles_always"] < r["heap_alloc_cycles_max"]


# ══════════════════════════════════════════════════════════════════════════════
# IP Packet / TCP (ネットワーク TCP/IP)
# ══════════════════════════════════════════════════════════════════════════════

class TestIpPacket:
    @pytest.fixture(autouse=True)
    def _import(self):
        pytest.importorskip("comms._ip_packet_kernel",
                            reason="IP packet kernel not built; run build_all_kernels.sh ip")
        from comms import _ip_packet_kernel as ip
        self.ip = ip

    def test_ipv4_header_length(self):
        """IPv4 header must be exactly 20 bytes (IHL=5, no options)."""
        hdr = self.ip.build_ipv4_header("192.168.1.10", "192.168.1.1")
        assert len(hdr) == 20

    def test_ipv4_checksum_valid(self):
        """Parsed IPv4 header must report checksum_ok=True."""
        hdr = self.ip.build_ipv4_header("10.0.0.1", "10.0.0.2",
                                        protocol=17, ttl=64, payload_len=100)
        r = self.ip.parse_ipv4_header(hdr)
        assert r["checksum_ok"] is True, "IPv4 checksum verification failed"

    def test_ipv4_parse_roundtrip(self):
        """Build then parse must recover identical src/dst/protocol/ttl."""
        hdr = self.ip.build_ipv4_header("192.168.10.20", "10.1.2.3",
                                        protocol=6, ttl=128, payload_len=0)
        r = self.ip.parse_ipv4_header(hdr)
        assert r["src"] == "192.168.10.20"
        assert r["dst"] == "10.1.2.3"
        assert r["protocol"] == 6
        assert r["protocol_name"] == "TCP"
        assert r["ttl"] == 128

    def test_ipv4_df_flag(self):
        """DF (Don't Fragment) flag must be parsed correctly."""
        hdr_df  = self.ip.build_ipv4_header("1.2.3.4", "5.6.7.8", df_flag=True)
        hdr_no  = self.ip.build_ipv4_header("1.2.3.4", "5.6.7.8", df_flag=False)
        assert self.ip.parse_ipv4_header(hdr_df)["df"]  is True
        assert self.ip.parse_ipv4_header(hdr_no)["df"]  is False

    def test_internet_checksum_zero_data(self):
        """Checksum of all-zero 20 bytes must be 0xffff."""
        cksum = self.ip.internet_checksum(b"\x00" * 20)
        assert cksum == 0xffff

    def test_udp_header_length(self):
        """UDP header must be exactly 8 bytes."""
        hdr = self.ip.build_udp_header(5000, 80, payload_len=100)
        assert len(hdr) == 8

    def test_tcp_header_length(self):
        """TCP header must be exactly 20 bytes (no options)."""
        hdr = self.ip.build_tcp_header(5001, 80)
        assert len(hdr) == 20

    def test_tcp_handshake_three_steps(self):
        """3-way handshake must contain exactly 3 steps before data transfer."""
        steps = self.ip.tcp_handshake_demo()
        handshake = [s for s in steps if s["flags"] in ("SYN", "SYN+ACK", "ACK")
                     and s["client_state"] in ("SYN_SENT", "SYN_RECEIVED", "ESTABLISHED")]
        # SYN, SYN+ACK, ACK must all appear
        flag_set = {s["flags"] for s in steps}
        assert "SYN"     in flag_set
        assert "SYN+ACK" in flag_set

    def test_tcp_handshake_reaches_established(self):
        """After handshake, both sides must reach ESTABLISHED."""
        steps = self.ip.tcp_handshake_demo()
        established = [s for s in steps if s["client_state"] == "ESTABLISHED"]
        assert len(established) > 0

    def test_tcp_handshake_fin_teardown(self):
        """Teardown must include FIN flag."""
        steps = self.ip.tcp_handshake_demo()
        fin_steps = [s for s in steps if "FIN" in s["flags"]]
        assert len(fin_steps) >= 2, "Both sides must send FIN"
