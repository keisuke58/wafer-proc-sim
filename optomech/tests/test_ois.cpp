// test_ois.cpp — optical image stabilization rejection.
#include <cmath>
#include <iostream>

#include "optomech/ois.hpp"

using namespace optm;

static int g_total = 0, g_failed = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        ++g_total;                                                      \
        if (!(cond)) {                                                  \
            ++g_failed;                                                 \
            std::cerr << "FAIL [" << __LINE__ << "]: " #cond "\n";      \
        }                                                               \
    } while (0)

static void test_stabilization() {
    ois::Result r = ois::simulate(35.0);
    std::cout << "OIS off=" << r.off_rms_um << "µm on=" << r.on_rms_um
              << "µm ratio=" << r.ratio << " stops=" << r.stops << "\n";
    CHECK(r.demand_um.size() == r.residual_um.size());
    CHECK(r.off_rms_um > r.on_rms_um);   // OIS reduces image motion
    CHECK(r.ratio > 2.0);
    CHECK(r.stops > 1.0);                 // at least ~1 stop of stabilization
}

static void test_bandwidth() {
    ois::Plant slow; slow.act_freq_hz = 12.0;
    ois::Plant fast; fast.act_freq_hz = 60.0;
    ois::Result rs = ois::simulate(35.0, slow);
    ois::Result rf = ois::simulate(35.0, fast);
    std::cout << "slow stops=" << rs.stops << " fast stops=" << rf.stops << "\n";
    // A faster actuator tracks more of the tremor => more stabilization.
    CHECK(rf.stops > rs.stops);
    // Longer focal length has larger absolute image motion to reject.
    CHECK(ois::simulate(85.0).off_rms_um > ois::simulate(24.0).off_rms_um);
}

int main() {
    test_stabilization();
    test_bandwidth();
    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) { std::cerr << g_failed << " FAILED\n"; return 1; }
    std::cout << "OIS OK\n";
    return 0;
}
