// test_fab.cpp — digital-twin line: bottleneck, throughput, yield, OEE.
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "wproc/fab.hpp"

using namespace wproc;

static int g_total = 0, g_failed = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        ++g_total;                                                      \
        if (!(cond)) {                                                  \
            ++g_failed;                                                 \
            std::cerr << "FAIL [" << __LINE__ << "]: " #cond "\n";      \
        }                                                               \
    } while (0)

static void test_line() {
    std::vector<fab::Station> line = {
        {"dice",    30.0, 28.0, 0.95, 0.99},
        {"grind",   45.0, 42.0, 0.92, 0.98},   // bottleneck
        {"polish",  40.0, 38.0, 0.90, 0.97},
        {"inspect", 20.0, 20.0, 0.99, 0.995},
    };
    fab::LineResult r = fab::analyze_line(line);
    std::cout << "bottleneck=" << r.stations[r.bottleneck].name
              << " thru=" << r.throughput_wph << " wph, yield=" << r.line_yield
              << ", good=" << r.good_wph << ", OEE=" << r.line_oee << "\n";

    // Effective cycles: dice 31.6, grind 48.9, polish 44.4, inspect 20.2.
    CHECK(r.bottleneck == 1);                       // grind
    CHECK(r.stations[1].bottleneck);
    CHECK(std::fabs(r.throughput_wph - 3600.0 / (45.0 / 0.92)) < 1e-6);
    CHECK(r.throughput_wph > 73.0 && r.throughput_wph < 74.0);
    CHECK(std::fabs(r.line_yield - 0.99 * 0.98 * 0.97 * 0.995) < 1e-9);
    CHECK(r.good_wph < r.throughput_wph);           // yield loss
    CHECK(r.line_oee > 0.75 && r.line_oee < 0.85);
    // Bottleneck runs at 100% utilization; others below.
    CHECK(std::fabs(r.stations[1].utilization - 1.0) < 1e-9);
    CHECK(r.stations[3].utilization < 0.5);         // inspect is lightly loaded
    // Per-station OEE = A*P*Q, in (0,1).
    for (const auto& s : r.stations) {
        CHECK(s.oee > 0.0 && s.oee <= 1.0);
        CHECK(std::fabs(s.oee - s.availability * s.performance * s.quality) < 1e-12);
    }

    // Speeding up the bottleneck raises throughput.
    line[1].cycle_s = 32.0;
    fab::LineResult r2 = fab::analyze_line(line);
    CHECK(r2.throughput_wph > r.throughput_wph);
}

static void test_validation() {
    bool threw = false;
    try { fab::analyze_line({}); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { fab::analyze_line({{"bad", 0.0, 0.0, 1.0, 1.0}}); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    threw = false;
    try { fab::analyze_line({{"bad", 10.0, 0.0, 1.5, 1.0}}); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

int main() {
    test_line();
    test_validation();
    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "FAB OK\n";
    return 0;
}
