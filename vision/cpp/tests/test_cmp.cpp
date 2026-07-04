// test_cmp.cpp — CMP Preston control, planarity (dishing/erosion), endpoint.
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "wproc/cmp.hpp"

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

static double rms_err(const std::vector<double>& v, double target) {
    double s = std::accumulate(
        v.begin(), v.end(), 0.0,
        [target](double a, double x) { return a + (x - target) * (x - target); });
    return std::sqrt(s / v.size());
}

static void test_control() {
    cmp::PolishParams bad;
    bad.kp = 0.0;
    bool threw = false;
    try { cmp::PolishController c(bad); (void)c; } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    cmp::PolishPlant plant;
    plant.kp0 = 1.0e-3;
    plant.pressure = 30.0;
    plant.velocity = 0.5;
    plant.kp_decay_per_run = 1.0e-6;   // pad glazes: same time removes less
    plant.noise_sigma = 2.0;

    const double target = 300.0;       // nm
    // Fresh rate = 1e-3*30*0.5 = 0.015 nm/s -> nominal time = 300/0.015 = 20000?
    // Scale so times are sane: use kp=1e-1 effective via pressure? Keep nominal:
    const double rate0 = plant.kp0 * plant.pressure * plant.velocity;  // 0.015
    const double nominal = target / rate0;   // 20000 s (units are illustrative)
    const int N = 200;
    std::mt19937 gen(5);
    std::normal_distribution<double> noise(0.0, plant.noise_sigma);

    std::vector<double> open_r, cl_r;
    cmp::PolishParams p;
    p.target_removal_nm = target;
    p.kp = plant.kp0;
    p.pressure = plant.pressure;
    p.velocity = plant.velocity;
    p.lambda = 0.3;
    p.time0 = nominal;
    p.time_max = 1e6;
    cmp::PolishController ctrl(p);

    double t = ctrl.current_time();
    for (long k = 1; k <= N; ++k) {
        open_r.push_back(plant.removed(nominal, k, noise(gen)));
        const double r = plant.removed(t, k, noise(gen));
        cl_r.push_back(r);
        t = ctrl.update(r);
    }
    const double open_rms = rms_err(open_r, target);
    const double cl_rms = rms_err(cl_r, target);
    std::cout << "CMP open RMS=" << open_rms << " closed RMS=" << cl_rms
              << " final t=" << ctrl.current_time() << "\n";
    CHECK(open_rms > 10.0);
    CHECK(cl_rms < 5.0);
    CHECK(cl_rms < 0.4 * open_rms);
    CHECK(ctrl.current_time() > p.time0);   // more time to offset pad glazing
}

static void test_planarity() {
    // Field at 0 nm; Cu features recessed 25 nm (dishing); small field noise.
    std::mt19937 gen(3);
    std::normal_distribution<double> noise(0.0, 1.0);
    std::vector<double> profile;
    std::vector<char> feat;
    for (int i = 0; i < 400; ++i) {
        const bool is_feat = (i % 40) >= 20;  // alternating field / Cu bands
        double h = is_feat ? -25.0 : 0.0;
        h += noise(gen);
        profile.push_back(h);
        feat.push_back(is_feat ? 1 : 0);
    }
    cmp::PlanaritySpec spec;
    spec.nominal_nm = 0.0;
    spec.max_dishing_nm = 40.0;
    spec.max_erosion_nm = 30.0;
    cmp::PlanarityStats s = cmp::analyze_planarity(profile, feat, spec);
    std::cout << "dishing=" << s.dishing_nm << " erosion=" << s.erosion_nm
              << " unif=" << s.uniformity_pct << " ok=" << s.ok << "\n";
    CHECK(std::fabs(s.dishing_nm - 25.0) < 3.0);
    CHECK(std::fabs(s.erosion_nm) < 3.0);
    CHECK(s.ok);

    // Excess dishing (60 nm) must fail the spec.
    std::vector<double> p2;
    std::vector<char> f2;
    for (int i = 0; i < 400; ++i) {
        const bool is_feat = (i % 40) >= 20;
        p2.push_back(is_feat ? -60.0 : 0.0);
        f2.push_back(is_feat ? 1 : 0);
    }
    cmp::PlanarityStats s2 = cmp::analyze_planarity(p2, f2, spec);
    CHECK(!s2.ok);
    CHECK(s2.dishing_nm > 40.0);

    // Degenerate inputs throw.
    bool threw = false;
    try { cmp::analyze_planarity({1.0, 2.0}, {0}, spec); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

static void test_endpoint() {
    // Friction plateaus high while polishing the film, then drops when cleared.
    const double fs = 10.0;   // Hz
    std::mt19937 gen(9);
    std::normal_distribution<double> noise(0.0, 0.03);
    std::vector<double> sig;
    const int clear_i = 180;  // endpoint sample
    for (int i = 0; i < 300; ++i) {
        double v = (i < clear_i) ? 1.0 : 0.3;  // step down at endpoint
        sig.push_back(v + noise(gen));
    }
    cmp::Endpoint e = cmp::detect_endpoint(sig, fs, /*drop_frac=*/0.5, /*smooth=*/5);
    std::cout << "endpoint idx=" << e.index << " t=" << e.time_s
              << " drop=" << e.drop << "\n";
    CHECK(e.detected);
    CHECK(std::abs(e.index - clear_i) < 12);          // near the true endpoint
    CHECK(std::fabs(e.time_s - clear_i / fs) < 1.5);

    // A flat signal has no endpoint.
    std::vector<double> flat(200, 0.5);
    cmp::Endpoint fe = cmp::detect_endpoint(flat, fs);
    CHECK(!fe.detected);
}

int main() {
    test_control();
    test_planarity();
    test_endpoint();
    std::cout << (g_total - g_failed) << "/" << g_total << " checks passed\n";
    if (g_failed) {
        std::cerr << g_failed << " FAILED\n";
        return 1;
    }
    std::cout << "CMP OK\n";
    return 0;
}
