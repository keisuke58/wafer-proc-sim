// ois.hpp — optical image stabilization (hand-shake rejection).
//
// Hand tremor rotates the camera by a small angle theta(t); at the sensor this
// becomes a lateral image motion of focal_length * theta that blurs the photo.
// Optical image stabilization measures the motion with a gyro and drives a
// shift actuator (a lens group or the sensor) to cancel it. The catch is the
// actuator has finite bandwidth, so fast tremor leaks through.
//
// This synthesizes a realistic multi-tone tremor, integrates it to an image
// demand, and passes the cancellation command through a second-order actuator
// model, returning the residual image motion with OIS off vs on — and the
// stabilization expressed in photographic "stops" (log2 of the RMS ratio).
//
// Pure C++17, no external dependencies.
#pragma once

#include <vector>

namespace optm::ois {

struct Plant {
    double act_freq_hz = 40.0;  // actuator servo bandwidth (natural freq)
    double damping = 0.7;       // actuator damping ratio
    double gyro_noise_urad = 40.0;  // gyro angle noise (1-sigma) [µrad]
    double duration_s = 2.0;
    double dt_s = 5e-4;
};

struct Result {
    double focal_mm = 0.0;
    double off_rms_um = 0.0;    // image-motion RMS, OIS off
    double on_rms_um = 0.0;     // image-motion RMS, OIS on
    double ratio = 0.0;         // off / on
    double stops = 0.0;         // log2(ratio)
    std::vector<double> t_s;
    std::vector<double> demand_um;   // image motion without OIS
    std::vector<double> residual_um; // image motion with OIS
};

// Simulate OIS for a lens of `focal_mm`. A fixed seed makes the tremor and gyro
// noise reproducible.
Result simulate(double focal_mm, const Plant& plant = Plant{}, unsigned seed = 11);

}  // namespace optm::ois
