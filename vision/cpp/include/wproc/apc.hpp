// apc.hpp — advanced process control for closed-loop kerf-width regulation.
//
// A run-to-run (R2R) EWMA controller, the workhorse of semiconductor APC:
// after each cut it updates an exponentially-weighted estimate of the process
// offset from the measured kerf width, then sets the next feed so the predicted
// width lands on target. This rejects slow disturbances — notably the width
// drift caused by blade wear — that a fixed recipe cannot.
//
//   model:   width = offset + gain * feed
//   update:  offset_hat = lambda*(width - gain*feed) + (1-lambda)*offset_hat
//   control: feed_next  = clamp((target - offset_hat) / gain)
#pragma once

namespace wproc::apc {

struct R2RParams {
    double target_um = 40.0;    // desired kerf width
    double gain = 5.0;          // d(width)/d(feed), model estimate (gain != 0)
    double lambda = 0.3;        // EWMA weight in (0, 1]
    double feed0 = 1.0;         // initial feed setting
    double feed_min = 0.2;      // actuator bounds
    double feed_max = 3.0;
};

class R2RController {
public:
    explicit R2RController(R2RParams p);

    // Feed the measured width of the cut made with the last commanded feed;
    // returns the feed to command for the next cut.
    double update(double measured_width_um);

    double current_feed() const { return feed_; }
    double offset_estimate() const { return offset_hat_; }

private:
    R2RParams p_;
    double feed_;         // last commanded feed
    double offset_hat_;   // EWMA estimate of the process offset
};

// Simple first-order dicing "plant" for simulation/tests: the kerf width for a
// given feed at cut index k, including a linear wear drift and Gaussian noise.
struct DicingPlant {
    double offset0_um = 35.0;      // width = offset0 + gain*feed at k=0
    double gain = 5.0;             // true process gain
    double drift_um_per_cut = 0.0; // blade-wear drift
    double noise_sigma = 0.0;

    // width produced by `feed` at cut k, with a deterministic noise value n
    // (caller supplies noise so simulations stay reproducible).
    double width(double feed, long k, double n) const {
        return offset0_um + gain * feed + drift_um_per_cut * k + n;
    }
};

}  // namespace wproc::apc
