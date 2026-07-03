#include "wproc/apc.hpp"

#include <algorithm>
#include <stdexcept>

namespace wproc::apc {

R2RController::R2RController(R2RParams p) : p_(p) {
    if (p_.gain == 0.0) throw std::invalid_argument("R2R: gain must be non-zero");
    if (p_.lambda <= 0.0 || p_.lambda > 1.0)
        throw std::invalid_argument("R2R: lambda must be in (0, 1]");
    feed_ = std::clamp(p_.feed0, p_.feed_min, p_.feed_max);
    // Initialize the offset estimate so the first control move keeps feed0.
    offset_hat_ = p_.target_um - p_.gain * feed_;
}

double R2RController::update(double measured_width_um) {
    // EWMA update of the process offset from this cut's result...
    offset_hat_ = p_.lambda * (measured_width_um - p_.gain * feed_) +
                  (1.0 - p_.lambda) * offset_hat_;
    // ...then invert the model to hit target on the next cut.
    double next = (p_.target_um - offset_hat_) / p_.gain;
    feed_ = std::clamp(next, p_.feed_min, p_.feed_max);
    return feed_;
}

}  // namespace wproc::apc
