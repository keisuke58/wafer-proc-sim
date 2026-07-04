// bopt.hpp — Gaussian-process Bayesian optimization for process-recipe search.
//
// Finding a good dicing/grinding recipe (feed rate, spindle load, coolant, …)
// by grid search is expensive: every trial is a real wafer. Bayesian
// optimization instead builds a probabilistic surrogate of the objective from
// the trials so far — a Gaussian process — and uses it to pick the single next
// recipe that is most promising, balancing exploiting the current best against
// exploring where the model is uncertain. It typically reaches the optimum in a
// fraction of the trials a grid would need.
//
// This implements a GP with an RBF (squared-exponential) kernel — fitted by a
// hand-written Cholesky solve — plus Expected-Improvement and Lower-Confidence-
// Bound acquisition functions, and a minimize() loop that drives them.
//
// Pure C++17, no external dependencies.
#pragma once

#include <functional>
#include <vector>

namespace wproc::bopt {

using Point = std::vector<double>;

struct Kernel {
    double signal_var = 1.0;   // output scale
    double lengthscale = 1.0;  // input scale (isotropic)
    double noise = 1e-4;       // observation noise variance (jitter)
};

// Posterior prediction at a query point.
struct Posterior {
    double mean = 0.0;
    double variance = 0.0;
    double stddev = 0.0;
};

// A fitted GP regressor over the observed (X, y).
class GP {
public:
    GP() = default;
    explicit GP(const Kernel& k) : kernel_(k) {}

    // Condition on the data; returns false if the kernel matrix is not SPD.
    bool fit(const std::vector<Point>& X, const std::vector<double>& y);

    Posterior predict(const Point& x) const;

    const std::vector<Point>& inputs() const { return X_; }
    const std::vector<double>& targets() const { return y_; }

private:
    double k(const Point& a, const Point& b) const;

    Kernel kernel_;
    std::vector<Point> X_;
    std::vector<double> y_;
    std::vector<std::vector<double>> L_;  // Cholesky factor of (K + noise*I)
    std::vector<double> alpha_;           // (K + noise*I)^{-1} y
    bool ok_ = false;
};

// Acquisition (both assume MINIMIZATION; larger return = more desirable next x).
// Expected improvement below the incumbent best `f_best`.
double expected_improvement(const Posterior& p, double f_best, double xi = 0.01);
// Lower confidence bound acquisition: -(mean - kappa*sigma).
double lcb_acquisition(const Posterior& p, double kappa = 2.0);

// Box bounds for the search space.
struct Bounds {
    Point lo;
    Point hi;
};

enum class Acq { EI, LCB };

struct Trace {
    Point x;
    double y = 0.0;
    double best_so_far = 0.0;
};

struct Result {
    bool ok = false;
    Point best_x;
    double best_y = 0.0;
    int evaluations = 0;
    std::vector<Trace> history;   // one entry per evaluation (init + iterations)
};

struct Config {
    int n_init = 5;          // random initial design points
    int n_iter = 20;         // BO iterations after the initial design
    int candidates = 512;    // random candidates scored per iteration
    Acq acq = Acq::EI;
    double kappa = 2.0;      // LCB exploration weight
    unsigned seed = 12345;
    Kernel kernel;
};

// Minimize `f` over the box `b`. `f` is called once per evaluation.
Result minimize(const std::function<double(const Point&)>& f, const Bounds& b,
                const Config& cfg = Config{});

}  // namespace wproc::bopt
