// bopt.cpp — GP surrogate + acquisition + Bayesian-optimization loop.
#include "wproc/bopt.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

namespace wproc::bopt {
namespace {

// Standard normal pdf/cdf (cdf via erfc).
double norm_pdf(double z) { return std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI); }
double norm_cdf(double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }

// In-place Cholesky: A = L L^T (lower). Returns false if not SPD.
bool cholesky(std::vector<std::vector<double>>& A) {
    const std::size_t n = A.size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double s = A[i][j];
            for (std::size_t k = 0; k < j; ++k) s -= A[i][k] * A[j][k];
            if (i == j) {
                if (s <= 0.0) return false;
                A[i][j] = std::sqrt(s);
            } else {
                A[i][j] = s / A[j][j];
            }
        }
        for (std::size_t j = i + 1; j < n; ++j) A[i][j] = 0.0;  // keep lower only
    }
    return true;
}

// Solve L L^T x = b given lower-triangular L (forward then back substitution).
std::vector<double> chol_solve(const std::vector<std::vector<double>>& L,
                               const std::vector<double>& b) {
    const std::size_t n = L.size();
    std::vector<double> y(n), x(n);
    for (std::size_t i = 0; i < n; ++i) {
        double s = b[i];
        for (std::size_t k = 0; k < i; ++k) s -= L[i][k] * y[k];
        y[i] = s / L[i][i];
    }
    for (std::size_t ii = n; ii-- > 0;) {
        double s = y[ii];
        for (std::size_t k = ii + 1; k < n; ++k) s -= L[k][ii] * x[k];
        x[ii] = s / L[ii][ii];
    }
    return x;
}

}  // namespace

double GP::k(const Point& a, const Point& b) const {
    double d2 = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        d2 += d * d;
    }
    return kernel_.signal_var *
           std::exp(-0.5 * d2 / (kernel_.lengthscale * kernel_.lengthscale));
}

bool GP::fit(const std::vector<Point>& X, const std::vector<double>& y) {
    X_ = X; y_ = y;
    ok_ = false;
    const std::size_t n = X_.size();
    if (n == 0) return false;
    std::vector<std::vector<double>> K(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            K[i][j] = k(X_[i], X_[j]) + (i == j ? kernel_.noise : 0.0);
    if (!cholesky(K)) return false;
    L_ = std::move(K);
    alpha_ = chol_solve(L_, y_);
    ok_ = true;
    return true;
}

Posterior GP::predict(const Point& x) const {
    Posterior p;
    if (!ok_) return p;
    const std::size_t n = X_.size();
    std::vector<double> ks(n);
    for (std::size_t i = 0; i < n; ++i) ks[i] = k(X_[i], x);
    for (std::size_t i = 0; i < n; ++i) p.mean += ks[i] * alpha_[i];
    // variance = k(x,x) - v^T v, with v = L^{-1} ks (forward substitution).
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        double s = ks[i];
        for (std::size_t j = 0; j < i; ++j) s -= L_[i][j] * v[j];
        v[i] = s / L_[i][i];
    }
    const double vv = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    p.variance = std::max(0.0, k(x, x) - vv);
    p.stddev = std::sqrt(p.variance);
    return p;
}

double expected_improvement(const Posterior& p, double f_best, double xi) {
    if (p.stddev < 1e-12) return 0.0;
    const double imp = f_best - p.mean - xi;   // minimization
    const double z = imp / p.stddev;
    return imp * norm_cdf(z) + p.stddev * norm_pdf(z);
}

double lcb_acquisition(const Posterior& p, double kappa) {
    return -(p.mean - kappa * p.stddev);
}

Result minimize(const std::function<double(const Point&)>& f, const Bounds& b,
                const Config& cfg) {
    Result r;
    const std::size_t dim = b.lo.size();
    if (dim == 0 || b.hi.size() != dim) return r;

    std::mt19937 gen(cfg.seed);
    std::vector<std::uniform_real_distribution<double>> u;
    u.reserve(dim);
    for (std::size_t d = 0; d < dim; ++d) u.emplace_back(b.lo[d], b.hi[d]);
    auto sample = [&]() {
        Point x(dim);
        for (std::size_t d = 0; d < dim; ++d) x[d] = u[d](gen);
        return x;
    };

    std::vector<Point> X;
    std::vector<double> Y;
    double best = std::numeric_limits<double>::infinity();
    Point best_x;
    auto record = [&](const Point& x, double y) {
        X.push_back(x); Y.push_back(y);
        if (y < best) { best = y; best_x = x; }
        r.history.push_back({x, y, best});
    };

    const int n_init = std::max(2, cfg.n_init);
    for (int i = 0; i < n_init; ++i) {
        Point x = sample();
        record(x, f(x));
    }

    for (int it = 0; it < cfg.n_iter; ++it) {
        GP gp(cfg.kernel);
        if (!gp.fit(X, Y)) {  // degenerate surrogate — fall back to random
            Point x = sample();
            record(x, f(x));
            continue;
        }
        Point next = sample();
        double best_acq = -std::numeric_limits<double>::infinity();
        for (int c = 0; c < cfg.candidates; ++c) {
            Point cand = sample();
            const Posterior p = gp.predict(cand);
            const double a = cfg.acq == Acq::EI
                                 ? expected_improvement(p, best)
                                 : lcb_acquisition(p, cfg.kappa);
            if (a > best_acq) { best_acq = a; next = cand; }
        }
        record(next, f(next));
    }

    r.ok = true;
    r.best_x = best_x;
    r.best_y = best;
    r.evaluations = static_cast<int>(Y.size());
    return r;
}

}  // namespace wproc::bopt
