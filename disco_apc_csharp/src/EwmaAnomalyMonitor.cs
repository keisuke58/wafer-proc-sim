using System;

namespace Disco.Apc
{
    /// <summary>
    /// EWMA control-chart anomaly monitor for equipment / process time-series.
    /// Smooths the incoming signal with an EWMA and flags a point when the
    /// smoothed value crosses time-varying control limits derived from the
    /// in-control standard deviation. Sensitive to small sustained shifts
    /// (tool drift, consumable faults) that a simple 3σ check on raw samples
    /// would miss.
    ///
    /// z_k   = λ·x_k + (1-λ)·z_{k-1}
    /// σ_z,k = σ · sqrt( (λ/(2-λ)) · (1 - (1-λ)^{2k}) )
    /// flag  = |z_k - center| > L · σ_z,k
    /// </summary>
    public sealed class EwmaAnomalyMonitor
    {
        private readonly double _lambda;
        private readonly double _L;
        private readonly double _sigma;
        private readonly double _center;
        private double _z;
        private int _k;

        /// <param name="center">In-control mean of the monitored signal.</param>
        /// <param name="sigma">In-control standard deviation (must be &gt; 0).</param>
        /// <param name="lambda">EWMA weight in (0,1]; 0.2 is a common default.</param>
        /// <param name="L">Control-limit width in sigmas (≈3 for ~0.3% false alarm).</param>
        public EwmaAnomalyMonitor(double center, double sigma,
                                  double lambda = 0.2, double L = 3.0)
        {
            if (sigma <= 0.0)
                throw new ArgumentOutOfRangeException(nameof(sigma), "sigma must be positive");
            if (lambda <= 0.0 || lambda > 1.0)
                throw new ArgumentOutOfRangeException(nameof(lambda), "lambda must be in (0,1]");

            _center = center;
            _sigma = sigma;
            _lambda = lambda;
            _L = L;
            _z = center;
        }

        /// <summary>Feed one sample; returns true if it is flagged as anomalous.
        /// The EWMA starts from the in-control center (not the first sample) so
        /// early samples are not spuriously flagged against the tight startup
        /// control limits.</summary>
        public bool Update(double x)
        {
            _z = _lambda * x + (1.0 - _lambda) * _z;
            _k++;

            double sez = _sigma * Math.Sqrt(
                (_lambda / (2.0 - _lambda)) *
                (1.0 - Math.Pow(1.0 - _lambda, 2 * _k)));
            double ucl = _center + _L * sez;
            double lcl = _center - _L * sez;
            return _z > ucl || _z < lcl;
        }

        /// <summary>Current EWMA statistic.</summary>
        public double Ewma { get { return _z; } }
    }
}
