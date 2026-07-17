using System;

namespace Disco.Apc
{
    /// <summary>
    /// Exponentially-weighted-moving-average (EWMA) Run-to-Run controller — the
    /// workhorse of semiconductor Advanced Process Control. After each lot it
    /// estimates the slowly-drifting process offset and picks the next recipe
    /// input so the predicted output lands on target.
    ///
    /// Process model:  y = offset + gain * u + noise
    ///   * <c>gain</c>   is the known/estimated sensitivity Δoutput/Δinput.
    ///   * <c>offset</c> drifts (tool wear, consumable ageing) and is what the
    ///     controller tracks with an EWMA filter to reject noise.
    ///
    /// Control law:  u_next = (target - offsetEstimate) / gain
    /// Update law :  offsetEstimate ← λ·(y - gain·u) + (1-λ)·offsetEstimate
    /// </summary>
    public sealed class RunToRunController
    {
        private readonly double _gain;
        private readonly double _lambda;
        private readonly double _target;
        private double _offsetEstimate;
        private bool _initialised;

        /// <param name="gain">Process gain (must be non-zero).</param>
        /// <param name="target">Desired process output.</param>
        /// <param name="lambda">EWMA weight in (0,1]; smaller = smoother, slower.</param>
        /// <param name="initialOffset">Prior guess of the process offset.</param>
        public RunToRunController(double gain, double target,
                                  double lambda = 0.3, double initialOffset = 0.0)
        {
            if (Math.Abs(gain) < 1e-12)
                throw new ArgumentException("gain must be non-zero", nameof(gain));
            if (lambda <= 0.0 || lambda > 1.0)
                throw new ArgumentOutOfRangeException(nameof(lambda), "lambda must be in (0,1]");

            _gain = gain;
            _target = target;
            _lambda = lambda;
            _offsetEstimate = initialOffset;
        }

        /// <summary>Recipe input to apply on the next lot.</summary>
        public double NextInput()
        {
            return (_target - _offsetEstimate) / _gain;
        }

        /// <summary>Fold a completed lot's measurement into the offset estimate.</summary>
        public void Update(double input, double measuredOutput)
        {
            double observedOffset = measuredOutput - _gain * input;
            _offsetEstimate = _initialised
                ? _lambda * observedOffset + (1.0 - _lambda) * _offsetEstimate
                : observedOffset;
            _initialised = true;
        }

        /// <summary>Current EWMA estimate of the process offset.</summary>
        public double OffsetEstimate { get { return _offsetEstimate; } }
    }
}
