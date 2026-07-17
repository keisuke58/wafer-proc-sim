using System;

namespace Disco.Apc
{
    /// <summary>
    /// Deterministic, seedable pseudo-random source (splitmix64 + Box-Muller) so
    /// that simulations and tests are reproducible across machines — independent
    /// of the platform RNG. splitmix64 gives high-quality, well-decorrelated
    /// draws, which matters for Box-Muller (it consumes two uniforms per sample).
    /// </summary>
    public sealed class DeterministicRandom
    {
        private ulong _state;

        public DeterministicRandom(ulong seed)
        {
            _state = seed == 0 ? 0x9E3779B97F4A7C15UL : seed;
        }

        private double NextUnit()
        {
            _state += 0x9E3779B97F4A7C15UL;
            ulong z = _state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            z = z ^ (z >> 31);
            ulong bits = (z >> 11) & 0x1FFFFFFFFFFFFFUL;
            return bits / (double)0x20000000000000UL;
        }

        public double NextGaussian(double mean, double std)
        {
            double u1 = NextUnit();
            double u2 = NextUnit();
            double z = Math.Sqrt(-2.0 * Math.Log(u1 + 1e-12)) *
                       Math.Cos(2.0 * Math.PI * u2);
            return mean + std * z;
        }
    }

    /// <summary>
    /// Synthetic single-input single-output equipment process with a slow
    /// linear drift, a step disturbance (fault injection) at a chosen lot, and
    /// measurement noise:  y = offset(k) + gain·u + noise.
    /// </summary>
    public sealed class ProcessSimulator
    {
        private readonly double _gain;
        private readonly double _baseOffset;
        private readonly double _driftPerLot;
        private readonly double _noiseStd;
        private readonly int _stepAtLot;
        private readonly double _stepMagnitude;
        private readonly DeterministicRandom _rng;

        public ProcessSimulator(double gain, double baseOffset, double driftPerLot,
                                double noiseStd, int stepAtLot, double stepMagnitude,
                                ulong seed)
        {
            _gain = gain;
            _baseOffset = baseOffset;
            _driftPerLot = driftPerLot;
            _noiseStd = noiseStd;
            _stepAtLot = stepAtLot;
            _stepMagnitude = stepMagnitude;
            _rng = new DeterministicRandom(seed);
        }

        /// <summary>True (noise-free) process offset at a given lot index.</summary>
        public double TrueOffset(int lot)
        {
            double o = _baseOffset + _driftPerLot * lot;
            if (lot >= _stepAtLot) o += _stepMagnitude;
            return o;
        }

        /// <summary>Run one lot with recipe input <paramref name="input"/>.</summary>
        public double Run(int lot, double input)
        {
            return TrueOffset(lot) + _gain * input + _rng.NextGaussian(0.0, _noiseStd);
        }

        public double Gain { get { return _gain; } }
    }
}
