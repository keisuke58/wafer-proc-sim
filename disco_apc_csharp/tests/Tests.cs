using System;
using Disco.Apc;

namespace Disco.Apc.Tests
{
    /// <summary>
    /// Dependency-free test harness (no NuGet/xUnit needed so it builds under
    /// both `dotnet test` shims and the Mono compiler). Exit code 0 = all pass,
    /// 1 = a failure. Each check prints PASS/FAIL with a label.
    /// </summary>
    public static class Tests
    {
        private static int _failures;

        private static void Check(bool cond, string label)
        {
            Console.WriteLine((cond ? "  PASS  " : "  FAIL  ") + label);
            if (!cond) _failures++;
        }

        public static int Main()
        {
            Console.WriteLine("Disco.Apc test suite");
            Console.WriteLine("--------------------");

            // 1. Controller offset/inversion math (deterministic).
            //    gain=2, target=10, observe y=4 at u=0 => offset=4 => next u=(10-4)/2=3
            var c1 = new RunToRunController(2.0, 10.0, 0.3, 0.0);
            c1.Update(0.0, 4.0);
            Check(Math.Abs(c1.OffsetEstimate - 4.0) < 1e-9, "controller first-update offset = y - gain*u");
            Check(Math.Abs(c1.NextInput() - 3.0) < 1e-9, "controller inverts model to hit target");

            // 2. EWMA blends new observation with prior estimate.
            var c2 = new RunToRunController(1.0, 0.0, 0.5, 0.0);
            c2.Update(0.0, 2.0);   // offset -> 2 (first)
            c2.Update(0.0, 4.0);   // offset -> 0.5*4 + 0.5*2 = 3
            Check(Math.Abs(c2.OffsetEstimate - 3.0) < 1e-9, "EWMA update blends 50/50 at lambda=0.5");

            // 3. Constructor guards.
            Check(Throws(() => new RunToRunController(0.0, 1.0)), "zero gain rejected");
            Check(Throws(() => new RunToRunController(1.0, 1.0, 1.5)), "lambda>1 rejected");
            Check(Throws(() => new EwmaAnomalyMonitor(0.0, -1.0)), "negative sigma rejected");

            // 4. Closed loop beats open loop under drift (RMSE + yield).
            RunLoops(out double rmseClosed, out double rmseOpen,
                     out double yieldClosed, out double yieldOpen);
            Check(rmseClosed < rmseOpen, "closed-loop RMSE < open-loop RMSE under drift");
            Check(yieldClosed > yieldOpen + 20.0, "closed-loop in-spec yield >> open-loop");

            // 5. Controller drives steady-state error small under constant drift.
            Check(SteadyStateError() < 0.4, "R2R steady-state |error| stays small under drift");

            // 6. Anomaly monitor: detects an injected step, few false alarms before it.
            AnomalyOnStep(out bool detected, out int falseBefore);
            Check(detected, "EWMA monitor flags an injected 6-sigma step");
            Check(falseBefore <= 2, "EWMA monitor keeps pre-fault false alarms low");

            Console.WriteLine("--------------------");
            Console.WriteLine(_failures == 0
                ? "ALL TESTS PASSED"
                : (_failures + " TEST(S) FAILED"));
            return _failures == 0 ? 0 : 1;
        }

        private static bool Throws(Action a)
        {
            try { a(); return false; } catch { return true; }
        }

        private static void RunLoops(out double rmseClosed, out double rmseOpen,
                                     out double yieldClosed, out double yieldOpen)
        {
            const int n = 150; const double gain = 2.0, target = 10.0, tol = 1.0;
            const double baseOff = 1.0, drift = 0.05, noise = 0.3;
            var pc = new ProcessSimulator(gain, baseOff, drift, noise, 100000, 0.0, 7);
            var po = new ProcessSimulator(gain, baseOff, drift, noise, 100000, 0.0, 7);
            var ctrl = new RunToRunController(gain, target, 0.3, baseOff);
            double u0 = (target - baseOff) / gain;
            double seC = 0, seO = 0; int inC = 0, inO = 0;
            for (int k = 0; k < n; k++)
            {
                double u = ctrl.NextInput();
                double yc = pc.Run(k, u); ctrl.Update(u, yc);
                double yo = po.Run(k, u0);
                seC += (yc - target) * (yc - target);
                seO += (yo - target) * (yo - target);
                if (Math.Abs(yc - target) <= tol) inC++;
                if (Math.Abs(yo - target) <= tol) inO++;
            }
            rmseClosed = Math.Sqrt(seC / n); rmseOpen = Math.Sqrt(seO / n);
            yieldClosed = 100.0 * inC / n; yieldOpen = 100.0 * inO / n;
        }

        private static double SteadyStateError()
        {
            const int n = 200; const double gain = 1.5, target = 5.0;
            var p = new ProcessSimulator(gain, 0.5, 0.04, 0.05, 100000, 0.0, 3);
            var ctrl = new RunToRunController(gain, target, 0.4, 0.5);
            double acc = 0; int cnt = 0;
            for (int k = 0; k < n; k++)
            {
                double u = ctrl.NextInput();
                double y = p.Run(k, u); ctrl.Update(u, y);
                if (k >= n - 50) { acc += Math.Abs(y - target); cnt++; }
            }
            return acc / cnt;
        }

        private static void AnomalyOnStep(out bool detected, out int falseBefore)
        {
            // 100 in-control points, then a +6-sigma step for 40 points.
            var rng = new DeterministicRandom(99);
            const double sigma = 1.0; const int pre = 100, post = 40;
            var mon = new EwmaAnomalyMonitor(0.0, sigma, 0.2, 3.0);
            detected = false; falseBefore = 0;
            for (int k = 0; k < pre + post; k++)
            {
                double x = rng.NextGaussian(k < pre ? 0.0 : 6.0 * sigma, sigma);
                bool flag = mon.Update(x);
                if (flag && k < pre) falseBefore++;
                if (flag && k >= pre) detected = true;
            }
        }
    }
}
