using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Disco.Apc;

namespace Disco.Apc.Demo
{
    /// <summary>
    /// Operator-console demo: runs a drifting equipment process for N lots under
    /// (a) open loop and (b) EWMA Run-to-Run control, injects a step fault, and
    /// monitors residuals with an EWMA anomaly chart. Prints an ASCII dashboard,
    /// writes a results JSON and a self-contained SVG chart.
    /// </summary>
    public static class Program
    {
        public static int Main(string[] args)
        {
            const int lots = 200;
            const double gain = 2.0;      // Δoutput / Δinput
            const double baseOffset = 1.0;
            const double drift = 0.03;    // per-lot upward drift (tool wear)
            const double noise = 0.35;
            const int stepAt = 130;       // fault injection lot
            const double stepMag = 2.2;   // sudden disturbance
            const double target = 10.0;
            const double tol = 1.0;       // spec = target +/- tol
            double usl = target + tol, lsl = target - tol;

            var procClosed = new ProcessSimulator(gain, baseOffset, drift, noise,
                                                  stepAt, stepMag, 12345);
            var procOpen = new ProcessSimulator(gain, baseOffset, drift, noise,
                                                stepAt, stepMag, 12345);
            var ctrl = new RunToRunController(gain, target, 0.3, baseOffset);

            double nominalInput = (target - baseOffset) / gain; // fixed open-loop recipe

            var closed = new double[lots];
            var openLoop = new double[lots];
            var residual = new double[lots];

            for (int k = 0; k < lots; k++)
            {
                double u = ctrl.NextInput();
                double y = procClosed.Run(k, u);
                ctrl.Update(u, y);
                closed[k] = y;
                residual[k] = y - target;
                openLoop[k] = procOpen.Run(k, nominalInput);
            }

            // anomaly monitor: estimate in-control sigma from an early warmup window
            double warmSigma = Std(residual, 5, 40);
            var monitor = new EwmaAnomalyMonitor(0.0, warmSigma, 0.2, 3.0);
            var flags = new List<int>();
            for (int k = 0; k < lots; k++)
                if (monitor.Update(residual[k])) flags.Add(k);

            var mClosed = Metrics(closed, target, usl, lsl);
            var mOpen = Metrics(openLoop, target, usl, lsl);
            int firstFlagAfterStep = -1;
            foreach (int f in flags) if (f >= stepAt) { firstFlagAfterStep = f; break; }
            int falseAlarmsBeforeStep = 0;
            foreach (int f in flags) if (f < stepAt) falseAlarmsBeforeStep++;

            PrintDashboard(lots, target, usl, lsl, stepAt, mClosed, mOpen,
                           flags.Count, firstFlagAfterStep, falseAlarmsBeforeStep);

            string outDir = args.Length > 0 ? args[0] : ".";
            Directory.CreateDirectory(outDir);
            File.WriteAllText(Path.Combine(outDir, "disco_apc_results.json"),
                WriteJson(lots, target, usl, lsl, stepAt, mClosed, mOpen,
                          flags.Count, firstFlagAfterStep, falseAlarmsBeforeStep, warmSigma));
            File.WriteAllText(Path.Combine(outDir, "disco_apc_chart.svg"),
                SvgChart.Render(closed, openLoop, target, usl, lsl, flags.ToArray(), stepAt));

            Console.WriteLine();
            Console.WriteLine("wrote disco_apc_results.json and disco_apc_chart.svg to " + outDir);
            return 0;
        }

        private struct Result { public double Rmse, Cpk, InSpecPct; }

        private static Result Metrics(double[] y, double target, double usl, double lsl)
        {
            int n = y.Length;
            double se = 0, mean = 0;
            for (int i = 0; i < n; i++) { se += (y[i] - target) * (y[i] - target); mean += y[i]; }
            mean /= n;
            double var = 0; int inSpec = 0;
            for (int i = 0; i < n; i++)
            {
                var += (y[i] - mean) * (y[i] - mean);
                if (y[i] <= usl && y[i] >= lsl) inSpec++;
            }
            double sd = Math.Sqrt(var / Math.Max(1, n - 1));
            double cpk = Math.Min(usl - mean, mean - lsl) / (3.0 * Math.Max(sd, 1e-9));
            return new Result
            {
                Rmse = Math.Sqrt(se / n),
                Cpk = cpk,
                InSpecPct = 100.0 * inSpec / n
            };
        }

        private static double Std(double[] a, int from, int to)
        {
            int n = 0; double mean = 0;
            for (int i = from; i < to && i < a.Length; i++) { mean += a[i]; n++; }
            mean /= Math.Max(1, n);
            double v = 0;
            for (int i = from; i < to && i < a.Length; i++) v += (a[i] - mean) * (a[i] - mean);
            return Math.Sqrt(v / Math.Max(1, n - 1));
        }

        private static void PrintDashboard(int lots, double target, double usl, double lsl,
            int stepAt, Result c, Result o, int nFlags, int firstFlag, int falseAlarms)
        {
            string bar = new string('=', 58);
            Console.WriteLine(bar);
            Console.WriteLine("  DISCO APC console  -  Run-to-Run control + anomaly monitor");
            Console.WriteLine(bar);
            Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
                "  lots={0}   target={1:0.0}   spec=[{2:0.0},{3:0.0}]   fault@lot {4}",
                lots, target, lsl, usl, stepAt));
            Console.WriteLine("  " + new string('-', 54));
            Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
                "  {0,-22}{1,12}{2,12}", "metric", "open-loop", "closed-loop"));
            Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
                "  {0,-22}{1,12:0.000}{2,12:0.000}", "RMSE to target", o.Rmse, c.Rmse));
            Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
                "  {0,-22}{1,12:0.00}{2,12:0.00}", "Cpk", o.Cpk, c.Cpk));
            Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
                "  {0,-22}{1,11:0.0}%{2,11:0.0}%", "in-spec yield", o.InSpecPct, c.InSpecPct));
            Console.WriteLine("  " + new string('-', 54));
            Console.WriteLine(string.Format(CultureInfo.InvariantCulture,
                "  anomaly monitor: {0} flag(s); first after fault @lot {1}; " +
                "{2} false alarm(s) pre-fault", nFlags, firstFlag, falseAlarms));
            Console.WriteLine(bar);
        }

        private static string WriteJson(int lots, double target, double usl, double lsl,
            int stepAt, Result c, Result o, int nFlags, int firstFlag, int falseAlarms,
            double warmSigma)
        {
            var s = new StringBuilder();
            Func<string, double, string> kv = (k, v) =>
                "\"" + k + "\": " + v.ToString("0.####", CultureInfo.InvariantCulture);
            s.Append("{\n");
            s.Append("  \"lots\": " + lots + ",\n");
            s.Append("  " + kv("target", target) + ", " + kv("usl", usl) + ", " + kv("lsl", lsl) + ",\n");
            s.Append("  \"fault_at_lot\": " + stepAt + ",\n");
            s.Append("  \"open_loop\":   { " + kv("rmse", o.Rmse) + ", " + kv("cpk", o.Cpk) +
                     ", " + kv("in_spec_pct", o.InSpecPct) + " },\n");
            s.Append("  \"closed_loop\": { " + kv("rmse", c.Rmse) + ", " + kv("cpk", c.Cpk) +
                     ", " + kv("in_spec_pct", c.InSpecPct) + " },\n");
            s.Append("  \"anomaly\": { \"n_flags\": " + nFlags +
                     ", \"first_flag_after_fault\": " + firstFlag +
                     ", \"false_alarms_pre_fault\": " + falseAlarms +
                     ", " + kv("warmup_sigma", warmSigma) + " }\n");
            s.Append("}\n");
            return s.ToString();
        }
    }
}
