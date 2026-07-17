using System;
using System.Globalization;
using System.Text;

namespace Disco.Apc
{
    /// <summary>
    /// Minimal dependency-free SVG line chart writer. Produces a self-contained
    /// vector figure of the closed-loop vs open-loop process output with spec
    /// limits and flagged anomalies — no plotting library, cross-platform.
    /// </summary>
    public static class SvgChart
    {
        private static string F(double v)
        {
            return v.ToString("0.###", CultureInfo.InvariantCulture);
        }

        public static string Render(double[] closed, double[] openLoop,
                                    double target, double usl, double lsl,
                                    int[] anomalyLots, int stepAtLot)
        {
            int n = closed.Length;
            double w = 900, h = 380, ml = 60, mr = 20, mt = 30, mb = 40;
            double pw = w - ml - mr, ph = h - mt - mb;

            double ymin = lsl, ymax = usl;
            for (int i = 0; i < n; i++)
            {
                ymin = Math.Min(ymin, Math.Min(closed[i], openLoop[i]));
                ymax = Math.Max(ymax, Math.Max(closed[i], openLoop[i]));
            }
            double pad = 0.08 * (ymax - ymin + 1e-9);
            ymin -= pad; ymax += pad;

            Func<int, double> X = k => ml + pw * k / Math.Max(1, n - 1);
            Func<double, double> Y = v => mt + ph * (1.0 - (v - ymin) / (ymax - ymin + 1e-9));

            var sb = new StringBuilder();
            sb.Append("<svg xmlns='http://www.w3.org/2000/svg' width='" + F(w) +
                      "' height='" + F(h) + "' viewBox='0 0 " + F(w) + " " + F(h) +
                      "' font-family='system-ui,Arial' font-size='12'>");
            sb.Append("<rect width='" + F(w) + "' height='" + F(h) + "' fill='white'/>");

            // spec band
            sb.Append("<rect x='" + F(ml) + "' y='" + F(Y(usl)) + "' width='" + F(pw) +
                      "' height='" + F(Y(lsl) - Y(usl)) +
                      "' fill='#2e7d57' opacity='0.08'/>");
            foreach (var pair in new[] { new[] { usl, 0.0 }, new[] { lsl, 0.0 }, new[] { target, 1.0 } })
            {
                double yv = Y(pair[0]);
                string dash = pair[1] > 0.5 ? "" : " stroke-dasharray='4 3'";
                string col = pair[1] > 0.5 ? "#444" : "#2e7d57";
                sb.Append("<line x1='" + F(ml) + "' y1='" + F(yv) + "' x2='" + F(w - mr) +
                          "' y2='" + F(yv) + "' stroke='" + col + "'" + dash + " stroke-width='1'/>");
            }
            sb.Append("<text x='" + F(w - mr) + "' y='" + F(Y(target) - 4) +
                      "' text-anchor='end' fill='#444'>target</text>");
            sb.Append("<text x='" + F(w - mr) + "' y='" + F(Y(usl) - 4) +
                      "' text-anchor='end' fill='#2e7d57'>USL/LSL (spec)</text>");

            // step-disturbance marker
            double xs = X(stepAtLot);
            sb.Append("<line x1='" + F(xs) + "' y1='" + F(mt) + "' x2='" + F(xs) +
                      "' y2='" + F(h - mb) + "' stroke='#c0392b' stroke-dasharray='2 3'/>");
            sb.Append("<text x='" + F(xs + 3) + "' y='" + F(mt + 12) +
                      "' fill='#c0392b'>fault injected</text>");

            sb.Append(Poly(openLoop, X, Y, n, "#c8102e", 1.4, 0.6));
            sb.Append(Poly(closed, X, Y, n, "#0e6ba8", 1.8, 1.0));

            // anomaly flags on closed loop
            foreach (int lot in anomalyLots)
            {
                if (lot < 0 || lot >= n) continue;
                sb.Append("<circle cx='" + F(X(lot)) + "' cy='" + F(Y(closed[lot])) +
                          "' r='3' fill='none' stroke='#e08a00' stroke-width='1.6'/>");
            }

            // legend
            sb.Append("<rect x='" + F(ml + 8) + "' y='" + F(mt + 6) +
                      "' width='210' height='54' fill='white' opacity='0.85' stroke='#ddd'/>");
            sb.Append(LegLine(ml + 16, mt + 22, "#0e6ba8", "closed-loop (Run-to-Run)"));
            sb.Append(LegLine(ml + 16, mt + 38, "#c8102e", "open-loop (no control)"));
            sb.Append("<circle cx='" + F(ml + 24) + "' cy='" + F(mt + 52) +
                      "' r='3' fill='none' stroke='#e08a00'/>");
            sb.Append("<text x='" + F(ml + 36) + "' y='" + F(mt + 56) +
                      "'>anomaly flagged</text>");

            sb.Append("<text x='" + F(ml) + "' y='" + F(h - 12) +
                      "'>lot index</text>");
            sb.Append("<text x='14' y='" + F(mt + 10) + "' fill='#333'>output</text>");
            sb.Append("</svg>");
            return sb.ToString();
        }

        private static string Poly(double[] d, Func<int, double> X, Func<double, double> Y,
                                   int n, string color, double sw, double op)
        {
            var sb = new StringBuilder();
            sb.Append("<polyline fill='none' stroke='" + color + "' stroke-width='" +
                      F(sw) + "' opacity='" + F(op) + "' points='");
            for (int i = 0; i < n; i++)
                sb.Append(F(X(i)) + "," + F(Y(d[i])) + " ");
            sb.Append("'/>");
            return sb.ToString();
        }

        private static string LegLine(double x, double y, string color, string label)
        {
            return "<line x1='" + F(x) + "' y1='" + F(y - 4) + "' x2='" + F(x + 16) +
                   "' y2='" + F(y - 4) + "' stroke='" + color + "' stroke-width='2'/>" +
                   "<text x='" + F(x + 22) + "' y='" + F(y) + "'>" + label + "</text>";
        }
    }
}
