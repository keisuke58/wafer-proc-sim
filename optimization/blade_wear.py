"""
Blade Wear Model — Chipping Degradation & Replacement Alerting.

Physics:
    Blade wear follows a power-law degradation (Taylor-type tool life model).
    Chipping increases exponentially with cumulative number of cuts:

        chip(n) = chip_0 × exp(α × n / N_life)

    where:
        chip_0  = initial chipping [µm] (from GP surrogate at t=0)
        α       = wear coefficient (material/process dependent)
        n       = cumulative wafer count on current blade
        N_life  = nominal blade life [wafers]

Alert strategy:
    1. Predictive: extrapolate wear curve → predict when chip will hit USL
    2. Prescriptive: recommend replacement n_margin wafers before predicted USL
    3. Reactive: immediate stop if chip > USL (Shewhart rule)

DISCO relevance:
    - Blade life management is a core cost driver (blade = consumable)
    - Premature replacement wastes cost; late replacement causes yield loss
    - This model enables Just-in-Time blade replacement

Usage:
    python optimization/blade_wear.py --demo
    python optimization/blade_wear.py --n-life 150 --alpha 2.5 --usl 15
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

# DISCO typical parameters (conservative)
DEFAULT_N_LIFE  = 200    # wafers per blade (DISCO recommendation)
DEFAULT_ALPHA   = 2.0    # wear coefficient (gentle → 2.0, aggressive → 4.0)
DEFAULT_USL     = 15.0   # µm
DEFAULT_WARNING = 12.0   # µm — warning before USL
DEFAULT_MARGIN  = 10     # wafers — replace this many wafers before predicted breach


class BladeWearModel:
    """
    Exponential wear model with GP surrogate integration.

    chip(n) = chip_0 × exp(α × n / N_life)

    Supports:
      - Initialisation from GP surrogate prediction
      - Online update: fit α from observations
      - Prediction: time-to-threshold
      - Replacement scheduling
    """

    def __init__(self, chip_0: float, n_life=DEFAULT_N_LIFE,
                 alpha=DEFAULT_ALPHA, usl=DEFAULT_USL,
                 warning=DEFAULT_WARNING, margin=DEFAULT_MARGIN):
        self.chip_0  = chip_0
        self.n_life  = n_life
        self.alpha   = alpha
        self.usl     = usl
        self.warning = warning
        self.margin  = margin

        self.history_n: list    = []  # wafer indices
        self.history_chip: list = []  # observed chipping

    # ── Degradation model ──────────────────────────────────────────────────────

    def predict_chip(self, n: float) -> float:
        """Predicted chipping at wafer n (from blade start)."""
        return self.chip_0 * np.exp(self.alpha * n / self.n_life)

    def predict_array(self, n_arr: np.ndarray) -> np.ndarray:
        return self.chip_0 * np.exp(self.alpha * n_arr / self.n_life)

    def wafers_to_threshold(self, threshold: float) -> float:
        """Wafers remaining until chipping hits threshold."""
        if threshold <= self.chip_0:
            return 0.0
        return self.n_life * np.log(threshold / self.chip_0) / self.alpha

    def replace_at(self) -> int:
        """Recommend blade replacement n_margin wafers before USL breach."""
        n_usl = self.wafers_to_threshold(self.usl)
        return max(0, int(n_usl) - self.margin)

    # ── Online update ─────────────────────────────────────────────────────────

    def observe(self, n: int, chip: float):
        """Record a new chipping observation."""
        self.history_n.append(n)
        self.history_chip.append(chip)
        if len(self.history_n) >= 3:
            self._fit_alpha()

    def _fit_alpha(self):
        """Fit α from observed data using log-linear regression."""
        n_arr   = np.array(self.history_n, dtype=float)
        c_arr   = np.array(self.history_chip, dtype=float)
        valid   = c_arr > 0
        if valid.sum() < 2:
            return
        log_c   = np.log(c_arr[valid] / self.chip_0 + 1e-9)
        t_arr   = n_arr[valid] / self.n_life
        # Least squares: log_c = alpha * t
        self.alpha = float(np.dot(t_arr, log_c) / (np.dot(t_arr, t_arr) + 1e-9))
        self.alpha = max(0.1, min(self.alpha, 10.0))  # clamp

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self, n_current: int) -> dict:
        chip_now  = self.predict_chip(n_current)
        n_replace = self.replace_at()
        n_usl     = self.wafers_to_threshold(self.usl)
        remaining = n_usl - n_current

        if chip_now >= self.usl:
            alert = "🛑 STOP: USL breached — replace blade immediately"
        elif chip_now >= self.warning:
            alert = f"⚠️  WARNING: approaching USL ({chip_now:.1f}/{self.usl:.0f}µm)"
        elif n_current >= n_replace:
            alert = f"🔔 SCHEDULE: recommend replacement within {n_replace-n_current} wafers"
        else:
            alert = "✅ OK"

        return {
            "n_current":     n_current,
            "chip_now":      chip_now,
            "n_to_warning":  max(0, self.wafers_to_threshold(self.warning) - n_current),
            "n_to_usl":      max(0, remaining),
            "replace_at":    n_replace,
            "alpha_fitted":  self.alpha,
            "alert":         alert,
        }


# ── Simulation ─────────────────────────────────────────────────────────────────

def simulate_production(chip_0=4.5, n_life=200, alpha=2.5,
                         usl=DEFAULT_USL, seed=42,
                         n_wafers=250, noise_std=0.8):
    """
    Simulate n_wafers of production with a wearing blade.
    Returns (wafer indices, observed chipping, model, events).
    """
    model  = BladeWearModel(chip_0=chip_0, n_life=n_life, alpha=alpha, usl=usl)
    rng    = np.random.default_rng(seed)
    chips  = []
    events = []

    n_current = 0
    blade_no  = 1

    while n_current < n_wafers:
        true_chip = model.predict_chip(n_current)
        obs_chip  = max(0.1, true_chip + rng.normal(0, noise_std))
        chips.append((n_current, blade_no, obs_chip))
        model.observe(n_current, obs_chip)

        s = model.status(n_current)
        if "STOP" in s["alert"] or "SCHEDULE" in s["alert"]:
            if not events or events[-1][1] != s["alert"]:
                events.append((n_current, s["alert"], blade_no))

        # Auto-replace at recommended wafer count
        if n_current >= model.replace_at():
            blade_no += 1
            model = BladeWearModel(chip_0=chip_0, n_life=n_life,
                                    alpha=alpha, usl=usl)
            n_current = 0  # reset for new blade
        else:
            n_current += 1

    return chips, events, model


def plot_wear(chip_0=4.5, n_life=200, out_dir=RESULTS):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ── Panel 1: Wear curves for different α ──────────────────────────────
    ax = axes[0]
    n_arr = np.linspace(0, n_life, 200)
    colors = ["#2166ac", "#f97316", "#d62728", "#7b2d8b"]
    for alpha, color in zip([1.0, 2.0, 3.0, 4.0], colors):
        m = BladeWearModel(chip_0=chip_0, n_life=n_life, alpha=alpha)
        ax.plot(n_arr, m.predict_array(n_arr), color=color, lw=2,
                label=f"α={alpha:.1f}  replace@{m.replace_at()}W")

    ax.axhline(DEFAULT_USL, color="#d62728", ls="--", lw=1.5, label=f"USL={DEFAULT_USL}µm")
    ax.axhline(DEFAULT_WARNING, color="orange", ls=":", lw=1.2, label=f"Warning={DEFAULT_WARNING}µm")
    ax.axhline(chip_0, color="k", ls=":", lw=0.8, label=f"Initial={chip_0}µm")
    ax.set_xlabel("Wafers on Current Blade", fontsize=11)
    ax.set_ylabel("Front Chipping [µm]", fontsize=11)
    ax.set_title("Blade Wear Curves\n(exponential degradation, varying α)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_ylim(0, DEFAULT_USL * 1.3)

    # ── Panel 2: Simulated production run ─────────────────────────────────
    ax2 = axes[1]
    chips, events, _ = simulate_production(chip_0=chip_0, n_life=n_life,
                                            n_wafers=300)
    wafer_global = np.arange(len(chips))
    chip_vals    = np.array([c[2] for c in chips])
    blade_nos    = np.array([c[1] for c in chips])

    # Colour by blade
    cmap = plt.cm.tab10
    for b in np.unique(blade_nos):
        mask = blade_nos == b
        ax2.plot(wafer_global[mask], chip_vals[mask],
                 color=cmap(b % 10), lw=1.5, alpha=0.85,
                 label=f"Blade #{b}")
        ax2.scatter(wafer_global[mask], chip_vals[mask],
                    color=cmap(b % 10), s=15, zorder=3)

    ax2.axhline(DEFAULT_USL, color="#d62728", ls="--", lw=1.5, label="USL")
    ax2.axhline(DEFAULT_WARNING, color="orange", ls=":", lw=1.2, label="Warning")
    for ev_w, ev_msg, ev_b in events[:5]:
        ax2.axvline(ev_w, color="orange", alpha=0.4, lw=8)

    ax2.set_xlabel("Cumulative Wafer # (production)", fontsize=11)
    ax2.set_ylabel("Front Chipping [µm]", fontsize=11)
    ax2.set_title(f"Production Simulation: 300 Wafers\n"
                  f"({len(np.unique(blade_nos))} blade changes, α={2.5})",
                  fontsize=10)
    ax2.legend(fontsize=7, loc="upper left", ncol=2)
    ax2.grid(alpha=0.25)
    ax2.set_ylim(0, DEFAULT_USL * 1.3)

    fig.suptitle("Blade Wear Model — Chipping Degradation & Replacement Scheduling\n"
                 "(chip(n) = chip₀ × exp(α × n / N_life))", fontsize=10)
    plt.tight_layout()
    out = os.path.join(out_dir, "blade_wear.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Blade wear plot → {out}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-life",  type=int,   default=DEFAULT_N_LIFE)
    parser.add_argument("--alpha",   type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--usl",     type=float, default=DEFAULT_USL)
    parser.add_argument("--chip0",   type=float, default=4.5,
                        help="Initial chipping [µm] (from GP at fresh blade)")
    parser.add_argument("--demo",    action="store_true")
    parser.add_argument("--plot",    action="store_true")
    args = parser.parse_args()

    model = BladeWearModel(chip_0=args.chip0, n_life=args.n_life,
                            alpha=args.alpha, usl=args.usl)

    if args.demo:
        print(f"[*] Blade Wear Demo  (chip_0={args.chip0}µm, N_life={args.n_life}W, α={args.alpha})")
        print(f"    USL={args.usl}µm  Warning={DEFAULT_WARNING}µm\n")
        print(f"  {'Wafer':>6}  {'chip(n)':>9}  {'Alert'}")
        checkpoints = [0, 25, 50, 75, 100, 125, 150, 175, model.replace_at(),
                       int(model.wafers_to_threshold(args.usl))]
        for n in sorted(set(max(0, c) for c in checkpoints)):
            s = model.status(n)
            print(f"  {n:>6}    {s['chip_now']:>6.1f}µm  {s['alert']}")

        print(f"\n  → Recommended replacement: wafer #{model.replace_at()}")
        print(f"  → Predicted USL breach:    wafer #{model.wafers_to_threshold(args.usl):.0f}")

    if args.plot:
        plot_wear(chip_0=args.chip0, n_life=args.n_life)


if __name__ == "__main__":
    main()
