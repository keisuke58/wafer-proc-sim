"""
Process Capability Analysis — Cpk / Cp / Cpm for SiC Dicing.

Standard QC metrics used on actual production lines (DISCO included).

Metrics:
    Cp  = (USL - LSL) / (6σ)              — inherent capability
    Cpk = min(Cpu, Cpl)                    — actual (centred) capability
         Cpu = (USL - μ) / (3σ)
         Cpl = (μ - LSL) / (3σ)
    Cpm = Cp / sqrt(1 + ((μ-T)/σ)²)       — Taguchi capability (vs target T)

Industry benchmarks:
    Cpk < 1.00  → process out of control (immediate action)
    1.00 ≤ Cpk < 1.33 → marginal
    1.33 ≤ Cpk < 1.67 → capable (automotive standard)
    Cpk ≥ 1.67 → highly capable (semiconductor standard)

Usage:
    python optimization/process_capability.py --demo
    python optimization/process_capability.py --csv my_chipping_log.csv \\
        --usl 15 --lsl 0 --target 8
"""

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

# DISCO production defaults
DEFAULT_USL    = 15.0   # µm  upper spec limit (chipping)
DEFAULT_LSL    = 0.0    # µm  lower spec limit
DEFAULT_TARGET = 5.0    # µm  nominal target
SIGMA_MULT     = 3.0    # ±3σ for Cp/Cpk


@dataclass
class CapabilityResult:
    n:       int
    mean:    float
    std:     float
    usl:     float
    lsl:     float
    target:  float
    cp:      float
    cpk:     float
    cpm:     float
    cpu:     float
    cpl:     float
    ppm_estimated: float   # defects per million (assuming normal)
    status:  str           # OK / MARGINAL / OUT_OF_CONTROL

    def __str__(self):
        lines = [
            f"  n={self.n}  μ={self.mean:.2f}µm  σ={self.std:.2f}µm",
            f"  Cp={self.cp:.3f}  Cpk={self.cpk:.3f}  Cpm={self.cpm:.3f}",
            f"  Cpu={self.cpu:.3f}  Cpl={self.cpl:.3f}",
            f"  Est. defect rate: {self.ppm_estimated:.0f} ppm",
            f"  Status: {self.status}",
        ]
        return "\n".join(lines)


def compute_capability(data: np.ndarray,
                        usl=DEFAULT_USL,
                        lsl=DEFAULT_LSL,
                        target=DEFAULT_TARGET) -> CapabilityResult:
    n   = len(data)
    mu  = float(np.mean(data))
    # Unbiased std (sample)
    std = float(np.std(data, ddof=1)) if n > 1 else 1e-9

    span = usl - lsl
    cp  = span / (6 * std)
    cpu = (usl - mu) / (3 * std)
    # For chipping: LSL=0 is a physical lower bound, not a spec limit.
    # Use one-sided Cpk (Cpu only) when LSL=0 to avoid LSL dominating.
    if lsl <= 0:
        cpl = cpu   # effectively one-sided — Cpk = Cpu
    else:
        cpl = (mu - lsl) / (3 * std)
    cpk = min(cpu, cpl)

    # Taguchi Cpm
    tau = std**2 + (mu - target)**2
    cpm = (usl - lsl) / (6 * np.sqrt(tau))

    # Estimated PPM (assuming normality)
    from scipy.stats import norm
    p_above = norm.sf(usl, loc=mu, scale=std)
    p_below = norm.cdf(lsl, loc=mu, scale=std)
    ppm = (p_above + p_below) * 1e6

    if cpk >= 1.67:
        status = "✅ Highly capable (semiconductor grade)"
    elif cpk >= 1.33:
        status = "✅ Capable (automotive grade)"
    elif cpk >= 1.00:
        status = "⚠️ Marginal — process improvement recommended"
    else:
        status = "❌ Out of control — immediate action required"

    return CapabilityResult(n=n, mean=mu, std=std, usl=usl, lsl=lsl,
                             target=target, cp=cp, cpk=cpk, cpm=cpm,
                             cpu=cpu, cpl=cpl, ppm_estimated=ppm, status=status)


# ── Control chart ─────────────────────────────────────────────────────────────

def plot_capability(data: np.ndarray, result: CapabilityResult,
                    title="SiC Blade Dicing — Chipping Control Chart",
                    out_dir=RESULTS) -> str:
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ── Left: Individual chart ─────────────────────────────────────────────
    ax = axes[0]
    n = len(data)
    ax.plot(data, "o-", color="#2166ac", lw=1.2, ms=5, label="Chipping [µm]")
    ax.axhline(result.mean, color="k", lw=1.5, label=f"μ={result.mean:.2f}")
    ax.axhline(result.usl, color="#d62728", ls="--", lw=1.5,
               label=f"USL={result.usl:.0f}")
    if result.lsl > 0:
        ax.axhline(result.lsl, color="#d62728", ls="--", lw=1.5,
                   label=f"LSL={result.lsl:.0f}")
    ax.axhline(result.target, color="#2ca02c", ls=":", lw=1.2,
               label=f"Target={result.target:.0f}")
    # ±3σ control limits
    ucl = result.mean + 3 * result.std
    lcl = max(result.lsl, result.mean - 3 * result.std)
    ax.axhline(ucl, color="orange", ls="-.", lw=1.0, label=f"UCL={ucl:.2f}")
    ax.axhline(lcl, color="orange", ls="-.", lw=1.0, label=f"LCL={lcl:.2f}")

    # Highlight violations
    viol = np.where(data > result.usl)[0]
    if len(viol):
        ax.scatter(viol, data[viol], color="red", s=100, zorder=5, marker="x",
                   lw=2, label=f"USL breach ({len(viol)})")

    ax.set_xlabel("Wafer / Lot #", fontsize=11)
    ax.set_ylabel("Front Chipping [µm]", fontsize=11)
    ax.set_title("Individual Control Chart (I-Chart)", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_ylim(bottom=0)

    # ── Right: Capability histogram ────────────────────────────────────────
    ax2 = axes[1]
    ax2.hist(data, bins=min(20, n//3+2), density=True,
             color="#2166ac", alpha=0.6, edgecolor="k", lw=0.5, label="Data")

    x_range = np.linspace(min(data.min(), result.lsl) - result.std,
                          max(data.max(), result.usl) + result.std, 200)
    ax2.plot(x_range, norm.pdf(x_range, result.mean, result.std),
             color="#1a1a2e", lw=2, label="Normal fit")

    ax2.axvline(result.usl, color="#d62728", ls="--", lw=2,
                label=f"USL={result.usl:.0f}")
    ax2.axvline(result.target, color="#2ca02c", ls=":", lw=1.5,
                label=f"Target={result.target:.0f}")
    ax2.axvline(result.mean, color="k", lw=1.5, label=f"μ={result.mean:.2f}")

    # Shade out-of-spec
    x_oos = x_range[x_range > result.usl]
    if len(x_oos):
        ax2.fill_between(x_oos, norm.pdf(x_oos, result.mean, result.std),
                         alpha=0.35, color="#d62728", label="Out-of-spec")

    ax2.set_xlabel("Front Chipping [µm]", fontsize=11)
    ax2.set_ylabel("Probability Density", fontsize=11)
    ax2.set_title(
        f"Capability Histogram\n"
        f"Cp={result.cp:.2f}  Cpk={result.cpk:.2f}  Cpm={result.cpm:.2f}  "
        f"PPM≈{result.ppm_estimated:.0f}",
        fontsize=10,
    )
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()
    out = os.path.join(out_dir, "process_capability.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Capability chart → {out}")
    return out


# ── Simulation: GP-based capability prediction ────────────────────────────────

def simulate_production_cpk(model, recipe, n_wafers=100, noise_std=1.0,
                              usl=DEFAULT_USL, target=DEFAULT_TARGET):
    """
    Simulate n_wafers of production chipping, compute Cpk.
    model: ExperimentalGPSurrogate
    recipe: [cut_depth_um, blade_W_um, feed_mm_s, spindle_rpm]
    """
    import numpy as np
    x = np.array([recipe])
    mu_gp, sig_gp = model.predict(x, return_std=True)
    mu_gp, sig_gp = float(mu_gp[0]), float(sig_gp[0])

    rng = np.random.default_rng(42)
    total_std = np.sqrt(sig_gp**2 + noise_std**2)
    data = rng.normal(mu_gp, total_std, n_wafers)
    data = np.maximum(data, 0.0)

    return compute_capability(data, usl=usl, lsl=0.0, target=target)


# ── Demo ──────────────────────────────────────────────────────────────────────

def run_demo():
    """Simulate 3 recipes: nominal / over-aggressive / optimal"""
    try:
        from ml.train_from_experimental import ExperimentalGPSurrogate, MODEL_PATH
        model = ExperimentalGPSurrogate.load(MODEL_PATH)
        has_model = True
    except Exception:
        has_model = False

    recipes = [
        ([200, 23, 1.0, 30000], "Nominal (deep=200, feed=1.0)"),
        ([350, 23, 3.0, 30000], "Aggressive (deep=350, feed=3.0)"),
        ([150, 23, 0.8, 30000], "Conservative (deep=150, feed=0.8)"),
    ]

    print("\n=== Process Capability Demo ===")
    print(f"  USL={DEFAULT_USL}µm  Target={DEFAULT_TARGET}µm  "
          f"Semiconductor standard: Cpk≥1.67\n")

    all_data, labels = [], []
    for recipe, label in recipes:
        if has_model:
            result = simulate_production_cpk(model, recipe, n_wafers=200)
        else:
            rng = np.random.default_rng(sum(recipe))
            mean = 3.0 + recipe[0]/100 + recipe[2]*0.5
            data = rng.normal(mean, 1.5, 200)
            result = compute_capability(np.maximum(data, 0), usl=DEFAULT_USL, target=DEFAULT_TARGET)
        print(f"  Recipe: {label}")
        print(result)
        print()
        all_data.append(result.mean)
        labels.append(label)

    return all_data


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    help="CSV with chipping_um column")
    parser.add_argument("--usl",    type=float, default=DEFAULT_USL)
    parser.add_argument("--lsl",    type=float, default=DEFAULT_LSL)
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET)
    parser.add_argument("--demo",   action="store_true")
    parser.add_argument("--plot",   action="store_true")
    args = parser.parse_args()

    if args.demo:
        run_demo()

        # Generate demo plot with simulated data
        try:
            from ml.train_from_experimental import ExperimentalGPSurrogate, MODEL_PATH
            model = ExperimentalGPSurrogate.load(MODEL_PATH)
            rng = np.random.default_rng(0)

            # Simulate 50-wafer production run with drift
            recipe = [200., 23., 1.0, 30000.]
            x = np.array([recipe])
            mu_gp = float(model.predict(x)[0])
            # Add blade wear drift: chip increases over wafers
            data = np.array([
                max(0, rng.normal(mu_gp + i * 0.08, 1.5))
                for i in range(60)
            ])
            result = compute_capability(data, usl=args.usl, lsl=args.lsl,
                                         target=args.target)
            if args.plot:
                plot_capability(data, result,
                                title="SiC Blade Dicing — 60 Wafer Production Run\n"
                                      "(Nominal recipe with blade wear drift)")
        except Exception as e:
            print(f"[Warn] GP-based demo failed: {e}")
            rng = np.random.default_rng(0)
            data = np.maximum(rng.normal(6.0, 1.8, 60) + np.linspace(0, 3, 60), 0)
            result = compute_capability(data, usl=args.usl, target=args.target)
            if args.plot:
                plot_capability(data, result)
        return

    if args.csv:
        df = pd.read_csv(args.csv)
        col = "chipping_um" if "chipping_um" in df.columns else df.columns[0]
        data = df[col].dropna().values.astype(float)
        result = compute_capability(data, usl=args.usl, lsl=args.lsl,
                                     target=args.target)
        print(f"\n=== Capability Analysis: {col} ===")
        print(result)
        if args.plot:
            plot_capability(data, result)


if __name__ == "__main__":
    main()
