"""
Cost-Per-Die Optimiser for SiC Blade Dicing.

Total cost per die:
    C_die = (C_blade / (N_life × N_die_per_wafer)
            + C_machine × t_cut_per_die
            + C_yield_loss × P_defect)

    Where:
        C_blade         = blade cost [¥/blade]
        N_life          = blade life [wafers]
        N_die_per_wafer = die count per wafer (depends on die size + kerf)
        C_machine       = machine cost [¥/hour]
        t_cut_per_die   = cutting time per die = kerf_length / (feed × rpm)
        P_defect        = defect probability (chipping > USL)

Optimisation:
    Minimise C_die subject to:
        chipping_um < USL (GP constraint)
        Cpk ≥ 1.33 (process capability constraint)

DISCO relevance:
    - DISCO bills customers per die/wafer → cost optimisation is core business
    - Trade-off: higher feed → faster (cheaper) but more chipping (higher yield loss)
    - This model quantifies the trade-off and finds the sweet spot

Usage:
    python optimization/cost_per_die.py --demo
    python optimization/cost_per_die.py --plot
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

# ── Cost parameters (DISCO-grade defaults) ────────────────────────────────────
COST_DEFAULTS = {
    "blade_cost_jpy":      8000,     # ¥ per blade (DISCO #327 equivalent)
    "blade_life_wafers":   200,      # wafers per blade
    "machine_cost_jpy_hr": 15000,    # ¥/hr (dicing saw fully loaded)
    "die_size_mm":         10.0,     # mm (10×10 mm die)
    "wafer_diam_mm":       200.0,    # mm (8-inch wafer)
    "defect_cost_jpy":     500,      # ¥ per defective die (rework + test)
    "usl_um":              15.0,     # µm chipping USL
    "cpk_min":             1.33,     # minimum acceptable Cpk
}


# ── Geometry helpers ──────────────────────────────────────────────────────────

def dies_per_wafer(die_size_mm: float, wafer_diam_mm: float,
                    blade_W_um: float) -> int:
    """Estimate usable dies per wafer (rectangular grid, 3mm edge exclusion)."""
    kerf_mm = blade_W_um / 1000.0
    usable  = wafer_diam_mm - 6.0  # 3mm exclusion each side
    pitch   = die_size_mm + kerf_mm
    n_per_row = int(usable / pitch)
    # Circular wafer approximation
    total = 0
    for row in range(-n_per_row, n_per_row + 1):
        y = row * pitch
        if abs(y) > usable / 2:
            continue
        half_chord = np.sqrt((usable / 2) ** 2 - y ** 2)
        cols = int(2 * half_chord / pitch)
        total += cols
    return max(1, total)


def cut_time_per_wafer(die_size_mm: float, wafer_diam_mm: float,
                        blade_W_um: float, feed_mm_s: float) -> float:
    """Approximate cutting time [s] per wafer (X + Y passes)."""
    n_dies    = dies_per_wafer(die_size_mm, wafer_diam_mm, blade_W_um)
    n_rows    = int(np.sqrt(n_dies))
    kerf_mm   = blade_W_um / 1000.0
    usable_mm = wafer_diam_mm - 6.0
    # Total kerf length = n_passes × wafer_usable_diameter
    kerf_len_mm = n_rows * 2 * usable_mm  # X + Y
    return kerf_len_mm / feed_mm_s / 1000.0  # s


# ── Cost model ────────────────────────────────────────────────────────────────

class CostPerDieModel:

    def __init__(self, costs=None):
        self.c = dict(COST_DEFAULTS)
        if costs:
            self.c.update(costs)

    def compute(self, recipe: dict, chip_um: float,
                 chip_std: float = 1.5) -> dict:
        """
        recipe: dict with cut_depth_um, blade_W_um, feed_mm_s, spindle_rpm
        chip_um: predicted chipping (GP mean)
        chip_std: GP uncertainty + process noise
        """
        from scipy.stats import norm

        bw   = recipe.get("blade_W_um", 23.0)
        feed = recipe.get("feed_mm_s", 1.0)

        n_die   = dies_per_wafer(self.c["die_size_mm"], self.c["wafer_diam_mm"], bw)
        t_wafer = cut_time_per_wafer(self.c["die_size_mm"], self.c["wafer_diam_mm"],
                                      bw, feed)

        # Cost components
        c_blade = self.c["blade_cost_jpy"] / (self.c["blade_life_wafers"] * n_die)
        c_mach  = self.c["machine_cost_jpy_hr"] / 3600 * t_wafer / n_die
        p_def   = float(norm.sf(self.c["usl_um"], loc=chip_um, scale=chip_std))
        c_yield = self.c["defect_cost_jpy"] * p_def
        c_total = c_blade + c_mach + c_yield

        # Throughput
        takt_s   = t_wafer / n_die
        wph      = 3600.0 / t_wafer  # wafers per hour

        return {
            "n_die_per_wafer":  n_die,
            "cut_time_s":       t_wafer,
            "takt_time_s":      takt_s,
            "wph":              wph,
            "cost_blade_jpy":   c_blade,
            "cost_machine_jpy": c_mach,
            "cost_yield_jpy":   c_yield,
            "cost_total_jpy":   c_total,
            "p_defect":         p_def,
            "chip_um":          chip_um,
        }


# ── Grid optimisation ─────────────────────────────────────────────────────────

def optimise_recipe(model_gp, cost_model, usl=15.0,
                     n_grid=40, cpk_min=1.33):
    """
    Sweep (depth, feed) grid; return Pareto front of cost vs chipping.
    """
    from optimization.process_capability import compute_capability

    depths = np.linspace(80, 380, n_grid)
    feeds  = np.linspace(0.5, 3.5, n_grid)
    D, F   = np.meshgrid(depths, feeds)

    X_grid = np.column_stack([D.ravel(), np.full(D.size, 23.0),
                               F.ravel(), np.full(D.size, 30000.0)])
    mu, sigma = model_gp.predict(X_grid, return_std=True)

    results = []
    for i, (d, f) in enumerate(zip(D.ravel(), F.ravel())):
        chip = float(mu[i])
        sig  = float(sigma[i])

        # Cpk check (one-sided, USL only)
        cpu = (usl - chip) / (3 * max(sig, 0.1))
        if cpu < cpk_min:
            continue

        recipe = {"cut_depth_um": d, "blade_W_um": 23.0,
                  "feed_mm_s": f, "spindle_rpm": 30000.0}
        r = cost_model.compute(recipe, chip, sig)
        r.update({"depth": d, "feed": f, "cpk": cpu})
        results.append(r)

    if not results:
        return []

    # Sort by total cost
    results.sort(key=lambda x: x["cost_total_jpy"])
    return results


def plot_cost(model_gp, cost_model, out_dir=RESULTS):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    depths = np.linspace(80, 380, 50)
    feeds  = np.linspace(0.5, 3.5, 50)
    D, F   = np.meshgrid(depths, feeds)
    X_grid = np.column_stack([D.ravel(), np.full(D.size, 23.0),
                               F.ravel(), np.full(D.size, 30000.0)])
    mu, sigma = model_gp.predict(X_grid, return_std=True)

    cost_grid = np.zeros(D.size)
    for i, (d, f) in enumerate(zip(D.ravel(), F.ravel())):
        r = cost_model.compute(
            {"cut_depth_um": d, "blade_W_um": 23.0,
             "feed_mm_s": f, "spindle_rpm": 30000.0},
            float(mu[i]), float(sigma[i]),
        )
        cost_grid[i] = r["cost_total_jpy"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Chip heatmap
    im0 = axes[0].contourf(D, F, mu.reshape(D.shape), levels=20, cmap="YlOrRd")
    plt.colorbar(im0, ax=axes[0], label="Predicted Chipping [µm]")
    axes[0].contour(D, F, mu.reshape(D.shape), levels=[15.0],
                    colors="white", linewidths=2, linestyles="--")
    axes[0].set_title("GP: Predicted Chipping\n(white dashed = USL 15µm)", fontsize=10)

    # Cost heatmap
    im1 = axes[1].contourf(D, F, cost_grid.reshape(D.shape), levels=20, cmap="viridis")
    plt.colorbar(im1, ax=axes[1], label="Cost per Die [¥]")
    # Safe zone overlay
    safe = mu.reshape(D.shape) < 15.0
    axes[1].contourf(D, F, safe.astype(float), levels=[0.5, 1.5],
                     colors=["none", "white"], alpha=0.1)
    axes[1].contour(D, F, mu.reshape(D.shape), levels=[15.0],
                    colors="white", linewidths=2, linestyles="--")
    axes[1].set_title("Cost per Die [¥]\n(white = safe zone)", fontsize=10)

    # Throughput
    takt = np.array([
        cost_model.compute(
            {"cut_depth_um": d, "blade_W_um": 23.0,
             "feed_mm_s": f, "spindle_rpm": 30000.0},
            float(mu[i]), 1.5)["wph"]
        for i, (d, f) in enumerate(zip(D.ravel(), F.ravel()))
    ])
    im2 = axes[2].contourf(D, F, takt.reshape(D.shape), levels=20, cmap="Blues")
    plt.colorbar(im2, ax=axes[2], label="Throughput [wafers/hour]")
    axes[2].contour(D, F, mu.reshape(D.shape), levels=[15.0],
                    colors="red", linewidths=2, linestyles="--")
    axes[2].set_title("Throughput [WPH]\n(red = USL boundary)", fontsize=10)

    # Find optimal point (min cost in safe zone)
    safe_mask = mu < 15.0
    if safe_mask.any():
        cost_safe = np.where(safe_mask, cost_grid, np.inf)
        best_i = int(np.argmin(cost_safe))
        bd, bf = D.ravel()[best_i], F.ravel()[best_i]
        for ax in axes:
            ax.scatter(bd, bf, color="lime", s=200, marker="*", zorder=10,
                       edgecolors="k", lw=0.8, label=f"Optimal ({bd:.0f}µm,{bf:.1f}mm/s)")
            ax.legend(fontsize=7)

    for ax in axes:
        ax.set_xlabel("Cut Depth [µm]", fontsize=10)
        ax.set_ylabel("Feed Speed [mm/s]", fontsize=10)

    fig.suptitle("Cost-Per-Die Optimisation — SiC Blade Dicing\n"
                 "(blade=¥8000/200W, machine=¥15000/hr, defect=¥500/die, "
                 "8\" wafer, 10×10mm die)",
                 fontsize=10)
    plt.tight_layout()
    out = os.path.join(out_dir, "cost_per_die.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Cost plot → {out}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",  action="store_true")
    parser.add_argument("--plot",  action="store_true")
    args = parser.parse_args()

    try:
        from ml.train_from_experimental import ExperimentalGPSurrogate, MODEL_PATH
        model_gp = ExperimentalGPSurrogate.load(MODEL_PATH)
    except Exception as e:
        print(f"[!] GP model not found: {e}")
        return

    cost_model = CostPerDieModel()

    if args.demo:
        recipes = [
            ({"cut_depth_um": 200, "blade_W_um": 23, "feed_mm_s": 1.0, "spindle_rpm": 30000},
             "Nominal"),
            ({"cut_depth_um": 200, "blade_W_um": 23, "feed_mm_s": 2.5, "spindle_rpm": 30000},
             "High feed"),
            ({"cut_depth_um": 150, "blade_W_um": 23, "feed_mm_s": 0.8, "spindle_rpm": 30000},
             "Conservative"),
        ]
        import numpy as np
        print("\n=== Cost-Per-Die Analysis ===")
        print(f"  {'Recipe':>14}  {'chip':>7}  {'¥/die':>7}  "
              f"{'blade':>7}  {'mach':>7}  {'yield':>7}  {'WPH':>6}  Cpk")
        for recipe, label in recipes:
            x = np.array([[recipe["cut_depth_um"], recipe["blade_W_um"],
                           recipe["feed_mm_s"], recipe["spindle_rpm"]]])
            mu, sig = model_gp.predict(x, return_std=True)
            r = cost_model.compute(recipe, float(mu[0]), float(sig[0]))
            cpu = (15.0 - float(mu[0])) / (3 * max(float(sig[0]), 0.1))
            print(f"  {label:>14}  {r['chip_um']:>6.1f}µm  "
                  f"¥{r['cost_total_jpy']:>5.1f}  "
                  f"¥{r['cost_blade_jpy']:>5.2f}  "
                  f"¥{r['cost_machine_jpy']:>5.2f}  "
                  f"¥{r['cost_yield_jpy']:>5.2f}  "
                  f"{r['wph']:>5.1f}  {cpu:.2f}")

        print("\n[*] Optimised recipes (Cpk≥1.33, min cost):")
        results = optimise_recipe(model_gp, cost_model)
        for r in results[:5]:
            print(f"  depth={r['depth']:.0f}µm  feed={r['feed']:.1f}mm/s  "
                  f"chip={r['chip_um']:.1f}µm  ¥{r['cost_total_jpy']:.1f}/die  "
                  f"Cpk={r['cpk']:.2f}  {r['wph']:.1f}WPH")

    if args.plot:
        plot_cost(model_gp, cost_model)


if __name__ == "__main__":
    main()
