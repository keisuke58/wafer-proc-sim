"""
Sony Semiconductor Solutions — BSI CMOS Image Sensor Thin-Wafer Dicing Simulation.

Key differences from 4H-SiC dicing:
  - Material  : Si (Young's modulus 130 GPa, isotropic for dicing purposes)
  - Thickness : 30–100 µm (TAIKO / back-side thinning, BSI process)
  - Failure   : chipping AND pixel-array delamination at SiO2/Cu-TSV interface
  - Blade     : fine-grit resin-bond, 15–50 µm kerf width
  - Metric    : chipping ≤ 3 µm (tighter than SiC) + delamination risk < 5%

Physics models
--------------
1. Warpage (Stoney curvature):
     κ = (6 E_f h_f ε_f) / (E_s h_s²)   [1/m]
     w_max = κ L² / 8                     [µm]
   where E_s=130 GPa, E_f=120 GPa (SiO2 stack), h_f=5µm film, h_s=wafer thickness

2. Chipping oracle:
     chip = C × depth^0.55 × feed^0.35 / (spindle^0.25 × (h_s/100)^0.4)
   Thinner wafer → less rigidity → more chipping at same recipe.

3. Delamination risk:
     G_I = 0.02 × depth × feed² / h_s   [J/m²]
     risk_pct = Φ((G_I - G_c) / σ_G)   Gaussian CDF, G_c=2.0 J/m²

4. Die strength (Weibull):
     F(σ) = 1 - exp(-(σ/σ_0)^m)
   Characteristic strength σ_0 decreases with wafer thinness.

Usage:
    python ml/sony_cmos_thin_wafer.py          # run all analyses + save figures
    python ml/sony_cmos_thin_wafer.py --quick  # skip GP fitting (fast demo)
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "figure.dpi": 300,
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})
import matplotlib.pyplot as plt
from scipy.special import ndtr   # Gaussian CDF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ── Material constants ────────────────────────────────────────────────────────

E_S   = 130e3   # Si Young's modulus [MPa]
E_F   = 120e3   # SiO2/metal stack effective modulus [MPa]
H_F   = 5.0     # film stack thickness [µm]
EPS_F = 2e-4    # biaxial residual strain in BSI metal stack
G_C   = 2.0     # critical energy release rate [J/m²]
SIGMA_G = 0.3   # scatter in G_I estimate

# ── Physics oracles ───────────────────────────────────────────────────────────

def warpage_um(thickness_um: float, die_length_mm: float = 15.0) -> float:
    """Stoney warpage at wafer centre [µm]."""
    h_s = thickness_um * 1e-6   # m
    h_f = H_F * 1e-6
    kappa = (6 * E_F * 1e6 * h_f * EPS_F) / (E_S * 1e6 * h_s**2)   # 1/m
    L = die_length_mm * 1e-3
    return kappa * L**2 / 8 * 1e6   # µm


def chipping_oracle(depth, feed, spindle_krpm, thickness_um):
    """
    Front chipping [µm] for thin Si wafer.
    depth [µm], feed [mm/s], spindle [krpm], thickness [µm].
    """
    t_norm = np.clip(thickness_um / 100.0, 0.2, 1.5)
    chip = (
        0.6 * depth**0.55 * feed**0.35
        / (spindle_krpm**0.25 * t_norm**0.4)
        + np.random.normal(0, 0.15, size=np.shape(depth))
    )
    return np.clip(chip, 0.05, 20.0)


def delamination_risk(depth, feed, thickness_um):
    """Delamination probability [0, 1] at BSI Cu-TSV interface."""
    # G_I scales with MRR / thickness: higher feed + deeper cut → more interfacial loading
    G_I = 0.12 * depth**0.6 * feed**0.8 / (thickness_um / 50.0)**0.5
    return ndtr((G_I - G_C) / SIGMA_G)


# ── Dataset ───────────────────────────────────────────────────────────────────

THICKNESSES = [30, 50, 75, 100]   # µm — representative CMOS thinning targets

def make_dataset(N: int = 400):
    rng = np.random.default_rng(0)
    depth   = rng.uniform(10, 80,  N)
    feed    = rng.uniform(0.5, 5,  N)
    spindle = rng.uniform(20, 50,  N)   # krpm
    thick   = rng.choice(THICKNESSES, N).astype(float)
    chip    = chipping_oracle(depth, feed, spindle, thick)
    delam   = delamination_risk(depth, feed, thick)
    X = np.column_stack([depth, feed, spindle, thick])
    return X, chip, delam


# ── GP surrogate ──────────────────────────────────────────────────────────────

def fit_gp(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kernel = ConstantKernel(1.0, (0.1, 100)) * RBF(
        length_scale=[1.0]*4,
        length_scale_bounds=[(0.1, 10)]*4
    ) + WhiteKernel(0.1, (0.01, 5))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.0)
    gp.fit(Xs, y)
    return gp, scaler


def gp_predict(gp, scaler, X):
    return gp.predict(scaler.transform(X), return_std=True)


# ── Weibull analysis ──────────────────────────────────────────────────────────

def weibull_params(thickness_um: float):
    """
    Weibull scale σ_0 and shape m for die bending strength [MPa].
    σ_0 decreases ~15% per 50µm thinning (literature values for Si).
    """
    sigma_0_ref = 420.0   # MPa at 100µm
    m = 8.0
    sigma_0 = sigma_0_ref * (thickness_um / 100.0)**0.3
    return sigma_0, m


def weibull_cdf(sigma, sigma_0, m):
    return 1.0 - np.exp(-((sigma / sigma_0)**m))


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_all(gp, scaler, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 12))
    gs  = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.38)

    # ── Panel 1: Warpage vs. thickness ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    thicknesses = np.linspace(25, 120, 200)
    warp = [warpage_um(t) for t in thicknesses]
    ax1.plot(thicknesses, warp, color="#003087", lw=2)
    ax1.axhline(10, color="red", ls="--", lw=1.2, label="10 µm limit")
    ax1.axvline(50, color="orange", ls=":", lw=1.2, label="BSI target 50 µm")
    ax1.set_xlabel("Wafer Thickness [µm]")
    ax1.set_ylabel("Max Warpage [µm]")
    ax1.set_title("(a) TAIKO Warpage vs. Thickness")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.25)

    # ── Panel 2: Chipping vs. feed, by thickness ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    feeds = np.linspace(0.5, 5, 100)
    colors = ["#003087", "#0055cc", "#66aaff", "#aaccff"]
    for t, c in zip(THICKNESSES, colors):
        X_sw = np.column_stack([
            np.full(100, 40.0),  # depth=40µm
            feeds,
            np.full(100, 30.0),  # spindle=30krpm
            np.full(100, float(t)),
        ])
        mu, _ = gp_predict(gp, scaler, X_sw)
        ax2.plot(feeds, mu, color=c, lw=2, label=f"{t} µm")
    ax2.axhline(3.0, color="red", ls="--", lw=1.2, label="3 µm spec")
    ax2.set_xlabel("Feed Speed [mm/s]")
    ax2.set_ylabel("Front Chipping [µm]")
    ax2.set_title("(b) Chipping vs. Feed (depth=40 µm)")
    ax2.legend(title="Thickness", fontsize=10)
    ax2.grid(alpha=0.25)

    # ── Panel 3: Delamination risk heatmap ───────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    depths = np.linspace(10, 80, 60)
    feeds2 = np.linspace(0.5, 5, 60)
    D, F = np.meshgrid(depths, feeds2)
    risk = delamination_risk(D, F, thickness_um=50.0)
    im = ax3.contourf(D, F, risk * 100, levels=20, cmap="Reds")
    plt.colorbar(im, ax=ax3, label="Risk [%]")
    ax3.contour(D, F, risk * 100, levels=[5.0],
                colors="white", linewidths=1.5, linestyles="--")
    ax3.set_xlabel("Cut Depth [µm]")
    ax3.set_ylabel("Feed Speed [mm/s]")
    ax3.set_title("(c) Delamination Risk (h=50 µm)\nWhite dashed: 5% limit")
    ax3.grid(alpha=0.15)

    # ── Panel 4: GP response surface (depth vs. feed, 50µm) ──────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    X_grid = np.column_stack([
        D.ravel(), F.ravel(),
        np.full(D.size, 30.0),
        np.full(D.size, 50.0),
    ])
    mu_g, _ = gp_predict(gp, scaler, X_grid)
    im4 = ax4.contourf(D, F, mu_g.reshape(D.shape), levels=20, cmap="YlOrRd")
    plt.colorbar(im4, ax=ax4, label="Chipping [µm]")
    ax4.contour(D, F, mu_g.reshape(D.shape), levels=[3.0],
                colors="white", linewidths=1.5, linestyles="--")
    ax4.set_xlabel("Cut Depth [µm]")
    ax4.set_ylabel("Feed Speed [mm/s]")
    ax4.set_title("(d) GP Mean Chipping (h=50 µm)\nWhite: 3 µm spec limit")
    ax4.grid(alpha=0.15)

    # ── Panel 5: Weibull die strength by thickness ────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    sigma_vals = np.linspace(100, 700, 300)
    for t, c in zip(THICKNESSES, colors):
        s0, m = weibull_params(t)
        ax5.plot(sigma_vals, weibull_cdf(sigma_vals, s0, m) * 100,
                 color=c, lw=2, label=f"{t} µm (σ₀={s0:.0f})")
    ax5.axhline(10, color="red", ls="--", lw=1.2, label="10% failure")
    ax5.set_xlabel("Bending Stress [MPa]")
    ax5.set_ylabel("Cumulative Failure [%]")
    ax5.set_title("(e) Weibull Die Strength vs. Thickness")
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.25)

    # ── Panel 6: Safe operating window summary ────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    summary_thick = np.array(THICKNESSES, dtype=float)
    max_feed_safe = []
    for t in summary_thick:
        # Max feed such that chipping oracle < 3 µm at depth=40, spindle=30
        feeds_t = np.linspace(0.5, 5, 200)
        X_t = np.column_stack([
            np.full(200, 40.0), feeds_t,
            np.full(200, 30.0), np.full(200, t)
        ])
        mu_t, _ = gp_predict(gp, scaler, X_t)
        idx = np.where(mu_t < 3.0)[0]
        max_feed_safe.append(feeds_t[idx[-1]] if len(idx) else 0.0)

    ax6.bar(summary_thick, max_feed_safe, width=12,
            color=colors, edgecolor="k", linewidth=0.8)
    ax6.set_xlabel("Wafer Thickness [µm]")
    ax6.set_ylabel("Max Safe Feed [mm/s]\n(chipping < 3 µm)")
    ax6.set_title("(f) Safe Feed Limit vs. Thickness")
    ax6.grid(alpha=0.25, axis="y")
    for x, y in zip(summary_thick, max_feed_safe):
        ax6.text(x, y + 0.05, f"{y:.1f}", ha="center", fontsize=11, fontweight="bold")

    fig.suptitle(
        "Sony BSI CMOS Thin-Wafer Dicing Simulation\n"
        "4H-Si (isotropic), 30–100 µm, BSI/Cu-TSV stack",
        fontsize=14, fontweight="bold",
    )

    out_path = os.path.join(out_dir, "sony_thin_wafer_analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[✓] Figure → {out_path}")
    plt.close()
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Skip GP fitting (load oracle only)")
    args = parser.parse_args()

    print("[Sony BSI CMOS Thin-Wafer Dicing Simulation]")
    print(f"  Material : Si  E_s={E_S/1e3:.0f} GPa (isotropic)")
    print(f"  Film     : SiO2/Cu stack  h_f={H_F} µm  ε={EPS_F}")
    print(f"  G_c      : {G_C} J/m² (SiO2/Cu interface)")

    print("\n[1] Warpage by thickness:")
    for t in THICKNESSES:
        print(f"  h={t:3d} µm → warpage = {warpage_um(t):.1f} µm")

    print("\n[2] Generating dataset …")
    X, chip, delam = make_dataset(600)
    print(f"  Chipping : {chip.min():.2f} – {chip.max():.2f} µm")
    print(f"  Delam    : {delam.min():.3f} – {delam.max():.3f}")

    if not args.quick:
        print("\n[3] Fitting GP surrogate …")
        gp, scaler = fit_gp(X, chip)
        print(f"  kernel: {gp.kernel_}")
    else:
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        gp, scaler = fit_gp(X[:100], chip[:100])

    print("\n[4] Weibull die strength:")
    for t in THICKNESSES:
        s0, m = weibull_params(t)
        p10 = s0 * (-np.log(0.9))**(1/m)
        print(f"  h={t:3d} µm  σ_0={s0:.0f} MPa  m={m}  P10%={p10:.0f} MPa")

    print("\n[5] Plotting …")
    plot_all(gp, scaler)
    print("\nDone.")


if __name__ == "__main__":
    main()
