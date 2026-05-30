"""
GP surrogate + TMCMC calibration on DiSECt real cutting force data.

DiSECt (NVIDIA, Huang et al. 2022): LS-DYNA knife cutting force measurements
for cylinder/prism/sphere soft-material objects at 35/45/55/65 mm/s.

Demonstrates the wafer-proc-sim pipeline on REAL cutting data:
    Input : (geometry_id, velocity_mm_s)
    Output: peak contact force [N]
    TMCMC : given observed force → infer velocity posterior

Usage:
    python ml/disect_surrogate_demo.py --plot
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FORCE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "external",
    "DiSECt_cutting_dataset", "forces",
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

GEOM_MAP = {"cylinder": 0, "prism": 1, "sphere": 2}
VELOCITY_BOUNDS = np.array([[30.0, 70.0]])   # mm/s range for TMCMC prior


# ── Data loading ──────────────────────────────────────────────────────────────

def load_disect_forces() -> pd.DataFrame:
    """Parse all velocity-sweep CSVs; return tidy dataframe."""
    records = []
    for path in sorted(glob.glob(os.path.join(FORCE_DIR, "*.csv"))):
        fname = os.path.basename(path).replace(".csv", "")
        parts = fname.split("_")
        geom = parts[0]
        if geom not in GEOM_MAP:
            continue
        vel = None
        for p in parts:
            if p.startswith("vel"):
                vel = int(p[3:])
        if vel is None:
            if "props" in fname:
                continue          # cross-material property files
            vel = 35              # default velocity for base files

        try:
            df = pd.read_csv(path, skiprows=2, header=None)
            Fx, Fy, Fz = df[1].values, df[3].values, df[5].values
            F = np.sqrt(Fx**2 + Fy**2 + Fz**2)
            nonzero = F[F > 0.01]
            if len(nonzero) == 0:
                continue
            records.append({
                "geometry":       geom,
                "geometry_id":    GEOM_MAP[geom],
                "velocity_mm_s":  float(vel),
                "peak_force_N":   float(nonzero.max()),
                "mean_force_N":   float(nonzero.mean()),
            })
        except Exception:
            pass

    return pd.DataFrame(records)


# ── GP surrogate ──────────────────────────────────────────────────────────────

def build_surrogate(df: pd.DataFrame):
    """Train a 2-feature GP: [geometry_id, velocity_mm_s] → peak_force_N."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.preprocessing import StandardScaler

    X = df[["geometry_id", "velocity_mm_s"]].values.astype(float)
    y = df["peak_force_N"].values.astype(float)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    Xs = scaler_x.fit_transform(X)
    ys = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    kernel = (
        ConstantKernel(1.0, (0.1, 10.0))
        * RBF(length_scale=[1.0, 10.0],
              length_scale_bounds=[(0.1, 10), (1, 50)])
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(0.01, 2.0))
    )
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                  normalize_y=False, alpha=0.0)
    gp.fit(Xs, ys)

    def predict(X_new, return_std=False):
        Xs_new = scaler_x.transform(np.atleast_2d(X_new))
        mu, sigma = gp.predict(Xs_new, return_std=True)
        mu    = scaler_y.inverse_transform(mu.reshape(-1, 1)).ravel()
        sigma = sigma * float(scaler_y.scale_[0])
        return (mu, sigma) if return_std else mu

    print(f"  kernel: {gp.kernel_}")
    return predict, scaler_x, scaler_y


# ── TMCMC calibration ─────────────────────────────────────────────────────────

def calibrate_velocity(predict_fn, observed_force_N: float,
                       geometry: str = "cylinder",
                       sigma_obs: float = 3.0,
                       n_samples: int = 800) -> dict:
    """
    TMCMC: given observed peak force, infer cutting velocity posterior.
    Geometry is fixed; only velocity is inferred.
    """
    from scipy.stats import norm as sp_norm
    from optimization.tmcmc_dicing import tmcmc

    geom_id = float(GEOM_MAP[geometry])
    bounds  = VELOCITY_BOUNDS     # [[30, 70]]

    def log_likelihood(theta):
        vel = float(theta[0])
        X   = np.array([[geom_id, vel]])
        mu, std = predict_fn(X, return_std=True)
        total_std = float(np.sqrt(std[0]**2 + sigma_obs**2))
        return float(sp_norm.logpdf(observed_force_N, loc=mu[0], scale=total_std))

    def log_prior(theta):
        if bounds[0, 0] <= theta[0] <= bounds[0, 1]:
            return 0.0
        return -np.inf

    result  = tmcmc(log_likelihood, log_prior,
                    n_samples=n_samples, param_bounds=bounds)
    samples = result["samples"].ravel()   # (n,1) → (n,)
    weights = result["weights"]
    mean_v  = float(np.average(samples, weights=weights))
    std_v   = float(np.sqrt(np.average((samples - mean_v)**2, weights=weights)))
    map_v   = float(samples[np.argmax(weights)])

    print(f"  geometry   : {geometry}")
    print(f"  observed F : {observed_force_N:.1f} N")
    print(f"  posterior  : velocity = {mean_v:.1f} ± {std_v:.1f} mm/s")
    print(f"  MAP        : {map_v:.1f} mm/s")

    return {"geometry": geometry, "observed_N": observed_force_N,
            "mean_vel": mean_v, "std_vel": std_v, "map_vel": map_v,
            "samples": samples, "weights": weights}


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_surrogate(predict_fn, df: pd.DataFrame, out_dir: str = RESULTS_DIR):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    v_grid = np.linspace(28, 72, 100)
    colors  = {"cylinder": "#2166ac", "prism": "#d6604d", "sphere": "#4dac26"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: GP mean ± 2σ per geometry
    ax = axes[0]
    for geom, gid in GEOM_MAP.items():
        X_sw = np.column_stack([np.full(len(v_grid), gid), v_grid])
        mu, sigma = predict_fn(X_sw, return_std=True)
        c = colors[geom]
        ax.plot(v_grid, mu, color=c, lw=2, label=geom.capitalize())
        ax.fill_between(v_grid, mu - 2*sigma, mu + 2*sigma, alpha=0.15, color=c)

        # Training points for this geometry
        sub = df[df["geometry"] == geom]
        ax.scatter(sub["velocity_mm_s"], sub["peak_force_N"],
                   color=c, s=80, zorder=5, edgecolors="k", lw=0.6)

    ax.set_xlabel("Cutting Velocity [mm/s]", fontsize=11)
    ax.set_ylabel("Peak Contact Force [N]", fontsize=11)
    ax.set_title("GP Surrogate — DiSECt Cutting Force\n(Real LS-DYNA data, NVIDIA)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)

    # Right: 2D heatmap (velocity × geometry)
    ax = axes[1]
    geom_ids = np.array([0, 1, 2])
    V, G = np.meshgrid(v_grid, geom_ids)
    X_grid = np.column_stack([G.ravel(), V.ravel()])
    mu_grid, _ = predict_fn(X_grid, return_std=True)
    im = ax.contourf(V, G, mu_grid.reshape(G.shape), levels=20, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Peak Force [N]")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Cylinder", "Prism", "Sphere"], fontsize=10)
    ax.set_xlabel("Cutting Velocity [mm/s]", fontsize=11)
    ax.set_title("GP Response Surface\n(geometry × velocity)", fontsize=11)

    plt.suptitle("DiSECt Knife Cutting — GP Surrogate Model", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "disect_gp_surrogate.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Surrogate plot → {path}")


def plot_tmcmc(result: dict, out_dir: str = RESULTS_DIR):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    samples = result["samples"].ravel()
    weights = result["weights"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.scatter(samples, np.zeros_like(samples),
               c=weights, cmap="plasma", s=20, alpha=0.7)
    ax.axvline(result["mean_vel"], color="lime", lw=2,
               label=f"Posterior mean = {result['mean_vel']:.1f} mm/s")
    ax.axvline(result["map_vel"],  color="cyan", lw=1.5, ls="--",
               label=f"MAP = {result['map_vel']:.1f} mm/s")
    ax.set_xlabel("Cutting Velocity [mm/s]", fontsize=11)
    ax.set_title(f"TMCMC Posterior\n({result['geometry'].capitalize()}, "
                 f"observed F = {result['observed_N']:.0f} N)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_yticks([])
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.hist(samples, bins=30, weights=weights, density=True,
            color="#2166ac", alpha=0.8, edgecolor="k", lw=0.4)
    ax.axvline(result["mean_vel"], color="red", lw=2, ls="--",
               label=f"Mean = {result['mean_vel']:.1f} mm/s")
    ax.set_xlabel("Cutting Velocity [mm/s]", fontsize=11)
    ax.set_ylabel("Posterior density", fontsize=11)
    ax.set_title("Posterior Marginal", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    plt.suptitle("TMCMC Calibration — DiSECt Real Cutting Data", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "disect_tmcmc_calibration.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] TMCMC plot → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--geometry", default="cylinder",
                        choices=["cylinder", "prism", "sphere"])
    parser.add_argument("--observed-force", type=float, default=80.0,
                        help="Observed peak force [N] for TMCMC calibration")
    parser.add_argument("--n-samples", type=int, default=800)
    args = parser.parse_args()

    print("[*] Loading DiSECt cutting force data …")
    df = load_disect_forces()
    print(f"  {len(df)} records: {df['geometry'].value_counts().to_dict()}")
    print(df[["geometry","velocity_mm_s","peak_force_N"]].to_string(index=False))

    print("\n[*] Training GP surrogate …")
    predict_fn, _, _ = build_surrogate(df)

    # Quick prediction table
    print("\n[*] GP predictions (velocity sweep, cylinder):")
    for v in [35, 45, 55, 65]:
        mu, s = predict_fn(np.array([[0, v]]), return_std=True)
        print(f"  vel={v} mm/s → F = {mu[0]:.1f} ± {s[0]:.1f} N")

    print(f"\n[*] TMCMC calibration: {args.geometry}, F_obs={args.observed_force:.0f} N")
    result = calibrate_velocity(predict_fn, args.observed_force,
                                geometry=args.geometry,
                                n_samples=args.n_samples)

    if args.plot:
        plot_surrogate(predict_fn, df)
        plot_tmcmc(result)


if __name__ == "__main__":
    main()
