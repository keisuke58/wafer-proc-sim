"""
Ensemble Kalman Filter — Dicing Process State Estimation
=========================================================
Layer 1 of the Quantum Stack: real-time state estimation from sensor data.

Directly inspired by:
    Muramatsu et al. (2026) "Data Assimilation Based on the Ensemble Kalman
    Filter for Dislocation Motion", Modelling and Simulation in Materials
    Science and Engineering.

    There:  state = dislocation positions + drag coefficient B
            obs   = TEM frame positions
            model = dislocation dynamics (DD) equations

    Here:   state = [d_eff, alpha_w, k_ic]   (hidden machining state)
            obs   = chipping_um + (optionally force_N)
            model = physics-based chipping model

State vector x ∈ R³:
    d_eff   : effective depth ratio (actual / nominal), ~1.0 ± 0.05
              (deviation due to wafer bow, chuck compliance, springback)
    alpha_w : blade wear coefficient, 1.0 (new) → 0.8 (worn)
              (abrasive grain exposure degrades with grinding distance)
    k_ic    : local K_Ic factor (crystal orientation), 0.84–1.00
              (m-plane → 1.00, a-plane → 0.84, varies along wafer lot)

Transition model (wafer-to-wafer):
    d_eff_{t+1}  = d_eff_t                  + ε_d   (small drift)
    alpha_w_{t+1}= alpha_w_t - γ_w * dt     + ε_w   (monotone wear)
    k_ic_{t+1}   = k_ic_t                   + ε_k   (slow crystal variation)

Observation model:
    chip_pred = base_chip(d_nom * d_eff, blade_W) / (alpha_w * k_ic)
    y = chip_pred + η,   η ~ N(0, R)

Usage:
    python ml/ekf_state_estimation.py                  # simulate 60 wafers
    python ml/ekf_state_estimation.py --n-wafers 100
    python ml/ekf_state_estimation.py --obs-noise 2.0  # noisier sensor
    python ml/ekf_state_estimation.py --ensemble 200   # larger ensemble
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# C++ EnKF kernel — build with: bash build_all_kernels.sh
try:
    from ml import _enkf_kernel as _cpp_enkf
    _CPP_ENKF = True
except ImportError:
    _CPP_ENKF = False

# ── Reference recipe (fixed during simulation) ───────────────────────────────
D_NOM    = 200.0   # µm  nominal cut depth
BLADE_W  = 23.0    # µm  blade kerf width
FEED     = 1.0     # mm/s
RPM      = 30000.0


# ════════════════════════════════════════════════════════════════════════════
# Physics model (identical to quantum_annealing_recipe.py)
# ════════════════════════════════════════════════════════════════════════════

def chipping_physics(depth_um: float, blade_w_um: float) -> float:
    """Base chipping [µm] at reference feed/spindle conditions."""
    depth_term   = 1.2 + 0.035 * (depth_um - 80.0) / 310.0 * 10.0
    width_term   = 0.8 + 0.4   * (blade_w_um - 20.0) / 35.0
    interaction  = 1.0 + 0.12  * (
        (depth_um - 80.0) / 310.0 * (blade_w_um - 20.0) / 35.0
    )
    return depth_term * width_term * interaction * 3.5


def observation_model(state: np.ndarray) -> float:
    """
    H(x): predicted chipping given hidden state.

    chip = base_chip(d_nom * d_eff, blade_W) / (alpha_w * k_ic)

    Higher d_eff  → deeper actual cut  → more chipping
    Lower alpha_w → more worn blade    → more chipping
    Lower k_ic    → a-plane crystal    → more chipping
    """
    d_eff, alpha_w, k_ic = state
    actual_depth = D_NOM * np.clip(d_eff, 0.5, 1.5)
    base = chipping_physics(actual_depth, BLADE_W)
    return base / (np.clip(alpha_w, 0.1, 2.0) * np.clip(k_ic, 0.5, 1.5))


# ════════════════════════════════════════════════════════════════════════════
# Ensemble Kalman Filter
# ════════════════════════════════════════════════════════════════════════════

class EnsembleKalmanFilter:
    """
    EnKF for dicing process state estimation.

    State: x = [d_eff, alpha_w, k_ic]

    Algorithm (stochastic EnKF — Burgers et al. 1998):
        For each observation y_t:
            1. Forecast:  x_f^(i) = F(x_a^(i)) + w^(i)
            2. Predicted obs: y^(i) = H(x_f^(i)) + v^(i)
            3. Kalman gain: K = P_xy (P_yy + R)^{-1}
            4. Analysis:  x_a^(i) = x_f^(i) + K(y_t - y^(i))

    Cross-covariances computed directly from ensemble spread
    (no Jacobians needed — this is why EnKF handles nonlinear H).
    """

    def __init__(self,
                 n_ensemble:   int   = 100,
                 obs_noise_std: float = 1.5,
                 seed:         int   = 42):
        self.N   = n_ensemble
        self.R   = obs_noise_std ** 2   # obs noise variance (scalar)
        self.rng = np.random.default_rng(seed)

        # Process noise covariance Q (diagonal)
        self.Q = np.diag([
            1e-4,   # d_eff  — slow drift
            5e-5,   # alpha_w — slow monotone wear
            1e-4,   # k_ic   — slow crystal variation
        ])

        # Wear rate γ_w per wafer
        self._gamma_w = 2e-3

        # Initialise ensemble near "new blade, m-plane crystal" conditions
        x0    = np.array([1.00, 1.00, 1.00])
        P0    = np.diag([0.02**2, 0.02**2, 0.03**2])
        L     = np.linalg.cholesky(P0)
        noise = (L @ self.rng.standard_normal((3, n_ensemble))).T  # (N,3)
        self.ensemble = x0 + noise                                 # (N,3)

    # ── Accessors ──────────────────────────────────────────────────────────

    @property
    def mean(self) -> np.ndarray:
        return self.ensemble.mean(axis=0)

    @property
    def std(self) -> np.ndarray:
        return self.ensemble.std(axis=0)

    # ── Transition model ───────────────────────────────────────────────────

    def _forecast(self) -> np.ndarray:
        """
        Apply transition F to every ensemble member + process noise.
        alpha_w has a deterministic wear drift.
        """
        xf = self.ensemble.copy()
        xf[:, 1] -= self._gamma_w                          # blade wear

        # add process noise
        L  = np.linalg.cholesky(self.Q)
        xf += (L @ self.rng.standard_normal((3, self.N))).T
        return xf

    # ── Analysis (update) step ─────────────────────────────────────────────

    def update(self, y_obs: float) -> None:
        """
        Assimilate scalar observation y_obs (measured chipping [µm]).
        Updates self.ensemble in-place.
        """
        # 1. Forecast
        xf = self._forecast()                              # (N, 3)

        # 2. Observation noise perturbations
        v = self.rng.normal(0.0, np.sqrt(self.R), self.N)

        if _CPP_ENKF:
            yf = _cpp_enkf.three_state_obs(
                np.ascontiguousarray(xf, np.float64), D_NOM, BLADE_W)
            self.ensemble = _cpp_enkf.enkf_analysis(
                np.ascontiguousarray(xf, np.float64), yf, y_obs, v, self.R)
            return

        # Python fallback
        yf = np.array([observation_model(xf[i]) for i in range(self.N)])  # (N,)
        yf_pert = yf + v

        xf_bar = xf.mean(axis=0, keepdims=True)
        yf_bar = yf.mean()
        dX = xf - xf_bar
        dY = yf - yf_bar
        P_xy = (dX.T @ dY) / (self.N - 1)
        P_yy = float((dY @ dY) / (self.N - 1))
        K = P_xy / (P_yy + self.R)
        innovation = y_obs - yf_pert
        self.ensemble = xf + np.outer(innovation, K)


# ════════════════════════════════════════════════════════════════════════════
# Synthetic sensor stream
# ════════════════════════════════════════════════════════════════════════════

def simulate_wafer_stream(n_wafers:       int   = 60,
                          obs_noise_std:  float = 1.5,
                          wear_rate:      float = 0.003,
                          seed:           int   = 7) -> dict:
    """
    Simulate a sequence of wafers with hidden state evolution.

    Returns dict with true states, observations, and recipe parameters.
    """
    rng = np.random.default_rng(seed)

    # True hidden state trajectories
    d_eff_true   = np.ones(n_wafers)
    alpha_w_true = np.ones(n_wafers)
    k_ic_true    = np.ones(n_wafers)

    # Introduce realistic perturbations
    # Slight depth drift (wafer bow + chuck variation)
    d_eff_true  += np.cumsum(rng.normal(0, 0.008, n_wafers))
    d_eff_true   = np.clip(d_eff_true, 0.85, 1.15)

    # Monotone blade wear
    alpha_w_true -= wear_rate * np.arange(n_wafers)
    alpha_w_true  = np.clip(alpha_w_true, 0.75, 1.0)

    # Crystal orientation variation (lot-to-lot, slow)
    k_ic_true += 0.05 * np.sin(2 * np.pi * np.arange(n_wafers) / 30)
    k_ic_true -= 0.08 * (np.arange(n_wafers) > 40)  # orientation shift at wafer 40
    k_ic_true  = np.clip(k_ic_true, 0.84, 1.0)

    # True chipping + noisy observation
    true_state = np.column_stack([d_eff_true, alpha_w_true, k_ic_true])
    chip_true  = np.array([observation_model(s) for s in true_state])
    chip_obs   = chip_true + rng.normal(0, obs_noise_std, n_wafers)

    return {
        "n_wafers":    n_wafers,
        "true_state":  true_state,
        "chip_true":   chip_true,
        "chip_obs":    chip_obs,
        "d_eff_true":  d_eff_true,
        "alpha_w_true": alpha_w_true,
        "k_ic_true":   k_ic_true,
    }


# ════════════════════════════════════════════════════════════════════════════
# Run EKF
# ════════════════════════════════════════════════════════════════════════════

def run_ekf(stream: dict,
            n_ensemble: int   = 100,
            obs_noise_std: float = 1.5) -> dict:
    """Run EnKF on the simulated wafer stream."""
    n_wafers = stream["n_wafers"]
    ekf      = EnsembleKalmanFilter(n_ensemble=n_ensemble,
                                    obs_noise_std=obs_noise_std)

    means = np.zeros((n_wafers, 3))
    stds  = np.zeros((n_wafers, 3))

    for t in range(n_wafers):
        ekf.update(stream["chip_obs"][t])
        means[t] = ekf.mean
        stds[t]  = ekf.std

    # Chipping prediction from EKF estimate
    chip_ekf = np.array([observation_model(means[t]) for t in range(n_wafers)])

    # Baseline: assume state = nominal (no correction)
    nominal_state = np.array([1.0, 1.0, 1.0])
    chip_nominal  = np.full(n_wafers, observation_model(nominal_state))

    return {
        "means":        means,
        "stds":         stds,
        "chip_ekf":     chip_ekf,
        "chip_nominal": chip_nominal,
    }


# ════════════════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════════════════

def print_metrics(stream: dict, ekf_result: dict) -> None:
    from sklearn.metrics import mean_squared_error, r2_score

    true = stream["chip_true"]
    rmse_ekf = np.sqrt(mean_squared_error(true, ekf_result["chip_ekf"]))
    rmse_nom = np.sqrt(mean_squared_error(true, ekf_result["chip_nominal"]))
    r2_ekf   = r2_score(true, ekf_result["chip_ekf"])
    r2_nom   = r2_score(true, ekf_result["chip_nominal"])

    print(f"\n{'':=<50}")
    print(f"  EKF   : RMSE={rmse_ekf:.2f}µm  R²={r2_ekf:.3f}")
    print(f"  Nominal: RMSE={rmse_nom:.2f}µm  R²={r2_nom:.3f}")
    print(f"  RMSE improvement: {100*(rmse_nom-rmse_ekf)/rmse_nom:.1f}%")

    means = ekf_result["means"]
    stds  = ekf_result["stds"]
    labels = ["d_eff", "alpha_w", "k_ic"]
    true_states = stream["true_state"]
    print()
    for i, lab in enumerate(labels):
        state_rmse = np.sqrt(np.mean((means[:, i] - true_states[:, i])**2))
        print(f"  State [{lab:8s}] RMSE = {state_rmse:.4f}")


# ════════════════════════════════════════════════════════════════════════════
# Plots
# ════════════════════════════════════════════════════════════════════════════

def plot_results(stream: dict, ekf_result: dict, save: bool = True) -> None:
    n  = stream["n_wafers"]
    t  = np.arange(n)

    means  = ekf_result["means"]
    stds   = ekf_result["stds"]
    labels = ["d_eff", "alpha_w  (blade wear)", "k_ic  (crystal)"]
    colors = ["#2166ac", "#d62728", "#1a9641"]
    true_arrs = [stream["d_eff_true"], stream["alpha_w_true"],
                 stream["k_ic_true"]]

    fig = plt.figure(figsize=(13, 10))
    gs  = gridspec.GridSpec(4, 1, hspace=0.45)

    # ── Row 0–2: State estimation ─────────────────────────────────────────
    for i in range(3):
        ax = fig.add_subplot(gs[i])
        ax.fill_between(t,
                         means[:, i] - 2*stds[:, i],
                         means[:, i] + 2*stds[:, i],
                         alpha=0.25, color=colors[i], label="EKF 2σ")
        ax.plot(t, means[:, i],  color=colors[i], lw=2, label="EKF mean")
        ax.plot(t, true_arrs[i], "k--", lw=1.5, label="True state")
        ax.set_ylabel(labels[i])
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_title("EnKF State Estimation — SiC Dicing (Layer 1 of Quantum Stack)",
                         fontsize=11)

    # ── Row 3: Chipping prediction ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[3])
    ax3.scatter(t, stream["chip_obs"],   s=20, color="gray", alpha=0.6,
                label="Sensor observation", zorder=2)
    ax3.plot(t, stream["chip_true"],     "k-", lw=1.5, label="True chipping")
    ax3.plot(t, ekf_result["chip_ekf"],  color=colors[0], lw=2,
             label="EKF prediction")
    ax3.axhline(ekf_result["chip_nominal"][0], color="orange", ls="--", lw=1.5,
                label="Nominal (no correction)")
    ax3.set_xlabel("Wafer index")
    ax3.set_ylabel("Chipping [µm]")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    plt.suptitle("Ensemble Kalman Filter for Dicing Process\n"
                 "(inspired by Muramatsu et al. 2026 — EKF for Dislocation Motion)",
                 fontsize=10, style="italic")

    if save:
        path = os.path.join(OUT_DIR, "ekf_state_estimation.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()


def plot_ensemble_spread(ekf_result: dict, stream: dict, save: bool = True) -> None:
    """Show ensemble spread evolution (uncertainty quantification)."""
    stds   = ekf_result["stds"]
    n      = stds.shape[0]
    t      = np.arange(n)
    labels = ["d_eff", "alpha_w", "k_ic"]
    colors = ["#2166ac", "#d62728", "#1a9641"]

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (lab, col) in enumerate(zip(labels, colors)):
        ax.plot(t, stds[:, i], color=col, lw=2, label=f"σ({lab})")

    ax.set_xlabel("Wafer index")
    ax.set_ylabel("Ensemble std (posterior uncertainty)")
    ax.set_title("Posterior uncertainty reduction — EnKF")
    ax.legend()
    ax.grid(alpha=0.3)

    # Annotation: sharp drop = filter has learned the state
    ax.annotate("Filter converges\n(uncertainty ↓)",
                xy=(15, stds[15].max()),
                xytext=(25, stds[15].max() * 1.5),
                arrowprops=dict(arrowstyle="->", color="k"),
                fontsize=8)

    if save:
        path = os.path.join(OUT_DIR, "ekf_uncertainty.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wafers",   type=int,   default=60)
    parser.add_argument("--ensemble",   type=int,   default=100)
    parser.add_argument("--obs-noise",  type=float, default=1.5,
                        help="Sensor noise std [µm]")
    parser.add_argument("--wear-rate",  type=float, default=0.003,
                        help="Blade wear per wafer (default 0.003)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 60)
    print("Ensemble Kalman Filter — Dicing State Estimation (Layer 1)")
    print(f"  n_wafers={args.n_wafers}  ensemble={args.ensemble}  "
          f"obs_noise={args.obs_noise}µm")
    print("=" * 60)

    stream     = simulate_wafer_stream(args.n_wafers, args.obs_noise,
                                       args.wear_rate)
    ekf_result = run_ekf(stream, args.ensemble, args.obs_noise)

    print_metrics(stream, ekf_result)
    plot_results(stream, ekf_result)
    plot_ensemble_spread(ekf_result, stream)

    # ── Feed best state estimate into Layer 2 ──────────────────────────────
    final_state  = ekf_result["means"][-1]
    final_depth  = D_NOM * final_state[0]
    final_alpha  = final_state[1]
    final_kic    = final_state[2]

    print("\n--- Final EKF state estimate (→ feeds into Layer 2 QKernel-GP) ---")
    print(f"  Effective depth : {final_depth:.1f} µm  "
          f"(nominal {D_NOM:.0f} µm, ratio {final_state[0]:.3f})")
    print(f"  Blade wear      : α_w = {final_alpha:.3f}  "
          f"({'new' if final_alpha > 0.95 else 'worn' if final_alpha < 0.85 else 'moderate'})")
    print(f"  K_Ic factor     : k = {final_kic:.3f}  "
          f"({'m-plane' if final_kic > 0.95 else 'a-plane mix' if final_kic < 0.88 else 'mixed'})")
    corrected_chip = observation_model(final_state)
    nominal_chip   = observation_model(np.array([1.0, 1.0, 1.0]))
    print(f"  Predicted chip  : {corrected_chip:.2f} µm  "
          f"(nominal model: {nominal_chip:.2f} µm, "
          f"Δ = {corrected_chip - nominal_chip:+.2f} µm)")


if __name__ == "__main__":
    main()
