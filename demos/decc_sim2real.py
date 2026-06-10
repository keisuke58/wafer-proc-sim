#!/usr/bin/env python3
"""DECC-style sim-to-real Bayesian calibration demo (Tier 3 / portfolio).

Motivated by DISCO's DECC (Discovery Channel Code Contest) equipment
division: bounce a ball into a target. Quoting the contest premise --
"simulation gives the answer from the laws of physics, but the real
machine deviates due to air drag and friction; the competition is about
sensing that difference and correcting for it."

This demo reproduces exactly that problem structure:

1. "Real machine" simulator (ground truth): 2D projectile with bounces,
   quadratic air drag (Cd), restitution (e) and Coulomb floor friction
   (mu). True values Cd*=0.47, e*=0.82, mu*=0.15, plus observation noise.
2. "Ideal" simulator: drag-free, perfectly elastic textbook physics ->
   visualizes the sim-to-real gap in landing position.
3. Bayesian calibration: from a handful of real-machine trials (bounce
   landing observations), infer the posterior of (Cd, e, mu) with a
   self-contained Metropolis-Hastings MCMC (numpy only; multiple
   independent chains vectorized for speed).
4. Calibrated control: invert launch angle / speed with scipy.optimize
   using the posterior mean, and compare hit error before vs after
   calibration.

The same structure (expensive forward physics + Bayesian parameter
inference + downstream control) is the author's thesis topic
(TMCMC x GPU Bayesian calibration, Physics-Informed ML).

Run:
    python demos/decc_sim2real.py
Output:
    results/decc_sim2real.png

Dependencies: numpy, scipy, matplotlib only. Runtime < 1 min.
"""
from __future__ import annotations

import os
import time

import numpy as np

# ----------------------------------------------------------------------
# Physics constants
# ----------------------------------------------------------------------
G = 9.81                 # gravity [m/s^2]
RHO_AIR = 1.225          # air density [kg/m^3]
BALL_MASS = 0.057        # tennis-ball-like [kg]
BALL_RADIUS = 0.033      # [m]
LAUNCH_Z0 = 0.25         # launcher height [m]
DT = 6e-3                # integration step [s]
T_MAX = 8.0              # safety cap [s]

# Unknown machine parameters: theta = (Cd, e, mu)
TRUE_PARAMS = np.array([0.47, 0.82, 0.15])   # "real machine" ground truth
IDEAL_PARAMS = np.array([0.0, 1.0, 0.0])     # textbook physics
PARAM_NAMES = ["$C_d$ (drag)", "$e$ (restitution)", r"$\mu$ (friction)"]
PARAM_KEYS = ["Cd", "e", "mu"]
BOUNDS = np.array([[0.05, 1.2], [0.40, 0.99], [0.0, 0.6]])  # uniform prior

SIGMA_OBS = 0.02         # observation noise on bounce x-positions [m]
N_BOUNCES_OBS = 3        # observe x of the first 3 bounce points per shot
X_TARGET = 5.0           # target: 2nd bounce must land here [m]


def _drag_k(cd):
    """Quadratic drag coefficient k such that a_drag = -k |v| v."""
    return 0.5 * RHO_AIR * np.pi * BALL_RADIUS**2 / BALL_MASS * cd


# ----------------------------------------------------------------------
# Forward simulators
# ----------------------------------------------------------------------
def simulate_bounces(theta_deg, v0, cd, e, mu, n_bounces=N_BOUNCES_OBS,
                     dt=DT, z0=LAUNCH_Z0, t_max=T_MAX):
    """Vectorized 2D projectile with quadratic drag and bouncing floor.

    All five leading arguments broadcast against each other; returns the
    x-positions of the first `n_bounces` floor impacts with shape
    ``broadcast_shape + (n_bounces,)`` (NaN if a bounce never happens
    within t_max). Deterministic (no RNG).

    Bounce model: vz -> -e * vz_in; Coulomb friction impulse removes
    mu * (1+e) * |vz_in| from |vx| (clamped at zero, no reversal).
    """
    th, v, cdb, eb, mub = np.broadcast_arrays(
        np.asarray(theta_deg, dtype=float), np.asarray(v0, dtype=float),
        cd, e, mu)
    shape = th.shape
    th, v = th.ravel(), v.ravel()
    eb, mub = np.asarray(eb, float).ravel(), np.asarray(mub, float).ravel()
    k = _drag_k(np.asarray(cdb, float).ravel())
    n = th.size

    rad = np.deg2rad(th)
    x = np.zeros(n)
    z = np.full(n, float(z0))
    vx = v * np.cos(rad)
    vz = v * np.sin(rad)
    bx = np.full((n, n_bounces), np.nan)
    count = np.zeros(n, dtype=int)
    active = np.ones(n, dtype=bool)

    t = 0.0
    while active.any() and t < t_max:
        # semi-implicit Euler with quadratic drag
        sp = np.hypot(vx, vz)
        vx = vx - dt * k * sp * vx
        vz = vz - dt * (G + k * sp * vz)
        x_new = x + dt * vx
        z_new = z + dt * vz

        hit = active & (z_new < 0.0) & (vz < 0.0)
        if hit.any():
            frac = z[hit] / (z[hit] - z_new[hit])      # linear interpolation
            x_hit = x[hit] + dt * vx[hit] * frac
            bx[hit, count[hit]] = x_hit
            count[hit] += 1
            # bounce response (restitution + Coulomb friction impulse)
            vz_in = vz[hit]                            # negative
            vz[hit] = -eb[hit] * vz_in
            dvx = mub[hit] * (1.0 + eb[hit]) * np.abs(vz_in)
            vx[hit] = np.sign(vx[hit]) * np.maximum(np.abs(vx[hit]) - dvx, 0.0)
            x_new[hit] = x_hit
            z_new[hit] = 0.0
            active = count < n_bounces

        x, z = x_new, z_new
        t += dt

    return bx.reshape(shape + (n_bounces,))


def simulate_trajectory(theta_deg, v0, params, n_bounces=2, dt=DT,
                        z0=LAUNCH_Z0, t_max=T_MAX):
    """Single-shot trajectory recorder for plotting. Same physics as
    `simulate_bounces`. Returns (xs, zs, bounce_xs)."""
    cd, e, mu = params
    k = _drag_k(cd)
    rad = np.deg2rad(theta_deg)
    x, z = 0.0, float(z0)
    vx, vz = v0 * np.cos(rad), v0 * np.sin(rad)
    xs, zs, bxs = [x], [z], []
    t = 0.0
    while len(bxs) < n_bounces and t < t_max:
        sp = np.hypot(vx, vz)
        vx -= dt * k * sp * vx
        vz -= dt * (G + k * sp * vz)
        x_new, z_new = x + dt * vx, z + dt * vz
        if z_new < 0.0 and vz < 0.0:
            frac = z / (z - z_new)
            x_hit = x + dt * vx * frac
            bxs.append(x_hit)
            vz_in = vz
            vz = -e * vz_in
            dvx = mu * (1.0 + e) * abs(vz_in)
            vx = np.sign(vx) * max(abs(vx) - dvx, 0.0)
            x_new, z_new = x_hit, 0.0
        xs.append(x_new)
        zs.append(z_new)
        x, z = x_new, z_new
        t += dt
    return np.array(xs), np.array(zs), np.array(bxs)


# ----------------------------------------------------------------------
# "Real machine" trials (synthetic observations)
# ----------------------------------------------------------------------
def shot_schedule(n_shots):
    """Fixed launch conditions for the real-machine trials. The first
    few shots already span the angle range (helps identifiability)."""
    thetas = np.array([35, 55, 45, 30, 60, 40, 50, 65, 38, 58,
                       33, 48, 62, 42, 52], dtype=float)
    v0s = np.array([8.0, 7.0, 9.0, 10.0, 6.5, 8.5, 7.5, 6.0, 9.5, 7.0,
                    10.5, 8.0, 6.8, 9.0, 7.8])
    if n_shots > thetas.size:
        raise ValueError(f"at most {thetas.size} shots available")
    return thetas[:n_shots], v0s[:n_shots]


def make_observations(n_shots=15, seed=0, params=TRUE_PARAMS):
    """Fire `n_shots` on the real machine and record noisy bounce
    x-positions. Returns (obs, thetas, v0s); obs shape (n_shots, 3)."""
    thetas, v0s = shot_schedule(n_shots)
    bx = simulate_bounces(thetas, v0s, *params)
    rng = np.random.default_rng(seed)
    obs = bx + rng.normal(0.0, SIGMA_OBS, size=bx.shape)
    return obs, thetas, v0s


# ----------------------------------------------------------------------
# Bayesian calibration: Metropolis-Hastings MCMC (numpy only)
# ----------------------------------------------------------------------
def log_likelihood(params, obs, thetas, v0s):
    """Gaussian observation model. params: (C, 3) -> (C,) log-likelihood."""
    params = np.atleast_2d(params)
    cd = params[:, 0:1]
    e = params[:, 1:2]
    mu = params[:, 2:3]
    model = simulate_bounces(thetas[None, :], v0s[None, :],
                             cd, e, mu)            # (C, S, B)
    resid = (model - obs[None, :, :]) / SIGMA_OBS
    ll = -0.5 * np.sum(resid**2, axis=(1, 2))
    ll[~np.isfinite(ll)] = -np.inf
    return ll


def run_mcmc(obs, thetas, v0s, n_chains=10, n_steps=750, n_burn=250,
             seed=1, scales=(0.02, 0.012, 0.0035)):
    """Random-walk Metropolis-Hastings, `n_chains` independent chains
    advanced in lockstep (likelihood vectorized across chains).

    Returns (samples, acc_rate): samples shape
    (n_chains * (n_steps - n_burn), 3).
    """
    rng = np.random.default_rng(seed)
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    # start chains spread over the central half of the prior box
    cur = lo + (0.25 + 0.5 * rng.random((n_chains, 3))) * (hi - lo)
    ll_cur = log_likelihood(cur, obs, thetas, v0s)
    scales = np.asarray(scales)

    kept = []
    n_acc = 0
    for step in range(n_steps):
        prop = cur + rng.normal(size=(n_chains, 3)) * scales
        in_bounds = np.all((prop >= lo) & (prop <= hi), axis=1)
        ll_prop = np.full(n_chains, -np.inf)
        if in_bounds.any():
            ll_prop[in_bounds] = log_likelihood(prop[in_bounds],
                                                obs, thetas, v0s)
        accept = np.log(rng.random(n_chains)) < (ll_prop - ll_cur)
        cur = np.where(accept[:, None], prop, cur)
        ll_cur = np.where(accept, ll_prop, ll_cur)
        n_acc += int(accept.sum())
        if step >= n_burn:
            kept.append(cur.copy())

    samples = np.concatenate(kept, axis=0)
    acc_rate = n_acc / (n_steps * n_chains)
    return samples, acc_rate


# ----------------------------------------------------------------------
# Control: invert (launch angle, speed) for a target
# ----------------------------------------------------------------------
def aim(params, x_target=X_TARGET, x0=(45.0, 7.0)):
    """Solve for (theta_deg, v0) such that the 2nd bounce lands on the
    target, under the model parameters `params` (scipy Nelder-Mead)."""
    from scipy.optimize import minimize

    def objective(u):
        th, v = u
        if not (15.0 <= th <= 75.0 and 2.0 <= v <= 20.0):
            return 1e6 + (th - 45.0) ** 2 + (v - 7.0) ** 2
        x2 = float(simulate_bounces(th, v, *params, n_bounces=2)[..., 1])
        if not np.isfinite(x2):
            return 1e6
        return (x2 - x_target) ** 2

    res = minimize(objective, np.asarray(x0, float), method="Nelder-Mead",
                   options=dict(xatol=1e-4, fatol=1e-10, maxiter=400))
    return float(res.x[0]), float(res.x[1])


def hit_error(theta_deg, v0, x_target=X_TARGET, params=TRUE_PARAMS):
    """Fire the planned shot on the (noise-free) real machine and return
    the absolute miss distance of the 2nd bounce [m]."""
    x2 = float(simulate_bounces(theta_deg, v0, *params, n_bounces=2)[..., 1])
    return abs(x2 - x_target)


# ----------------------------------------------------------------------
# Demo driver
# ----------------------------------------------------------------------
def main():
    t0 = time.time()
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.abspath(os.path.join(out_dir, "decc_sim2real.png"))

    print("=" * 64)
    print("DECC sim-to-real Bayesian calibration demo")
    print("=" * 64)

    # --- (1) sim-to-real gap for a nominal shot --------------------------
    th_demo, v_demo = 45.0, 8.0
    xi, zi, bxi = simulate_trajectory(th_demo, v_demo, IDEAL_PARAMS)
    xr, zr, bxr = simulate_trajectory(th_demo, v_demo, TRUE_PARAMS)
    gap = bxi[1] - bxr[1]
    print(f"\n[1] Sim-to-real gap (theta={th_demo:.0f} deg, v0={v_demo:.1f} m/s)")
    print(f"    ideal physics 2nd-bounce x : {bxi[1]:6.3f} m")
    print(f"    real machine  2nd-bounce x : {bxr[1]:6.3f} m")
    print(f"    gap                        : {gap:6.3f} m")

    # --- (2) uncalibrated control (textbook physics) ---------------------
    th_u, v_u = aim(IDEAL_PARAMS)
    err_uncal = hit_error(th_u, v_u)
    print(f"\n[2] Uncalibrated aim at target x={X_TARGET:.1f} m")
    print(f"    plan: theta={th_u:.2f} deg, v0={v_u:.3f} m/s "
          f"-> miss = {err_uncal:.3f} m")

    # --- (3) Bayesian calibration with 5/10/15 real trials ---------------
    obs_all, thetas_all, v0s_all = make_observations(15, seed=0)
    n_obs_list = [5, 10, 15]
    errors, results = [], {}
    for n in n_obs_list:
        samples, acc = run_mcmc(obs_all[:n], thetas_all[:n], v0s_all[:n],
                                n_chains=10, n_steps=750, n_burn=250, seed=1)
        post_mean = samples.mean(axis=0)
        post_std = samples.std(axis=0)
        th_c, v_c = aim(post_mean)
        err = hit_error(th_c, v_c)
        errors.append(err)
        results[n] = dict(samples=samples, mean=post_mean, std=post_std,
                          acc=acc, aim=(th_c, v_c), err=err)
        print(f"\n[3] MCMC with {n:2d} shots ({samples.shape[0]} samples, "
              f"acceptance {acc:.2f})")
        for j, key in enumerate(PARAM_KEYS):
            print(f"    {key:>3s}: posterior {post_mean[j]:.3f} "
                  f"+/- {post_std[j]:.3f}   (truth {TRUE_PARAMS[j]:.3f})")
        print(f"    calibrated aim: theta={th_c:.2f} deg, v0={v_c:.3f} m/s "
              f"-> miss = {err:.4f} m")

    final = results[15]
    print(f"\n[4] Hit error: {err_uncal:.3f} m (uncalibrated) -> "
          f"{final['err']:.4f} m (calibrated, 15 shots), "
          f"x{err_uncal / max(final['err'], 1e-6):.0f} improvement")

    # --- (5) figure -------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0])

    # (a) trajectories: ideal vs real
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(xi, zi, "--", color="tab:blue", lw=1.8,
            label="ideal sim (no drag, $e$=1)")
    ax.plot(xr, zr, "-", color="tab:red", lw=1.8,
            label="real machine ($C_d$=0.47, $e$=0.82, $\\mu$=0.15)")
    ax.plot(bxi, np.zeros_like(bxi), "v", color="tab:blue", ms=8)
    ax.plot(bxr, np.zeros_like(bxr), "v", color="tab:red", ms=8)
    ax.annotate("", xy=(bxr[1], 0.12), xytext=(bxi[1], 0.12),
                arrowprops=dict(arrowstyle="<->", color="k"))
    ax.text(0.5 * (bxi[1] + bxr[1]), 0.18,
            f"sim-to-real gap {gap:.2f} m", ha="center", fontsize=10)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(f"(a) Same launch command ({th_demo:.0f} deg, "
                 f"{v_demo:.0f} m/s): ideal vs real")
    ax.legend(loc="upper right", fontsize=9)

    # (c) convergence curve
    ax = fig.add_subplot(gs[0, 2])
    ax.axhline(err_uncal, color="0.3", ls="--",
               label=f"uncalibrated ({err_uncal:.2f} m)")
    ax.plot(n_obs_list, errors, "o-", color="tab:green", lw=2,
            label="after calibration")
    ax.set_yscale("log")
    ax.set_xticks(n_obs_list)
    ax.set_xlabel("number of real-machine trials")
    ax.set_ylabel("hit error at target [m]")
    ax.set_title("(c) Trials vs hit error")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # (b) posterior corner-style histograms (15-shot run)
    for j in range(3):
        ax = fig.add_subplot(gs[1, j])
        ax.hist(final["samples"][:, j], bins=40, color="tab:purple",
                alpha=0.65, density=True)
        ax.axvline(TRUE_PARAMS[j], color="k", lw=2, label="truth")
        ax.axvline(final["mean"][j], color="tab:orange", lw=2, ls="--",
                   label="posterior mean")
        ax.set_xlabel(PARAM_NAMES[j])
        if j == 0:
            ax.set_ylabel("posterior density")
            ax.legend(fontsize=8)
        ax.set_title(f"(b{j + 1}) {PARAM_KEYS[j]}: "
                     f"{final['mean'][j]:.3f}$\\pm${final['std'][j]:.3f}")

    fig.suptitle("DECC-style sim-to-real: Bayesian calibration of "
                 "(drag, restitution, friction) from few real trials",
                 fontsize=12)
    fig.savefig(out_png, dpi=150)
    print(f"\nFigure saved: {out_png}")
    print(f"Total runtime: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
