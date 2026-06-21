"""
Inverse phase-field forensics (MVP).

Idea A: given an observed AT2 damage field d_obs, recover the posterior over the
fracture parameters (Gc, ell) that produced it -- using the trained flow-matching
generator (field_fm.pt) as a fast *forward emulator* inside the likelihood.

Most SHM work does detection / localisation. This instead asks the forensic
question: "which material parameters & loading produced this damage?" and returns
a *posterior*, so the Gc-ell identifiability trade-off is visible rather than
hidden behind a point estimate.

This MVP runs fully on synthetic data with known ground truth:
  1. fix a true (Gc*, ell*); render d_obs as the mean of K generator samples
  2. evaluate an unnormalised log-posterior on a 2D grid of (Gc, ell)
  3. plot the posterior, mark the truth, show marginals + MAP / posterior mean

Forward emulator is stochastic, so each likelihood averages K samples to get a
smooth field estimate. Grid inference (not MCMC) keeps it reproducible and
tuning-free for the WCCM demo.

Run:
  cd /home/nishioka/git/wafer-proc-sim
  python research/phasefield/inverse_pf_forensics.py
"""
from __future__ import annotations
import json
import time
import numpy as np
import torch

from research.phasefield.diffusion_field_ddpm import (
    FieldDenoiser, sample_fields_fm, FULL_CFG,
)
from research.phasefield.at2_simulator_2d import crack_deflection_angle

# --- config -----------------------------------------------------------------
CKPT     = "research/phasefield/results/field_fm.pt"
OUT_PNG  = "research/phasefield/results/inverse_pf_posterior.png"
OUT_JSON = "research/phasefield/results/inverse_pf_posterior.json"
DEVICE   = torch.device("cpu")

# generator condition ranges (must match training; see _norm_cond)
GC_RANGE  = (2e-4, 3e-3)     # fracture toughness  [log-sampled]
ELL_RANGE = (0.02, 0.10)     # length scale        [log-sampled]

# fixed / "known" nuisance conditions for this 2D demo
BETA_FIX  = -0.40
THETA_FIX = 45.0

# ground truth to recover
GC_TRUE   = 8.0e-4
ELL_TRUE  = 0.045

# inference settings
GRID_N    = 16             # grid points per axis -> GRID_N^2 likelihood evals
K_SAMPLES = 6              # generator samples averaged per field estimate
N_STEPS   = 20             # FM Euler steps
N_CALIB   = 10             # independent renders at truth -> feature-noise covariance
SEED      = 0

# Physically-interpretable damage descriptors. We invert on THESE rather than on
# 3200 raw pixels: the field is spatially smooth, so pixels are highly correlated
# and treating them as independent observations makes the posterior absurdly
# overconfident (collapses to one grid cell). A handful of mechanics-relevant
# summary features with empirically calibrated noise gives honest posterior width.
FEATURE_NAMES = ["mean_d", "std_d", "broken_frac", "crack_angle_deg"]


def features(d: np.ndarray) -> np.ndarray:
    """Map a damage field (ny, nx) -> physical descriptor vector."""
    return np.array([
        float(d.mean()),                       # overall damage intensity (~Gc)
        float(d.std()),                        # localization / contrast (~ell)
        float((d > 0.5).mean()),               # broken-area fraction
        crack_deflection_angle(d, FULL_CFG),   # crack morphology / orientation
    ])


def load_generator() -> FieldDenoiser:
    ck = torch.load(CKPT, map_location="cpu")
    m = FieldDenoiser(ny=int(ck["ny"]), nx=int(ck["nx"]))
    m.load_state_dict(ck["model"])
    m.eval()
    return m


def forward_field(model, Gc, ell, k=K_SAMPLES, seed=None) -> np.ndarray:
    """Mean damage field predicted by the emulator at (Gc, ell)."""
    if seed is not None:
        torch.manual_seed(seed)
    d = sample_fields_fm(model, Gc=float(Gc), ell=float(ell),
                         beta=BETA_FIX, theta=THETA_FIX,
                         n_samples=k, n_steps=N_STEPS, device=DEVICE)
    return d.mean(axis=0)          # (ny, nx)


def log_likelihood(f_pred, f_obs, f_sigma) -> float:
    """Gaussian likelihood in physical-feature space (diagonal covariance)."""
    r = (f_pred - f_obs) / f_sigma
    return -0.5 * float(np.dot(r, r))


# ── reusable inversion core (shared by main, coverage test, real-data inject) ──

def make_grids(n=GRID_N):
    gc_grid  = np.exp(np.linspace(np.log(GC_RANGE[0]),  np.log(GC_RANGE[1]),  n))
    ell_grid = np.exp(np.linspace(np.log(ELL_RANGE[0]), np.log(ELL_RANGE[1]), n))
    return gc_grid, ell_grid


def calibrate_obs(model, gc_true, ell_true, n_calib=N_CALIB, k=K_SAMPLES, seed0=900):
    """Render n_calib independent fields at truth -> (f_obs, f_sigma, d_display)."""
    feats = np.array([
        features(forward_field(model, gc_true, ell_true, k=k, seed=seed0 + r))
        for r in range(n_calib)
    ])
    f_obs   = feats.mean(axis=0)
    f_sigma = np.maximum(feats.std(axis=0), 0.02 * (np.abs(f_obs) + 1e-6))
    d_disp  = forward_field(model, gc_true, ell_true, k=k, seed=seed0 + 99)
    return f_obs, f_sigma, d_disp


def grid_logpost(model, f_obs, f_sigma, gc_grid, ell_grid, k=K_SAMPLES, seed0=0):
    """Evaluate the unnormalised log-posterior on the (ell x Gc) grid."""
    n_ell, n_gc = len(ell_grid), len(gc_grid)
    lp = np.full((n_ell, n_gc), -np.inf)
    for i, ell in enumerate(ell_grid):
        for j, gc in enumerate(gc_grid):
            f_pred = features(forward_field(model, gc, ell, k=k, seed=seed0 + i * n_gc + j))
            lp[i, j] = log_likelihood(f_pred, f_obs, f_sigma)
    return lp


def normalise_post(logpost):
    p = np.exp(logpost - logpost.max())
    return p / p.sum()


def posterior_summary(post, gc_grid, ell_grid):
    mi, mj = np.unravel_index(np.argmax(post), post.shape)
    pgc, pell = post.sum(axis=0), post.sum(axis=1)
    gc_mean  = float((pgc * gc_grid).sum())
    ell_mean = float((pell * ell_grid).sum())
    gc_std   = float(np.sqrt((pgc * (gc_grid - gc_mean) ** 2).sum()))
    ell_std  = float(np.sqrt((pell * (ell_grid - ell_mean) ** 2).sum()))
    return {"gc_map": float(gc_grid[mj]), "ell_map": float(ell_grid[mi]),
            "gc_mean": gc_mean, "ell_mean": ell_mean,
            "gc_std": gc_std, "ell_std": ell_std}


def hpd_contains(post, gc_grid, ell_grid, gc_true, ell_true, q):
    """Is (gc_true, ell_true) inside the smallest grid region holding mass >= q?"""
    flat = post.ravel()
    order = np.argsort(flat)[::-1]
    cum = np.cumsum(flat[order])
    kmax = np.searchsorted(cum, q) + 1
    in_region = np.zeros_like(flat, dtype=bool)
    in_region[order[:kmax]] = True
    in_region = in_region.reshape(post.shape)
    # nearest grid cell to truth
    j = int(np.argmin(np.abs(gc_grid - gc_true)))
    i = int(np.argmin(np.abs(ell_grid - ell_true)))
    return bool(in_region[i, j])


def main() -> None:
    t_start = time.time()
    print(f"[setup] loading generator from {CKPT}")
    model = load_generator()

    gc_grid, ell_grid = make_grids(GRID_N)

    # render observation at truth + calibrate feature noise (emulator stochasticity)
    print(f"[obs] rendering {N_CALIB} fields at Gc*={GC_TRUE:.2e}, ell*={ELL_TRUE:.3f}")
    f_obs, f_sigma, d_obs = calibrate_obs(model, GC_TRUE, ELL_TRUE,
                                          n_calib=N_CALIB, k=K_SAMPLES, seed0=SEED + 900)
    print("[calib] feature  obs            sigma")
    for n, fo, fs in zip(FEATURE_NAMES, f_obs, f_sigma):
        print(f"        {n:<16} {fo:8.4f}    {fs:8.4f}")

    print(f"[grid] {GRID_N}x{GRID_N}={GRID_N*GRID_N} evals, K={K_SAMPLES} samples each")
    logpost = grid_logpost(model, f_obs, f_sigma, gc_grid, ell_grid, k=K_SAMPLES, seed0=SEED)
    post = normalise_post(logpost)
    s = posterior_summary(post, gc_grid, ell_grid)

    print("\n=== posterior summary ===")
    print(f"  truth : Gc*={GC_TRUE:.3e}  ell*={ELL_TRUE:.4f}")
    print(f"  MAP   : Gc ={s['gc_map']:.3e}  ell ={s['ell_map']:.4f}")
    print(f"  mean  : Gc ={s['gc_mean']:.3e}  ell ={s['ell_mean']:.4f}")
    print(f"  std   : Gc ={s['gc_std']:.3e} ({100*s['gc_std']/s['gc_mean']:.0f}%)  "
          f"ell ={s['ell_std']:.4f} ({100*s['ell_std']/s['ell_mean']:.0f}%)")
    gc_cov  = abs(s['gc_mean'] - GC_TRUE)  <= s['gc_std']
    ell_cov = abs(s['ell_mean'] - ELL_TRUE) <= s['ell_std']
    print(f"  truth within 1sd:  Gc {'YES' if gc_cov else 'no'}   "
          f"ell {'YES' if ell_cov else 'no'}")

    result = {
        "truth":      {"Gc": GC_TRUE, "ell": ELL_TRUE},
        "map":        {"Gc": s['gc_map'], "ell": s['ell_map']},
        "post_mean":  {"Gc": s['gc_mean'], "ell": s['ell_mean']},
        "post_std":   {"Gc": s['gc_std'], "ell": s['ell_std']},
        "feature_names": FEATURE_NAMES,
        "feature_obs":   f_obs.tolist(),
        "feature_sigma": f_sigma.tolist(),
        "truth_within_1sd": {"Gc": bool(gc_cov), "ell": bool(ell_cov)},
        "gc_grid":    gc_grid.tolist(),
        "ell_grid":   ell_grid.tolist(),
        "posterior":  post.tolist(),
        "config":     {"GRID_N": GRID_N, "K_SAMPLES": K_SAMPLES, "N_STEPS": N_STEPS,
                       "N_CALIB": N_CALIB, "BETA_FIX": BETA_FIX, "THETA_FIX": THETA_FIX},
        "runtime_s":  round(time.time() - t_start, 1),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[save] {OUT_JSON}")

    _plot(post, gc_grid, ell_grid, d_obs,
          (s['gc_map'], s['ell_map']), (s['gc_mean'], s['ell_mean']))
    print(f"[save] {OUT_PNG}")
    print(f"[done] total {time.time()-t_start:.0f}s")


def _plot(post, gc_grid, ell_grid, d_obs, mapxy, meanxy):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 4.2))

    # (1) observation
    ax0 = fig.add_subplot(1, 3, 1)
    im0 = ax0.imshow(d_obs, origin="lower", aspect="auto", cmap="inferno", vmin=0, vmax=1)
    ax0.set_title(f"observed damage  d_obs\nGc*={GC_TRUE:.2e}, ell*={ELL_TRUE:.3f}")
    ax0.set_xlabel("x"); ax0.set_ylabel("y")
    fig.colorbar(im0, ax=ax0, fraction=0.046)

    # (2) posterior heatmap over (Gc, ell)
    ax1 = fig.add_subplot(1, 3, 2)
    im1 = ax1.pcolormesh(gc_grid, ell_grid, post, shading="auto", cmap="viridis")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("Gc"); ax1.set_ylabel("ell")
    ax1.set_title("posterior  p(Gc, ell | d_obs)")
    ax1.scatter([GC_TRUE], [ELL_TRUE], marker="*", s=260, c="red",
                edgecolor="white", linewidth=1.0, label="truth", zorder=5)
    ax1.scatter([mapxy[0]], [mapxy[1]], marker="X", s=120, c="cyan",
                edgecolor="black", label="MAP", zorder=5)
    ax1.legend(loc="upper right", fontsize=8)
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    # (3) marginals
    ax2 = fig.add_subplot(1, 3, 3)
    pgc  = post.sum(axis=0); pgc  /= pgc.max()
    pell = post.sum(axis=1); pell /= pell.max()
    ax2.plot(gc_grid / GC_TRUE,  pgc,  "-o", ms=3, label="Gc / Gc*")
    ax2.plot(ell_grid / ELL_TRUE, pell, "-s", ms=3, label="ell / ell*")
    ax2.axvline(1.0, color="red", ls="--", lw=1, label="truth")
    ax2.set_xscale("log")
    ax2.set_xlabel("parameter / truth")
    ax2.set_ylabel("marginal (norm.)")
    ax2.set_title("marginal posteriors")
    ax2.legend(fontsize=8)

    fig.suptitle("Inverse phase-field forensics: damage field -> (Gc, ell) posterior",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
