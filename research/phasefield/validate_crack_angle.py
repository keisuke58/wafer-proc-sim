"""
validate_crack_angle.py — Quantitative validation of the conditional generative
model against the AT2 simulator.

The visual crack-band figures show the model *looks* like it deflects cracks
with theta_cut.  This script proves it *quantitatively*: for a sweep of cleavage
angles theta, it
  1. samples N crack fields from the trained model,
  2. measures each field's deflection angle phi with the SAME estimator used on
     simulator output (crack_deflection_angle),
  3. compares the generated phi distribution (mean ± std band) against the
     deterministic AT2 simulator ground truth at the same (Gc, ell, beta, theta).

A model that learned the physics — not just plausible textures — tracks the
simulator's phi(theta) curve.  The generated std is the model's own uncertainty.

Usage
-----
  python validate_crack_angle.py --ckpt results/field_ddpm.pt --method ddpm \
      --out results/validate_phi.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from research.phasefield.at2_simulator_2d import (
    PFParams2D, run_forward_2d, crack_deflection_angle,
)
from research.phasefield.diffusion_field_ddpm import (
    FieldDenoiser, DDPMSchedule, sample_fields, sample_fields_fm,
    predict_fields_mse, FULL_CFG,
)

# Fixed material params; sweep theta_cut
GC, ELL, BETA = 8e-4, 0.04, -0.40
THETA_GRID = [0, 15, 30, 45, 60, 75, 90]


def load_model(ckpt_path, device):
    ck    = torch.load(ckpt_path, map_location=device)
    model = FieldDenoiser(ny=int(ck.get("ny", 40)), nx=int(ck.get("nx", 80))).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def sample(model, ck, method, theta, n, device):
    if method == "fm":
        return sample_fields_fm(model, Gc=GC, ell=ELL, beta=BETA, theta=theta,
                                n_samples=n, n_steps=20, guidance_scale=3.0,
                                device=device)
    if method == "ddpm":
        sched = DDPMSchedule(T=int(ck.get("schedule_T", 300)), device=device)
        return sample_fields(model, sched, Gc=GC, ell=ELL, beta=BETA, theta=theta,
                             n_samples=n, guidance_scale=3.0, device=device)
    return predict_fields_mse(model, Gc=GC, ell=ELL, beta=BETA, theta=theta,
                              n_samples=n, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--method", choices=["fm", "ddpm", "mse"], default="ddpm")
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--out", default="results/validate_phi.png")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, ck = load_model(args.ckpt, device)

    gen_mean, gen_std, sim_phi = [], [], []
    print(f"{'theta':>6} {'sim_phi':>9} {'gen_mean':>9} {'gen_std':>8}")
    for theta in THETA_GRID:
        # Simulator ground truth (deterministic)
        p = PFParams2D(Gc=GC, ell=ELL, E=1.0, nu=0.25, beta=BETA,
                       theta_cut=np.radians(theta))
        sim = run_forward_2d(p, FULL_CFG)
        sim_phi.append(sim.crack_angle)

        # Generated fields → measure angle on each
        fields = sample(model, ck, args.method, float(theta), args.n_samples, device)
        angles = np.array([crack_deflection_angle(f, FULL_CFG) for f in fields])
        angles = angles[np.isfinite(angles)]
        gen_mean.append(float(angles.mean()))
        gen_std.append(float(angles.std()))
        print(f"{theta:>6} {sim.crack_angle:>9.2f} {angles.mean():>9.2f} "
              f"{angles.std():>8.2f}")

    th = np.array(THETA_GRID, dtype=float)
    gm, gs, sp = map(np.array, (gen_mean, gen_std, sim_phi))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(th, sp, "k-o", lw=2, label="AT2 simulator (ground truth)")
    ax.plot(th, gm, "C0-s", lw=2, label=f"generated mean ({args.method.upper()})")
    ax.fill_between(th, gm - gs, gm + gs, color="C0", alpha=0.25,
                    label="generated ±1σ (model uncertainty)")
    ax.set_xlabel("cleavage angle  θ_cut  [deg]")
    ax.set_ylabel("crack deflection angle  φ  [deg]")
    ax.set_title(f"Crack-angle validation: generated vs simulator\n"
                 f"(Gc={GC:.0e}, ℓ={ELL}, β={BETA}, N={args.n_samples})")
    ax.legend(); ax.grid(alpha=0.3)

    # Quantitative fit metrics
    mae  = float(np.mean(np.abs(gm - sp)))
    corr = float(np.corrcoef(gm, sp)[0, 1])
    ax.text(0.03, 0.97, f"MAE={mae:.2f}°\ncorr={corr:+.3f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, dpi=130)
    print(f"\nMAE={mae:.2f}°  corr={corr:+.3f}")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
