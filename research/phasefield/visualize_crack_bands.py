"""
visualize_crack_bands.py — Visualise crack-pattern uncertainty from a trained
conditional generative model (Flow Matching or DDPM).

For each (Gc, ell, beta) setting, draws N samples and plots:
  Row 1: a few individual sampled damage fields d(x,y)
  Row 2: pixel-wise mean field  E[d(x,y)]   (expected crack path)
  Row 3: pixel-wise std  field  Std[d(x,y)] (spatial uncertainty band)

This is the headline figure for "diffusion as a stochastic surrogate":
the simulator is deterministic, but the learned model expresses the
distribution of crack patterns consistent with the material parameters.

Usage
-----
  python visualize_crack_bands.py \
      --ckpt results/field_fm.pt --method fm \
      --out  results/crack_bands.png
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
from research.phasefield.diffusion_field_ddpm import (
    FieldDenoiser, DDPMSchedule, sample_fields, sample_fields_fm,
    predict_fields_mse,
)


# Parameter settings to visualise: (label, Gc, ell, beta)
SETTINGS = [
    ("isotropic\nβ=0.0",      8e-4, 0.04,  0.0),
    ("SiC-like\nβ=−0.40",     8e-4, 0.04, -0.40),
    ("strong aniso\nβ=−0.70", 8e-4, 0.04, -0.70),
]


def load_model(ckpt_path: str, device: torch.device
               ) -> tuple[FieldDenoiser, dict]:
    ck    = torch.load(ckpt_path, map_location=device)
    ny    = int(ck.get("ny", 40))
    nx    = int(ck.get("nx", 80))
    model = FieldDenoiser(ny=ny, nx=nx).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def draw(model: FieldDenoiser, ck: dict, method: str, n_samples: int,
         guidance: float, out_path: str, device: torch.device) -> None:
    n_show = 3                              # individual samples per setting
    n_set  = len(SETTINGS)
    fig, axes = plt.subplots(n_show + 2, n_set,
                             figsize=(3.2 * n_set, 2.0 * (n_show + 2)))

    for col, (label, Gc, ell, beta) in enumerate(SETTINGS):
        if method == "fm":
            fields = sample_fields_fm(model, Gc=Gc, ell=ell, beta=beta,
                                      n_samples=n_samples, n_steps=20,
                                      guidance_scale=guidance, device=device)
        elif method == "ddpm":
            sched  = DDPMSchedule(T=int(ck.get("schedule_T", 300)), device=device)
            fields = sample_fields(model, sched, Gc=Gc, ell=ell, beta=beta,
                                   n_samples=n_samples,
                                   guidance_scale=guidance, device=device)
        else:  # mse
            fields = predict_fields_mse(model, Gc=Gc, ell=ell, beta=beta,
                                        n_samples=n_samples, device=device)

        mean = fields.mean(axis=0)
        std  = fields.std(axis=0)

        # Individual samples
        for r in range(n_show):
            ax = axes[r, col]
            ax.imshow(fields[r], cmap="inferno", vmin=0, vmax=1,
                      origin="lower", aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(label, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"sample {r+1}", fontsize=8)

        # Mean field
        ax = axes[n_show, col]
        ax.imshow(mean, cmap="inferno", vmin=0, vmax=1,
                  origin="lower", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("mean E[d]", fontsize=8)

        # Std field (uncertainty band)
        ax = axes[n_show + 1, col]
        im = ax.imshow(std, cmap="viridis", origin="lower", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("std[d]\n(uncertainty)", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    method_name = "Flow Matching" if method == "fm" else "DDPM"
    fig.suptitle(f"Crack-pattern uncertainty — {method_name} "
                 f"(N={n_samples}, w={guidance})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    print(f"Saved figure → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to trained .pt checkpoint")
    ap.add_argument("--method", choices=["fm", "ddpm", "mse"], default="fm")
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--out", default="results/crack_bands.png")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                          else "cuda")
    print(f"Device: {device}  method={args.method}")

    model, ck = load_model(args.ckpt, device)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    draw(model, ck, args.method, args.n_samples, args.guidance, args.out, device)


if __name__ == "__main__":
    main()
