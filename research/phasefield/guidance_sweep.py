"""
guidance_sweep.py — Tune Flow-Matching inference (guidance scale, #steps) to
best reproduce the simulator's phi(theta) deflection law.

Reuses a trained FM checkpoint (no retraining).  For each (w, steps) it samples
crack fields across theta, measures phi with the (adequate) original estimator,
and reports correlation / MAE vs the AT2 simulator, plus the high-theta bias
that motivated this sweep.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from research.phasefield.at2_simulator_2d import (
    PFParams2D, run_forward_2d, crack_deflection_angle)
from research.phasefield.diffusion_field_ddpm import (
    FieldDenoiser, sample_fields_fm, FULL_CFG)

GC, ELL, BETA = 8e-4, 0.04, -0.40
THETA_GRID = [0, 15, 30, 45, 60, 75, 90]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    ck = torch.load(args.ckpt, map_location=dev)
    model = FieldDenoiser(ny=int(ck.get("ny", 40)), nx=int(ck.get("nx", 80))).to(dev)
    model.load_state_dict(ck["model"]); model.eval()

    # Simulator ground truth
    sim = []
    for th in THETA_GRID:
        p = PFParams2D(Gc=GC, ell=ELL, E=1.0, nu=0.25, beta=BETA,
                       theta_cut=np.radians(th))
        sim.append(run_forward_2d(p, FULL_CFG).crack_angle)
    sim = np.array(sim)

    print(f"{'w':>5} {'steps':>6} {'corr':>7} {'MAE':>6} {'hi-θ bias':>10}")
    best = None
    for steps in (20, 50):
        for w in (1.0, 3.0, 5.0, 8.0):
            gen = []
            for th in THETA_GRID:
                f = sample_fields_fm(model, Gc=GC, ell=ELL, beta=BETA, theta=float(th),
                                     n_samples=args.n_samples, n_steps=steps,
                                     guidance_scale=w, device=dev)
                a = np.array([crack_deflection_angle(x, FULL_CFG) for x in f])
                a = a[np.isfinite(a)]
                gen.append(a.mean())
            gen = np.array(gen)
            corr = float(np.corrcoef(gen, sim)[0, 1])
            mae = float(np.mean(np.abs(gen - sim)))
            hi_bias = float(np.mean((gen - sim)[-3:]))   # θ=60,75,90 mean bias
            print(f"{w:>5.1f} {steps:>6} {corr:>7.3f} {mae:>6.2f} {hi_bias:>10.2f}")
            score = corr - 0.1 * mae
            if best is None or score > best[0]:
                best = (score, w, steps, corr, mae, hi_bias)

    print(f"\nbest: w={best[1]} steps={best[2]}  corr={best[3]:.3f} "
          f"MAE={best[4]:.2f} hi-θ bias={best[5]:+.2f}")


if __name__ == "__main__":
    main()
