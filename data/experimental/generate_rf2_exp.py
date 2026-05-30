"""
Generate synthetic "experimental" RF2 dataset for SiC wafer dicing.

Simulates what a laboratory measurement would look like:
  - RF2 (thrust force) measured at 5 cut depths
  - Multiple repeat measurements per depth (3 reps) → mean + std
  - Realistic noise: 2–5% CV (coefficient of variation)

Ground-truth material params (matching Disco Micro2026 SiC blade):
  Gc = 17.5 J/m²,  dp_cohesion = 1.0 GPa,  eps_fracture_scale = 25

Output:
    data/experimental/rf2_exp_sic.csv

Usage:
    python data/experimental/generate_rf2_exp.py
    python data/experimental/generate_rf2_exp.py --noise-cv 0.03 --seed 7
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from optimization.bayesian_opt_gc_fit import _physics_rf2, CUT_DEPTHS_UM

OUT_DIR = os.path.dirname(__file__)

# Ground-truth params (SiC Micro2026 reference)
TRUE_PARAMS = {
    "Gc":                17.5,   # J/m²
    "dp_cohesion_GPa":    1.0,   # GPa
    "eps_fracture_scale": 25.0,
}


def generate(noise_cv: float = 0.03, n_reps: int = 3, seed: int = 42) -> pd.DataFrame:
    rng      = np.random.default_rng(seed)
    rf2_true = _physics_rf2(CUT_DEPTHS_UM, **TRUE_PARAMS)

    rows = []
    for depth, rf2 in zip(CUT_DEPTHS_UM, rf2_true):
        sigma  = rf2 * noise_cv
        reps   = rf2 + rng.normal(0, sigma, size=n_reps)
        rows.append({
            "cut_depth_um":   depth,
            "RF2_peak_N":     round(float(reps.mean()), 2),
            "RF2_std_N":      round(float(reps.std()), 2),
            "n_reps":         n_reps,
            "source":         "synthetic_Micro2026",
        })

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-cv", type=float, default=0.03,
                        help="Coefficient of variation for measurement noise (default 0.03)")
    parser.add_argument("--n-reps",   type=int,   default=3)
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()

    df   = generate(noise_cv=args.noise_cv, n_reps=args.n_reps, seed=args.seed)
    path = os.path.join(OUT_DIR, "rf2_exp_sic.csv")
    df.to_csv(path, index=False)

    print(f"[✓] Generated experimental RF2 data → {path}")
    print(f"\n  True params: {TRUE_PARAMS}")
    print(f"  Noise CV: {args.noise_cv*100:.1f}%  |  reps: {args.n_reps}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
