"""
GPU training script for AT2 surrogate.

Runs on vancouver/stuttgart GPU nodes (JAX with CUDA backend).
Generates 2000-sample training set, trains 256-256-128 MLP for 15k epochs,
saves weights + normalization to results/surrogate_gpu.npz.

Usage (on GPU server):
    cd /home/nishioka/git/wafer-proc-sim
    CUDA_VISIBLE_DEVICES=0 \
    LD_LIBRARY_PATH=/home/nishioka/miniconda3/lib:$LD_LIBRARY_PATH \
    python research/phasefield/train_surrogate_gpu.py [--samples 2000] [--epochs 15000]
"""
from __future__ import annotations
import argparse, time, sys, os
import numpy as np

# Print device info before importing JAX (lets us diagnose CPU fallback)
print(f"[env] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','(unset)')}")
print(f"[env] JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS','(unset)')}")

# ── JAX import and device check ───────────────────────────────────────────────
import jax
import jax.numpy as jnp
import optax

devices = jax.devices()
print(f"[jax] devices: {devices}")
backend = devices[0].platform
if backend != "gpu":
    print(f"[WARN] JAX is using {backend}, not GPU. "
          f"Set CUDA_VISIBLE_DEVICES or install jax[cuda12].")

# ── AT2 simulator (pure NumPy, runs on CPU for data gen) ─────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from research.phasefield.at2_simulator import PFParams, SimConfig, run_forward
from research.phasefield.surrogate import (
    AT2Surrogate, generate_training_data, TrainingData,
)

# ─────────────────────────────────────────────────────────────────────────────
#  CLI args
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--samples",  type=int,   default=2000)
parser.add_argument("--epochs",   type=int,   default=15000)
parser.add_argument("--hidden",   type=str,   default="256,256,128")
parser.add_argument("--lr",       type=float, default=3e-3)
parser.add_argument("--batch",    type=int,   default=128)
parser.add_argument("--seed",     type=int,   default=0)
parser.add_argument("--out",      type=str,   default="results/surrogate_gpu.npz")
parser.add_argument("--cfg_n",    type=int,   default=200,
                    help="n_steps for training data simulator (higher=finer init_U)")
args = parser.parse_args()

hidden = [int(x) for x in args.hidden.split(",")]
print(f"\n[config] samples={args.samples}  epochs={args.epochs}  "
      f"hidden={hidden}  lr={args.lr}  batch={args.batch}  "
      f"cfg_n_steps={args.cfg_n}")

# ─────────────────────────────────────────────────────────────────────────────
#  1. Generate training data  (CPU — pure NumPy)
# ─────────────────────────────────────────────────────────────────────────────

cfg = SimConfig(n_elem=200, n_steps=args.cfg_n, u_max_factor=5.0)

print(f"\n[data] Generating {args.samples} AT2 samples (n_steps={args.cfg_n}) ...")
t0 = time.perf_counter()
data = generate_training_data(args.samples, cfg=cfg, seed=args.seed, verbose=True)
dt = time.perf_counter() - t0
print(f"[data] Done in {dt:.1f}s  "
      f"init_U: min={data.init_U.min():.4f}  max={data.init_U.max():.4f}  "
      f"mean={data.init_U.mean():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
#  2. Train on GPU
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[train] Training MLP {hidden} on {devices[0]} ...")
surr = AT2Surrogate(hidden=hidden)

t0 = time.perf_counter()
loss_hist = surr.fit(
    data,
    n_epochs=args.epochs,
    lr=args.lr,
    batch_size=args.batch,
    seed=args.seed,
    verbose=True,
)
dt = time.perf_counter() - t0
print(f"[train] Done in {dt:.1f}s  "
      f"final_loss={loss_hist[-1]:.6f}  best_loss={min(loss_hist):.6f}")

# ─────────────────────────────────────────────────────────────────────────────
#  3. Evaluate on holdout
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[eval] Holdout accuracy ...")
holdout = generate_training_data(200, cfg=cfg, seed=args.seed + 999, verbose=False)
pred = surr.predict_batch(np.exp(holdout.log_Gc), np.exp(holdout.log_ell))
rel_err = np.abs(pred - holdout.init_U) / (holdout.init_U + 1e-12)
print(f"  mean rel error : {rel_err.mean()*100:.2f}%")
print(f"  max  rel error : {rel_err.max()*100:.2f}%")
print(f"  p95  rel error : {np.percentile(rel_err, 95)*100:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
#  4. Save weights and normalization stats
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

# Flatten JAX pytree to numpy arrays for portability
flat_weights = {}
for i, layer in enumerate(surr._mlp):
    flat_weights[f"layer_{i}_W"] = np.array(layer["W"])
    flat_weights[f"layer_{i}_b"] = np.array(layer["b"])

np.savez(
    args.out,
    **flat_weights,
    X_mean=surr._X_mean,
    X_std=surr._X_std,
    y_mean=np.array([surr._y_mean]),
    y_std=np.array([surr._y_std]),
    hidden=np.array(hidden),
    n_layers=np.array([len(surr._mlp)]),
    loss_hist=np.array(loss_hist),
    mean_rel_err=np.array([rel_err.mean()]),
    cfg_n_steps=np.array([args.cfg_n]),
    n_samples=np.array([args.samples]),
)
print(f"\n[save] Saved to {args.out}  ({os.path.getsize(args.out)/1024:.1f} KB)")
print("[done] ")
