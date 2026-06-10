"""
train_field_model.py — Train a conditional generative model on AT2 damage
fields and save a checkpoint for visualize_crack_bands.py.

Loads a dataset produced by generate_field_dataset(...save_path=...) and trains
either Flow Matching (default) or DDPM. CPU-friendly: set --threads to the
number of cores when running on a frontale node.

Usage
-----
  python train_field_model.py \
      --data research/phasefield/at2_fields_1k.npz \
      --method fm --epochs 500 --threads 24 \
      --out results/field_fm.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from research.phasefield.diffusion_field_ddpm import (
    load_field_dataset, train_flow_matching, train_ddpm,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to dataset .npz")
    ap.add_argument("--method", choices=["fm", "ddpm"], default="fm")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--T", type=int, default=300, help="DDPM steps (ignored for fm)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0=auto)")
    ap.add_argument("--out", required=True, help="output checkpoint path")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
        print(f"torch CPU threads: {args.threads}")

    ds = load_field_dataset(args.data)
    print(f"Dataset: fields={ds.fields.shape}  cond={ds.cond.shape}  "
          f"grid={ds.ny}x{ds.nx}")

    ckpt = {"ny": ds.ny, "nx": ds.nx, "method": args.method}

    if args.method == "fm":
        model = train_flow_matching(ds, epochs=args.epochs,
                                    batch_size=args.batch_size, lr=args.lr)
        ckpt["model"] = model.state_dict()
    else:
        model, sched = train_ddpm(ds, epochs=args.epochs,
                                  batch_size=args.batch_size, lr=args.lr, T=args.T)
        ckpt["model"]       = model.state_dict()
        ckpt["schedule_T"]  = sched.T

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"Saved checkpoint → {args.out}")


if __name__ == "__main__":
    main()
