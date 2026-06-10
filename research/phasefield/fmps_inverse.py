"""
fmps_inverse.py — Flow-Matching Posterior Sampler for fracture-parameter
identification (Idea A: replace TMCMC with an amortized posterior).

Forward problem (already solved by diffusion_field_ddpm):
    params θ=(Gc, ℓ, β, θ_cut)  →  crack field d(x,y)

Inverse problem (here):
    observed field  →  posterior  p(θ | observation)

Method — Flow-Matching Posterior Estimation (Dax et al. 2023, "Flow Matching for
Scalable Simulation-Based Inference"):
    Train a conditional flow whose TARGET is the 4-D parameter vector and whose
    CONDITION is a low-dimensional summary s(d) of the observed crack field:
        s(d) = [deflection_angle, cracked_fraction, d_mean, centroid_x]
    Straight-line FM path in parameter space:
        θ_t = (1-t)·θ_data + t·noise,   target velocity = noise − θ_data
    At inference, integrate the ODE from noise back to t=0 conditioned on s(d_obs)
    → samples from the amortized posterior p(θ | d_obs).

Unlike TMCMC this is AMORTIZED: train once, then posterior sampling for any new
observation is a handful of network evaluations (no per-observation MCMC).

Usage
-----
  # train
  python fmps_inverse.py train --data research/phasefield/at2_fields_1k.npz \
      --epochs 2000 --out results/fmps.pt
  # validate posterior calibration on held-out simulations
  python fmps_inverse.py validate --ckpt results/fmps.pt \
      --out results/fmps_posterior.png
"""
from __future__ import annotations
import argparse, math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from research.phasefield.diffusion_field_ddpm import (
    load_field_dataset, _x_to_field, _norm_cond,
    _GC_LO, _GC_HI, _ELL_LO, _ELL_HI, _BETA_LO, _BETA_HI, _THETA_LO, _THETA_HI,
    FULL_CFG)
from research.phasefield.crack_metrics import crack_summary

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PARAM_NAMES = ["Gc", "ell", "beta", "theta"]


# ── Summary-statistic conditioning ────────────────────────────────────────────

def fields_to_summaries(ds, cfg=FULL_CFG) -> np.ndarray:
    """Compute s(d) for every field in the dataset → (N, 4), then z-normalise."""
    S = np.stack([crack_summary(_x_to_field(ds.fields[i, 0]), cfg)
                  for i in range(ds.fields.shape[0])]).astype(np.float32)
    return S


# ── Tiny conditional FM in 4-D parameter space (MLP velocity field) ───────────

class SinEmb(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        half = dim // 2
        self.register_buffer("freqs",
            torch.exp(-math.log(10000) * torch.arange(half).float() / half))
    def forward(self, t):
        e = t.float()[:, None] * self.freqs[None]
        return torch.cat([e.sin(), e.cos()], -1)


class ParamVelocity(nn.Module):
    """v_θ(θ_t, t, s) for 4-D params conditioned on summary s (dim=4)."""
    def __init__(self, p_dim=4, s_dim=4, t_dim=64, hidden=256):
        super().__init__()
        self.t_emb = SinEmb(t_dim)
        self.net = nn.Sequential(
            nn.Linear(p_dim + t_dim + s_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, p_dim))
    def forward(self, p, t, s):
        return self.net(torch.cat([p, self.t_emb(t), s], -1))


# ── Train ──────────────────────────────────────────────────────────────────────

def train(ds, epochs=2000, batch_size=128, lr=2e-4, device=None):
    dev = device or DEVICE
    params = torch.from_numpy(ds.cond.astype(np.float32)).to(dev)   # (N,4) in [0,1]
    S_np = fields_to_summaries(ds)
    s_mean = S_np.mean(0); s_std = S_np.std(0) + 1e-6
    S = torch.from_numpy((S_np - s_mean) / s_std).to(dev)            # z-normalised

    model = ParamVelocity().to(dev)
    print(f"ParamVelocity: {sum(p.numel() for p in model.parameters()):,} params "
          f"device={dev}")
    dl = DataLoader(TensorDataset(params, S), batch_size=batch_size,
                    shuffle=True, drop_last=True)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        model.train(); tot = 0.0
        for p0, s in dl:
            B = p0.shape[0]
            t = torch.rand(B, device=dev)
            noise = torch.randn_like(p0)
            pt = (1 - t[:, None]) * p0 + t[:, None] * noise
            target = noise - p0
            loss = nn.functional.mse_loss(model(pt, t * 1000, s), target)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * B
        sch.step()
        if (ep + 1) % 200 == 0:
            print(f"  Epoch {ep+1:>4}/{epochs}  loss={tot/len(params):.5f}")
    return model, s_mean, s_std


@torch.no_grad()
def sample_posterior(model, s_obs_norm, n=2000, n_steps=50, device=None):
    """Sample n parameter vectors from p(θ | s_obs).  Returns (n,4) in [0,1]."""
    dev = device or DEVICE
    s = torch.from_numpy(np.asarray(s_obs_norm, np.float32)).to(dev)[None].expand(n, -1)
    x = torch.randn(n, 4, device=dev)
    dt = -1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((n,), (1.0 + k * dt) * 1000, device=dev)
        x = x + dt * model(x, t, s)
    return x.clamp(0, 1).cpu().numpy()


# ── Denormalise params [0,1] → physical ───────────────────────────────────────

def denorm_params(c):
    Gc   = np.exp(c[..., 0] * (math.log(_GC_HI)  - math.log(_GC_LO))  + math.log(_GC_LO))
    ell  = np.exp(c[..., 1] * (math.log(_ELL_HI) - math.log(_ELL_LO)) + math.log(_ELL_LO))
    beta = c[..., 2] * (_BETA_HI - _BETA_LO) + _BETA_LO
    th   = c[..., 3] * (_THETA_HI - _THETA_LO) + _THETA_LO
    return np.stack([Gc, ell, beta, th], -1)


# ── Validate posterior calibration on held-out simulations ────────────────────

def validate(model, s_mean, s_std, out_path, n_test=6, device=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from research.phasefield.at2_simulator_2d import PFParams2D, run_forward_2d
    dev = device or DEVICE

    rng = np.random.default_rng(123)
    fig, axes = plt.subplots(n_test, 4, figsize=(13, 2.4 * n_test))
    cover = np.zeros(4)
    for r in range(n_test):
        # draw a random TRUE param set, simulate, summarise, infer posterior
        u = rng.uniform(size=4)
        true_c = u.astype(np.float32)
        tp = denorm_params(true_c)
        p = PFParams2D(Gc=float(tp[0]), ell=float(tp[1]), E=1.0, nu=0.25,
                       beta=float(tp[2]), theta_cut=math.radians(float(tp[3])))
        res = run_forward_2d(p, FULL_CFG)
        s_obs = (crack_summary(res.d_field, FULL_CFG) - s_mean) / s_std
        post = sample_posterior(model, s_obs, n=2000, device=dev)   # (2000,4) [0,1]
        post_phys = denorm_params(post)
        true_phys = tp
        for k in range(4):
            ax = axes[r, k]
            ax.hist(post_phys[:, k], bins=40, color="C0", alpha=0.7, density=True)
            ax.axvline(true_phys[k], color="r", lw=2)
            lo, hi = np.percentile(post_phys[:, k], [5, 95])
            if lo <= true_phys[k] <= hi:
                cover[k] += 1
            if r == 0:
                ax.set_title(PARAM_NAMES[k])
            ax.set_yticks([])
    cover /= n_test
    fig.suptitle("FMPS posterior p(θ|obs) — red=truth, blue=posterior  "
                 f"(90% coverage: " + ", ".join(
                     f"{PARAM_NAMES[k]}={cover[k]*100:.0f}%" for k in range(4)) + ")",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print("90% coverage per param:",
          {PARAM_NAMES[k]: f"{cover[k]*100:.0f}%" for k in range(4)})
    print(f"Saved → {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True); t.add_argument("--epochs", type=int, default=2000)
    t.add_argument("--threads", type=int, default=0); t.add_argument("--out", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--ckpt", required=True); v.add_argument("--out", default="results/fmps_posterior.png")
    v.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if args.cmd == "train":
        if args.threads > 0:
            torch.set_num_threads(args.threads)
        ds = load_field_dataset(args.data)
        model, s_mean, s_std = train(ds, epochs=args.epochs)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save({"model": model.state_dict(), "s_mean": s_mean, "s_std": s_std},
                   args.out)
        print(f"Saved → {args.out}")
    else:
        dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
        ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
        model = ParamVelocity().to(dev); model.load_state_dict(ck["model"]); model.eval()
        validate(model, ck["s_mean"], ck["s_std"], args.out, device=dev)


if __name__ == "__main__":
    main()
