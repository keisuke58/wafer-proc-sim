"""
fmps_multiobs.py — Multi-observation FMPS: identify (Gc, ℓ, β) from crack
deflection angles measured at SEVERAL known cleavage angles θ_cut.

Why
---
fmps_inverse.py showed that a single crack field cannot identify β or θ — the
deflection depends on the β×θ combination, so one observation is degenerate.
But experimentally θ_cut is CONTROLLED (the cut angle relative to the cleavage
plane is known).  Measuring φ at a sweep of known θ_cut traces out the material's
anisotropic response φ(θ; Gc, ℓ, β), which DOES pin down β.

Setup
-----
  observation  s = [φ(θ_1), …, φ(θ_K)]   at fixed THETA_OBS (known, K=5)
  target       θ = (Gc, ℓ, β)             (3-D, the unknown material params)
  model        conditional Flow Matching  p(θ | s)   (amortized posterior)

This is the multi-observation analogue of the thesis Phase-2 identifiability
argument, and the direct replacement for per-observation TMCMC.

Usage
-----
  python fmps_multiobs.py gen   --n-params 800 --workers 22 --out research/phasefield/multiobs.npz
  python fmps_multiobs.py train --data research/phasefield/multiobs.npz --epochs 3000 --out results/fmps_multi.pt
  python fmps_multiobs.py validate --ckpt results/fmps_multi.pt --out results/fmps_multi_posterior.png
"""
from __future__ import annotations
import argparse, math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from research.phasefield.at2_simulator_2d import (
    PFParams2D, SimConfig2D, run_forward_2d, crack_deflection_angle)
from research.phasefield.diffusion_field_ddpm import (
    _GC_LO, _GC_HI, _ELL_LO, _ELL_HI, _BETA_LO, _BETA_HI, FULL_CFG)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THETA_OBS = [0.0, 30.0, 45.0, 60.0, 90.0]    # known cleavage angles (deg)
PARAM_NAMES = ["Gc", "ell", "beta"]


# ── Param normalisation (3-D: Gc, ell, beta) ──────────────────────────────────

def norm3(Gc, ell, beta):
    c0 = (np.log(Gc)  - math.log(_GC_LO))  / (math.log(_GC_HI)  - math.log(_GC_LO))
    c1 = (np.log(ell) - math.log(_ELL_LO)) / (math.log(_ELL_HI) - math.log(_ELL_LO))
    c2 = (beta - _BETA_LO) / (_BETA_HI - _BETA_LO)
    return np.stack([c0, c1, c2], -1).astype(np.float32)


def denorm3(c):
    Gc   = np.exp(c[..., 0] * (math.log(_GC_HI)  - math.log(_GC_LO))  + math.log(_GC_LO))
    ell  = np.exp(c[..., 1] * (math.log(_ELL_HI) - math.log(_ELL_LO)) + math.log(_ELL_LO))
    beta = c[..., 2] * (_BETA_HI - _BETA_LO) + _BETA_LO
    return np.stack([Gc, ell, beta], -1)


# ── Data generation: φ-vector at known θ_cut sweep ────────────────────────────

def _phi_vector(args):
    """Worker: (i, Gc, ell, beta, cfg_dict) → (i, [φ(θ) for θ in THETA_OBS])."""
    i, Gc, ell, beta, cfg_dict = args
    cfg = SimConfig2D(**cfg_dict)
    phis = []
    for th in THETA_OBS:
        p = PFParams2D(Gc=Gc, ell=ell, E=1.0, nu=0.25, beta=beta,
                       theta_cut=math.radians(th))
        r = run_forward_2d(p, cfg)
        phis.append(r.crack_angle)
    return i, np.array(phis, dtype=np.float32)


def generate_multiobs(n_params=800, n_workers=22, seed=0, save_path=None,
                      verbose=True):
    import dataclasses
    cfg = FULL_CFG
    cfg_dict = dataclasses.asdict(cfg)
    try:
        from scipy.stats import qmc
        lhs = qmc.LatinHypercube(d=3, seed=seed).random(n=n_params)
    except ImportError:
        lhs = np.random.default_rng(seed).uniform(size=(n_params, 3))

    Gc   = np.exp(lhs[:, 0] * (math.log(_GC_HI)  - math.log(_GC_LO))  + math.log(_GC_LO))
    ell  = np.exp(lhs[:, 1] * (math.log(_ELL_HI) - math.log(_ELL_LO)) + math.log(_ELL_LO))
    beta = lhs[:, 2] * (_BETA_HI - _BETA_LO) + _BETA_LO

    params = norm3(Gc, ell, beta)
    phi = np.empty((n_params, len(THETA_OBS)), dtype=np.float32)
    args = [(i, float(Gc[i]), float(ell[i]), float(beta[i]), cfg_dict)
            for i in range(n_params)]

    import multiprocessing as mp
    with mp.Pool(n_workers) as pool:
        for done, (i, pv) in enumerate(pool.imap_unordered(_phi_vector, args)):
            phi[i] = pv
            if verbose and (done + 1) % 50 == 0:
                print(f"  [{done+1:>4}/{n_params}] done", flush=True)

    if save_path:
        np.savez_compressed(save_path, params=params, phi=phi,
                            theta_obs=np.array(THETA_OBS, np.float32))
        if verbose:
            print(f"Saved → {save_path}")
    return params, phi


# ── Conditional FM in 3-D param space, conditioned on K-D φ-vector ────────────

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
    def __init__(self, p_dim=3, s_dim=5, t_dim=64, hidden=256):
        super().__init__()
        self.t_emb = SinEmb(t_dim)
        self.net = nn.Sequential(
            nn.Linear(p_dim + t_dim + s_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, p_dim))
    def forward(self, p, t, s):
        return self.net(torch.cat([p, self.t_emb(t), s], -1))


def train(params, phi, epochs=3000, batch_size=128, lr=2e-4, device=None):
    dev = device or DEVICE
    s_mean = phi.mean(0); s_std = phi.std(0) + 1e-6
    P = torch.from_numpy(params).to(dev)
    S = torch.from_numpy(((phi - s_mean) / s_std).astype(np.float32)).to(dev)
    model = ParamVelocity(p_dim=params.shape[1], s_dim=phi.shape[1]).to(dev)
    print(f"ParamVelocity: {sum(p.numel() for p in model.parameters()):,} params "
          f"device={dev}", flush=True)
    dl = DataLoader(TensorDataset(P, S), batch_size=batch_size,
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
            loss = nn.functional.mse_loss(model(pt, t * 1000, s), noise - p0)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * B
        sch.step()
        if (ep + 1) % 300 == 0:
            print(f"  Epoch {ep+1:>4}/{epochs}  loss={tot/len(P):.5f}", flush=True)
    return model, s_mean, s_std


@torch.no_grad()
def sample_posterior(model, s_obs_norm, n=2000, n_steps=50, device=None):
    dev = device or DEVICE
    s = torch.from_numpy(np.asarray(s_obs_norm, np.float32)).to(dev)[None].expand(n, -1)
    x = torch.randn(n, model.net[-1].out_features, device=dev)
    dt = -1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((n,), (1.0 + k * dt) * 1000, device=dev)
        x = x + dt * model(x, t, s)
    return x.clamp(0, 1).cpu().numpy()


# ── Validate: posterior coverage + β-width vs single-obs ──────────────────────

def validate(model, s_mean, s_std, out_path, n_test=6, device=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dev = device or DEVICE
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(n_test, 3, figsize=(10, 2.4 * n_test))
    cover = np.zeros(3); beta_widths = []
    for r in range(n_test):
        true_c = rng.uniform(size=3).astype(np.float32)
        tp = denorm3(true_c)
        # simulate φ-vector at the known θ_cut sweep
        phis = []
        for th in THETA_OBS:
            p = PFParams2D(Gc=float(tp[0]), ell=float(tp[1]), E=1.0, nu=0.25,
                           beta=float(tp[2]), theta_cut=math.radians(th))
            phis.append(run_forward_2d(p, FULL_CFG).crack_angle)
        s_obs = (np.array(phis, np.float32) - s_mean) / s_std
        post = denorm3(sample_posterior(model, s_obs, n=2000, device=dev))
        beta_widths.append(np.diff(np.percentile(post[:, 2], [5, 95]))[0])
        for k in range(3):
            ax = axes[r, k]
            ax.hist(post[:, k], bins=40, color="C2", alpha=0.7, density=True)
            ax.axvline(tp[k], color="r", lw=2)
            lo, hi = np.percentile(post[:, k], [5, 95])
            if lo <= tp[k] <= hi:
                cover[k] += 1
            if r == 0:
                ax.set_title(PARAM_NAMES[k])
            ax.set_yticks([])
    cover /= n_test
    bw = float(np.mean(beta_widths)) / (_BETA_HI - _BETA_LO)   # fraction of prior
    fig.suptitle("Multi-obs FMPS p(Gc,ℓ,β | φ-sweep) — red=truth  "
                 f"(90% cov: " + ", ".join(f"{PARAM_NAMES[k]}={cover[k]*100:.0f}%"
                 for k in range(3)) + f";  β 90%-width = {bw*100:.0f}% of prior)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print("90% coverage:", {PARAM_NAMES[k]: f"{cover[k]*100:.0f}%" for k in range(3)})
    print(f"β posterior 90%-width: {bw*100:.0f}% of prior "
          f"(single-obs was ~near 100% = uninformative)")
    print(f"Saved → {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--n-params", type=int, default=800)
    g.add_argument("--workers", type=int, default=22)
    g.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True); t.add_argument("--epochs", type=int, default=3000)
    t.add_argument("--threads", type=int, default=0); t.add_argument("--out", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--ckpt", required=True)
    v.add_argument("--out", default="results/fmps_multi_posterior.png")
    v.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if args.cmd == "gen":
        generate_multiobs(args.n_params, n_workers=args.workers, save_path=args.out)
    elif args.cmd == "train":
        if args.threads > 0:
            torch.set_num_threads(args.threads)
        d = np.load(args.data)
        model, s_mean, s_std = train(d["params"], d["phi"], epochs=args.epochs)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save({"model": model.state_dict(), "s_mean": s_mean, "s_std": s_std},
                   args.out)
        print(f"Saved → {args.out}")
    else:
        dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
        ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
        model = ParamVelocity(p_dim=3, s_dim=len(THETA_OBS)).to(dev)
        model.load_state_dict(ck["model"]); model.eval()
        validate(model, ck["s_mean"], ck["s_std"], args.out, device=dev)


if __name__ == "__main__":
    main()
