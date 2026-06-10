"""
diffusion_field_ddpm.py — Conditional DDPM for AT2 damage fields.

Learns p(d(x,y) | Gc, ℓ, β) via denoising diffusion so that diverse crack
patterns — including their spatial uncertainty — can be sampled for any
material / fracture parameter triplet in the training range.

The AT2 simulator (at2_simulator_2d.py) provides training data: 2D damage
fields d(x,y) ∈ [0,1] on a (ny=40, nx=80) grid given (Gc, ℓ, β).

Architecture
------------
3-level UNet denoiser (B, 1, 40, 80):
  Stem:    Conv2d(1 → 16)
  Down 1:  stride-2 conv → (32, 20, 40) + ResBlock
  Down 2:  stride-2 conv → (64, 10, 20) + ResBlock
  Down 3:  stride-2 conv → (128, 5, 10) + ResBlock (bottleneck)
  Up 1:    ConvTranspose2d + skip + ResBlock → (64, 10, 20)
  Up 2:    ConvTranspose2d + skip + ResBlock → (32, 20, 40)
  Up 3:    ConvTranspose2d + skip + ResBlock → (16, 40, 80)
  Out:     Conv2d(16 → 1)
Time and condition are jointly embedded (128-D) and injected into every
ResBlock as an additive bias.

Guidance
--------
Classifier-free (Ho & Salimans 2022): cond dropped with prob 0.2 during
training; at inference ε̂ = ε̂_uncond + w·(ε̂_cond − ε̂_uncond).

Typical workflow
----------------
  # 1. Generate dataset (one-time, ~3 min for 200 samples on CPU):
  ds = generate_field_dataset(200, save_path="at2_fields.npz")

  # 2. Train (~5 min GPU / ~1 h CPU for 300 epochs):
  model, sched = train_ddpm(ds, epochs=300)
  torch.save({"model": model.state_dict(), "ny": ds.ny, "nx": ds.nx},
             "field_ddpm.pt")

  # 3. Sample conditioned on SiC fracture params:
  fields = sample_fields(model, sched, Gc=8e-4, ell=0.04, beta=-0.40,
                         n_samples=16)   # (16, 40, 80) in [0, 1]

References
----------
Ho et al. (2020) Denoising Diffusion Probabilistic Models
Ho & Salimans (2022) Classifier-Free Diffusion Guidance
Ronneberger et al. (2015) U-Net
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# AT2 simulator lives two levels up in the package hierarchy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from research.phasefield.at2_simulator_2d import PFParams2D, SimConfig2D, run_forward_2d

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Physical parameter ranges ─────────────────────────────────────────────────
_GC_LO,   _GC_HI   = 2e-4, 3e-3   # fracture energy [N/m]
_ELL_LO,  _ELL_HI  = 0.02, 0.10   # length scale [m]
_BETA_LO, _BETA_HI = -0.8,  0.8   # anisotropy coefficient

_E_FIXED  = 1.0
_NU_FIXED = 0.25

# Both configs produce (ny=40, nx=80) fields — divisible by 8 — so the same
# UNet architecture works for both.  FAST_CFG uses fewer load steps.
FAST_CFG = SimConfig2D(nx=80, ny=40, n_steps=60)   # ~300 ms/sim on CPU
FULL_CFG = SimConfig2D(nx=80, ny=40, n_steps=100)  # ~500 ms/sim on CPU


# ── Normalisation ─────────────────────────────────────────────────────────────

def _norm_cond(Gc: np.ndarray, ell: np.ndarray,
               beta: np.ndarray) -> np.ndarray:
    """Map (Gc, ell, beta) → c ∈ [0,1]³  (log-scale for Gc, ell)."""
    lg = math.log(_GC_HI)  - math.log(_GC_LO)
    le = math.log(_ELL_HI) - math.log(_ELL_LO)
    c0 = (np.log(Gc)  - math.log(_GC_LO))  / lg
    c1 = (np.log(ell) - math.log(_ELL_LO)) / le
    c2 = (beta - _BETA_LO) / (_BETA_HI - _BETA_LO)
    return np.stack([c0, c1, c2], axis=-1).astype(np.float32)


def _field_to_x(d: np.ndarray) -> np.ndarray:
    """d ∈ [0, 1]  →  x ∈ [-1, 1]  (DDPM convention)."""
    return (2.0 * d - 1.0).astype(np.float32)


def _x_to_field(x: np.ndarray) -> np.ndarray:
    """x ∈ [-1, 1]  →  d ∈ [0, 1]."""
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0)


# ── Dataset ───────────────────────────────────────────────────────────────────

@dataclass
class AT2FieldDataset:
    fields: np.ndarray   # (N, 1, ny, nx)  in [-1, 1]
    cond:   np.ndarray   # (N, 3)          normalised [log_Gc_n, log_ell_n, beta_n]
    ny:     int
    nx:     int


def _run_one(args: tuple) -> tuple[int, np.ndarray, float]:
    """Worker for multiprocessing: (index, params_tuple, cfg) → (i, d_field, crack_angle)."""
    i, Gc, ell, beta, cfg_dict = args
    cfg = SimConfig2D(**cfg_dict)
    params = PFParams2D(Gc=Gc, ell=ell, E=_E_FIXED, nu=_NU_FIXED, beta=beta)
    res = run_forward_2d(params, cfg)
    return i, res.d_field, res.crack_angle


def generate_field_dataset(
    n_samples:  int = 1000,
    cfg:        SimConfig2D | None = None,
    save_path:  str | None = None,
    seed:       int = 0,
    n_workers:  int = 1,
    verbose:    bool = True,
) -> AT2FieldDataset:
    """
    Run AT2 forward simulations on a Latin-hypercube of (Gc, ell, beta) params.

    Parameters
    ----------
    n_samples : number of simulations (≥200 recommended for training)
    cfg       : SimConfig2D instance; defaults to FULL_CFG (40×80 grid)
    save_path : if given, saves compressed npz for later re-use
    seed      : RNG seed for reproducible LHS sampling
    n_workers : parallel workers via multiprocessing.Pool (1 = serial)
    """
    import dataclasses
    cfg = cfg or FULL_CFG
    ny, nx = cfg.ny, cfg.nx
    cfg_dict = dataclasses.asdict(cfg)

    try:
        from scipy.stats import qmc
        lhs = qmc.LatinHypercube(d=3, seed=seed).random(n=n_samples)
    except ImportError:
        lhs = np.random.default_rng(seed).uniform(size=(n_samples, 3))

    lg = math.log(_GC_HI)  - math.log(_GC_LO)
    le = math.log(_ELL_HI) - math.log(_ELL_LO)
    Gc_arr   = np.exp(lhs[:, 0] * lg + math.log(_GC_LO))
    ell_arr  = np.exp(lhs[:, 1] * le + math.log(_ELL_LO))
    beta_arr = lhs[:, 2] * (_BETA_HI - _BETA_LO) + _BETA_LO

    fields = np.empty((n_samples, 1, ny, nx), dtype=np.float32)
    cond   = _norm_cond(Gc_arr, ell_arr, beta_arr)
    angles = np.empty(n_samples, dtype=np.float32)

    args = [(i, float(Gc_arr[i]), float(ell_arr[i]), float(beta_arr[i]), cfg_dict)
            for i in range(n_samples)]

    if n_workers > 1:
        import multiprocessing as mp
        if verbose:
            print(f"  Launching {n_workers} workers for {n_samples} simulations ...")
        with mp.Pool(n_workers) as pool:
            for done, (i, d_field, phi) in enumerate(pool.imap_unordered(_run_one, args)):
                fields[i, 0] = _field_to_x(d_field)
                angles[i]    = phi
                if verbose and (done + 1) % 50 == 0:
                    print(f"  [{done+1:>4}/{n_samples}] completed")
    else:
        for i, Gc, ell, beta, _ in args:
            params = PFParams2D(Gc=Gc, ell=ell, E=_E_FIXED, nu=_NU_FIXED, beta=beta)
            res = run_forward_2d(params, cfg)
            fields[i, 0] = _field_to_x(res.d_field)
            angles[i]    = res.crack_angle
            if verbose and (i + 1) % 50 == 0:
                print(f"  [{i+1:>4}/{n_samples}]  Gc={Gc_arr[i]:.1e}  "
                      f"ell={ell_arr[i]:.3f}  beta={beta_arr[i]:+.2f}  "
                      f"phi={res.crack_angle:+.1f}°")

    ds = AT2FieldDataset(fields=fields, cond=cond, ny=ny, nx=nx)
    if save_path:
        np.savez_compressed(save_path, fields=fields, cond=cond,
                            angles=angles, ny=np.int32(ny), nx=np.int32(nx))
        if verbose:
            print(f"Saved → {save_path}")
    return ds


def load_field_dataset(path: str) -> AT2FieldDataset:
    d = np.load(path)
    return AT2FieldDataset(fields=d["fields"], cond=d["cond"],
                           ny=int(d["ny"]), nx=int(d["nx"]))


# ── DDPM schedule ─────────────────────────────────────────────────────────────

class DDPMSchedule:
    def __init__(self, T: int = 300, beta_min: float = 1e-4,
                 beta_max: float = 0.02, device: torch.device | None = None):
        dev = device or DEVICE
        self.T = T
        beta      = torch.linspace(beta_min, beta_max, T, device=dev)
        alpha     = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.beta              = beta
        self.alpha             = alpha
        self.alpha_bar         = alpha_bar
        self.sqrt_ab           = alpha_bar.sqrt()
        self.sqrt_one_minus_ab = (1.0 - alpha_bar).sqrt()

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 eps: torch.Tensor) -> torch.Tensor:
        """Forward: x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε."""
        ab  = self.sqrt_ab[t][:, None, None, None]
        mab = self.sqrt_one_minus_ab[t][:, None, None, None]
        return ab * x0 + mab * eps

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: int,
                 cond: torch.Tensor, w: float = 3.0) -> torch.Tensor:
        """Reverse step with classifier-free guidance (guidance scale w)."""
        t_vec  = torch.full((x_t.shape[0],), t, dtype=torch.long,
                            device=x_t.device)
        eps_c  = model(x_t, t_vec, cond)
        eps_u  = model(x_t, t_vec, torch.zeros_like(cond))
        eps    = eps_u + w * (eps_c - eps_u)

        beta  = self.beta[t]
        alpha = self.alpha[t]
        ab    = self.alpha_bar[t]
        mean  = (x_t - (1.0 - alpha) / (1.0 - ab).sqrt() * eps) / alpha.sqrt()
        noise = beta.sqrt() * torch.randn_like(x_t) if t > 0 else 0.0
        return mean + noise


# ── Time embedding ─────────────────────────────────────────────────────────────

class SinusoidalEmb(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / half)
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = t.float()[:, None] * self.freqs[None]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.mlp(emb)


# ── UNet building block ────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Conv residual block with additive time+cond embedding injection."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        g_in  = min(8, in_ch)
        g_out = min(8, out_ch)
        self.conv1    = nn.Sequential(
            nn.GroupNorm(g_in, in_ch), nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.conv2    = nn.Sequential(
            nn.GroupNorm(g_out, out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = (nn.Conv2d(in_ch, out_ch, 1)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x) + self.emb_proj(emb)[:, :, None, None]
        return self.conv2(h) + self.skip(x)


# ── Field denoiser UNet ────────────────────────────────────────────────────────

class FieldDenoiser(nn.Module):
    """
    3-level UNet denoiser for AT2 damage fields.

    Input  : (B, 1, ny, nx)  noisy damage field x_t ∈ [-1, 1]
    Cond   : (B, 3)          normalised [log_Gc_n, log_ell_n, beta_n];
                              pass zero vector for unconditional forward pass
    Output : (B, 1, ny, nx)  predicted noise ε̂  (DDPM) or velocity v̂  (Flow Matching)

    Spatial sizes at each level (default ny=40, nx=80):
      s0: (16, 40, 80)   s1: (32, 20, 40)   s2: (64, 10, 20)
      bottleneck: (128, 5, 10)

    ny and nx must each be divisible by 8 — satisfied by both FAST_CFG and
    FULL_CFG (which both use ny=40, nx=80).
    """

    def __init__(self, ny: int = 40, nx: int = 80,
                 ch: tuple[int, ...] = (16, 32, 64, 128),
                 emb_dim: int = 128):
        super().__init__()
        assert ny % 8 == 0 and nx % 8 == 0, \
            f"ny={ny}, nx={nx} must both be divisible by 8"
        self.ny, self.nx = ny, nx
        c0, c1, c2, c3 = ch

        # Embeddings: time (sinusoidal) + cond (MLP) → emb_dim each, summed
        self.t_emb = SinusoidalEmb(emb_dim)
        self.c_emb = nn.Sequential(
            nn.Linear(3, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # Encoder
        self.stem    = nn.Conv2d(1, c0, 3, padding=1)
        self.d1_conv = nn.Conv2d(c0, c1, 3, stride=2, padding=1)
        self.d1_res  = ResBlock(c1, c1, emb_dim)
        self.d2_conv = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.d2_res  = ResBlock(c2, c2, emb_dim)
        self.d3_conv = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.bot     = ResBlock(c3, c3, emb_dim)

        # Decoder — skip connections double in_ch before each ResBlock
        self.u1_up  = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.u1_res = ResBlock(c2 + c2, c2, emb_dim)
        self.u2_up  = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.u2_res = ResBlock(c1 + c1, c1, emb_dim)
        self.u3_up  = nn.ConvTranspose2d(c1, c0, 2, stride=2)
        self.u3_res = ResBlock(c0 + c0, c0, emb_dim)

        self.out_head = nn.Sequential(
            nn.GroupNorm(min(8, c0), c0), nn.SiLU(),
            nn.Conv2d(c0, 1, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        emb = self.t_emb(t) + self.c_emb(cond)

        s0 = self.stem(x)                               # (B, c0, ny,   nx  )
        s1 = self.d1_res(self.d1_conv(s0), emb)        # (B, c1, ny/2, nx/2)
        s2 = self.d2_res(self.d2_conv(s1), emb)        # (B, c2, ny/4, nx/4)
        h  = self.bot(self.d3_conv(s2), emb)           # (B, c3, ny/8, nx/8)

        h = self.u1_res(torch.cat([self.u1_up(h), s2], 1), emb)
        h = self.u2_res(torch.cat([self.u2_up(h), s1], 1), emb)
        h = self.u3_res(torch.cat([self.u3_up(h), s0], 1), emb)

        return self.out_head(h)                         # (B, 1, ny,   nx  )


# ── Training ───────────────────────────────────────────────────────────────────

def train_ddpm(
    dataset:        AT2FieldDataset,
    epochs:         int   = 500,
    batch_size:     int   = 32,
    lr:             float = 1e-4,
    T:              int   = 300,
    cond_drop_rate: float = 0.20,
    device:         torch.device | None = None,
) -> tuple[FieldDenoiser, DDPMSchedule]:
    """
    Train a DDPM on a pre-generated AT2FieldDataset.

    Returns (model, schedule) ready for sample_fields().
    Prints loss every 50 epochs.
    """
    dev   = device or DEVICE
    sched = DDPMSchedule(T=T, device=dev)
    model = FieldDenoiser(ny=dataset.ny, nx=dataset.nx).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FieldDenoiser: {n_params:,} parameters  device={dev}")

    fields = torch.from_numpy(dataset.fields).to(dev)
    cond   = torch.from_numpy(dataset.cond).to(dev)

    dl = DataLoader(TensorDataset(fields, cond),
                    batch_size=batch_size, shuffle=True, drop_last=True)
    opt      = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for x0, c in dl:
            t   = torch.randint(0, T, (x0.shape[0],), device=dev)
            eps = torch.randn_like(x0)
            x_t = sched.q_sample(x0, t, eps)

            keep = (torch.rand(c.shape[0], 1, device=dev) > cond_drop_rate)
            pred = model(x_t, t, c * keep.float())
            loss = nn.functional.mse_loss(pred, eps)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * x0.shape[0]

        lr_sched.step()
        if (ep + 1) % 50 == 0:
            print(f"  Epoch {ep+1:>4}/{epochs}  "
                  f"loss={ep_loss / len(dataset.fields):.5f}")

    return model, sched


# ── Sampling ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_fields(
    model:          FieldDenoiser,
    sched:          DDPMSchedule,
    Gc:             float,
    ell:            float,
    beta:           float,
    n_samples:      int   = 8,
    guidance_scale: float = 3.0,
    device:         torch.device | None = None,
) -> np.ndarray:
    """
    Draw n_samples damage fields conditioned on (Gc, ell, beta).

    Returns
    -------
    np.ndarray  shape (n_samples, ny, nx), values in [0, 1].
    """
    dev  = device or DEVICE
    c_np = _norm_cond(np.array([Gc]), np.array([ell]), np.array([beta]))
    cond = torch.from_numpy(c_np).to(dev).expand(n_samples, -1)

    x = torch.randn(n_samples, 1, model.ny, model.nx, device=dev)
    for t in reversed(range(sched.T)):
        x = sched.p_sample(model, x, t, cond, w=guidance_scale)
        x = x.clamp(-1.5, 1.5)

    return _x_to_field(x[:, 0].cpu().numpy())


# ── Flow Matching ─────────────────────────────────────────────────────────────

def train_flow_matching(
    dataset:        AT2FieldDataset,
    epochs:         int   = 500,
    batch_size:     int   = 32,
    lr:             float = 1e-4,
    cond_drop_rate: float = 0.20,
    device:         torch.device | None = None,
) -> FieldDenoiser:
    """
    Train a Conditional Flow Matching model on an AT2FieldDataset.

    The model learns the velocity field v_θ(x_t, t, c) that transports
    Gaussian noise to the data distribution along straight-line paths:

        x_t = (1-t)·x_data + t·x_noise,  t ~ Uniform[0,1]
        target velocity: u_t = x_noise − x_data  (constant along path)
        loss: E[ ||v_θ(x_t, t, c) − u_t||² ]

    Same FieldDenoiser as DDPM — only the training objective changes.
    No schedule object is needed; inference uses plain Euler integration.
    """
    dev   = device or DEVICE
    model = FieldDenoiser(ny=dataset.ny, nx=dataset.nx).to(dev)
    n_p   = sum(p.numel() for p in model.parameters())
    print(f"FieldDenoiser (Flow Matching): {n_p:,} params  device={dev}")

    fields = torch.from_numpy(dataset.fields).to(dev)
    cond   = torch.from_numpy(dataset.cond).to(dev)

    dl = DataLoader(TensorDataset(fields, cond),
                    batch_size=batch_size, shuffle=True, drop_last=True)
    opt      = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for x0, c in dl:
            B     = x0.shape[0]
            t     = torch.rand(B, device=dev)                       # t ~ U[0,1]
            noise = torch.randn_like(x0)
            xt    = (1 - t[:, None, None, None]) * x0 \
                  +      t[:, None, None, None]  * noise            # linear interp
            target = noise - x0                                     # constant velocity

            keep = (torch.rand(B, 1, device=dev) > cond_drop_rate)
            pred  = model(xt, t * 1000, c * keep.float())          # scale t for emb
            loss  = nn.functional.mse_loss(pred, target)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * B

        lr_sched.step()
        if (ep + 1) % 50 == 0:
            print(f"  Epoch {ep+1:>4}/{epochs}  "
                  f"loss={ep_loss / len(dataset.fields):.5f}")

    return model


@torch.no_grad()
def sample_fields_fm(
    model:          FieldDenoiser,
    Gc:             float,
    ell:            float,
    beta:           float,
    n_samples:      int   = 8,
    n_steps:        int   = 20,
    guidance_scale: float = 3.0,
    device:         torch.device | None = None,
) -> np.ndarray:
    """
    Draw n_samples damage fields via Flow Matching Euler integration.

    Integrates dx/dt = v_θ(x, t, c) backward from t=1 (noise) to t=0 (data).
    20 steps is typically sufficient; use 50 for higher-fidelity samples.

    Returns np.ndarray shape (n_samples, ny, nx), values in [0, 1].
    """
    dev  = device or DEVICE
    c_np = _norm_cond(np.array([Gc]), np.array([ell]), np.array([beta]))
    cond = torch.from_numpy(c_np).to(dev).expand(n_samples, -1)

    x  = torch.randn(n_samples, 1, model.ny, model.nx, device=dev)
    dt = -1.0 / n_steps                        # integrate t: 1 → 0

    for step in range(n_steps):
        t_val = 1.0 + step * dt                # 1.0, 1-1/n, 1-2/n, ..., 1/n
        t_vec = torch.full((n_samples,), t_val * 1000, device=dev)

        v_c = model(x, t_vec, cond)
        v_u = model(x, t_vec, torch.zeros_like(cond))
        v   = v_u + guidance_scale * (v_c - v_u)   # classifier-free guidance

        x = x + dt * v

    return _x_to_field(x[:, 0].cpu().numpy())


# ── Demo ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import time
    print(f"Device: {DEVICE}\n")

    print("[1/3] Generating 200-sample dataset (FAST_CFG) ...")
    ds = generate_field_dataset(200, cfg=FAST_CFG, verbose=True)
    print(f"  fields: {ds.fields.shape}  cond: {ds.cond.shape}\n")

    print("[2/3] Training Flow Matching (200 epochs) ...")
    model_fm = train_flow_matching(ds, epochs=200, batch_size=16)

    print("[2b/3] Training DDPM for comparison (200 epochs, T=200) ...")
    model_ddpm, sched = train_ddpm(ds, epochs=200, batch_size=16, T=200)

    print("\n[3/3] Sampling comparison (Gc=8e-4, ell=0.04, beta=−0.40, n=4):\n")
    params = dict(Gc=8e-4, ell=0.04, beta=-0.40, n_samples=4, guidance_scale=3.0)

    t0 = time.perf_counter()
    s_fm = sample_fields_fm(model_fm, **params, n_steps=20)
    t_fm = time.perf_counter() - t0

    t0 = time.perf_counter()
    s_ddpm = sample_fields(model_ddpm, sched, **params)
    t_ddpm = time.perf_counter() - t0

    for label, samples, elapsed in [("FM  (20 steps)", s_fm, t_fm),
                                     ("DDPM(200 steps)", s_ddpm, t_ddpm)]:
        d_max   = samples.max(axis=(1, 2))
        cracked = (samples > 0.5).sum(axis=(1, 2))
        print(f"  {label}  {elapsed*1000:6.0f}ms  "
              f"d_max={d_max.mean():.3f}±{d_max.std():.3f}  "
              f"cracked_px={cracked.mean():.0f}")


if __name__ == "__main__":
    main()
