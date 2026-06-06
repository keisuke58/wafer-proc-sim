"""
Conditional Denoising Diffusion Probabilistic Model (DDPM) for SiC dicing recipe generation.

Motivation
----------
Bayesian optimisation finds ONE optimal recipe per run.
A diffusion model learns the DISTRIBUTION of good recipes:
  p(recipe | target_quality)
and generates diverse candidates — useful for:
  - Robustness analysis (spread of feasible recipes)
  - Multi-constraint satisfaction (chipping AND throughput AND cost)
  - Transfer to new wafer grades without rerunning BO

Recipe space (4D continuous):
  x = [depth_um, feed_mms, spindle_krpm, kerf_width_um]
  normalised to [-1, 1]  (recipe hypercube)

Conditioning:
  c = [target_chipping_um]   (scalar quality target)
  Classifier-free guidance: train with c dropped 20% → enable guidance at inference

Architecture:
  Small MLP denoiser  (physics is low-dimensional → no need for UNet)
  Time embedding: sinusoidal → linear → ReLU

References
----------
Ho et al. (2020) Denoising Diffusion Probabilistic Models
Ho & Salimans (2022) Classifier-Free Diffusion Guidance
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)
np.random.seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ── Recipe parameter bounds ───────────────────────────────────────────────────
BOUNDS = np.array([
    [30,  150],   # depth_um
    [1.0, 20.0],  # feed_mms
    [10,  60],    # spindle_krpm
    [50,  200],   # kerf_width_um
], dtype=np.float32)


def normalise_recipe(x: np.ndarray) -> np.ndarray:
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    return 2 * (x - lo) / (hi - lo) - 1.0


def denormalise_recipe(x: np.ndarray) -> np.ndarray:
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    return (x + 1.0) / 2.0 * (hi - lo) + lo


# ── Quality oracle (same physics as surrogate_transformer) ────────────────────

def quality_oracle(X: np.ndarray) -> np.ndarray:
    """Chipping proxy: depth^0.6 × feed^0.4 / spindle^0.3 + anisotropy"""
    depth, feed, spin, kerf = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    chip = (
        0.8 * depth**0.6 * feed**0.4 / spin**0.3
        + 0.3 * np.sin(4 * kerf / 200 * math.pi)
    )
    return np.clip(chip, 0.1, 30.0).astype(np.float32)


# ── Synthetic training data ───────────────────────────────────────────────────

def make_training_data(N: int = 5000):
    rng = np.random.default_rng(1)
    X = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(N, 4)).astype(np.float32)
    y = quality_oracle(X)
    return X, y


# ── DDPM schedule ─────────────────────────────────────────────────────────────

class DDPMSchedule:
    def __init__(self, T: int = 100, beta_min: float = 1e-4, beta_max: float = 0.02,
                 device: torch.device | None = None):
        self.T = T
        dev = device or DEVICE
        beta      = torch.linspace(beta_min, beta_max, T, device=dev)
        alpha     = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        # Pre-place everything on device — no per-step transfers
        self.beta               = beta
        self.alpha              = alpha
        self.alpha_bar          = alpha_bar
        self.sqrt_ab            = alpha_bar.sqrt()
        self.sqrt_one_minus_ab  = (1 - alpha_bar).sqrt()

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε"""
        sqrt_ab   = self.sqrt_ab[t][:, None]
        sqrt_1mab = self.sqrt_one_minus_ab[t][:, None]
        return sqrt_ab * x0 + sqrt_1mab * noise

    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: int,
                 cond: torch.Tensor, guidance_scale: float = 3.0) -> torch.Tensor:
        """
        Reverse step with classifier-free guidance:
        ε̂_guided = ε̂_uncond + w × (ε̂_cond - ε̂_uncond)
        """
        t_tensor = torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device)
        model.eval()
        with torch.no_grad():
            eps_cond   = model(x_t, t_tensor, cond)
            eps_uncond = model(x_t, t_tensor, torch.zeros_like(cond))
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        beta  = self.beta[t]
        alpha = self.alpha[t]
        ab    = self.alpha_bar[t]

        mean = (x_t - (1 - alpha) / (1 - ab).sqrt() * eps) / alpha.sqrt()
        if t > 0:
            return mean + beta.sqrt() * torch.randn_like(x_t)
        return mean


# ── Denoiser MLP ──────────────────────────────────────────────────────────────

class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = t.float()[:, None] * self.freqs[None]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class RecipeDenoiser(nn.Module):
    """
    Input : [x_t (4), t_emb (32), cond (1)] → noise estimate (4)
    """
    def __init__(self, x_dim: int = 4, t_dim: int = 32, c_dim: int = 1,
                 hidden: int = 256):
        super().__init__()
        self.time_emb = SinusoidalTimeEmb(t_dim)
        in_dim = x_dim + t_dim + c_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        t_emb = self.time_emb(t)
        h = torch.cat([x, t_emb, cond], dim=-1)
        return self.net(h)


# ── Training ──────────────────────────────────────────────────────────────────

def train(model: RecipeDenoiser, schedule: DDPMSchedule,
          X: np.ndarray, y: np.ndarray,
          epochs: int = 30, lr: float = 2e-4, batch_size: int = 512,
          cond_drop_rate: float = 0.2) -> list:

    X_norm = torch.tensor(normalise_recipe(X)).to(DEVICE)
    # Normalise quality to [0, 1] for conditioning
    y_norm = torch.tensor((y - y.min()) / (y.max() - y.min() + 1e-8),
                          dtype=torch.float32).to(DEVICE).unsqueeze(-1)

    ds = TensorDataset(X_norm, y_norm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = optim.AdamW(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    losses = []

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for x0, cond in dl:
            t = torch.randint(0, schedule.T, (x0.shape[0],), device=DEVICE)
            noise = torch.randn_like(x0)
            x_t = schedule.q_sample(x0, t, noise)

            # Classifier-free guidance: randomly drop condition
            mask = (torch.rand(cond.shape[0], 1, device=DEVICE) > cond_drop_rate).float()
            cond_masked = cond * mask

            pred_noise = model(x_t, t, cond_masked)
            loss = nn.functional.mse_loss(pred_noise, noise)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(x0)

        sched.step()
        losses.append(ep_loss / len(X))

    return losses


# ── Sampling / generation ─────────────────────────────────────────────────────

def generate_recipes(model: RecipeDenoiser, schedule: DDPMSchedule,
                     target_chipping_um: float, y_min: float, y_max: float,
                     n_samples: int = 200, guidance_scale: float = 5.0) -> np.ndarray:
    """
    Sample recipes conditioned on target_chipping_um.
    Returns (n_samples, 4) array of denormalised process parameters.
    """
    c_norm = (target_chipping_um - y_min) / (y_max - y_min + 1e-8)
    cond = torch.full((n_samples, 1), c_norm, dtype=torch.float32, device=DEVICE)

    x = torch.randn(n_samples, 4, device=DEVICE)
    for t in reversed(range(schedule.T)):
        x = schedule.p_sample(model, x, t, cond, guidance_scale)
        x = x.clamp(-1.5, 1.5)  # soft clamp during diffusion

    x_np = x.cpu().numpy()
    x_np = np.clip(x_np, -1, 1)
    return denormalise_recipe(x_np)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")

    print(f"\n[Data] Generating 3000 recipe-quality pairs via oracle ({DEVICE})...")
    X, y = make_training_data(3000)
    y_min, y_max = float(y.min()), float(y.max())
    print(f"  Chipping range: [{y_min:.2f}, {y_max:.2f}] µm")

    schedule = DDPMSchedule(T=100)
    model = RecipeDenoiser(hidden=256).to(DEVICE)

    print(f"\n[Training] DDPM denoiser, 1000 epochs...")
    losses = train(model, schedule, X, y, epochs=1000, batch_size=512)
    print(f"  Final loss: {losses[-1]:.5f}")

    # Generate for three quality targets (within training distribution [y_min, y_max])
    targets = [4.0, 8.0, 15.0]
    print(f"\n[Generation] Conditional sampling (guidance_scale=20):")
    print(f"  {'target':>8}  {'depth':>8}  {'feed':>6}  {'spin':>8}  {'kerf':>8}  "
          f"{'oracle_chip':>12}  {'frac_feasible':>14}")

    results = {}
    for tgt in targets:
        recipes = generate_recipes(model, schedule, tgt, y_min, y_max,
                                   n_samples=200, guidance_scale=20.0)
        # Clip to valid bounds
        recipes = np.clip(recipes, BOUNDS[:, 0], BOUNDS[:, 1])
        chip_oracle = quality_oracle(recipes)

        # Feasibility: oracle chipping within ±2 µm of target (absolute tolerance)
        frac = np.mean(np.abs(chip_oracle - tgt) < 2.0)
        median = recipes.mean(0)
        print(f"  {tgt:>8.1f}  {median[0]:>8.1f}  {median[1]:>6.2f}  "
              f"{median[2]:>8.1f}  {median[3]:>8.1f}  "
              f"{chip_oracle.mean():>12.2f}  {frac:>14.2%}")
        results[f"target_{tgt}"] = {
            "recipes": recipes,
            "oracle_chipping": chip_oracle,
            "feasibility": float(frac),
        }

    # Diversity analysis
    r_lo  = results[f"target_{targets[0]}"]["recipes"]
    r_hi  = results[f"target_{targets[-1]}"]["recipes"]
    print(f"\n[Diversity] Std of generated recipes:")
    labels = ["depth", "feed", "spindle", "kerf"]
    for i, lab in enumerate(labels):
        print(f"  {lab:>10}: target={targets[0]}µm {r_lo[:,i].std():.2f}  |  target={targets[-1]}µm {r_hi[:,i].std():.2f}")

    # Save model
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "schedule_T": schedule.T,
                "y_min": y_min, "y_max": y_max},
               os.path.join(OUT_DIR, "recipe_diffusion.pt"))
    print(f"\nModel saved: results/recipe_diffusion.pt")


if __name__ == "__main__":
    main()
