"""
Transformer surrogate for SiC dicing — Foundation Model approach.

Strategy (mirrors "small-n beats large-n" paper finding):
  Phase 1  Pre-train  on N=2000 FEM simulation samples (physics-rich, label-cheap)
  Phase 2  Fine-tune  on n=11  Grade-A experimental points (label-expensive)

Architecture: Physics-Aware Transformer (PAT)
  - Input: 4 process params  [depth, feed, spindle, kerf_width]
  - Physics embedding: crystal anisotropy angle (SiC 4H: 4-fold) appended as sin/cos features
  - Encoder: 4 layers × 4 heads × d_model=64 (small — data-efficient)
  - Output head: mean + log_var (heteroscedastic, matches paper GP model)

Pre-training objective: MSE on FEM chipping proxy (deletion_fraction)
Fine-tuning:           NLL with heteroscedastic output on experimental chipping_um

Key result benchmark (target):
  LOO RMSE (GP baseline):          1.62 µm
  LOO RMSE (PAT fine-tuned, n=11): ≤ 1.40 µm   ← target improvement
"""

from __future__ import annotations

import os
import math
import numpy as np
# torch is only needed by the transformer model / training functions below.
# The data helpers (make_fem_data / make_experimental_data / physics_features)
# are pure NumPy, so keep them importable on torch-less environments (CI).
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    _HAS_TORCH = True
except ImportError:               # pragma: no cover - exercised on CI only
    torch = None
    _HAS_TORCH = False

    class _NnStub:                # placeholder so class definition below parses
        class Module:             # noqa: D401 - minimal stand-in
            def __init__(self, *a, **k):
                raise ImportError(
                    "PyTorch is required for PhysicsAwareTransformer; "
                    "install torch to use the surrogate model.")
    nn = _NnStub()
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
if _HAS_TORCH:
    torch.manual_seed(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = None
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ── Physics feature engineering ───────────────────────────────────────────────

def physics_features(X: np.ndarray) -> np.ndarray:
    """
    Append physics-motivated features to raw process params.
    X columns: [depth_um, feed_mms, spindle_rpm, kerf_width_um]

    Added features:
      MRR       = depth × feed  (material removal rate proxy)
      SFR       = spindle × feed / depth  (specific force ratio)
      aniso_sin = sin(4θ) where θ ∝ kerf_width (SiC 4-fold anisotropy)
      aniso_cos = cos(4θ)
    """
    depth  = X[:, 0:1]
    feed   = X[:, 1:2]
    spin   = X[:, 2:3]
    kerf   = X[:, 3:4]

    mrr     = depth * feed
    sfr     = spin * feed / (depth + 1e-3)
    theta   = kerf / 200.0 * math.pi  # normalise to [0, π]
    a_sin   = np.sin(4 * theta)
    a_cos   = np.cos(4 * theta)

    return np.concatenate([X, mrr, sfr, a_sin, a_cos], axis=1).astype(np.float32)


# ── Synthetic data generators ─────────────────────────────────────────────────

def make_fem_data(N: int = 2000):
    """
    Simulate FEM-derived chipping data with known physics:
    chipping ∝ depth^0.6 × feed^0.4 / spindle^0.3  + anisotropy + noise
    """
    rng = np.random.default_rng(0)
    depth  = rng.uniform(30,  150, N)
    feed   = rng.uniform(1,   20,  N)
    spin   = rng.uniform(10,  60,  N) * 1000  # [rpm]
    kerf   = rng.uniform(50,  200, N)

    chip = (
        0.8 * depth**0.6 * feed**0.4 / (spin / 1000)**0.3
        + 0.3 * np.sin(4 * kerf / 200 * math.pi)
        + rng.normal(0, 0.5, N)
    )
    chip = np.clip(chip, 0.1, 30.0)

    X = np.column_stack([depth, feed, spin / 1000, kerf])
    y = chip.astype(np.float32)
    return X.astype(np.float32), y


def make_experimental_data():
    """
    11 Grade-A experimental points (matches paper dataset).
    Reconstructed from Micro2026 + Mat2022 digitised values.
    """
    data = np.array([
        [40,  2.0, 30, 100,  3.1],
        [40,  2.0, 26, 100,  2.8],
        [60,  2.0, 30, 120,  4.2],
        [60,  1.5, 34, 120,  3.5],
        [80,  2.0, 30, 140,  5.8],
        [80,  1.5, 26, 140,  5.1],
        [100, 2.0, 30, 160,  7.3],
        [100, 1.5, 26, 160,  6.6],
        [100, 2.0, 34, 160,  6.9],
        [120, 1.5, 30, 180,  9.2],
        [120, 2.0, 26, 180,  9.8],
    ], dtype=np.float32)
    # cols: depth, feed, spindle_krpm, kerf, chipping_um
    X = data[:, :4].copy()
    X[:, 2] *= 1000  # krpm → rpm
    y = data[:, 4]
    return X, y


# ── Model ─────────────────────────────────────────────────────────────────────

class PhysicsAwareTransformer(nn.Module):
    """
    Small Transformer encoder → heteroscedastic output head.
    Input dim = 8 (4 raw + 4 physics features).
    CPU default: 2 layers × 2 heads × d=32 (fast).  GPU: 4 layers × 4 heads × d=64.
    """
    def __init__(self, in_dim: int = 8, d_model: int = 32,
                 n_heads: int = 2, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head_mean = nn.Linear(d_model, 1)
        self.head_logvar = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor):
        # x: (B, F) → treat features as sequence length=1 token
        h = self.input_proj(x).unsqueeze(1)          # (B, 1, d_model)
        h = self.encoder(h).squeeze(1)                # (B, d_model)
        mean   = self.head_mean(h).squeeze(-1)
        logvar = self.head_logvar(h).squeeze(-1)
        return mean, logvar


# ── Training utilities ─────────────────────────────────────────────────────────

def mse_loss(pred_mean, pred_logvar, target):
    return nn.functional.mse_loss(pred_mean, target)

def nll_loss(pred_mean, pred_logvar, target):
    """Gaussian NLL: 0.5 × [log σ² + (y - µ)² / σ²]"""
    return 0.5 * (pred_logvar + (target - pred_mean)**2 / pred_logvar.exp()).mean()


def normalise(X, mu=None, sigma=None):
    if mu is None:
        mu, sigma = X.mean(0), X.std(0) + 1e-8
    return (X - mu) / sigma, mu, sigma


def pretrain(model: PhysicsAwareTransformer, X_fem: np.ndarray, y_fem: np.ndarray,
             epochs: int = 200, lr: float = 1e-3, batch_size: int = 256) -> list:
    Xf = physics_features(X_fem)
    Xf_t, mu, sigma = normalise(Xf)
    # Normalise y to zero-mean, unit-variance so MSE loss is well-conditioned
    y_mu, y_sigma = y_fem.mean(), y_fem.std() + 1e-8
    y_norm = (y_fem - y_mu) / y_sigma
    y_t = torch.tensor(y_norm).to(DEVICE)
    ds = TensorDataset(torch.tensor(Xf_t).to(DEVICE), y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    losses = []
    for _ in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            mean, logvar = model(xb)
            loss = mse_loss(mean, logvar, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(xb)
        sched.step()
        losses.append(ep_loss / len(y_fem))
    return losses, mu, sigma, y_mu, y_sigma


def finetune_loo(model_state: dict, X_exp: np.ndarray, y_exp: np.ndarray,
                 mu: np.ndarray, sigma: np.ndarray,
                 y_mu: float = 0.0, y_sigma: float = 1.0,
                 epochs: int = 300, lr: float = 5e-4) -> dict:
    """
    LOO-CV: for each held-out point, load pre-trained weights and fine-tune
    on remaining n-1 points, then predict the held-out point.
    """
    n = len(y_exp)
    preds, stds = np.zeros(n), np.zeros(n)

    for i in range(n):
        mask = np.arange(n) != i
        X_tr = physics_features(X_exp[mask])
        y_tr = torch.tensor(y_exp[mask]).to(DEVICE)
        X_te = physics_features(X_exp[[i]])

        Xtr_t = torch.tensor((X_tr - mu) / sigma).to(DEVICE)
        Xte_t = torch.tensor((X_te - mu) / sigma).to(DEVICE)
        # Normalise targets with same scale used in pre-training
        y_tr_norm = torch.tensor((y_exp[mask] - y_mu) / y_sigma).to(DEVICE)

        # Reload pre-trained weights
        m = PhysicsAwareTransformer().to(DEVICE)
        m.load_state_dict(model_state)

        # Discriminative fine-tuning: encoder at LR/10, head at LR
        # (freezing encoder causes mean-collapse with n=10 experimental points)
        opt = optim.Adam([
            {"params": m.input_proj.parameters(), "lr": lr / 10},
            {"params": m.encoder.parameters(),    "lr": lr / 10},
            {"params": m.head_mean.parameters(),  "lr": lr},
            {"params": m.head_logvar.parameters(),"lr": lr},
        ])

        for _ in range(epochs):
            m.train()
            opt.zero_grad()
            mean, logvar = m(Xtr_t)
            loss = nll_loss(mean, logvar, y_tr_norm)
            loss.backward()
            opt.step()

        m.eval()
        with torch.no_grad():
            mean_norm, logvar = m(Xte_t)
        # Denormalise back to µm
        preds[i] = float(mean_norm.item() * y_sigma + y_mu)
        stds[i]  = float(logvar.exp().sqrt().item() * y_sigma)

    rmse = float(np.sqrt(np.mean((preds - y_exp)**2)))
    mae  = float(np.mean(np.abs(preds - y_exp)))
    r2   = float(1 - np.sum((preds - y_exp)**2) / np.sum((y_exp - y_exp.mean())**2))

    return {"preds": preds, "stds": stds, "rmse": rmse, "mae": mae, "r2": r2}


def extract_features(model: PhysicsAwareTransformer, X: np.ndarray,
                     mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Extract encoder representations (frozen) for downstream regression."""
    Xf = physics_features(X)
    Xf_t = torch.tensor((Xf - mu) / sigma, dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        h = model.input_proj(Xf_t).unsqueeze(1)
        h = model.encoder(h).squeeze(1)
    return h.cpu().numpy()


def finetune_loo_hybrid(model: PhysicsAwareTransformer,
                        X_exp: np.ndarray, y_exp: np.ndarray,
                        mu: np.ndarray, sigma: np.ndarray) -> dict:
    """
    LOO-CV: freeze encoder, extract features, fit kernel ridge on n-1 points.
    Works better than gradient fine-tuning at n=10 (avoids mean collapse).
    """
    feats = extract_features(model, X_exp, mu, sigma)  # (11, d_model)
    n = len(y_exp)
    preds = np.zeros(n)

    for i in range(n):
        mask = np.arange(n) != i
        scaler = StandardScaler().fit(feats[mask])
        X_tr = scaler.transform(feats[mask])
        X_te = scaler.transform(feats[[i]])
        # KRR with RBF kernel — works well for small n
        kr = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.5)
        kr.fit(X_tr, y_exp[mask])
        preds[i] = kr.predict(X_te)[0]

    rmse = float(np.sqrt(np.mean((preds - y_exp) ** 2)))
    mae  = float(np.mean(np.abs(preds - y_exp)))
    r2   = float(1 - np.sum((preds - y_exp) ** 2) / np.sum((y_exp - y_exp.mean()) ** 2))
    return {"preds": preds, "rmse": rmse, "mae": mae, "r2": r2}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"\n[Phase 1] Pre-training on N=2000 FEM samples ({DEVICE})...")
    X_fem, y_fem = make_fem_data(2000)
    model = PhysicsAwareTransformer(in_dim=8).to(DEVICE)
    losses, mu, sigma, y_mu, y_sigma = pretrain(model, X_fem, y_fem, epochs=200)
    print(f"  Pre-train loss (final): {losses[-1]:.4f}")

    print("\n[Phase 2a] LOO gradient fine-tune on n=11 experimental points...")
    X_exp, y_exp = make_experimental_data()
    result_ft = finetune_loo(model.state_dict(), X_exp, y_exp, mu, sigma,
                             y_mu=y_mu, y_sigma=y_sigma, epochs=300)

    print(f"\n  [Gradient FT]  LOO RMSE: {result_ft['rmse']:.3f} µm  MAE: {result_ft['mae']:.3f}  R²: {result_ft['r2']:.3f}")

    print("\n[Phase 2b] LOO hybrid (frozen encoder + kernel ridge) on n=11 points...")
    result = finetune_loo_hybrid(model, X_exp, y_exp, mu, sigma)

    print(f"\n  [Hybrid KRR]  LOO RMSE: {result['rmse']:.3f} µm  (GP baseline: 1.62 µm)")
    print(f"                LOO MAE : {result['mae']:.3f} µm")
    print(f"                LOO R²  : {result['r2']:.3f}")
    print(f"\n  Per-sample predictions vs truth:")
    print(f"  {'i':>3}  {'true':>8}  {'pred':>8}  {'err':>8}")
    _, y_exp_check = make_experimental_data()
    for i, (t, p) in enumerate(zip(y_exp_check, result['preds'])):
        print(f"  {i:>3}  {t:>8.2f}  {p:>8.2f}  {abs(t-p):>8.2f}")

    # Save
    out_path = os.path.join(OUT_DIR, "pat_loo_results.npz")
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(out_path, **{k: v for k, v in result.items() if isinstance(v, np.ndarray)},
             rmse=result["rmse"], mae=result["mae"], r2=result["r2"])
    print(f"\nSaved: {out_path}")

    torch.save({"state_dict": model.state_dict(), "mu": mu, "sigma": sigma},
               os.path.join(OUT_DIR, "pat_pretrained.pt"))
    print(f"Pre-trained weights saved: results/pat_pretrained.pt")


if __name__ == "__main__":
    main()
