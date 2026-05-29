"""
Fourier Neural Operator (FNO) surrogate for continuous stress field prediction.

Reference: Li et al. (2020) "Fourier Neural Operator for Parametric PDEs"
           https://arxiv.org/abs/2010.08895

Input  : processing parameters (cut_depth, blade_W) → broadcast to grid
Output : full 2D max-principal-stress field on the wafer domain

Why FNO over GP:
  - GP maps scalar params → scalar outputs
  - FNO maps params → continuous function (entire field)
  - Resolution-independent: train on coarse mesh, infer on fine mesh
  - 1000× faster than FEM at inference

Data format:
  results/<job_name>/stress_maxprincipal.npy   shape: (N_elements,)
  We reshape to 2D grid after loading.

Usage:
    python surrogate_fno.py --data-dir results --train
    python surrogate_fno.py --data-dir results --predict --depth 40 --kerf 30
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "fno_model.pt")

# ── FNO building blocks ───────────────────────────────────────────────────────

class SpectralConv2d(nn.Module):
    """
    2D Fourier layer: FFT → multiply complex weights → IFFT.
    Keeps only the lowest `modes` Fourier modes in each spatial dim.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2,
                                dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2,
                                dtype=torch.cfloat))

    def _compl_mul2d(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """(batch, in_ch, h, w) × (in_ch, out_ch, h, w) → (batch, out_ch, h, w)."""
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNOBlock(nn.Module):
    """One FNO layer: spectral conv + bypass conv + GELU."""

    def __init__(self, channels: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes1, modes2)
        self.bypass   = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.bypass(x))


class FNO2d(nn.Module):
    """
    Full 2-D FNO architecture.

    Input  channels: 2 (cut_depth, blade_W) broadcast over spatial grid
                   + 2 (x, y grid coordinates)  → 4 total
    Output channels: 1 (max principal stress field)
    """

    def __init__(self,
                 modes1:    int = 12,
                 modes2:    int = 12,
                 width:     int = 32,
                 n_layers:  int = 4,
                 grid_h:    int = 64,
                 grid_w:    int = 64):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w

        self.input_proj  = nn.Conv2d(4, width, kernel_size=1)
        self.blocks      = nn.Sequential(
            *[FNOBlock(width, modes1, modes2) for _ in range(n_layers)])
        self.output_proj = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, 1, kernel_size=1))

    def _make_grid(self, batch_size: int, device: str) -> torch.Tensor:
        """Returns (B, 2, H, W) normalized [0,1] coordinate grid."""
        xs = torch.linspace(0, 1, self.grid_w, device=device)
        ys = torch.linspace(0, 1, self.grid_h, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=0)   # (2, H, W)
        return grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        params: (B, 2) — [cut_depth_norm, blade_W_norm] in [0, 1]
        Returns: (B, 1, H, W) stress field (normalised)
        """
        B = params.shape[0]
        H, W = self.grid_h, self.grid_w

        # Broadcast parameters over spatial grid
        p = params.view(B, 2, 1, 1).expand(B, 2, H, W)
        g = self._make_grid(B, params.device)

        x = torch.cat([p, g], dim=1)   # (B, 4, H, W)
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output_proj(x)      # (B, 1, H, W)


# ── Dataset ───────────────────────────────────────────────────────────────────

class DicingFieldDataset(Dataset):
    """
    Loads stress field .npy files + metadata from parametric_summary.csv.
    Reshapes flat element arrays to (grid_h, grid_w) via bilinear interpolation.
    """

    def __init__(self, manifest_path: str, data_dir: str,
                 grid_h: int = 64, grid_w: int = 64):
        import pandas as pd
        self.grid_h = grid_h
        self.grid_w = grid_w

        df = pd.read_csv(manifest_path)
        self.records = []

        # Normalisation stats (computed over dataset)
        depths  = df["cut_depth_um"].values.astype(np.float32)
        kerfs   = df["blade_W_um"].values.astype(np.float32)
        self.depth_min, self.depth_max = depths.min(), depths.max()
        self.kerf_min,  self.kerf_max  = kerfs.min(),  kerfs.max()

        stress_fields = []
        for _, row in df.iterrows():
            npy_path = os.path.join(data_dir, row["job"],
                                    "stress_maxprincipal.npy")
            if not os.path.exists(npy_path):
                continue
            field = np.load(npy_path).astype(np.float32)
            field = self._to_grid(field)
            stress_fields.append(field)
            self.records.append({
                "depth": float(row["cut_depth_um"]),
                "kerf":  float(row["blade_W_um"]),
                "field": field})

        if stress_fields:
            all_stress = np.concatenate([f.ravel() for f in stress_fields])
            self.stress_mean = float(all_stress.mean())
            self.stress_std  = float(all_stress.std()) + 1e-8
        else:
            self.stress_mean = 0.0
            self.stress_std  = 1.0

        print(f"[*] Dataset: {len(self.records)} samples loaded")

    def _to_grid(self, flat_field: np.ndarray) -> np.ndarray:
        """Reshape flat FEM output to (grid_h, grid_w) via interpolation."""
        n = len(flat_field)
        side = int(np.sqrt(n))
        if side * side == n:
            return flat_field.reshape(side, side)
        # Non-square: interpolate to target grid
        from scipy.interpolate import RegularGridInterpolator
        n_approx = int(np.sqrt(n))
        src = flat_field[:n_approx * n_approx].reshape(n_approx, n_approx)
        xs = np.linspace(0, 1, n_approx)
        interp = RegularGridInterpolator((xs, xs), src, method="linear")
        xg = np.linspace(0, 1, self.grid_w)
        yg = np.linspace(0, 1, self.grid_h)
        YY, XX = np.meshgrid(yg, xg, indexing="ij")
        pts = np.stack([YY.ravel(), XX.ravel()], axis=1)
        return interp(pts).reshape(self.grid_h, self.grid_w).astype(np.float32)

    def _norm_param(self, val, lo, hi):
        return (val - lo) / (hi - lo + 1e-8)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        params = torch.tensor([
            self._norm_param(rec["depth"], self.depth_min, self.depth_max),
            self._norm_param(rec["kerf"],  self.kerf_min,  self.kerf_max)],
            dtype=torch.float32)
        field_norm = (rec["field"] - self.stress_mean) / self.stress_std
        target = torch.tensor(field_norm, dtype=torch.float32).unsqueeze(0)
        return params, target


# ── Training ──────────────────────────────────────────────────────────────────

def train(data_dir: str,
          manifest: str,
          epochs:    int = 200,
          batch:     int = 4,
          lr:        float = 1e-3,
          grid_h:    int = 64,
          grid_w:    int = 64):

    dataset = DicingFieldDataset(manifest, data_dir, grid_h, grid_w)
    if len(dataset) == 0:
        print("[!] No field data found. Run ABAQUS first and extract results.")
        return

    loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=0)

    model  = FNO2d(modes1=12, modes2=12, width=32, n_layers=4,
                   grid_h=grid_h, grid_w=grid_w).to(DEVICE)
    optim  = torch.optim.Adam(model.parameters(), lr=lr)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    print(f"[*] Training FNO on {DEVICE}  |  {len(dataset)} samples  |  "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for params, targets in loader:
            params  = params.to(DEVICE)
            targets = targets.to(DEVICE)
            pred    = model(params)
            loss    = F.mse_loss(pred, targets)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

        sched.step()
        avg_loss = total_loss / len(loader)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{epochs}  loss={avg_loss:.6f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state":  model.state_dict(),
                "stress_mean":  dataset.stress_mean,
                "stress_std":   dataset.stress_std,
                "depth_min":    dataset.depth_min,
                "depth_max":    dataset.depth_max,
                "kerf_min":     dataset.kerf_min,
                "kerf_max":     dataset.kerf_max,
                "grid_h":       grid_h,
                "grid_w":       grid_w,
            }, MODEL_PATH)

    print(f"[✓] Best loss: {best_loss:.6f}  model → {MODEL_PATH}")
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_field(cut_depth_um: float,
                   blade_W_um:  float,
                   model_path:  str = MODEL_PATH) -> np.ndarray:
    """
    Returns predicted stress field (GPa) as (grid_h, grid_w) numpy array.
    """
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    model = FNO2d(grid_h=ckpt["grid_h"], grid_w=ckpt["grid_w"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    p_norm = np.array([
        (cut_depth_um - ckpt["depth_min"]) / (ckpt["depth_max"] - ckpt["depth_min"] + 1e-8),
        (blade_W_um   - ckpt["kerf_min"])  / (ckpt["kerf_max"]  - ckpt["kerf_min"]  + 1e-8),
    ], dtype=np.float32)

    with torch.no_grad():
        params = torch.tensor(p_norm).unsqueeze(0)
        pred   = model(params)[0, 0].numpy()

    # Denormalise
    stress_Pa = pred * ckpt["stress_std"] + ckpt["stress_mean"]
    return stress_Pa / 1e9   # → GPa


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="results")
    parser.add_argument("--manifest", default="results/parametric_summary.csv")
    parser.add_argument("--train",   action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--depth",   type=float, default=50.0)
    parser.add_argument("--kerf",    type=float, default=30.0)
    parser.add_argument("--epochs",  type=int,   default=200)
    parser.add_argument("--plot",    action="store_true")
    args = parser.parse_args()

    if args.train:
        train(data_dir=args.data_dir,
              manifest=args.manifest,
              epochs=args.epochs)

    if args.predict:
        if not os.path.exists(MODEL_PATH):
            print("[!] No trained model found. Run --train first.")
            return
        field_GPa = predict_field(args.depth, args.kerf)
        print(f"[✓] Predicted field: depth={args.depth}µm, kerf={args.kerf}µm")
        print(f"    Max stress: {field_GPa.max():.3f} GPa")
        print(f"    Mean stress: {field_GPa.mean():.3f} GPa")
        print(f"    Field shape: {field_GPa.shape}")

        if args.plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 5))
            plt.imshow(field_GPa, origin="lower", cmap="hot",
                       extent=[0, 500, 0, 200])
            plt.colorbar(label="Max Principal Stress [GPa]")
            plt.xlabel("Width [µm]")
            plt.ylabel("Depth [µm]")
            plt.title(f"FNO Prediction: depth={args.depth}µm, kerf={args.kerf}µm")
            plt.tight_layout()
            path = f"results/fno_pred_d{int(args.depth)}_bw{int(args.kerf)}.png"
            plt.savefig(path, dpi=150)
            print(f"[✓] Plot → {path}")


if __name__ == "__main__":
    main()
