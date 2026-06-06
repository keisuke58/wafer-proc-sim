"""
Physics-Informed DeepONet for SiC dicing stress field prediction.

Why DeepONet over FNO (surrogate_fno.py)?
  FNO   : fixed grid, learns grid-to-grid map
  DeepONet: mesh-free, predicts u(x,y) at ANY query point
  → natural for adaptive refinement near crack tips (SiC dicing failure zone)

Architecture: vanilla DeepONet  +  physics residual loss
  Branch net : f(θ) = process params [depth, feed, spindle, kerf]  → R^p
  Trunk  net : g(x, y)  = spatial coordinate                        → R^p
  Output : σ_max(x, y | θ) = Σ_k  branch_k × trunk_k              (inner product)

Physics loss (plane stress, isotropic approx for 4H-SiC):
  PDE residual at N_col collocation points:
    ∂σ_xx/∂x + ∂τ_xy/∂y = 0   (x-equilibrium)
    ∂τ_xy/∂x + ∂σ_yy/∂y = 0   (y-equilibrium)
  We predict max-principal stress and enforce its Laplacian smoothness
  as a soft physics constraint (full tensor PDE needs two output heads —
  future extension; current constraint is necessary but not sufficient).

Training data:
  Synthetic FEM-like: σ_max ~ Hertz contact + SiC anisotropy field
  N_data = 500 parameter sets, M_sensors = 50 spatial points each

References
----------
Lu et al. (2021) Learning nonlinear operators via DeepONet
Raissi et al. (2019) Physics-informed neural networks
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(7)
np.random.seed(7)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ── SiC material constants ────────────────────────────────────────────────────
E_SIC_GPA   = 448.0   # Young's modulus [GPa]
NU_SIC      = 0.21    # Poisson's ratio
K1C_MPAM    = 2.5     # fracture toughness [MPa·√m]


# ── Synthetic stress field oracle ─────────────────────────────────────────────

def stress_oracle(params: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """
    Simplified Hertz + anisotropy stress field for blade dicing.
    params: (4,) [depth, feed, spindle, kerf]
    xy    : (M, 2) [x, y] spatial coordinates in [0,1]²
    Returns: (M,) max-principal-stress normalised [MPa]
    """
    depth, feed, spindle, kerf = params
    x, y = xy[:, 0], xy[:, 1]

    # Hertz contact: peak pressure ∝ sqrt(depth × E)
    p0 = 0.01 * np.sqrt(depth * E_SIC_GPA)   # MPa (simplified)
    a  = 0.1 * np.sqrt(depth / 100)            # contact half-width normalised

    # σ_max decays from blade contact zone (x≈0.3, y≈1 — blade entry)
    r  = np.sqrt((x - 0.3)**2 + (y - 1.0)**2) + 1e-3
    s_hertz = p0 * np.exp(-r**2 / (2 * a**2))

    # SiC 4-fold anisotropy: crystal planes at 45° stress concentration
    angle = np.arctan2(y - 0.5, x - 0.5)
    s_aniso = 0.15 * p0 * np.abs(np.cos(4 * angle))

    # Feed rate modulation
    s_feed = 0.02 * feed * np.exp(-y**2 / 0.1)

    # Spindle speed: higher speed → lower chip load → lower stress (empirical)
    s_spin = -0.001 * spindle * np.ones_like(x)

    return (s_hertz + s_aniso + s_feed + s_spin).clip(0).astype(np.float32)


# ── Data generation ───────────────────────────────────────────────────────────

def make_dataset(N_params: int = 500, M_sensors: int = 50, M_col: int = 200):
    """
    N_params  : number of process parameter sets
    M_sensors : fixed sensor locations (same for all params) for branch net input
    M_col     : collocation points for physics residual
    """
    rng = np.random.default_rng(42)

    # Fixed sensor grid (branch net input layout)
    xs = np.linspace(0, 1, int(M_sensors**0.5))
    ys = np.linspace(0, 1, int(M_sensors**0.5))
    Xs, Ys = np.meshgrid(xs, ys)
    sensors_xy = np.column_stack([Xs.ravel(), Ys.ravel()])[:M_sensors]  # (M_sensors, 2)

    # Process parameters
    params = rng.uniform(
        [30, 1, 10, 50],
        [150, 20, 60, 200],
        size=(N_params, 4)
    ).astype(np.float32)

    # Branch net input: stress at sensor locations for each param set
    u_sensors = np.array([
        stress_oracle(params[i], sensors_xy) for i in range(N_params)
    ], dtype=np.float32)  # (N_params, M_sensors)

    # Query points (random, one set per param for training)
    query_xy = rng.uniform(0, 1, size=(N_params, 2)).astype(np.float32)
    u_query  = np.array([
        stress_oracle(params[i], query_xy[[i]])[0] for i in range(N_params)
    ], dtype=np.float32)  # (N_params,)

    # Collocation points (physics loss, shared)
    col_xy = rng.uniform(0, 1, size=(M_col, 2)).astype(np.float32)

    return {
        "params": params,
        "u_sensors": u_sensors,
        "sensors_xy": sensors_xy,
        "query_xy": query_xy,
        "u_query": u_query,
        "col_xy": col_xy,
    }


# ── DeepONet ──────────────────────────────────────────────────────────────────

class BranchNet(nn.Module):
    """Input: sensor readings u(x_1,...,x_m) → basis coefficients p"""
    def __init__(self, in_dim: int, p: int = 64, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, p),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)


class TrunkNet(nn.Module):
    """Input: query coordinate (x, y) → basis functions p"""
    def __init__(self, p: int = 64, hidden: int = 128):
        super().__init__()
        # Positional encoding for coordinate input
        self.pos_enc = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
        )
        self.net = nn.Sequential(
            nn.Linear(32, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, p), nn.Tanh(),
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(self.pos_enc(xy))


class PhysicsDeepONet(nn.Module):
    def __init__(self, M_sensors: int = 50, p: int = 64):
        super().__init__()
        self.branch = BranchNet(in_dim=M_sensors, p=p)
        self.trunk  = TrunkNet(p=p)
        self.bias   = nn.Parameter(torch.zeros(1))

    def forward(self, u_sensors: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """
        u_sensors: (B, M_sensors)  branch input
        xy       : (B, 2)          trunk input
        Returns  : (B,)  predicted stress
        """
        b = self.branch(u_sensors)   # (B, p)
        t = self.trunk(xy)           # (B, p)
        return (b * t).sum(-1) + self.bias

    def physics_residual(self, u_sensors: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """
        Laplacian smoothness loss as a proxy physics constraint.
        ∇²σ ≈ 0  (stress harmonic in elastic body away from loads)
        Computed via autograd on trunk net output.
        """
        xy_r = xy.requires_grad_(True)
        sigma = self.forward(u_sensors, xy_r)
        # ∂σ/∂x, ∂σ/∂y
        grad = torch.autograd.grad(
            sigma.sum(), xy_r, create_graph=True
        )[0]  # (M_col, 2)
        # ∂²σ/∂x²
        d2x = torch.autograd.grad(grad[:, 0].sum(), xy_r, create_graph=True)[0][:, 0]
        d2y = torch.autograd.grad(grad[:, 1].sum(), xy_r, create_graph=True)[0][:, 1]
        laplacian = d2x + d2y
        return laplacian.pow(2).mean()


# ── Training ──────────────────────────────────────────────────────────────────

def train(model: PhysicsDeepONet, data: dict,
          epochs: int = 500, lr: float = 1e-3,
          lambda_phy: float = 0.01) -> dict:

    u_s  = torch.tensor(data["u_sensors"]).to(DEVICE)
    xy_q = torch.tensor(data["query_xy"]).to(DEVICE)
    u_q  = torch.tensor(data["u_query"]).to(DEVICE)

    # Collocation: repeat a fixed representative param set's sensors
    col_idx = 0
    u_col_fixed = u_s[[col_idx]].expand(len(data["col_xy"]), -1)
    xy_col = torch.tensor(data["col_xy"]).to(DEVICE)

    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=50)

    data_losses, phy_losses = [], []

    for ep in range(epochs):
        model.train()
        opt.zero_grad()

        # Data loss
        pred = model(u_s, xy_q)
        loss_data = nn.functional.mse_loss(pred, u_q)

        # Physics residual (Laplacian smoothness)
        loss_phy = model.physics_residual(u_col_fixed, xy_col.clone())

        loss = loss_data + lambda_phy * loss_phy
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step(loss_data)

        data_losses.append(loss_data.item())
        phy_losses.append(loss_phy.item())

    return {"data_losses": data_losses, "phy_losses": phy_losses}


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: PhysicsDeepONet, data: dict, n_test: int = 100) -> dict:
    model.eval()
    rng = np.random.default_rng(99)

    # New parameter set (unseen)
    params_test = rng.uniform([30, 1, 10, 50], [150, 20, 60, 200], size=(n_test, 4)).astype(np.float32)
    sensors_xy = data["sensors_xy"]
    u_sens_test = np.array([
        stress_oracle(params_test[i], sensors_xy) for i in range(n_test)
    ], dtype=np.float32)

    # Dense grid for field prediction
    xg = np.linspace(0, 1, 20)
    yg = np.linspace(0, 1, 20)
    Xg, Yg = np.meshgrid(xg, yg)
    xy_grid = np.column_stack([Xg.ravel(), Yg.ravel()]).astype(np.float32)

    all_true, all_pred = [], []

    with torch.no_grad():
        for i in range(min(n_test, 20)):  # evaluate first 20
            u_s_i  = torch.tensor(u_sens_test[[i]]).expand(len(xy_grid), -1).to(DEVICE)
            xy_g_t = torch.tensor(xy_grid).to(DEVICE)
            pred_field = model(u_s_i, xy_g_t).cpu().numpy()
            true_field = stress_oracle(params_test[i], xy_grid)
            all_true.append(true_field)
            all_pred.append(pred_field)

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    rmse = float(np.sqrt(np.mean((all_true - all_pred)**2)))
    r2   = float(1 - np.sum((all_true - all_pred)**2) / np.sum((all_true - all_true.mean())**2))
    # Relative error: only compute where |true| > 5% of max (avoid near-zero division)
    thresh = 0.05 * np.abs(all_true).max()
    mask = np.abs(all_true) > thresh
    rel_err = float(np.mean(np.abs(all_true[mask] - all_pred[mask]) / np.abs(all_true[mask]))) if mask.any() else float('nan')

    return {"rmse_MPa": rmse, "r2": r2, "rel_err_pct": rel_err * 100,
            "all_true": all_true, "all_pred": all_pred}


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_field(model: PhysicsDeepONet, data: dict, param_idx: int = 0):
    model.eval()
    sensors_xy = data["sensors_xy"]
    params = data["params"]

    xg = np.linspace(0, 1, 40)
    yg = np.linspace(0, 1, 40)
    Xg, Yg = np.meshgrid(xg, yg)
    xy_grid = np.column_stack([Xg.ravel(), Yg.ravel()]).astype(np.float32)

    u_s_i = data["u_sensors"][[param_idx]]
    true_field = stress_oracle(params[param_idx], xy_grid).reshape(40, 40)

    with torch.no_grad():
        u_s_t = torch.tensor(u_s_i).expand(len(xy_grid), -1).to(DEVICE)
        pred_field = model(u_s_t, torch.tensor(xy_grid).to(DEVICE)).cpu().numpy().reshape(40, 40)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    vmax = max(true_field.max(), pred_field.max())

    for ax, field, title in zip(axes[:2], [true_field, pred_field], ["FEM (oracle)", "DeepONet"]):
        im = ax.contourf(Xg, Yg, field, levels=20, cmap="hot_r", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label="σ_max [MPa]")
        ax.set_title(f"{title}\n(depth={params[param_idx,0]:.0f}µm, feed={params[param_idx,1]:.1f}mm/s)")
        ax.set_xlabel("x"); ax.set_ylabel("y")

    err = np.abs(true_field - pred_field)
    im3 = axes[2].contourf(Xg, Yg, err, levels=20, cmap="Blues")
    plt.colorbar(im3, ax=axes[2], label="|error| [MPa]")
    axes[2].set_title(f"Absolute error\nmax={err.max():.3f} MPa")
    axes[2].set_xlabel("x")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "deeponet_field_comparison.png")
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Field comparison saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")

    print(f"\n[Data] Generating operator dataset ({DEVICE})...")
    data = make_dataset(N_params=500, M_sensors=49, M_col=300)
    print(f"  Branch input shape : {data['u_sensors'].shape}")
    print(f"  Query xy shape     : {data['query_xy'].shape}")
    print(f"  Collocation points : {data['col_xy'].shape}")

    model = PhysicsDeepONet(M_sensors=49, p=64).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters   : {n_params:,}")

    print("\n[Training] Physics-Informed DeepONet, 500 epochs...")
    hist = train(model, data, epochs=500, lr=1e-3, lambda_phy=0.01)
    print(f"  Final data loss : {hist['data_losses'][-1]:.5f}")
    print(f"  Final phy  loss : {hist['phy_losses'][-1]:.5f}")

    print("\n[Evaluation] On 20 unseen parameter sets (40×40 grid)...")
    metrics = evaluate(model, data, n_test=20)
    print(f"  RMSE   = {metrics['rmse_MPa']:.4f} MPa")
    print(f"  R²     = {metrics['r2']:.4f}")
    print(f"  RelErr = {metrics['rel_err_pct']:.2f} %")

    plot_field(model, data, param_idx=3)

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "deeponet_physics.pt"))
    print(f"Model saved: results/deeponet_physics.pt")

    print("\n[vs FNO] Key differences:")
    print("  DeepONet : mesh-free, any query point, physics loss in loop")
    print("  FNO      : fixed grid, spectral conv, faster on uniform grids")
    print("  → DeepONet preferred near crack tips (adaptive sampling)")


if __name__ == "__main__":
    main()
