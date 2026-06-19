"""
dicing_thermal_3d_gpu.py — GPU 3D transient thermal FEM/FD for wafer dicing.

A PyTorch explicit finite-difference solver for the 3-D heat equation during
dicing, runnable on GPU (CUDA) with automatic CPU fallback. Fills the gap that
the repo's CUDA kernel is 2-D only and the ABAQUS 3-D scripts are SiC-only /
license-gated / serial. Default material is **silicon**.

    ∂T/∂t = α ∇²T + q/(ρ·c_p)          (7-point Laplacian, replicate BC)

Two moving heat sources (both processes, per the build request):
    process="laser"  Gaussian surface flux, Beer–Lambert depth (ps grooving)
    process="blade"  frictional line source over the kerf contact column

Grid convention: T[z, y, x], z=0 is the top (worked) surface, x = feed.

Run:
    python fem/dicing_thermal_3d_gpu.py --process laser --power 20 --speed 100
    python fem/dicing_thermal_3d_gpu.py --dataset --n 60          # Si sweep → CSV
    python fem/dicing_thermal_3d_gpu.py --process blade --plot
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
from data.materials.material_properties import ALL_MATERIALS


def get_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── recipe / config ─────────────────────────────────────────────────────────────
@dataclass
class Sim3DConfig:
    material: str = "Si"
    process: str = "laser"            # "laser" | "blade"
    # domain (µm) — sized for quasi-steady local field, not a full lane
    lx_um: float = 120.0              # feed extent
    ly_um: float = 120.0             # cross-lane extent
    lz_um: float = 110.0             # wafer thickness (depth)
    dx_um: float = 5.0               # isotropic cell size
    # process knobs
    power_W: float = 20.0
    speed_mm_s: float = 100.0
    beam_radius_um: float = 5.0      # laser 1/e² radius (Gaussian σ≈r/√2 used)
    absorptivity: float = 0.7
    abs_depth_um: float = 3.0        # Beer–Lambert absorption depth (laser)
    blade_W_um: float = 23.0         # blade kerf width (blade)
    cut_depth_um: float = 60.0       # blade engagement depth (blade)
    friction_eff: float = 0.5        # fraction of cutting power → heat (blade)
    # numerics
    cfl: float = 0.8                 # fraction of 3-D explicit stability limit
    t_amb_C: float = 25.0
    haz_threshold_C: float = 500.0   # damage onset (low-k/Si subsurface)
    t_ablation_C: float = 2000.0     # cells above this ablate (energy ejected) → T capped
    max_steps: int = 60_000
    plateau_tol: float = 1e-3        # early-stop when peak-T plateaus

    def copy(self, **kw) -> "Sim3DConfig":
        d = asdict(self); d.update(kw); return Sim3DConfig(**d)


@dataclass
class Sim3DResult:
    peak_T_C: float
    groove_depth_um: float           # ablated depth at the lane centre (z-extent ≥ T_abl)
    haz_depth_um: float              # total z-extent above HAZ threshold
    haz_excess_um: float             # thermal shell BEYOND the groove (low-k relevant)
    haz_width_um: float
    haz_volume_um3: float
    haz_volume_frac: float
    ablated_volume_um3: float
    max_grad_C_per_um: float
    n_steps: int
    device: str
    field: Optional[np.ndarray] = field(default=None, repr=False)  # (nz,ny,nx) °C

    def kpis(self) -> dict:
        d = asdict(self); d.pop("field"); return d


# ── solver ───────────────────────────────────────────────────────────────────────
class DicingThermal3D:
    """3-D explicit-FD transient heat solver (PyTorch, GPU-capable)."""

    def __init__(self, cfg: Sim3DConfig, device: Optional[torch.device] = None):
        self.cfg = cfg
        self.device = device or get_device()
        m = ALL_MATERIALS.get(cfg.material, ALL_MATERIALS["Si"])
        self.alpha = m["k_thermal"] / (m["density"] * m["Cp"])      # m²/s
        self.rho_cp = m["density"] * m["Cp"]                         # J/(m³·K)

        self.dx = cfg.dx_um * 1e-6
        self.nx = max(8, int(round(cfg.lx_um / cfg.dx_um)))
        self.ny = max(8, int(round(cfg.ly_um / cfg.dx_um)))
        self.nz = max(8, int(round(cfg.lz_um / cfg.dx_um)))
        # 3-D explicit stability: α·dt/dx² ≤ 1/6
        self.dt = cfg.cfl * self.dx ** 2 / (6.0 * self.alpha)

        self.T = torch.full((self.nz, self.ny, self.nx), cfg.t_amb_C,
                            dtype=torch.float32, device=self.device)

    # 7-point Laplacian with insulated (replicate) boundaries
    @staticmethod
    def _laplacian(T: torch.Tensor) -> torch.Tensor:
        Tp = F.pad(T[None, None], (1, 1, 1, 1, 1, 1), mode="replicate")[0, 0]
        return (Tp[2:, 1:-1, 1:-1] + Tp[:-2, 1:-1, 1:-1]
                + Tp[1:-1, 2:, 1:-1] + Tp[1:-1, :-2, 1:-1]
                + Tp[1:-1, 1:-1, 2:] + Tp[1:-1, 1:-1, :-2] - 6.0 * T)

    def _apply_farfield_bc(self, T: torch.Tensor) -> None:
        """Ambient Dirichlet on the bulk/backside faces (heat sinks); the top
        surface z=0 stays free (insulated). Prevents heat trapping that would
        otherwise push the whole box above the HAZ threshold."""
        amb = self.cfg.t_amb_C
        T[:, 0, :] = amb; T[:, -1, :] = amb        # y faces (bulk wafer)
        T[:, :, 0] = amb; T[:, :, -1] = amb        # x faces (feed in/out)
        T[-1, :, :] = amb                          # z=max (backside chuck)

    def _coords(self):
        dxu = self.cfg.dx_um
        xg = torch.arange(self.nx, device=self.device) * dxu
        yg = torch.arange(self.ny, device=self.device) * dxu
        zg = torch.arange(self.nz, device=self.device) * dxu
        return xg, yg, zg

    def _source(self, x_center_um: float) -> torch.Tensor:
        """Volumetric heat source q [W/m³] as (nz,ny,nx) for source at x_center."""
        cfg = self.cfg
        xg, yg, zg = self._coords()
        yc = self.ny * cfg.dx_um / 2.0
        X = xg[None, None, :]; Y = yg[None, :, None]; Z = zg[:, None, None]

        if cfg.process == "laser":
            sig = cfg.beam_radius_um / math.sqrt(2.0)
            lateral = torch.exp(-((X - x_center_um) ** 2 + (Y - yc) ** 2)
                                / (2.0 * sig ** 2))
            depth = torch.exp(-Z / max(cfg.abs_depth_um, 1e-3))
            shape = lateral * depth
            P_abs = cfg.absorptivity * cfg.power_W
        else:  # blade: friction concentrated at the cutting front (walls + bottom)
            half_w = cfg.blade_W_um / 2.0
            sig_x = 2.0 * cfg.dx_um
            along = torch.exp(-((X - x_center_um) ** 2) / (2.0 * sig_x ** 2))
            # contact interface: near the two kerf walls OR the kerf bottom front
            near_wall = (torch.abs(torch.abs(Y - yc) - half_w) <= cfg.dx_um).float()
            in_depth = (Z <= cfg.cut_depth_um).float()
            near_bottom = (torch.abs(Z - cfg.cut_depth_um) <= cfg.dx_um).float()
            inkerf = (torch.abs(Y - yc) <= half_w).float()
            contact = torch.clamp(near_wall * in_depth + near_bottom * inkerf, max=1.0)
            shape = along * contact
            P_abs = cfg.friction_eff * cfg.power_W

        # normalize shape so ∫ q dV = P_abs  →  q = P_abs · shape / (Σ shape · dV)
        dV = self.dx ** 3
        denom = shape.sum() * dV
        if float(denom) <= 0:
            return torch.zeros_like(self.T)
        return (P_abs / denom) * shape           # W/m³

    def run(self, keep_field: bool = False) -> Sim3DResult:
        cfg = self.cfg
        v = cfg.speed_mm_s * 1e-3                 # m/s
        x_step_um = max((v * self.dt) * 1e6, 1e-6)   # µm advanced per step
        n_traverse = int(cfg.lx_um / x_step_um)
        n_steps = int(min(max(n_traverse, 50), cfg.max_steps))

        T = self.T
        diff = self.alpha * self.dt / self.dx ** 2
        scale = self.dt / self.rho_cp
        t_abl = cfg.t_ablation_C
        # keep the moving source mid-domain once it would run off the short box
        x_run = self.nx * cfg.dx_um
        prev_peak, plateau = 0.0, 0
        ablated = torch.zeros_like(T, dtype=torch.bool)
        s = 0
        for s in range(n_steps):
            xc = min((s + 0.5) * x_step_um, 0.6 * x_run)   # park near centre at steady state
            q = self._source(xc)
            T = T + diff * self._laplacian(T) + q * scale
            # ablation: cells above T_abl eject material → cap T, record removed volume
            hot = T > t_abl
            if bool(hot.any()):
                ablated |= hot
                T = torch.clamp(T, max=t_abl)
            self._apply_farfield_bc(T)
            if s % 50 == 0:
                peak = float(T.max())
                if s > 200 and abs(peak - prev_peak) <= cfg.plateau_tol * max(peak, 1.0):
                    plateau += 1
                    if plateau >= 3:               # quasi-steady reached
                        break
                else:
                    plateau = 0
                prev_peak = peak
        self.T = T
        return self._postprocess(T, ablated, s + 1, keep_field)

    def _postprocess(self, T, ablated, n_steps, keep_field) -> Sim3DResult:
        cfg = self.cfg
        dxu = cfg.dx_um
        yc_i = self.ny // 2
        nx_c = int(torch.argmax(T[0]).item() % self.nx)   # hottest surface column (x)

        haz = (T > cfg.haz_threshold_C)
        # groove = ablated material depth at the lane-centre column
        if bool(ablated.any()):
            col = ablated[:, yc_i, nx_c]
            zabl = torch.where(col)[0]
            groove_depth = float(zabl.max() + 1) * dxu if len(zabl) else 0.0
            ablated_vol = float(ablated.sum()) * dxu ** 3
        else:
            groove_depth = 0.0; ablated_vol = 0.0
        if bool(haz.any()):
            zidx = torch.where(haz.any(dim=(1, 2)))[0]
            yidx = torch.where(haz.any(dim=(0, 2)))[0]
            haz_depth = float(zidx.max() + 1) * dxu
            haz_width = float(yidx.max() - yidx.min() + 1) * dxu
            haz_vol = float(haz.sum()) * dxu ** 3
        else:
            haz_depth = haz_width = haz_vol = 0.0
        haz_excess = max(0.0, haz_depth - groove_depth)
        peak = float(T.max())

        gz, gy, gx = torch.gradient(T)
        grad = torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2) / dxu
        total_vol = self.nx * self.ny * self.nz * dxu ** 3
        return Sim3DResult(
            peak_T_C=round(peak, 2),
            groove_depth_um=round(groove_depth, 3),
            haz_depth_um=round(haz_depth, 3),
            haz_excess_um=round(haz_excess, 3),
            haz_width_um=round(haz_width, 3),
            haz_volume_um3=round(haz_vol, 2),
            haz_volume_frac=round(haz_vol / total_vol, 6),
            ablated_volume_um3=round(ablated_vol, 3),
            max_grad_C_per_um=round(float(grad.max()), 3),
            n_steps=int(n_steps), device=str(self.device),
            field=(T.detach().cpu().numpy() if keep_field else None),
        )


def simulate(cfg: Sim3DConfig, keep_field: bool = False,
             device: Optional[torch.device] = None) -> Sim3DResult:
    return DicingThermal3D(cfg, device).run(keep_field=keep_field)


# ── dataset sweep ────────────────────────────────────────────────────────────────
KPI_COLS = ["peak_T_C", "groove_depth_um", "haz_depth_um", "haz_excess_um",
            "haz_width_um", "haz_volume_um3", "haz_volume_frac",
            "ablated_volume_um3", "max_grad_C_per_um", "n_steps"]

_SWEEP = {
    "laser": {"power_W": (8.0, 40.0), "speed_mm_s": (80.0, 400.0),
              "beam_radius_um": (3.0, 8.0)},
    "blade": {"power_W": (10.0, 50.0), "speed_mm_s": (40.0, 120.0),
              "blade_W_um": (15.0, 30.0), "cut_depth_um": (40.0, 90.0)},
}


def _lhs(n, d, seed):
    rng = np.random.default_rng(seed)
    cut = np.linspace(0, 1, n + 1)
    u = rng.random((n, d))
    pts = cut[:n, None] + u * (1.0 / n)
    for j in range(d):
        rng.shuffle(pts[:, j])
    return pts


def build_thermal3d_dataset(material: str = "Si", processes=("laser", "blade"),
                            n_per_process: int = 40, seed: int = 0,
                            device: Optional[torch.device] = None):
    import pandas as pd
    device = device or get_device()
    rows = []
    for pi, proc in enumerate(processes):
        space = _SWEEP[proc]
        names = list(space)
        U = _lhs(n_per_process, len(names), seed + 17 * pi)
        for u in U:
            over = {"material": material, "process": proc}
            for val, name in zip(u, names):
                lo, hi = space[name]
                over[name] = float(lo + val * (hi - lo))
            cfg = Sim3DConfig(**over)
            r = simulate(cfg, device=device)
            row = {"material": material, "process": proc,
                   "power_W": cfg.power_W, "speed_mm_s": cfg.speed_mm_s,
                   "beam_radius_um": cfg.beam_radius_um, "blade_W_um": cfg.blade_W_um,
                   "cut_depth_um": cfg.cut_depth_um}
            row.update({k: getattr(r, k) for k in KPI_COLS})
            rows.append(row)
    return pd.DataFrame(rows)


# ── visualization ────────────────────────────────────────────────────────────────
def plot_field(cfg: Sim3DConfig, out: str, device=None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = simulate(cfg, keep_field=True, device=device)
    T = r.field                                  # (nz, ny, nx) °C
    nz, ny, nx = T.shape
    dxu = cfg.dx_um
    ext_xy = [0, nx * dxu, 0, ny * dxu]
    ext_xz = [0, nx * dxu, nz * dxu, 0]          # z downward
    yc, zc = ny // 2, 0

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    vmax = max(cfg.haz_threshold_C * 1.2, float(np.percentile(T, 99.9)))

    im0 = ax[0].imshow(T[:, yc, :], extent=ext_xz, aspect="auto", cmap="inferno",
                       vmin=cfg.t_amb_C, vmax=vmax, origin="upper")
    ax[0].contour(np.linspace(0, nx*dxu, nx), np.linspace(0, nz*dxu, nz),
                  T[:, yc, :], levels=[cfg.haz_threshold_C], colors="cyan", linewidths=1.2)
    ax[0].set(xlabel="feed x [µm]", ylabel="depth z [µm]",
              title=f"(a) x–z section (y=centre)\ncyan = HAZ {cfg.haz_threshold_C:.0f}°C")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, label="T [°C]")

    im1 = ax[1].imshow(T[zc, :, :], extent=ext_xy, aspect="auto", cmap="inferno",
                       vmin=cfg.t_amb_C, vmax=vmax, origin="lower")
    ax[1].set(xlabel="feed x [µm]", ylabel="cross-lane y [µm]",
              title="(b) x–y section (top surface z=0)")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, label="T [°C]")

    zdepth = np.arange(nz) * dxu
    ax[2].plot(T[:, yc, int(np.argmax(T[0]) % nx)], zdepth, color="#b5374a", lw=2)
    ax[2].axvline(cfg.haz_threshold_C, ls="--", color="cyan", label="HAZ")
    ax[2].axvline(cfg.t_ablation_C, ls="--", color="k", label="ablation")
    ax[2].invert_yaxis()
    ax[2].set(xlabel="T [°C]", ylabel="depth z [µm]",
              title=f"(c) centre depth profile\ngroove={r.groove_depth_um:.0f}µm "
                    f"HAZ_exc={r.haz_excess_um:.0f}µm")
    ax[2].legend(fontsize=8, frameon=False); ax[2].grid(alpha=.3)

    fig.suptitle(f"GPU 3D dicing thermal — {cfg.material} {cfg.process}  "
                 f"({cfg.power_W:.0f}W @ {cfg.speed_mm_s:.0f}mm/s, dev={r.device})",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description="GPU 3D dicing thermal solver")
    ap.add_argument("--process", choices=["laser", "blade"], default="laser")
    ap.add_argument("--material", default="Si")
    ap.add_argument("--power", type=float, default=20.0)
    ap.add_argument("--speed", type=float, default=100.0)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dataset", action="store_true", help="material sweep → CSV")
    ap.add_argument("--n", type=int, default=40, help="samples/process for --dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    dev = get_device(args.device)
    print(f"device: {dev}  (cuda available: {torch.cuda.is_available()})")

    if args.dataset:
        df = build_thermal3d_dataset(material=args.material, n_per_process=args.n,
                                     seed=args.seed, device=dev)
        out_dir = os.path.join(ROOT, "data", "hybrid_dicing")
        os.makedirs(out_dir, exist_ok=True)
        csv = os.path.join(out_dir, f"thermal3d_{args.material.lower()}_dataset.csv")
        df.to_csv(csv, index=False)
        g = df.groupby("process")
        print(g[["peak_T_C", "groove_depth_um", "haz_excess_um", "haz_width_um"]].median())
        print(f"\nwrote {csv}  ({len(df)} rows)")
        # exemplar full 3-D field per process (the actual 3D FEM data)
        for proc in df.process.unique():
            cfg = Sim3DConfig(material=args.material, process=proc, power_W=25.0)
            r = simulate(cfg, keep_field=True, device=dev)
            npy = os.path.join(out_dir, f"thermal3d_{args.material.lower()}_{proc}_field.npy")
            np.save(npy, r.field)
            print(f"wrote {npy}  field shape {r.field.shape}")
        return

    cfg = Sim3DConfig(material=args.material, process=args.process,
                      power_W=args.power, speed_mm_s=args.speed)
    if args.plot:
        out = plot_field(cfg, os.path.join(ROOT, "results",
                         f"thermal3d_{args.material.lower()}_{args.process}.png"),
                         device=dev)
        print(f"wrote {out}")
    r = simulate(cfg, device=dev)
    for k, v in r.kpis().items():
        print(f"  {k:22} {v}")


if __name__ == "__main__":
    main()
