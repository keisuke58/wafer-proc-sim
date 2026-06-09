"""
dicing_heat_sim.py — DISCO dicing saw wafer thermal simulation

Simulates transient heat conduction during dicing using the CUDA
_heat_diffusion_kernel when available, falling back to NumPy FD.

Physics:
  ∂T/∂t = α·∇²T + Q/(ρ·cp·dx)

  where Q [W/m²] is the blade-wafer contact heat flux, modeled as a
  moving Gaussian line source along each dicing lane.

Material defaults: SiC (DISCO DFL7160 blade on SiC wafer)
  α = 84e-6 m²/s,  ρ·cp = 2.5e6 J/(m³·K)

Usage:
    from fem.dicing_heat_sim import DicingHeatSim
    sim = DicingHeatSim(nx=256, ny=256, dx=20e-6)
    sim.set_ambient(25.0)
    T = sim.run_lane(y_mm=12.5, feed_mms=80.0, Q_W_per_m2=5e7)
    print(f"Peak wafer temp: {T.max():.1f} °C")
"""
from __future__ import annotations
import numpy as np

try:
    from fem._heat_diffusion_kernel import (
        heat_diffusion_solve,
        heat_diffusion_solve_with_source,
        gpu_info,
    )
    _CUDA = True
except ImportError:
    _CUDA = False


# ── NumPy fallback ────────────────────────────────────────────────────────────

def _numpy_step_with_source(T: np.ndarray, src_dt_rhocp: np.ndarray, r: float) -> np.ndarray:
    lap = (np.roll(T, 1, 0) + np.roll(T, -1, 0)
         + np.roll(T, 1, 1) + np.roll(T, -1, 1) - 4 * T)
    return T + r * lap + src_dt_rhocp


def _numpy_solve_with_source(T, source, alpha, rho_cp, dx, dt, n_steps):
    r     = alpha * dt / dx**2
    assert r <= 0.25, f"unstable: r={r:.4f}"
    scale = dt / (rho_cp * dx)
    src   = source * scale
    for _ in range(n_steps):
        T = _numpy_step_with_source(T, src, r)
    return T


# ── DicingHeatSim ─────────────────────────────────────────────────────────────

class DicingHeatSim:
    """
    2-D transient thermal simulation of a DISCO dicing lane.

    Grid: (ny, nx) cells, each dx × dx [m].
    Coordinate convention: x = feed direction, y = cross-lane.

    Parameters
    ----------
    nx, ny   : grid dimensions
    dx       : cell size [m] (uniform, default 20 µm)
    alpha    : thermal diffusivity [m²/s] (default SiC 84e-6)
    rho_cp   : volumetric heat capacity [J/(m³·K)] (default SiC 2.5e6)
    """

    # SiC defaults
    ALPHA_SiC  = 84e-6    # m²/s
    RHO_CP_SiC = 2.5e6    # J/(m³·K)

    def __init__(
        self,
        nx: int = 256,
        ny: int = 256,
        dx: float = 20e-6,
        alpha: float = ALPHA_SiC,
        rho_cp: float = RHO_CP_SiC,
    ):
        self.nx     = nx
        self.ny     = ny
        self.dx     = dx
        self.alpha  = alpha
        self.rho_cp = rho_cp

        # CFL-stable timestep at 90 % of limit
        self.dt = 0.9 * dx**2 / (4 * alpha)

        self.T: np.ndarray = np.full((ny, nx), 25.0)   # [°C]
        self.cuda = _CUDA

    def set_ambient(self, T_amb: float = 25.0) -> None:
        self.T[:] = T_amb

    def blade_source_field(
        self,
        x_m: float,          # blade center x position [m]
        y_m: float,          # blade center y position [m]
        Q: float = 5e7,      # peak heat flux [W/m²]
        sigma_x: float = 4,  # Gaussian half-width in x [cells]
        sigma_y: float = 100, # Gaussian extent in y — full lane width
    ) -> np.ndarray:
        """Build (ny, nx) source field for blade at (x_m, y_m)."""
        xi = x_m / self.dx
        yi = y_m / self.dx
        xx = np.arange(self.nx, dtype=float)
        yy = np.arange(self.ny, dtype=float)
        src = Q * np.exp(
            -0.5 * ((xx[None, :] - xi) / sigma_x)**2
            -0.5 * ((yy[:, None] - yi) / sigma_y)**2
        )
        return src.astype(np.float64, copy=False)

    def run_lane(
        self,
        y_mm: float,
        feed_mms: float = 80.0,
        Q_W_per_m2: float = 5e7,
        n_steps_per_pos: int = 50,
    ) -> np.ndarray:
        """
        Simulate blade traversing one full lane at y = y_mm.
        Returns final temperature field (ny, nx) [°C].
        """
        y_m  = y_mm * 1e-3
        dx   = self.dx
        T    = self.T.copy()
        nx   = self.nx

        # Blade moves from x=0 to x=nx*dx at feed_mms mm/s
        x_step_m = feed_mms * 1e-3 * self.dt * n_steps_per_pos
        n_pos    = max(1, int(nx * dx / x_step_m))

        for p in range(n_pos):
            x_m  = p * x_step_m
            src  = self.blade_source_field(x_m, y_m, Q=Q_W_per_m2)

            if self.cuda:
                T = np.array(heat_diffusion_solve_with_source(
                    T, src, self.alpha, self.rho_cp, dx, self.dt,
                    n_steps_per_pos))
            else:
                T = _numpy_solve_with_source(
                    T, src, self.alpha, self.rho_cp, dx, self.dt,
                    n_steps_per_pos)

        self.T = T
        return T

    def run_full_wafer(
        self,
        streets_y_mm: list[float],
        feed_mms: float = 80.0,
        Q_W_per_m2: float = 5e7,
    ) -> np.ndarray:
        """Run all dicing lanes; temperature accumulates across lanes."""
        for y in streets_y_mm:
            self.run_lane(y_mm=y, feed_mms=feed_mms, Q_W_per_m2=Q_W_per_m2)
        return self.T

    @property
    def peak_temp(self) -> float:
        return float(self.T.max())

    @property
    def mean_temp(self) -> float:
        return float(self.T.mean())

    @staticmethod
    def gpu_available() -> bool:
        return _CUDA

    @staticmethod
    def gpu_info() -> dict:
        if _CUDA:
            return dict(gpu_info())
        return {"cuda": False}
