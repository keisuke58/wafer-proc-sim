"""
at2_simulator.py  —  AT2 phase-field fracture, 1D bar, pure NumPy

Inference-backend-agnostic callable interface:
    params = PFParams(Gc=8e-4, ell=0.04, E=1.0, beta=-0.5)
    result = run_forward(params, cfg)           # ForwardResult
    p      = crack_probability(params, 45.0)    # float in (0,1)
    y      = simulate_observation(params, 45.0) # 0 or 1
    ll     = log_likelihood(params, obs)        # float

Physical model:
    E[u,d] = int g(d) psi_e+  +  Gc_eff(theta)/2 * [d^2/ell + ell*(d')^2]  dx
    g(d)   = (1-d)^2 + kappa_res                          (AT2 degradation)
    Anisotropic surface energy:  Gc_eff(theta) = Gc0*(1 + beta*sin^2(theta))
    Staggered solve  +  history variable for irreversibility

Units: dimensionless by default (scale to SI externally).
  Typical demo values: E=1.0, Gc~1e-3, ell~0.04 (= 4% of bar length L=1).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr   # standard normal CDF  Phi(x)


# ─────────────────────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PFParams:
    """Phase-field fracture material parameters (dimensionless / SI).

    For SI:  Gc [J/m²],  ell [m],  E [Pa].
    beta = 0  → isotropic.
    beta > 0  → off-cleavage angles harder (Gc increases away from theta=0).
    beta < 0  → off-cleavage angles easier.

    Semiconductor examples:
        4H-SiC:   a-plane K_IC=1.4, c-plane K_IC=2.0
                  Gc_a/Gc_c = (1.4/2.0)^2 = 0.49  →  beta ~ -0.51
        beta-Ga2O3:  strong (100) cleavage, K_IC_100 = 0.5 << off-axis
                  off-cleavage harder  →  beta > 0  (w.r.t. cleavage as theta=0)
    """
    Gc:   float
    ell:  float
    E:    float
    nu:   float = 0.20    # reserved for 2D extension
    beta: float = 0.00    # anisotropy parameter

    def at_angle(self, theta_rad: float) -> "PFParams":
        """Return copy with Gc = Gc_eff(theta_rad) and beta=0."""
        return PFParams(
            Gc=gc_anisotropic(theta_rad, self.Gc, self.beta),
            ell=self.ell,
            E=self.E,
            nu=self.nu,
            beta=0.0,
        )


@dataclass
class SimConfig:
    """Numerical configuration for the 1D solver."""
    L:            float = 1.0    # bar length
    n_elem:       int   = 300    # mesh elements  (nodes = n_elem + 1)
    n_steps:      int   = 150    # monotone load increments
    u_max_factor: float = 5.0    # u_max = u_max_factor * ell
    kappa_res:    float = 1e-6   # residual stiffness (prevents singular stiffness)
    defect_frac:  float = 0.04   # Gc reduction at bar centre (triggers localisation)


@dataclass
class ForwardResult:
    """Output of run_forward()."""
    U:         np.ndarray    # applied displacements  (n_steps+1,)
    F:         np.ndarray    # mean reaction stress    (n_steps+1,)
    d_field:   np.ndarray    # damage field at final step  (n_elem+1,)
    init_U:    float         # displacement at crack initiation  (d_max > 0.05)
    peak_F:    float         # peak mean stress
    converged: bool          # True when max(d) < 0.999 at final step


# ─────────────────────────────────────────────────────────────────────────────
#  Physics helpers
# ─────────────────────────────────────────────────────────────────────────────

def gc_anisotropic(theta_rad: float, Gc0: float, beta: float) -> float:
    """Effective fracture energy at angle theta from the preferred cleavage axis.

    Gc_eff(theta) = Gc0 * (1 + beta * sin^2(theta))
    """
    return Gc0 * (1.0 + beta * float(np.sin(theta_rad)) ** 2)


def critical_strain_estimate(params: PFParams) -> float:
    """
    Analytical crack-initiation strain for AT2 1D bar (Bourdin 2000):
        eps_c = (3/8) * sqrt(3*Gc / (2*E*ell))
    Useful as reference displacement for the observation model.
    """
    return (3.0 / 8.0) * np.sqrt(3.0 * params.Gc / (2.0 * params.E * params.ell))


# ─────────────────────────────────────────────────────────────────────────────
#  Tridiagonal solver — Thomas algorithm, O(n)
# ─────────────────────────────────────────────────────────────────────────────

def _thomas(sub: np.ndarray, diag: np.ndarray, sup: np.ndarray,
            rhs: np.ndarray) -> np.ndarray:
    """Solve Ax=rhs for tridiagonal A.
    sub[k] = A[k+1, k]  (length n-1)
    sup[k] = A[k, k+1]  (length n-1)
    diag   = A[k, k]    (length n)
    """
    n = len(rhs)
    b = diag.copy()
    d = rhs.copy()
    c = sup.copy()

    for i in range(1, n):
        if abs(b[i - 1]) < 1e-300:
            raise ZeroDivisionError(f"Near-zero pivot at row {i-1}; check Gc/ell/h ratio")
        m = sub[i - 1] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]

    x = np.empty(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


# ─────────────────────────────────────────────────────────────────────────────
#  Damage equation — FD, homogeneous Neumann BCs, spatially varying Gc
# ─────────────────────────────────────────────────────────────────────────────

def _solve_damage(H: np.ndarray, Gc_f: np.ndarray,
                  ell: float, h: float) -> np.ndarray:
    """Solve AT2 damage equation with homogeneous Neumann BCs.

    PDE:  -Gc_f(x)*ell*d''(x) + (Gc_f(x)/ell + 2*H(x))*d(x) = 2*H(x)
    BCs:  d'(0) = d'(L) = 0   (ghost-node symmetry)

    FD discretisation at interior node i  (i = 1,...,n-2):
        A[i,i-1] = -Gc_f[i]*ell/h^2 = -coeff[i]
        A[i,i]   =  2*coeff[i] + Gc_f[i]/ell + 2*H[i]
        A[i,i+1] = -coeff[i]
        rhs[i]   =  2*H[i]

    Boundary i=0 (Neumann: d_{-1}=d_1):
        sup[0] = -2*coeff[0]   (doubled, not -coeff[0])
    Boundary i=n-1 (Neumann: d_n=d_{n-2}):
        sub[-1] = -2*coeff[-1]
    """
    n = len(H)                          # number of nodes (n_elem + 1)
    coeff = Gc_f * ell / h ** 2         # per-node Laplacian coefficient

    diag = np.empty(n)
    sub  = np.empty(n - 1)
    sup  = np.empty(n - 1)

    # Interior  (diag[0] and diag[-1] are overwritten below)
    diag[:] = 2.0 * coeff + Gc_f / ell + 2.0 * H
    sub[:]  = -coeff[1:]     # A[i+1, i] = -coeff[i+1]
    sup[:]  = -coeff[:-1]    # A[i, i+1] = -coeff[i]

    # Left boundary: Neumann doubles the super-diagonal entry
    sup[0] = -2.0 * coeff[0]

    # Right boundary: Neumann doubles the sub-diagonal entry
    sub[-1] = -2.0 * coeff[-1]

    rhs = 2.0 * H
    d_new = _thomas(sub, diag, sup, rhs)
    return np.clip(d_new, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Main forward solver
# ─────────────────────────────────────────────────────────────────────────────

def run_forward(params: PFParams,
                cfg: SimConfig | None = None) -> ForwardResult:
    """1D AT2 bar under monotonically increasing end displacement.

    Algorithm (staggered alternating minimisation):
        For each load step k:
          1. Displacement step: u(x) = U_k * x/L  (exact 1D solution)
             → uniform strain  eps = U_k / L
          2. History update:   H = max(H, E/2 * eps^2)
          3. Damage step:      solve FD tridiagonal  (see _solve_damage)
          4. Irreversibility:  d = max(d_old, d_new)
          5. Reaction:         F = mean( g(d) * E * eps )

    A small defect (cfg.defect_frac) at the bar centre ensures
    damage localisation for isotropic loading; remove it (defect_frac=0)
    for homogeneous-damage studies.

    Returns:
        ForwardResult with load-displacement curve and final damage field.
    """
    if cfg is None:
        cfg = SimConfig()

    n_nodes = cfg.n_elem + 1
    h = cfg.L / cfg.n_elem

    # Gc field: optional weak zone at centre triggers localisation
    Gc_f = np.full(n_nodes, params.Gc)
    if cfg.defect_frac > 0.0:
        Gc_f[cfg.n_elem // 2] *= (1.0 - cfg.defect_frac)

    u_max = cfg.u_max_factor * params.ell
    U_arr = np.linspace(0.0, u_max, cfg.n_steps + 1)

    H = np.zeros(n_nodes)
    d = np.zeros(n_nodes)
    F_arr = np.zeros(len(U_arr))
    init_U = u_max      # updated at first d > INIT_THRESHOLD

    INIT_THRESHOLD = 0.05

    for k, U in enumerate(U_arr):
        eps = U / cfg.L

        psi_plus = 0.5 * params.E * eps ** 2
        H = np.maximum(H, psi_plus)

        d_new = _solve_damage(H, Gc_f, params.ell, h)
        d = np.maximum(d, d_new)

        g = (1.0 - d) ** 2 + cfg.kappa_res
        F_arr[k] = float(np.mean(g * params.E * eps))

        if init_U == u_max and np.max(d) >= INIT_THRESHOLD:
            init_U = float(U)

    return ForwardResult(
        U=U_arr,
        F=F_arr,
        d_field=d.copy(),
        init_U=init_U,
        peak_F=float(np.max(F_arr)),
        converged=bool(np.max(d) < 0.999),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Binary observation model  (Novel A: sign-only / probit likelihood)
# ─────────────────────────────────────────────────────────────────────────────

def crack_probability(
    params:      PFParams,
    theta_deg:   float,
    sigma_noise: float = 0.15,
    cfg:         SimConfig | None = None,
    _U_ref:      float | None = None,
) -> float:
    """Probability that a crack is observed at dicing angle theta_deg.

    Probit (Bernoulli-Phi) model:
        Gc_eff = Gc0 * (1 + beta * sin^2(theta))
        Run AT2 forward with Gc_eff  ->  crack initiation displacement  U_c
        U_ref  = run_forward(isotropic params).init_U  (same Gc/ell/E, beta=0)
        z      = (U_c - U_ref) / (sigma_noise * ell)
        p_crack = Phi(-z)

    Higher Gc_eff  ->  larger U_c  ->  z >> 0  ->  p_crack << 1  (fewer cracks).
    U_ref is from the actual simulator (not analytical), so both quantities
    are on the same scale regardless of defect geometry.

    _U_ref: pre-computed reference; pass from log_likelihood() to avoid
            running the reference simulation once per observation.
    """
    cfg_ = cfg or SimConfig()
    p_eff = params.at_angle(np.radians(theta_deg))
    result = run_forward(p_eff, cfg_)

    if _U_ref is None:
        p_iso = PFParams(Gc=params.Gc, ell=params.ell,
                         E=params.E, nu=params.nu, beta=0.0)
        _U_ref = run_forward(p_iso, cfg_).init_U

    denom = max(sigma_noise * p_eff.ell, 1e-14)
    z = (result.init_U - _U_ref) / denom
    return float(np.clip(1.0 - ndtr(z), 1e-9, 1.0 - 1e-9))


def simulate_observation(
    params:      PFParams,
    theta_deg:   float,
    sigma_noise: float = 0.15,
    cfg:         SimConfig | None = None,
    seed:        int | None = None,
) -> int:
    """Sample one binary dicing observation y in {0,1} at angle theta_deg.

    y = 1  →  crack / chipping observed.
    y = 0  →  clean cut (no crack).
    """
    p = crack_probability(params, theta_deg, sigma_noise, cfg)
    rng = np.random.default_rng(seed)
    return int(rng.random() < p)


def log_likelihood(
    params:      PFParams,
    obs:         list[tuple[float, int]],
    sigma_noise: float = 0.15,
    cfg:         SimConfig | None = None,
) -> float:
    """Bernoulli-probit log-likelihood for binary dicing observations.

    Args:
        params:      material parameters (Gc, ell, E, beta)
        obs:         [(theta_deg, y), ...]  with y in {0, 1}
        sigma_noise: noise level in probit model
        cfg:         solver config (shared across all observations)

    Returns:
        log L = sum_i [y_i * log(p_i) + (1-y_i) * log(1-p_i)]

    The isotropic reference simulation is computed once and reused across
    all observations to avoid redundant forward calls.
    """
    cfg_ = cfg or SimConfig()
    # Reference: isotropic run with same Gc/ell/E.  Computed once per LL call.
    p_iso = PFParams(Gc=params.Gc, ell=params.ell,
                     E=params.E, nu=params.nu, beta=0.0)
    U_ref = run_forward(p_iso, cfg_).init_U

    ll = 0.0
    for theta_deg, y in obs:
        p = crack_probability(params, theta_deg, sigma_noise, cfg_, _U_ref=U_ref)
        ll += y * np.log(p) + (1 - y) * np.log(1.0 - p)
    return ll


def log_prior(params: PFParams) -> float:
    """Log-prior for (Gc, ell, beta).  LogNormal on Gc and ell; Normal on beta.

    These are vague defaults; tighten with material-specific knowledge.
    """
    lp = 0.0
    # Gc: LogNormal(mu=log(Gc), sigma=0.5) — Gc > 0
    if params.Gc <= 0:
        return -np.inf
    lp += -0.5 * (np.log(params.Gc) ** 2) / (0.5 ** 2)

    # ell: LogNormal(mu=log(ell), sigma=0.5) — ell > 0
    if params.ell <= 0:
        return -np.inf
    lp += -0.5 * (np.log(params.ell) ** 2) / (0.5 ** 2)

    # beta: Normal(0, 1)
    lp += -0.5 * params.beta ** 2

    return lp


def log_posterior(
    params:      PFParams,
    obs:         list[tuple[float, int]],
    sigma_noise: float = 0.15,
    cfg:         SimConfig | None = None,
) -> float:
    """log p(params | obs) = log_likelihood + log_prior (unnormalized)."""
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, obs, sigma_noise, cfg)


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic dataset generator (for benchmark / SMC testing)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_dataset(
    true_params:  PFParams,
    theta_list:   list[float] | None = None,
    sigma_noise:  float = 0.15,
    cfg:          SimConfig | None = None,
    seed:         int = 42,
) -> list[tuple[float, int]]:
    """Generate N binary observations from known true_params.

    Default angles: 0, 10, 20, ..., 90 deg (10 observations).
    Use this to test MAP recovery and SMC convergence.
    """
    if theta_list is None:
        theta_list = list(np.arange(0.0, 91.0, 10.0))

    rng = np.random.default_rng(seed)
    obs = []
    for i, theta in enumerate(theta_list):
        s = int(rng.integers(0, 2 ** 31))
        y = simulate_observation(true_params, theta, sigma_noise, cfg, seed=s)
        obs.append((theta, y))
    return obs


# ─────────────────────────────────────────────────────────────────────────────
#  Smoke test / demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = PFParams(Gc=8e-4, ell=0.04, E=1.0, beta=-0.50)
    cfg    = SimConfig(n_elem=400, n_steps=200, u_max_factor=6.0)
    result = run_forward(params, cfg)

    print(f"Peak stress  :  {result.peak_F:.5f}")
    print(f"Crack at U   :  {result.init_U:.5f}  (u_max={cfg.u_max_factor*params.ell:.4f})")
    print(f"Max damage   :  {np.max(result.d_field):.4f}")
    print(f"Converged    :  {result.converged}")
    print(f"eps_c estim  :  {critical_strain_estimate(params):.5f}")

    # Load–displacement + damage field
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(result.U, result.F, "b-", lw=2)
    axes[0].axvline(result.init_U, color="r", ls="--",
                    label=f"initiation U={result.init_U:.4f}")
    axes[0].set_xlabel("Displacement U"); axes[0].set_ylabel("Mean stress")
    axes[0].set_title("Load-Displacement  (1D AT2)")
    axes[0].legend(fontsize=8)

    x = np.linspace(0.0, cfg.L, len(result.d_field))
    axes[1].plot(x, result.d_field, "r-", lw=2)
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].set_xlabel("x / L"); axes[1].set_ylabel("Damage d")
    axes[1].set_title("Final damage field")

    plt.tight_layout()
    plt.savefig("at2_demo.png", dpi=150, bbox_inches="tight")
    print("Saved: at2_demo.png")

    # Anisotropy sweep
    print("\n=== Crack probability vs angle (beta=-0.50) ===")
    cfg_fast = SimConfig(n_elem=200, n_steps=100, u_max_factor=5.0)
    for t in range(0, 91, 15):
        gc_eff = gc_anisotropic(np.radians(t), params.Gc, params.beta)
        p = crack_probability(params, float(t), cfg=cfg_fast)
        bar = "█" * int(p * 20)
        print(f"  theta={t:3d}°  Gc_eff={gc_eff:.2e}  p_crack={p:.3f}  {bar}")

    # Synthetic dataset + log-likelihood at true params
    obs = generate_synthetic_dataset(params, cfg=cfg_fast, seed=0)
    ll_true  = log_likelihood(params, obs, cfg=cfg_fast)
    ll_wrong = log_likelihood(PFParams(Gc=params.Gc*2, ell=params.ell, E=params.E), obs, cfg=cfg_fast)
    print(f"\nlog L (true params): {ll_true:.4f}")
    print(f"log L (Gc*2 wrong):  {ll_wrong:.4f}")
    print(f"True params give higher LL: {ll_true > ll_wrong}")
