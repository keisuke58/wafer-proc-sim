"""
map_estimator.py  —  Physics-informed MAP estimation of (Gc0, beta)

Approach (Novel B in thesis):
    - Forward model: AT2 1D simulator (at2_simulator.run_forward)
    - Observation model: Bernoulli-probit on binary crack/no-crack data
    - Optimization: L-BFGS-B with finite-difference gradients
    - This is "physics-informed" because the likelihood is computed from the
      full AT2 PDE solution, not a surrogate or analytic formula.

Parameters to identify:
    Gc0   — isotropic fracture energy (cleavage-axis baseline)
    beta  — anisotropy: Gc_eff(theta) = Gc0*(1 + beta*sin^2(theta))

ell and E are treated as known (from indentation / literature).
A 3-parameter variant that also fits ell is provided.

Usage:
    result = fit_map(obs, ell=0.04, E=1.0, n_restarts=5)
    print(result.params)   # PFParams with MAP (Gc0, beta)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize

from research.phasefield.at2_simulator import (
    PFParams, SimConfig, log_likelihood, log_posterior, critical_strain_estimate,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MapResult:
    params:       PFParams
    neg_log_post: float
    n_iter:       int
    success:      bool
    message:      str
    n_restarts_tried: int = 0
    all_neg_lp:   list[float] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
#  Parameter encoding  (unconstrained → PFParams)
# ─────────────────────────────────────────────────────────────────────────────

def _decode_2d(theta: np.ndarray, ell: float, E: float, nu: float = 0.2) -> PFParams:
    """theta = [log_Gc, beta]."""
    return PFParams(Gc=float(np.exp(theta[0])), ell=ell, E=E, nu=nu,
                    beta=float(theta[1]))


def _decode_3d(theta: np.ndarray, E: float, nu: float = 0.2) -> PFParams:
    """theta = [log_Gc, beta, log_ell]."""
    return PFParams(Gc=float(np.exp(theta[0])), ell=float(np.exp(theta[2])),
                    E=E, nu=nu, beta=float(theta[1]))


# ─────────────────────────────────────────────────────────────────────────────
#  Objective
# ─────────────────────────────────────────────────────────────────────────────

def _scale_aware_log_prior(params: PFParams,
                            Gc_center: float, ell_center: float,
                            sigma_log: float = 2.0) -> float:
    """LogNormal priors centered at problem-scale values.

    sigma_log=2.0 covers ±2 orders of magnitude — effectively vague.
    Using sigma=0.5 (at2_simulator default) centers at Gc=1 which is
    wrong for dimensionless Gc ~ 1e-3.  This version sets the center
    from the warm-start analytical estimate instead.
    """
    if params.Gc <= 0 or params.ell <= 0:
        return -np.inf
    lp  = -0.5 * ((np.log(params.Gc)  - np.log(Gc_center))  / sigma_log) ** 2
    lp += -0.5 * ((np.log(params.ell) - np.log(ell_center)) / sigma_log) ** 2
    lp += -0.5 * (params.beta / 1.0) ** 2   # N(0,1) on beta
    return lp


def _obj_2d(theta, obs, ell, E, sigma_noise, cfg, Gc_center):
    params = _decode_2d(theta, ell, E)
    lp = _scale_aware_log_prior(params, Gc_center, ell)
    if not np.isfinite(lp):
        return 1e10
    return -log_likelihood(params, obs, sigma_noise, cfg) - lp


def _obj_3d(theta, obs, E, sigma_noise, cfg, Gc_center, ell_center):
    params = _decode_3d(theta, E)
    lp = _scale_aware_log_prior(params, Gc_center, ell_center)
    if not np.isfinite(lp):
        return 1e10
    return -log_likelihood(params, obs, sigma_noise, cfg) - lp


# ─────────────────────────────────────────────────────────────────────────────
#  Warm-start initializer
# ─────────────────────────────────────────────────────────────────────────────

def _warm_start_Gc(ell: float, E: float) -> float:
    """Invert critical-strain formula: assume eps_c ≈ 0.05 as initial guess."""
    eps_c = 0.05
    Gc_est = (2.0 * E * ell / 3.0) * (eps_c * 8.0 / 3.0) ** 2 * (2.0 / 3.0)
    return max(Gc_est, 1e-12)


# ─────────────────────────────────────────────────────────────────────────────
#  2-parameter MAP  (Gc0, beta)  —  ell and E fixed
# ─────────────────────────────────────────────────────────────────────────────

def fit_map(
    obs:          list[tuple[float, int]],
    ell:          float,
    E:            float,
    sigma_noise:  float = 0.15,
    cfg:          SimConfig | None = None,
    n_restarts:   int = 5,
    seed:         int = 42,
    verbose:      bool = False,
) -> MapResult:
    """MAP estimate of (Gc0, beta) with ell and E fixed.

    Objective:
        theta* = argmin  -log L(obs | Gc0, beta)  -  log prior(Gc0, beta)

    Optimization space:
        theta[0] = log(Gc0)   [ensures Gc0 > 0]
        theta[1] = beta       [unconstrained]

    Uses L-BFGS-B with finite-difference Jacobian.
    n_restarts random initializations to reduce risk of local minima.

    Returns:
        MapResult with best params across all restarts.
    """
    cfg_    = cfg or SimConfig()
    rng     = np.random.default_rng(seed)

    Gc_ws   = _warm_start_Gc(ell, E)
    log_Gc0 = np.log(Gc_ws)

    # Restart initializations
    # First restart: analytical warm-start at beta=0
    # Remaining: random Gc + spread of beta starting points (include negative)
    log_Gc_inits = np.concatenate([[log_Gc0],
                                   rng.uniform(log_Gc0 - 2.0, log_Gc0 + 2.0,
                                               n_restarts - 1)])
    # Seed betas across full range including negative values (SiC-like)
    beta_spread  = np.linspace(-0.7, 0.5, n_restarts - 1)
    beta_inits   = np.concatenate([[0.0], beta_spread])

    bounds = [(np.log(1e-12), np.log(1.0)), (-0.99, 4.0)]
    # Per-parameter finite-difference step: beta needs larger eps (flat landscape near 0)
    fd_eps = np.array([1e-5, 0.05])

    best_nlp  = np.inf
    best_res  = None
    all_nlp   = []

    for i in range(n_restarts):
        theta0 = np.array([log_Gc_inits[i], beta_inits[i]])
        try:
            res = minimize(
                _obj_2d, theta0,
                args=(obs, ell, E, sigma_noise, cfg_, Gc_ws),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-6,
                         "eps": fd_eps},
            )
        except Exception as exc:
            if verbose:
                print(f"  restart {i}: error {exc}")
            continue

        all_nlp.append(float(res.fun))
        if verbose:
            p = _decode_2d(res.x, ell, E)
            print(f"  restart {i}: Gc={p.Gc:.3e} beta={p.beta:.3f} "
                  f"-logpost={res.fun:.4f} success={res.success}")

        if res.fun < best_nlp:
            best_nlp = res.fun
            best_res  = res

    if best_res is None:
        raise RuntimeError("All restarts failed")

    params = _decode_2d(best_res.x, ell, E)
    return MapResult(
        params=params,
        neg_log_post=float(best_nlp),
        n_iter=int(best_res.nit),
        success=bool(best_res.success),
        message=best_res.message,
        n_restarts_tried=n_restarts,
        all_neg_lp=all_nlp,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  3-parameter MAP  (Gc0, beta, ell)  —  only E fixed
# ─────────────────────────────────────────────────────────────────────────────

def fit_map_3d(
    obs:          list[tuple[float, int]],
    E:            float,
    sigma_noise:  float = 0.15,
    cfg:          SimConfig | None = None,
    n_restarts:   int = 5,
    seed:         int = 42,
    ell_hint:     float | None = None,
    verbose:      bool = False,
) -> MapResult:
    """MAP estimate of (Gc0, beta, ell) with only E fixed.

    theta = [log_Gc0, beta, log_ell]
    """
    cfg_    = cfg or SimConfig()
    rng     = np.random.default_rng(seed)

    ell0    = ell_hint if ell_hint is not None else cfg_.L * 0.04
    Gc_ws   = _warm_start_Gc(ell0, E)
    log_Gc0 = np.log(Gc_ws)
    log_ell0= np.log(ell0)

    log_Gc_inits  = np.concatenate([[log_Gc0],
                                    rng.uniform(log_Gc0 - 1.5, log_Gc0 + 1.5, n_restarts-1)])
    beta_inits    = np.concatenate([[0.0], rng.uniform(-0.7, 0.7, n_restarts-1)])
    log_ell_inits = np.concatenate([[log_ell0],
                                    rng.uniform(log_ell0 - 0.8, log_ell0 + 0.8, n_restarts-1)])

    bounds = [
        (np.log(1e-12), np.log(1.0)),   # log_Gc
        (-0.99, 4.0),                    # beta
        (np.log(cfg_.L * 0.005), np.log(cfg_.L * 0.3)),  # log_ell
    ]

    best_nlp, best_res = np.inf, None
    all_nlp = []

    Gc_ws_3d = _warm_start_Gc(ell0, E)

    for i in range(n_restarts):
        theta0 = np.array([log_Gc_inits[i], beta_inits[i], log_ell_inits[i]])
        try:
            res = minimize(
                _obj_3d, theta0,
                args=(obs, E, sigma_noise, cfg_, Gc_ws_3d, ell0),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-6},
            )
        except Exception as exc:
            if verbose:
                print(f"  restart {i}: error {exc}")
            continue

        all_nlp.append(float(res.fun))
        if verbose:
            p = _decode_3d(res.x, E)
            print(f"  restart {i}: Gc={p.Gc:.3e} beta={p.beta:.3f} "
                  f"ell={p.ell:.4f} -logpost={res.fun:.4f}")

        if res.fun < best_nlp:
            best_nlp = res.fun
            best_res  = res

    if best_res is None:
        raise RuntimeError("All restarts failed")

    params = _decode_3d(best_res.x, E)
    return MapResult(
        params=params,
        neg_log_post=float(best_nlp),
        n_iter=int(best_res.nit),
        success=bool(best_res.success),
        message=best_res.message,
        n_restarts_tried=n_restarts,
        all_neg_lp=all_nlp,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Recovery check utility
# ─────────────────────────────────────────────────────────────────────────────

def relative_error(estimated: PFParams, true: PFParams) -> dict[str, float]:
    """Relative errors |est - true| / |true| for Gc and beta."""
    return {
        "Gc":   abs(estimated.Gc   - true.Gc)   / max(abs(true.Gc),   1e-15),
        "beta": abs(estimated.beta - true.beta)  / max(abs(true.beta), 1e-3),
        "ell":  abs(estimated.ell  - true.ell)   / max(abs(true.ell),  1e-15),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from research.phasefield.at2_simulator import (
        generate_synthetic_dataset, log_likelihood as sim_ll,
    )

    TRUE = PFParams(Gc=8e-4, ell=0.04, E=1.0, beta=-0.40)
    CFG  = SimConfig(n_elem=100, n_steps=80, u_max_factor=5.0)

    # Use dense angle grid for better identifiability of beta
    # (Gc0 absolute value is NOT identifiable from sign-only obs alone;
    #  only beta and the relative Gc(theta)/Gc_ref are identifiable)
    theta_grid = list(np.arange(10, 91, 5.0))   # 17 angles, avoid theta=0 (uninformative)
    print(f"Generating {len(theta_grid)} synthetic binary observations ...")
    obs = generate_synthetic_dataset(TRUE, theta_list=theta_grid, cfg=CFG, seed=0)
    frac_crack = sum(y for _, y in obs) / len(obs)
    print(f"  obs = {[(int(t), y) for t,y in obs]}")
    print(f"  crack fraction = {frac_crack:.2f}  (expected > 0.5 for beta<0)")

    print("\n--- 2-param MAP: (Gc0, beta) with true ell fixed ---")
    print("Note: Gc0 absolute value is hard to identify from sign-only obs.")
    print("      beta (anisotropy) should recover well.\n")
    res = fit_map(obs, ell=TRUE.ell, E=TRUE.E, cfg=CFG, n_restarts=6, verbose=True)
    errs = relative_error(res.params, TRUE)
    print(f"\nMAP result:")
    print(f"  Gc   = {res.params.Gc:.4e}  (true: {TRUE.Gc:.4e})  err={errs['Gc']*100:.0f}%")
    print(f"  beta = {res.params.beta:.4f}  (true: {TRUE.beta:.4f})  err={errs['beta']*100:.0f}%")
    print(f"  -log post = {res.neg_log_post:.4f}  n_iter = {res.n_iter}  success = {res.success}")
    print(f"  MAP LL = {sim_ll(res.params, obs, cfg=CFG):.4f}  "
          f"true LL = {sim_ll(TRUE, obs, cfg=CFG):.4f}")
