"""
Co-evolutionary Multi-objective Optimisation for SiC Dicing.

Two populations compete and cooperate across generations:
  - Species A ("先行知識 / human prior"):  physics-guided agents initialised
    near the rule-of-thumb optimum.  They exploit domain knowledge.
  - Species B ("データ駆動 / machine"): random-initialised agents that explore
    the full parameter space via GP uncertainty.

Interaction rules (analogous to Lotka–Volterra co-evolution):
  - Cooperation  : best solutions from each species seed the other's elite pool.
  - Competition  : both species share the same feasibility constraint; the
    globally dominated are pruned from the final Pareto front.
  - Extinction events (every E_PERIOD gens): the weakest 20 % of each species
    is re-seeded from the other's elite — models speciation under pressure.

Objectives (minimise both):
    f1 = chipping_um           (quality)
    f2 = 1 / MRR              (throughput, MRR = depth × feed)

Usage:
    python optimization/coevolution.py [--gens 60] [--pop 60] [--plot]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.train_from_experimental import ExperimentalGPSurrogate, MODEL_PATH
from optimization.nsga2 import (
    fast_non_dominated_sort,
    crowding_distance,
    sbx_crossover,
    poly_mutation,
    dominates,
    USL_UM,
    CPK_MIN,
)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

BOUNDS = np.array([
    [60.0,  420.0],   # cut_depth_um
    [20.0,   55.0],   # blade_W_um
    [0.3,    3.5],    # feed_mm_s
    [18000., 42000.], # spindle_rpm
])

# Physics prior: rule-of-thumb centre for Species A
PRIOR_CENTRE = np.array([150.0, 30.0, 1.5, 30000.0])
PRIOR_STD    = np.array([ 40.0,  5.0, 0.4,  4000.0])

ELITE_FRAC  = 0.2   # fraction kept in elite pool for cross-seeding
E_PERIOD    = 15    # extinction event every N generations


# ── Objective / constraint evaluation ─────────────────────────────────────────

def evaluate(X: np.ndarray, model: ExperimentalGPSurrogate):
    """X: (N, 4) → F: (N, 2), G: (N,) constraint violation (≤0 = feasible)."""
    mu, sigma = model.predict(X, return_std=True)
    mrr = X[:, 0] * X[:, 2]
    f1  = mu
    f2  = 1.0 / (mrr + 1e-6)
    cpu = (USL_UM - mu) / (3 * np.maximum(sigma, 0.1))
    G   = CPK_MIN - cpu
    return np.column_stack([f1, f2]), G


# ── Uncertainty-weighted fitness bonus (exploitation of GP epistemic info) ─────

def uncertainty_bonus(X: np.ndarray, model: ExperimentalGPSurrogate,
                      weight: float = 0.05) -> np.ndarray:
    """Species B bonus: reward exploration by adding σ-scaled objective shift."""
    _, sigma = model.predict(X, return_std=True)
    sigma1d = sigma[:, 0] if sigma.ndim == 2 else sigma
    return -weight * sigma1d   # lower f1 proxy → encourage uncertain regions


# ── Tournament selection (shared utility) ─────────────────────────────────────

def tournament_select(ranks, crowd, G, rng, k=2):
    idx = rng.integers(0, len(ranks), k)
    for i in range(k - 1):
        a, b = idx[i], idx[i + 1]
        fa, fb = G[a] <= 0, G[b] <= 0
        if fa and not fb:
            idx[i + 1] = idx[i]
        elif fb and not fa:
            pass
        elif ranks[a] < ranks[b]:
            idx[i + 1] = idx[i]
        elif ranks[a] == ranks[b] and crowd[a] > crowd[b]:
            idx[i + 1] = idx[i]
    return int(idx[-1])


# ── Single-species NSGA-II step ────────────────────────────────────────────────

def _evolve_one_generation(X, F, G, model, rng, f1_bonus=None):
    """One NSGA-II generation for a single species; returns (X', F', G')."""
    pop_size = len(X)
    F_eff = F.copy()
    if f1_bonus is not None:
        F_eff[:, 0] += f1_bonus

    fronts = fast_non_dominated_sort(F_eff)
    ranks  = np.zeros(pop_size, dtype=int)
    crowd  = np.zeros(pop_size)
    for rank, front in enumerate(fronts):
        for i in front:
            ranks[i] = rank
        cd = crowding_distance(F_eff, front)
        for i, idx in enumerate(front):
            crowd[idx] = cd[i]

    offspring = []
    while len(offspring) < pop_size:
        p1 = tournament_select(ranks, crowd, G, rng)
        p2 = tournament_select(ranks, crowd, G, rng)
        c1, c2 = sbx_crossover(X[p1], X[p2], rng=rng)
        c1 = poly_mutation(c1, BOUNDS, rng=rng)
        c2 = poly_mutation(c2, BOUNDS, rng=rng)
        offspring.extend([np.clip(c1, BOUNDS[:, 0], BOUNDS[:, 1]),
                          np.clip(c2, BOUNDS[:, 0], BOUNDS[:, 1])])

    offspring = np.array(offspring[:pop_size])
    off_F, off_G = evaluate(offspring, model)

    combined_X = np.vstack([X, offspring])
    combined_F = np.vstack([F, off_F])
    combined_G = np.concatenate([G, off_G])
    combined_Feff = np.vstack([F_eff, off_F])

    fronts_all = fast_non_dominated_sort(combined_Feff)
    ranks_all  = np.zeros(len(combined_X), dtype=int)
    crowd_all  = np.zeros(len(combined_X))
    for rank, front in enumerate(fronts_all):
        for i in front:
            ranks_all[i] = rank
        cd = crowding_distance(combined_Feff, front)
        for i, idx in enumerate(front):
            crowd_all[idx] = cd[i]

    feasible  = combined_G <= 0
    infeas_pen = np.where(feasible, 0.0, combined_G * 1e6)
    order     = np.argsort(infeas_pen + ranks_all * 1e3 - crowd_all * 1e-3)
    sel       = order[:pop_size]
    return combined_X[sel], combined_F[sel], combined_G[sel]


# ── Elite pool ────────────────────────────────────────────────────────────────

def get_elite(X, G, frac=ELITE_FRAC):
    feas = np.where(G <= 0)[0]
    n_elite = max(1, int(frac * len(X)))
    if len(feas) >= n_elite:
        return X[feas[:n_elite]]
    return X[:n_elite]


# ── Co-evolutionary main loop ─────────────────────────────────────────────────

def coevolve(model: ExperimentalGPSurrogate,
             pop_size: int = 60,
             n_gens: int = 60,
             seed: int = 42):
    rng = np.random.default_rng(seed)

    # Species A: physics-guided initialisation (Gaussian around prior)
    Xa = np.clip(rng.normal(PRIOR_CENTRE, PRIOR_STD, (pop_size, 4)),
                 BOUNDS[:, 0], BOUNDS[:, 1])
    # Species B: random exploration
    Xb = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], (pop_size, 4))

    Fa, Ga = evaluate(Xa, model)
    Fb, Gb = evaluate(Xb, model)

    history = {"a_best_chip": [], "b_best_chip": [], "elite_shared": []}

    for gen in range(n_gens):
        # ── Species-specific bonus: B gets uncertainty reward ─────────────
        bonus_b = uncertainty_bonus(Xb, model)

        Xa, Fa, Ga = _evolve_one_generation(Xa, Fa, Ga, model, rng)
        Xb, Fb, Gb = _evolve_one_generation(Xb, Fb, Gb, model, rng,
                                             f1_bonus=bonus_b)

        # ── Cross-seeding: elite of A seeds B, and vice versa ─────────────
        elite_a = get_elite(Xa, Ga)
        elite_b = get_elite(Xb, Gb)

        n_seed = max(1, int(ELITE_FRAC * pop_size))
        # Weakest of B replaced by mutated elite of A
        Xb[-n_seed:] = np.clip(
            elite_a[:n_seed] + rng.normal(0, PRIOR_STD * 0.2, (n_seed, 4)),
            BOUNDS[:, 0], BOUNDS[:, 1])
        # Weakest of A replaced by mutated elite of B
        Xa[-n_seed:] = np.clip(
            elite_b[:n_seed] + rng.normal(0, PRIOR_STD * 0.5, (n_seed, 4)),
            BOUNDS[:, 0], BOUNDS[:, 1])
        Fa[-n_seed:], Ga[-n_seed:] = evaluate(Xa[-n_seed:], model)
        Fb[-n_seed:], Gb[-n_seed:] = evaluate(Xb[-n_seed:], model)

        # ── Extinction event: re-seed weakest 20% ─────────────────────────
        if (gen + 1) % E_PERIOD == 0:
            n_ext = max(1, int(0.2 * pop_size))
            # A: weakest → random near prior
            Xa[-n_ext:] = np.clip(
                rng.normal(PRIOR_CENTRE, PRIOR_STD * 1.5, (n_ext, 4)),
                BOUNDS[:, 0], BOUNDS[:, 1])
            # B: weakest → pure random
            Xb[-n_ext:] = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], (n_ext, 4))
            Fa[-n_ext:], Ga[-n_ext:] = evaluate(Xa[-n_ext:], model)
            Fb[-n_ext:], Gb[-n_ext:] = evaluate(Xb[-n_ext:], model)

        # ── Logging ──────────────────────────────────────────────────────
        feas_a = Fa[Ga <= 0, 0] if (Ga <= 0).any() else np.array([np.nan])
        feas_b = Fb[Gb <= 0, 0] if (Gb <= 0).any() else np.array([np.nan])
        history["a_best_chip"].append(float(np.nanmin(feas_a)))
        history["b_best_chip"].append(float(np.nanmin(feas_b)))
        n_shared = len(elite_a) + len(elite_b)
        history["elite_shared"].append(n_shared)

        if gen % 10 == 0 or gen == n_gens - 1:
            print(f"  Gen {gen+1:3d}  "
                  f"A(prior) best={np.nanmin(feas_a):.2f}µm  "
                  f"B(data)  best={np.nanmin(feas_b):.2f}µm  "
                  f"shared={n_shared}")

    # ── Merge both species and extract final Pareto front ─────────────────
    X_all = np.vstack([Xa, Xb])
    F_all = np.vstack([Fa, Fb])
    G_all = np.concatenate([Ga, Gb])
    label = np.array(["A"] * pop_size + ["B"] * pop_size)

    feas_mask = G_all <= 0
    if feas_mask.sum() < 2:
        feas_mask = np.ones(len(G_all), dtype=bool)

    X_f, F_f, L_f = X_all[feas_mask], F_all[feas_mask], label[feas_mask]
    fronts = fast_non_dominated_sort(F_f)
    pareto_idx = fronts[0]

    return X_f[pareto_idx], F_f[pareto_idx], L_f[pareto_idx], history


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_coevolution(X_pareto, F_pareto, labels, history, model, out_dir=RESULTS):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    mu, sigma = model.predict(X_pareto, return_std=True)
    mu1d    = mu[:, 0]    if mu.ndim    == 2 else mu
    sigma1d = sigma[:, 0] if sigma.ndim == 2 else sigma
    mrr = X_pareto[:, 0] * X_pareto[:, 2]
    cpu = (USL_UM - mu1d) / (3 * np.maximum(sigma1d, 0.1))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # ── Pareto front coloured by species origin ────────────────────────────
    ax = axes[0]
    colours = np.where(labels == "A", "#2196F3", "#FF5722")
    ax.scatter(mrr, mu1d, c=colours, s=90, edgecolors="k",
               lw=0.5, zorder=5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2196F3", label="Species A (human prior)"),
                        Patch(color="#FF5722", label="Species B (data-driven)")],
              fontsize=8)
    ax.axhline(USL_UM, color="#d62728", ls="--", lw=1.5,
               label=f"USL={USL_UM}µm")
    ax.set_xlabel("MRR proxy [µm·mm/s]", fontsize=10)
    ax.set_ylabel("Predicted Chipping [µm]", fontsize=10)
    ax.set_title("Co-evolutionary Pareto Front\n(species origin)", fontsize=10)
    ax.grid(alpha=0.25)

    # ── Convergence history: competition dynamics ──────────────────────────
    ax2 = axes[1]
    gens = np.arange(1, len(history["a_best_chip"]) + 1)
    ax2.plot(gens, history["a_best_chip"], color="#2196F3",
             lw=2, label="Species A (prior)")
    ax2.plot(gens, history["b_best_chip"], color="#FF5722",
             lw=2, label="Species B (data)")
    ax2.set_xlabel("Generation", fontsize=10)
    ax2.set_ylabel("Best Chipping [µm]", fontsize=10)
    ax2.set_title("Evolutionary Competition\n(lower = better survival)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)
    # Mark extinction events
    for ext_gen in range(E_PERIOD, len(gens), E_PERIOD):
        ax2.axvline(ext_gen, color="gray", ls=":", alpha=0.6)
    ax2.text(0.02, 0.05, "ext = extinction event", transform=ax2.transAxes,
             color="gray", fontsize=7)

    # ── Cpk distribution by species ────────────────────────────────────────
    ax3 = axes[2]
    cpk_a = cpu[labels == "A"]
    cpk_b = cpu[labels == "B"]
    ax3.hist(cpk_a, bins=12, alpha=0.6, color="#2196F3",
             label=f"A (n={len(cpk_a)})", edgecolor="k", lw=0.4)
    ax3.hist(cpk_b, bins=12, alpha=0.6, color="#FF5722",
             label=f"B (n={len(cpk_b)})", edgecolor="k", lw=0.4)
    ax3.axvline(1.33, color="red", ls="--", lw=1.5, label="Cpk 1.33")
    ax3.set_xlabel("Cpk", fontsize=10)
    ax3.set_ylabel("Count", fontsize=10)
    ax3.set_title("Process Capability Distribution\nby Species", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.25)

    fig.suptitle(
        "Co-evolutionary Optimisation — SiC Blade Dicing\n"
        "Human Prior (A) vs Data-Driven (B): competition → convergence → fusion",
        fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "coevolution_pareto.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Co-evolution plot → {out}")
    return out


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens", type=int, default=60)
    parser.add_argument("--pop",  type=int, default=60)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    model = ExperimentalGPSurrogate.load(MODEL_PATH)
    print(f"[*] Co-evolutionary NSGA-II: pop={args.pop}×2, gens={args.gens}")
    print(f"    Species A = physics prior,  Species B = data-driven exploration")

    X_pareto, F_pareto, labels, history = coevolve(
        model, pop_size=args.pop, n_gens=args.gens)

    mu, sigma = model.predict(X_pareto, return_std=True)
    mu1d    = mu[:, 0]    if mu.ndim    == 2 else mu
    sigma1d = sigma[:, 0] if sigma.ndim == 2 else sigma
    mrr = X_pareto[:, 0] * X_pareto[:, 2]
    cpu = (USL_UM - mu1d) / (3 * np.maximum(sigma1d, 0.1))

    feas = cpu >= CPK_MIN
    print(f"\n[*] Final Pareto: {len(X_pareto)} pts  ({feas.sum()} Cpk≥{CPK_MIN})")
    print(f"    Species A: {(labels=='A').sum()}  Species B: {(labels=='B').sum()}")
    print(f"\n  {'depth':>6}  {'feed':>5}  {'chip':>8}  {'MRR':>8}  Cpk   origin")
    for i in np.argsort(mrr)[::-1][:8]:
        print(f"  {X_pareto[i,0]:>6.0f}  {X_pareto[i,2]:>5.2f}  "
              f"{mu1d[i]:>6.1f}µm  {mrr[i]:>8.0f}  "
              f"{cpu[i]:.2f}   {labels[i]}")

    if args.plot:
        plot_coevolution(X_pareto, F_pareto, labels, history, model)


if __name__ == "__main__":
    main()
