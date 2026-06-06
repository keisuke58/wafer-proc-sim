"""
Prior Knowledge Advantage vs Dimensionality Study.

Research question: How does the advantage of human domain knowledge decay
as problem dimensionality increases?

Experiment design:
  - Synthetic d-dimensional optimisation landscape (shifted Rosenbrock)
  - Species A: initialised near true optimum (Gaussian prior, σ = 10% of range)
  - Species B: random initialisation
  - Metric: prior_advantage = (B_best - A_best) / |global_min|
             > 0  → human prior wins
             ≈ 0  → parity (convergence / fusion)
             < 0  → data-driven wins

Dimensionalities tested: d = 2, 4, 8, 16, 32, 64

Each condition: N_TRIALS independent runs → median ± IQR reported.

Usage:
    python optimization/prior_advantage_study.py [--plot]
"""

import argparse
import os
import time

import numpy as np

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

# ── Experiment hyperparameters ─────────────────────────────────────────────────
DIMS        = [2, 4, 8, 16, 32, 64]
POP_SIZE    = 80
N_GENS      = 80
N_TRIALS    = 10
PRIOR_SIGMA = 0.10   # std as fraction of [0,1] range — models "human knows roughly"
ELITE_FRAC  = 0.20
E_PERIOD    = 20


# ── Objective: shifted Rosenbrock on [0,1]^d ──────────────────────────────────

def make_landscape(d: int, seed: int = 0):
    """Return (f, x_opt) for a shifted Rosenbrock in d dimensions."""
    rng = np.random.default_rng(seed)
    shift = rng.uniform(0.1, 0.9, d)   # true optimum hidden from Species B

    def f(X: np.ndarray) -> np.ndarray:
        """X: (N, d) → (N,) objective values (minimise)."""
        # Scale to [-2, 2] for Rosenbrock, centred on shift
        Z = (X - shift) * 4.0
        val = np.sum(100 * (Z[:, 1:] - Z[:, :-1]**2)**2
                     + (1 - Z[:, :-1])**2, axis=1)
        return val

    return f, shift


# ── Differential Evolution operators (no external surrogate needed) ───────────

def sbx(p1, p2, eta=15.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random(len(p1))
    beta = np.where(u <= 0.5,
                    (2 * u) ** (1 / (eta + 1)),
                    (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    c1 = np.clip(0.5 * ((1 + beta) * p1 + (1 - beta) * p2), 0, 1)
    c2 = np.clip(0.5 * ((1 - beta) * p1 + (1 + beta) * p2), 0, 1)
    return c1, c2


def mutate(x, eta=20.0, prob=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    prob = prob or 1.0 / len(x)
    x_new = x.copy()
    for i in range(len(x)):
        if rng.random() < prob:
            delta = min(x[i], 1 - x[i])
            u = rng.random()
            if u < 0.5:
                dq = (2*u + (1-2*u)*(1-delta)**(eta+1))**(1/(eta+1)) - 1
            else:
                dq = 1 - (2*(1-u) + 2*(u-0.5)*(1-delta)**(eta+1))**(1/(eta+1))
            x_new[i] = np.clip(x[i] + dq, 0, 1)
    return x_new


# ── Single-species GA step ────────────────────────────────────────────────────

def ga_step(X, fit, rng):
    """Tournament selection + SBX + poly mutation → next generation."""
    pop = len(X)
    # Tournament selection
    idx1 = rng.integers(0, pop, pop)
    idx2 = rng.integers(0, pop, pop)
    winners = np.where(fit[idx1] <= fit[idx2], idx1, idx2)

    offspring = []
    for k in range(0, pop - 1, 2):
        c1, c2 = sbx(X[winners[k]], X[winners[k+1]], rng=rng)
        offspring.append(mutate(c1, rng=rng))
        offspring.append(mutate(c2, rng=rng))
    if len(offspring) < pop:
        offspring.append(mutate(X[winners[-1]], rng=rng))

    offspring = np.array(offspring[:pop])
    # (µ + λ) selection
    combined = np.vstack([X, offspring])
    combined_fit = np.concatenate([fit, np.zeros(len(offspring))])
    # evaluate offspring
    return combined, combined_fit, offspring


def run_species(x_init: np.ndarray, f, n_gens: int, rng,
                elite_seed=None, seed_frac=ELITE_FRAC, seed_std=0.05):
    """Evolve one species for n_gens; optionally cross-seed from elite_seed."""
    X = x_init.copy()
    pop = len(X)
    fit = f(X)

    for gen in range(n_gens):
        # Evaluate offspring
        _, _, offspring = ga_step(X, fit, rng)
        off_fit = f(offspring)
        combined = np.vstack([X, offspring])
        combined_fit = np.concatenate([fit, off_fit])
        order = np.argsort(combined_fit)
        X   = combined[order[:pop]]
        fit = combined_fit[order[:pop]]

        # Cross-seeding from the other species' elite
        if elite_seed is not None:
            n_seed = max(1, int(seed_frac * pop))
            noise = rng.normal(0, seed_std, (n_seed, X.shape[1]))
            X[-n_seed:] = np.clip(elite_seed[:n_seed] + noise, 0, 1)
            fit[-n_seed:] = f(X[-n_seed:])

        # Extinction event
        if (gen + 1) % E_PERIOD == 0:
            n_ext = max(1, int(0.20 * pop))
            X[-n_ext:] = rng.uniform(0, 1, (n_ext, X.shape[1]))
            fit[-n_ext:] = f(X[-n_ext:])

    return X, fit


# ── Single trial ──────────────────────────────────────────────────────────────

def one_trial(d: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    f, x_opt = make_landscape(d, seed=seed)

    # Species A: Gaussian prior centred on true optimum
    Xa = np.clip(
        rng.normal(x_opt, PRIOR_SIGMA, (POP_SIZE, d)), 0, 1)
    # Species B: uniform random
    Xb = rng.uniform(0, 1, (POP_SIZE, d))

    fit_a = f(Xa)
    fit_b = f(Xb)

    # Interleaved co-evolution (alternating generations with cross-seeding)
    half = N_GENS // 2
    for _ in range(half):
        elite_a = Xa[np.argsort(fit_a)[:max(1, int(ELITE_FRAC * POP_SIZE))]]
        elite_b = Xb[np.argsort(fit_b)[:max(1, int(ELITE_FRAC * POP_SIZE))]]
        Xa, fit_a = run_species(Xa, f, 2, rng, elite_seed=elite_b)
        Xb, fit_b = run_species(Xb, f, 2, rng, elite_seed=elite_a)

    a_best = float(fit_a.min())
    b_best = float(fit_b.min())
    f_opt  = float(f(x_opt.reshape(1, -1))[0])   # should be ≈ 0

    # prior_advantage > 0 means A wins (lower = better in minimisation)
    denom = max(abs(b_best), 1e-8)
    prior_advantage = (b_best - a_best) / denom

    return {
        "d": d, "seed": seed,
        "a_best": a_best, "b_best": b_best,
        "f_opt": f_opt,
        "prior_advantage": prior_advantage,
    }


# ── Main experiment ────────────────────────────────────────────────────────────

def run_experiment() -> dict:
    results = {d: [] for d in DIMS}
    total = len(DIMS) * N_TRIALS
    done = 0
    t0 = time.time()

    for d in DIMS:
        print(f"\n[d={d:2d}]", end="  ", flush=True)
        for trial in range(N_TRIALS):
            r = one_trial(d, seed=trial * 100 + d)
            results[d].append(r)
            done += 1
            print(f".", end="", flush=True)
        adv = [r["prior_advantage"] for r in results[d]]
        print(f"  median_advantage={np.median(adv):+.3f}  "
              f"(A wins: {sum(a>0 for a in adv)}/{N_TRIALS})")

    print(f"\n[✓] Elapsed: {time.time()-t0:.1f}s")
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(results: dict, out_dir=RESULTS):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    os.makedirs(out_dir, exist_ok=True)

    dims   = sorted(results.keys())
    median = []
    q25    = []
    q75    = []
    win_frac = []

    for d in dims:
        adv = [r["prior_advantage"] for r in results[d]]
        median.append(np.median(adv))
        q25.append(np.percentile(adv, 25))
        q75.append(np.percentile(adv, 75))
        win_frac.append(sum(a > 0 for a in adv) / len(adv))

    median   = np.array(median)
    q25      = np.array(q25)
    q75      = np.array(q75)
    win_frac = np.array(win_frac)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Prior advantage vs dimensionality ─────────────────────────────────
    ax = axes[0]
    ax.fill_between(dims, q25, q75, alpha=0.25, color="#2196F3",
                    label="IQR (25–75%)")
    ax.plot(dims, median, "o-", color="#2196F3", lw=2.5,
            ms=7, label="Median prior advantage")
    ax.axhline(0, color="gray", ls="--", lw=1.2,
               label="Parity (A = B)")
    ax.fill_between(dims, 0, np.maximum(median, 0), alpha=0.15,
                    color="green", label="A (prior) wins")
    ax.fill_between(dims, np.minimum(median, 0), 0, alpha=0.15,
                    color="red", label="B (data) wins")

    # Shade the "crossing" region
    cross_d = None
    for i in range(len(dims) - 1):
        if median[i] > 0 >= median[i+1]:
            cross_d = dims[i]
    if cross_d:
        ax.axvline(cross_d, color="orange", ls=":", lw=2,
                   label=f"Crossover ~d={cross_d}")

    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("Problem Dimensionality (d)", fontsize=12)
    ax.set_ylabel("Prior Advantage  (B_best − A_best) / |B_best|", fontsize=10)
    ax.set_title("Prior Knowledge Advantage vs Dimensionality\n"
                 "Species A = human prior, Species B = data-driven", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # ── Win rate ──────────────────────────────────────────────────────────
    ax2 = axes[1]
    colours = ["#2196F3" if w > 0.5 else "#FF5722" for w in win_frac]
    bars = ax2.bar([str(d) for d in dims], win_frac * 100,
                   color=colours, edgecolor="k", lw=0.5)
    ax2.axhline(50, color="gray", ls="--", lw=1.2, label="50% (parity)")
    ax2.set_xlabel("Problem Dimensionality (d)", fontsize=12)
    ax2.set_ylabel("% trials where prior (A) wins", fontsize=10)
    ax2.set_title(f"Prior Win Rate  (N={N_TRIALS} trials per d)", fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.25)
    for bar, w in zip(bars, win_frac):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1.5,
                 f"{w*100:.0f}%", ha="center", fontsize=9, fontweight="bold")

    patches = [mpatches.Patch(color="#2196F3", label="Prior (A) wins >50%"),
               mpatches.Patch(color="#FF5722", label="Data-driven (B) wins >50%")]
    ax2.legend(handles=patches, fontsize=8)

    fig.suptitle(
        "Curse of Dimensionality for Human Prior Knowledge\n"
        "Co-evolutionary GA — Shifted Rosenbrock Landscape",
        fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "prior_advantage_vs_dim.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Plot → {out}")
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    print(f"Prior Knowledge Advantage Study")
    print(f"  dims={DIMS}, pop={POP_SIZE}, gens={N_GENS}, trials={N_TRIALS}")

    results = run_experiment()

    # Summary table
    print(f"\n{'d':>4}  {'median_adv':>12}  {'win_rate':>10}  interpretation")
    print("-" * 55)
    for d in DIMS:
        adv = [r["prior_advantage"] for r in results[d]]
        med = np.median(adv)
        win = sum(a > 0 for a in adv) / len(adv)
        interp = ("Prior dominates" if win > 0.7
                  else "Parity / fusion" if win > 0.4
                  else "Data-driven wins")
        print(f"{d:>4}  {med:>+12.3f}  {win*100:>9.0f}%  {interp}")

    if args.plot:
        plot_results(results)


if __name__ == "__main__":
    main()
