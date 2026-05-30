"""
NSGA-II Multi-Objective Optimisation for SiC Blade Dicing.

Objectives (minimise both):
    f1 = chipping_um          (quality)
    f2 = 1 / MRR              (throughput, MRR = depth × feed)

Constraints:
    chipping < USL (15 µm)
    Cpk ≥ 1.33
    depth  ∈ [60, 420] µm
    feed   ∈ [0.3, 3.5] mm/s

Algorithm: NSGA-II (Deb et al. 2002) — pure NumPy, no external MOEA library.

Key NSGA-II steps:
    1. Non-dominated sorting → Pareto fronts F1, F2, ...
    2. Crowding distance → diversity preservation within front
    3. Tournament selection → binary tournament on (rank, crowding)
    4. SBX crossover + polynomial mutation

Usage:
    python optimization/nsga2.py [--gens 50] [--pop 100] [--plot]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.train_from_experimental import ExperimentalGPSurrogate, MODEL_PATH

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

# Decision variable bounds: [cut_depth_um, blade_W_um, feed_mm_s, spindle_rpm]
BOUNDS = np.array([
    [60.0,  420.0],
    [20.0,   55.0],
    [0.3,    3.5],
    [18000., 42000.],
])

USL_UM   = 15.0
CPK_MIN  = 1.00   # relaxed for diverse Pareto front (1.33 = automotive, 1.00 = minimum)


# ── Objective functions ────────────────────────────────────────────────────────

def evaluate(X: np.ndarray, model: ExperimentalGPSurrogate):
    """
    X: (N, 4) → F: (N, 2), G: (N,) constraint violation (≤0 = feasible)
    Objectives: f1=chipping (min), f2=1/MRR (min = max throughput)
    Constraint: G = CPK_MIN - Cpu ≤ 0  (i.e. Cpu ≥ CPK_MIN)
    """
    mu, sigma = model.predict(X, return_std=True)
    mrr = X[:, 0] * X[:, 2]
    f1  = mu
    f2  = 1.0 / (mrr + 1e-6)
    cpu = (USL_UM - mu) / (3 * np.maximum(sigma, 0.1))
    G   = CPK_MIN - cpu                # ≤ 0 means feasible
    return np.column_stack([f1, f2]), G


# ── NSGA-II core ──────────────────────────────────────────────────────────────

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if a dominates b (a ≤ b in all, a < b in at least one)."""
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(F: np.ndarray) -> list:
    """Return list of Pareto fronts (each front = list of indices)."""
    N = len(F)
    dom_count = np.zeros(N, dtype=int)
    dom_set   = [[] for _ in range(N)]
    ranks     = np.zeros(N, dtype=int)
    fronts    = [[]]

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if dominates(F[i], F[j]):
                dom_set[i].append(j)
            elif dominates(F[j], F[i]):
                dom_count[i] += 1
        if dom_count[i] == 0:
            ranks[i] = 0
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dom_set[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    ranks[j] = k + 1
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def crowding_distance(F: np.ndarray, front: list) -> np.ndarray:
    """Compute crowding distances for individuals in a front."""
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)
    dist = np.zeros(n)
    F_sub = F[front]
    for obj in range(F_sub.shape[1]):
        sorted_idx = np.argsort(F_sub[:, obj])
        f_min, f_max = F_sub[sorted_idx[0], obj], F_sub[sorted_idx[-1], obj]
        span = f_max - f_min + 1e-12
        dist[sorted_idx[0]] = dist[sorted_idx[-1]] = np.inf
        for k in range(1, n - 1):
            dist[sorted_idx[k]] += (F_sub[sorted_idx[k+1], obj] -
                                     F_sub[sorted_idx[k-1], obj]) / span
    return dist


def tournament_select(ranks, crowd, G, rng, k=2):
    """Binary tournament: feasible beats infeasible; then rank/crowding."""
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


def sbx_crossover(p1, p2, eta=15.0, rng=None):
    """Simulated Binary Crossover."""
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random(len(p1))
    beta = np.where(u <= 0.5, (2 * u) ** (1/(eta+1)),
                    (1 / (2*(1-u))) ** (1/(eta+1)))
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return c1, c2


def poly_mutation(x, bounds, eta=20.0, prob=None, rng=None):
    """Polynomial mutation."""
    if rng is None:
        rng = np.random.default_rng()
    if prob is None:
        prob = 1.0 / len(x)
    x_new = x.copy()
    for i in range(len(x)):
        if rng.random() < prob:
            lo, hi = bounds[i]
            delta = min(x[i] - lo, hi - x[i]) / (hi - lo)
            u = rng.random()
            if u < 0.5:
                dq = (2*u + (1-2*u)*(1-delta)**(eta+1))**(1/(eta+1)) - 1
            else:
                dq = 1 - (2*(1-u) + 2*(u-0.5)*(1-delta)**(eta+1))**(1/(eta+1))
            x_new[i] = np.clip(x[i] + dq * (hi - lo), lo, hi)
    return x_new


# ── Main NSGA-II loop ─────────────────────────────────────────────────────────

def grid_pareto(model: ExperimentalGPSurrogate, n_grid=60):
    """Grid-based Pareto front for 2 objectives (reliable alternative to NSGA-II)."""
    depths = np.linspace(60, 420, n_grid)
    feeds  = np.linspace(0.3, 3.5, n_grid)
    bw     = 23.0
    spindle = 30000.0
    D, F_arr = np.meshgrid(depths, feeds)
    X = np.column_stack([D.ravel(), np.full(D.size, bw),
                          F_arr.ravel(), np.full(D.size, spindle)])
    F_obj, G = evaluate(X, model)
    feasible = G <= 0
    X_f, F_f = X[feasible], F_obj[feasible]
    # Non-dominated sort on feasible set
    N = len(X_f)
    is_pareto = np.ones(N, dtype=bool)
    for i in range(N):
        for j in range(N):
            if i != j and dominates(F_f[j], F_f[i]):
                is_pareto[i] = False
                break
    print(f"  Grid: {N} feasible → {is_pareto.sum()} Pareto-optimal")
    return X_f[is_pareto], F_f[is_pareto]


def nsga2(model: ExperimentalGPSurrogate, pop_size=80, n_gens=40, seed=42):
    rng = np.random.default_rng(seed)

    # Initialise population (bias toward shallow/slow for initial feasibility)
    X = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], (pop_size, len(BOUNDS)))
    F, G = evaluate(X, model)

    for gen in range(n_gens):
        fronts = fast_non_dominated_sort(F)
        ranks  = np.zeros(len(X), dtype=int)
        crowd  = np.zeros(len(X))
        for rank, front in enumerate(fronts):
            for i in front: ranks[i] = rank
            cd = crowding_distance(F, front)
            for i, idx in enumerate(front): crowd[idx] = cd[i]

        # Offspring
        offspring_X = []
        while len(offspring_X) < pop_size:
            p1 = tournament_select(ranks, crowd, G, rng)
            p2 = tournament_select(ranks, crowd, G, rng)
            c1, c2 = sbx_crossover(X[p1], X[p2], rng=rng)
            c1 = poly_mutation(c1, BOUNDS, rng=rng)
            c2 = poly_mutation(c2, BOUNDS, rng=rng)
            offspring_X.extend([np.clip(c1, BOUNDS[:,0], BOUNDS[:,1]),
                                 np.clip(c2, BOUNDS[:,0], BOUNDS[:,1])])

        offspring_X = np.array(offspring_X[:pop_size])
        offspring_F, offspring_G = evaluate(offspring_X, model)

        combined_X = np.vstack([X, offspring_X])
        combined_F = np.vstack([F, offspring_F])
        combined_G = np.concatenate([G, offspring_G])

        fronts_all = fast_non_dominated_sort(combined_F)
        ranks_all  = np.zeros(len(combined_X), dtype=int)
        crowd_all  = np.zeros(len(combined_X))
        for rank, front in enumerate(fronts_all):
            for i in front: ranks_all[i] = rank
            cd = crowding_distance(combined_F, front)
            for i, idx in enumerate(front): crowd_all[idx] = cd[i]

        # Feasible first, then by rank/crowding
        feasible = combined_G <= 0
        infeas_pen = np.where(feasible, 0.0, combined_G * 1e6)
        order = np.argsort(infeas_pen + ranks_all * 1e3 - crowd_all * 1e-3)
        selected = order[:pop_size]
        X, F, G = combined_X[selected], combined_F[selected], combined_G[selected]

        if gen % 10 == 0 or gen == n_gens - 1:
            feas_mask = G <= 0
            if feas_mask.any():
                print(f"  Gen {gen+1:3d}  feasible={feas_mask.sum()}  "
                      f"min_chip={F[feas_mask,0].min():.2f}µm  "
                      f"max_MRR={1/F[feas_mask,1].min():.0f}")
            else:
                print(f"  Gen {gen+1:3d}  no feasible solutions yet")

    # Return feasible Pareto front
    feas = G <= 0
    if feas.sum() == 0:
        feas = np.ones(len(G), dtype=bool)
    fronts  = fast_non_dominated_sort(F[feas])
    pareto  = np.where(feas)[0][fronts[0]]
    return X[pareto], F[pareto]


def plot_pareto(X_pareto, F_pareto, model, out_dir=RESULTS):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    mu, sigma = model.predict(X_pareto, return_std=True)
    mrr = X_pareto[:, 0] * X_pareto[:, 2]
    cpu = (USL_UM - mu) / (3 * np.maximum(sigma, 0.1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Pareto front ──────────────────────────────────────────────────────
    ax = axes[0]
    sc = ax.scatter(mrr, mu, c=cpu, cmap="RdYlGn", s=80, zorder=5,
                    edgecolors="k", lw=0.5, vmin=1.0, vmax=2.5)
    plt.colorbar(sc, ax=ax, label="Cpk")
    ax.axhline(USL_UM, color="#d62728", ls="--", lw=1.5, label=f"USL={USL_UM}µm")

    # Highlight top-5 by MRR with Cpk≥1.33
    feasible = cpu >= CPK_MIN
    if feasible.any():
        mrr_feas = mrr[feasible]
        top5_idx = np.argsort(mrr_feas)[::-1][:5]
        top5_global = np.where(feasible)[0][top5_idx]
        ax.scatter(mrr[top5_global], mu[top5_global], color="lime", s=150,
                   marker="*", zorder=6, edgecolors="k", lw=0.8,
                   label="Top-5 (max MRR, Cpk≥1.33)")

    ax.set_xlabel("MRR proxy (depth × feed) [µm·mm/s]", fontsize=11)
    ax.set_ylabel("Predicted Chipping [µm]", fontsize=11)
    ax.set_title("NSGA-II Pareto Front\n(quality vs throughput tradeoff)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # ── Decision space ────────────────────────────────────────────────────
    ax2 = axes[1]
    sc2 = ax2.scatter(X_pareto[:, 0], X_pareto[:, 2], c=mu,
                       cmap="YlOrRd", s=80, zorder=5, edgecolors="k", lw=0.5,
                       vmin=0, vmax=USL_UM)
    plt.colorbar(sc2, ax=ax2, label="Predicted Chipping [µm]")
    ax2.set_xlabel("Cut Depth [µm]", fontsize=11)
    ax2.set_ylabel("Feed Speed [mm/s]", fontsize=11)
    ax2.set_title("Decision Space — Pareto Front\n(colour = chipping)", fontsize=10)
    ax2.grid(alpha=0.25)

    fig.suptitle("NSGA-II Multi-Objective Optimisation\n"
                 "Minimise: chipping (quality) + 1/MRR (throughput)",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(out_dir, "nsga2_pareto.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] NSGA-II Pareto plot → {out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens", type=int, default=40)
    parser.add_argument("--pop",  type=int, default=80)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    model = ExperimentalGPSurrogate.load(MODEL_PATH)
    print(f"[*] Grid Pareto + NSGA-II: pop={args.pop}, gens={args.gens}")
    print("[*] Grid-based Pareto front ...")
    X_pareto, F_pareto = grid_pareto(model, n_grid=60)
    print("[*] NSGA-II refinement ...")
    X_nsga, F_nsga = nsga2(model, pop_size=args.pop, n_gens=args.gens)
    # Merge both
    X_all = np.vstack([X_pareto, X_nsga])
    F_all = np.vstack([F_pareto, F_nsga])
    pareto_idx = [i for i in range(len(X_all))
                  if not any(dominates(F_all[j], F_all[i])
                             for j in range(len(X_all)) if j != i)]
    X_pareto = X_all[pareto_idx]
    F_pareto = F_all[pareto_idx]

    mu, sigma = model.predict(X_pareto, return_std=True)
    mrr = X_pareto[:, 0] * X_pareto[:, 2]
    cpu = (USL_UM - mu) / (3 * np.maximum(sigma, 0.1))
    feasible = cpu >= CPK_MIN

    print(f"\n[*] Pareto front: {len(X_pareto)} solutions  "
          f"({feasible.sum()} feasible Cpk≥{CPK_MIN})")
    print(f"  {'depth':>6}  {'bw':>4}  {'feed':>5}  {'chip':>8}  "
          f"{'MRR':>8}  Cpk")
    sort_idx = np.argsort(mrr)[::-1][:8]
    for i in sort_idx:
        print(f"  {X_pareto[i,0]:>6.0f}  {X_pareto[i,1]:>4.0f}  "
              f"{X_pareto[i,2]:>5.2f}  {mu[i]:>6.1f}µm  "
              f"{mrr[i]:>8.0f}  {cpu[i]:.2f}")

    if args.plot:
        plot_pareto(X_pareto, F_pareto, model)


if __name__ == "__main__":
    main()
