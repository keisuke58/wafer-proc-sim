"""
Hybrid laser+plasma process — NSGA-II multi-objective optimisation.

Decision variables (8):
    laser_W, scan_mm_s, pulse_kHz, n_passes,
    plasma_s, n_cycles, mask_um, wafer_um

Objectives (4, all minimised internally):
    f1 = street_width_um      (min → die yield ↑)
    f2 = -die_strength_MPa    (min → strength ↑)
    f3 = HAZ_depth_um         (min → damage ↓)
    f4 = -throughput_mm2_s    (min → throughput ↑)

Constraints:
    street_width_um  < 15 µm  (advanced node requirement)
    die_strength_MPa > 300 MPa
    total processing feasibility (wafer fully etched)

Algorithm: NSGA-II (non-dominated sorting + crowding distance)
  Uses primitives from optimization/nsga2.py adapted for 4-objective problem.

Usage:
    python optimization/hybrid_process_opt.py --gens 100 --pop 200 --plot
    python optimization/hybrid_process_opt.py --gens 50  --pop 100  --plot --no-loo
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.hybrid_process_gp import HybridGPSurrogate, BOUNDS, TARGET_COLS, MODEL_PATH

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

# Constraint limits
SW_MAX_UM      = 30.0    # street width upper spec limit [µm]  (vs blade ~80µm)
STRENGTH_MIN   = 120.0   # die strength lower spec limit [MPa]
DELAM_MAX      = 0.30    # ③ delamination risk upper limit (0–1)


# ── Objective evaluation ──────────────────────────────────────────────────────

def evaluate(X: np.ndarray, model: HybridGPSurrogate):
    """
    X : (N, 8) decision variable matrix
    Returns F: (N, 4) objective values (all to minimise)
            G: (N, 2) constraint violations (≤ 0 = feasible)
    """
    mu, sigma = model.predict(X, return_std=True)
    # mu columns: [street_width, die_strength, HAZ_depth, throughput]

    street_w = mu[:, 0]
    strength = mu[:, 1]
    haz      = mu[:, 2]
    tput     = mu[:, 3]
    delam    = mu[:, 4]

    F = np.column_stack([
        street_w,           # f1: minimise street width
        -strength,          # f2: maximise strength
        haz,                # f3: minimise HAZ depth
        -tput,              # f4: maximise throughput
    ])

    # Constraint violations (≤ 0 = feasible)
    G = np.column_stack([
        street_w - SW_MAX_UM,           # g1: street_width ≤ 30 µm
        STRENGTH_MIN - strength,        # g2: strength ≥ 120 MPa
        delam - DELAM_MAX,              # g3: ③ delamination_risk ≤ 0.30
    ])

    return F, G


# ── NSGA-II primitives ────────────────────────────────────────────────────────

def _fast_nondominated_sort(F: np.ndarray):
    """Return list of Pareto fronts (each a list of indices)."""
    n = len(F)
    S = [[] for _ in range(n)]
    n_dom = np.zeros(n, dtype=int)
    rank  = np.zeros(n, dtype=int)
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(F[p], F[q]):
                S[p].append(q)
            elif _dominates(F[q], F[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return [f for f in fronts if f], rank


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if a dominates b (a ≤ b in all objectives, a < b in at least one)."""
    return bool(np.all(a <= b) and np.any(a < b))


def _crowding_distance(F_front: np.ndarray) -> np.ndarray:
    """Crowding distance for a single Pareto front."""
    n, m = F_front.shape
    dist = np.zeros(n)
    for obj in range(m):
        order = np.argsort(F_front[:, obj])
        f_min = F_front[order[0],  obj]
        f_max = F_front[order[-1], obj]
        dist[order[0]]  = dist[order[-1]] = np.inf
        if f_max - f_min < 1e-10:
            continue
        for k in range(1, n - 1):
            dist[order[k]] += (F_front[order[k+1], obj] -
                                F_front[order[k-1], obj]) / (f_max - f_min)
    return dist


def _tournament_select(pop: np.ndarray, rank: np.ndarray,
                        crowd: np.ndarray, rng) -> np.ndarray:
    """Binary tournament selection → parent indices."""
    n = len(pop)
    idx = rng.integers(0, n, (n, 2))
    winners = np.where(
        (rank[idx[:, 0]] < rank[idx[:, 1]]) |
        ((rank[idx[:, 0]] == rank[idx[:, 1]]) & (crowd[idx[:, 0]] > crowd[idx[:, 1]])),
        idx[:, 0], idx[:, 1])
    return winners


def _sbx_crossover(p1: np.ndarray, p2: np.ndarray,
                    rng, eta_c: float = 20.0) -> tuple:
    """Simulated Binary Crossover."""
    n = len(p1)
    c1, c2 = p1.copy(), p2.copy()
    u = rng.random(n)
    beta = np.where(
        u <= 0.5,
        (2.0 * u) ** (1.0 / (eta_c + 1)),
        (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1)))
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return c1, c2


def _polynomial_mutation(x: np.ndarray, bounds: np.ndarray,
                           rng, eta_m: float = 20.0,
                           p_mut: float = None) -> np.ndarray:
    """Polynomial mutation."""
    n = len(x)
    if p_mut is None:
        p_mut = 1.0 / n
    xm = x.copy()
    lo, hi = bounds[:, 0], bounds[:, 1]
    for i in range(n):
        if rng.random() < p_mut:
            u = rng.random()
            delta = hi[i] - lo[i]
            if u < 0.5:
                delta_q = (2 * u) ** (1 / (eta_m + 1)) - 1.0
            else:
                delta_q = 1.0 - (2 * (1 - u)) ** (1 / (eta_m + 1))
            xm[i] = np.clip(x[i] + delta_q * delta, lo[i], hi[i])
    # Round integer variables (n_passes col 3, n_cycles col 5)
    xm[3] = round(xm[3])
    xm[5] = round(xm[5])
    return xm


# ── Main NSGA-II loop ─────────────────────────────────────────────────────────

def run_nsga2(model: HybridGPSurrogate,
              pop_size: int = 200,
              n_gens: int = 100,
              seed: int = 0) -> pd.DataFrame:
    """
    Run NSGA-II and return the final Pareto front as DataFrame.
    """
    rng = np.random.default_rng(seed)
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]

    # ── Initialise population ─────────────────────────────────────────────────
    pop = rng.uniform(lo, hi, (pop_size, len(BOUNDS)))
    pop[:, 3] = np.round(pop[:, 3]).clip(1, 5)    # n_passes integer
    pop[:, 5] = np.round(pop[:, 5]).clip(10, 100)  # n_cycles integer

    F, G = evaluate(pop, model)

    # Handle constraint violations via penalty (additive to all objectives)
    penalty = np.sum(np.maximum(G, 0), axis=1, keepdims=True) * 1e3
    F_pen   = F + penalty

    fronts, rank = _fast_nondominated_sort(F_pen)
    crowd = np.zeros(pop_size)
    for front in fronts:
        if len(front) > 1:
            d = _crowding_distance(F_pen[front])
            for k, idx in enumerate(front):
                crowd[idx] = d[k]

    for gen in range(n_gens):
        # ── Selection ──────────────────────────────────────────────────────────
        parent_idx = _tournament_select(pop, rank, crowd, rng)
        # ── Crossover + mutation ───────────────────────────────────────────────
        offspring = []
        for i in range(0, pop_size, 2):
            p1 = pop[parent_idx[i]]
            p2 = pop[parent_idx[min(i + 1, pop_size - 1)]]
            c1, c2 = _sbx_crossover(p1, p2, rng)
            c1 = _polynomial_mutation(np.clip(c1, lo, hi), BOUNDS, rng)
            c2 = _polynomial_mutation(np.clip(c2, lo, hi), BOUNDS, rng)
            offspring.extend([c1, c2])
        offspring = np.array(offspring[:pop_size])

        # ── Evaluate offspring ─────────────────────────────────────────────────
        F_off, G_off = evaluate(offspring, model)
        pen_off      = np.sum(np.maximum(G_off, 0), axis=1, keepdims=True) * 1e3
        F_off_pen    = F_off + pen_off

        # ── Combine parent + offspring ─────────────────────────────────────────
        pop_comb   = np.vstack([pop, offspring])
        F_comb     = np.vstack([F_pen, F_off_pen])
        fronts_c, rank_c = _fast_nondominated_sort(F_comb)

        crowd_c = np.zeros(len(pop_comb))
        for front in fronts_c:
            if len(front) > 1:
                d = _crowding_distance(F_comb[front])
                for k, idx in enumerate(front):
                    crowd_c[idx] = d[k]

        # ── Select best pop_size individuals ──────────────────────────────────
        selected = []
        for front in fronts_c:
            if len(selected) + len(front) <= pop_size:
                selected.extend(front)
            else:
                needed = pop_size - len(selected)
                d      = _crowding_distance(F_comb[front])
                order  = np.argsort(-d)
                selected.extend([front[k] for k in order[:needed]])
                break

        pop   = pop_comb[selected]
        F_pen = F_comb[selected]
        fronts, rank = _fast_nondominated_sort(F_pen)
        crowd = np.zeros(pop_size)
        for front in fronts:
            if len(front) > 1:
                d = _crowding_distance(F_pen[front])
                for k, idx in enumerate(front):
                    crowd[idx] = d[k]

        if gen % 10 == 0:
            n_pareto = len(fronts[0]) if fronts else 0
            best_sw  = -F_pen[fronts[0], 0].max() if fronts else float('nan')
            print(f"  Gen {gen:4d} | Pareto front size: {n_pareto:3d} | "
                  f"best street_w: {-best_sw:.2f} µm")

    # ── Extract Pareto front (rank 0) ─────────────────────────────────────────
    F_true, G_true = evaluate(pop, model)
    mu             = model.predict(pop)
    pareto_idx     = fronts[0] if fronts else list(range(len(pop)))

    from ml.hybrid_process_gp import FEATURE_COLS as cols_in
    df_out  = pd.DataFrame(pop[pareto_idx], columns=cols_in)
    for j, name in enumerate(TARGET_COLS):
        df_out[name] = mu[pareto_idx, j]
    df_out["feasible"] = np.all(G_true[pareto_idx] <= 0, axis=1)

    return df_out.sort_values("street_width_um")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_pareto(df: pd.DataFrame, out_dir: str = RESULTS):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    os.makedirs(out_dir, exist_ok=True)
    feas = df[df["feasible"]]
    infeas = df[~df["feasible"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # f1 vs f2: street width vs die strength
    ax = axes[0]
    sc = ax.scatter(feas["street_width_um"], feas["die_strength_MPa"],
                    c=feas["throughput_mm2_s"], cmap='viridis', s=30, label="Feasible")
    ax.scatter(infeas["street_width_um"], infeas["die_strength_MPa"],
               marker='x', c='red', s=20, label="Infeasible")
    plt.colorbar(sc, ax=ax, label="Throughput [mm²/s]")
    ax.axvline(15.0, color='r', ls='--', lw=0.8, label="SW limit 15µm")
    ax.axhline(300.0, color='g', ls='--', lw=0.8, label="Strength min 300MPa")
    ax.set(xlabel="Street width [µm]", ylabel="Die strength [MPa]",
           title="Street width vs Die strength")
    ax.legend(fontsize=7)

    # f1 vs f3: street width vs HAZ depth
    ax = axes[1]
    sc = ax.scatter(feas["street_width_um"], feas["HAZ_depth_um"],
                    c=feas["die_strength_MPa"], cmap='plasma', s=30)
    plt.colorbar(sc, ax=ax, label="Die strength [MPa]")
    ax.set(xlabel="Street width [µm]", ylabel="HAZ depth [µm]",
           title="Street width vs HAZ depth")

    # f3 vs f4: HAZ depth vs throughput
    ax = axes[2]
    sc = ax.scatter(feas["HAZ_depth_um"], feas["throughput_mm2_s"],
                    c=feas["street_width_um"], cmap='coolwarm', s=30)
    plt.colorbar(sc, ax=ax, label="Street width [µm]")
    ax.set(xlabel="HAZ depth [µm]", ylabel="Throughput [mm²/s]",
           title="HAZ depth vs Throughput")

    plt.suptitle("Hybrid Laser+Plasma Process — NSGA-II Pareto Front", fontsize=12)
    plt.tight_layout()
    out = os.path.join(out_dir, "hybrid_pareto_front.png")
    plt.savefig(out, dpi=150)
    print(f"Pareto plot → {out}")

    # Parallel coordinates of the full Pareto front
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    cols_plot = ["street_width_um", "die_strength_MPa", "HAZ_depth_um", "throughput_mm2_s"]
    norm_df   = (feas[cols_plot] - feas[cols_plot].min()) / \
                (feas[cols_plot].max() - feas[cols_plot].min() + 1e-9)
    for _, row in norm_df.iterrows():
        ax2.plot(range(len(cols_plot)), row.values, alpha=0.3, lw=0.8, color='steelblue')
    ax2.set_xticks(range(len(cols_plot)))
    ax2.set_xticklabels(cols_plot, rotation=15, fontsize=9)
    ax2.set_ylabel("Normalised value (0=best for each)")
    ax2.set_title("Parallel coordinates — Pareto front solutions")
    out2 = os.path.join(out_dir, "hybrid_pareto_parallel.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=150)
    print(f"Parallel coord plot → {out2}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens",   type=int,  default=100)
    ap.add_argument("--pop",    type=int,  default=200)
    ap.add_argument("--plot",   action="store_true")
    ap.add_argument("--no-loo", action="store_true",
                    help="Skip LOO if model not yet trained")
    ap.add_argument("--retrain", action="store_true",
                    help="Regenerate data and retrain GP before optimising")
    ap.add_argument("--seed",   type=int,  default=0)
    args = ap.parse_args()

    if args.retrain or not os.path.exists(MODEL_PATH):
        print("Generating synthetic data...")
        from fem.plasma_bosch_model import generate_dataset
        from ml.hybrid_process_gp import DEFAULT_CSV, train
        df_syn = generate_dataset(n=400)
        os.makedirs(os.path.dirname(DEFAULT_CSV), exist_ok=True)
        df_syn.to_csv(DEFAULT_CSV, index=False)
        model = train(run_loo=not args.no_loo, plot=False)
    else:
        print(f"Loading model from {MODEL_PATH}")
        model = HybridGPSurrogate.load()

    print(f"\nRunning NSGA-II (pop={args.pop}, gens={args.gens})...")
    pareto_df = run_nsga2(model, pop_size=args.pop,
                           n_gens=args.gens, seed=args.seed)

    os.makedirs(RESULTS, exist_ok=True)
    csv_out = os.path.join(RESULTS, "hybrid_pareto_front.csv")
    pareto_df.to_csv(csv_out, index=False)
    print(f"\nPareto front saved → {csv_out}  ({len(pareto_df)} solutions)")
    print("\nTop 5 by street width (feasible):")
    feas = pareto_df[pareto_df["feasible"]]
    if len(feas):
        print(feas[["street_width_um", "die_strength_MPa",
                     "HAZ_depth_um", "throughput_mm2_s",
                     "laser_W", "scan_mm_s", "plasma_s"]].head(5).to_string(index=False))

    if args.plot:
        plot_pareto(pareto_df)
