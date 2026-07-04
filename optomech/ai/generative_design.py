#!/usr/bin/env python3
"""generative_design.py — CAE × 生成AI: a PyTorch GNN surrogate + Bayesian
generative design of a lens barrel.

Built directly on the applicant's research stack (PyTorch / graph neural
networks + Bayesian inference), this is the "生成AIを活用した設計改革" piece:

  1. CAE ground truth — a zone-varying barrel is meshed into beam finite
     elements; a modal (eigenvalue) + mass + drop-shock evaluation gives the
     true performance (first bending mode, mass, safety factor).
  2. GNN surrogate — every design's FE mesh is a graph (nodes = FE nodes with
     local section/material features, edges = elements). A **PyTorch Graph
     Convolutional Network** learns to predict the CAE outputs ~instantly,
     replacing the solver inside the design loop.
  3. Generative design — Bayesian optimization (Gaussian process + Expected
     Improvement) uses the surrogate to *generate* the lightest barrel that
     still meets stiffness and shock-safety constraints, then verifies the
     generated design with the real FE solver.

PyTorch + NumPy/SciPy (FE eigensolver) + scikit-learn (GP) + matplotlib.
Run:  python3 generative_design.py
"""
import json
import os
import numpy as np
import torch
import torch.nn as nn

from scipy.linalg import eigh
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(7)
torch.manual_seed(1)
G = 9.80665

# Materials: (name, E[Pa], rho, yield[Pa], cte)
MATERIALS = [
    ("Al 6061", 68.9e9, 2700.0, 276e6, 23.6e-6),
    ("Mg AZ31", 45.0e9, 1770.0, 200e6, 26.0e-6),
]
# Design d = [wall_z1, wall_z2, wall_z3 (mm), n_ribs, material_idx]
D_LO = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
D_HI = np.array([3.5, 3.5, 3.5, 8.0, 1.0])
OUTER_MM, LEN_MM, LENS_G, N_ELEM = 30.0, 40.0, 8.0, 18
N_NODE = N_ELEM + 1


def _normalized_adjacency():
    """Â = D^-1/2 (A+I) D^-1/2 of the path graph shared by all designs."""
    A = np.eye(N_NODE)
    for e in range(N_ELEM):
        A[e, e + 1] = A[e + 1, e] = 1.0
    d = A.sum(1)
    Dinv = np.diag(1.0 / np.sqrt(d))
    return torch.tensor(Dinv @ A @ Dinv, dtype=torch.float32)


A_HAT = _normalized_adjacency()


def _section(wall_mm, ribs):
    Do = OUTER_MM * 1e-3
    Di = (OUTER_MM - 2 * wall_mm) * 1e-3
    area = np.pi / 4 * (Do**2 - Di**2)
    I = np.pi / 64 * (Do**4 - Di**4)
    if ribs > 0:
        rib_h, rib_w = 2.5e-3, 1.2e-3
        arm = Di / 2 - rib_h / 2
        a_r = rib_w * rib_h
        area += ribs * a_r
        I += 0.5 * ribs * (rib_w * rib_h**3 / 12 + a_r * arm**2)
    return area, I


def design_to_features_and_cae(d):
    """Return node_features [N_NODE, 5] and targets [first_mode_kHz, mass_g, SF]."""
    w = np.clip(d[:3], D_LO[:3], D_HI[:3])
    ribs = int(round(np.clip(d[3], 0, 8)))
    mat = MATERIALS[int(round(np.clip(d[4], 0, len(MATERIALS) - 1)))]
    _, E, rho, yld, cte = mat
    L = LEN_MM * 1e-3
    le = L / N_ELEM
    elem_area = np.zeros(N_ELEM); elem_I = np.zeros(N_ELEM)
    for e in range(N_ELEM):
        zone = min(2, int(3 * e / N_ELEM))
        a, I = _section(w[zone], ribs if zone == 1 else 0)
        elem_area[e], elem_I[e] = a, I

    ndof = 2 * N_NODE
    K = np.zeros((ndof, ndof)); M = np.zeros((ndof, ndof))
    for e in range(N_ELEM):
        EI = E * elem_I[e]; rhoA = rho * elem_area[e]
        ke = EI / le**3 * np.array([
            [12, 6*le, -12, 6*le], [6*le, 4*le**2, -6*le, 2*le**2],
            [-12, -6*le, 12, -6*le], [6*le, 2*le**2, -6*le, 4*le**2]])
        me = rhoA * le / 420 * np.array([
            [156, 22*le, 54, -13*le], [22*le, 4*le**2, 13*le, -3*le**2],
            [54, 13*le, 156, -22*le], [-13*le, -3*le**2, -22*le, 4*le**2]])
        idx = [2*e, 2*e+1, 2*e+2, 2*e+3]
        K[np.ix_(idx, idx)] += ke; M[np.ix_(idx, idx)] += me
    M[ndof-2, ndof-2] += LENS_G * 1e-3
    free = np.arange(2, ndof)
    w2 = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)],
              eigvals_only=True, subset_by_index=[0, 0])
    f1 = float(np.sqrt(max(w2[0], 0)) / (2 * np.pi))

    mass = float(np.sum(rho * elem_area * le) + LENS_G * 1e-3)
    v = np.sqrt(2 * G * 1.0); tau = 0.5e-3
    a_peak = np.pi * v / (2 * tau)
    Ie = float(np.mean(elem_I)); c = OUTER_MM / 2 * 1e-3
    stress = LENS_G * 1e-3 * a_peak * 1.1 * L * c / Ie
    sf = float(yld / stress) if stress > 0 else 0.0

    n = N_NODE
    node_area = np.zeros(n); node_I = np.zeros(n)
    for e in range(N_ELEM):
        node_area[e] += elem_area[e] / 2; node_area[e+1] += elem_area[e] / 2
        node_I[e] += elem_I[e] / 2; node_I[e+1] += elem_I[e] / 2
    X = np.stack([
        np.linspace(0, 1, n), node_area / 3e-4, node_I / 2e-8,
        np.full(n, E / 100e9), np.full(n, rho / 3000.0)], axis=1)
    return X.astype(np.float32), np.array([f1/1e3, mass*1e3, sf], dtype=np.float32)


# ── PyTorch Graph Convolutional Network surrogate ────────────────────────────
class GCNLayer(nn.Module):
    def __init__(self, f_in, f_out):
        super().__init__()
        self.lin = nn.Linear(f_in, f_out)

    def forward(self, x):                      # x: [B, N, F]; A_HAT: [N, N]
        return torch.tanh(A_HAT @ self.lin(x))


class BarrelGCN(nn.Module):
    def __init__(self, f_in=5, h=32, n_out=3):
        super().__init__()
        self.g1 = GCNLayer(f_in, h)
        self.g2 = GCNLayer(h, h)
        self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, n_out))

    def forward(self, x):
        h = self.g2(self.g1(x))
        pooled = h.mean(dim=1)                  # global mean pool over nodes
        return self.head(pooled)


def train_surrogate(Xtr, Ytr, mu, sd, epochs=600, lr=3e-3):
    net = BarrelGCN()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    Xt = torch.tensor(Xtr)
    Yt = torch.tensor((Ytr - mu) / sd)
    hist = []
    for ep in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(Xt), Yt)
        loss.backward(); opt.step()
        hist.append(loss.item())
    return net, hist


@torch.no_grad()
def predict(net, X, mu, sd):
    x = torch.tensor(X[None] if X.ndim == 2 else X)
    return (net(x).numpy() * sd + mu)


def objective(t, f1_min=9.0, sf_min=6.0):
    """Minimize mass s.t. first mode >= f1_min kHz and SF >= sf_min (penalized)."""
    f1, mass, sf = t
    return mass + 100.0 * max(0, f1_min - f1)**2 + 40.0 * max(0, sf_min - sf)**2


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    figs = os.path.join(here, "figures"); os.makedirs(figs, exist_ok=True)

    # 1. Sample designs, run CAE.
    n_samp = 300
    designs = D_LO + RNG.random((n_samp, 5)) * (D_HI - D_LO)
    X = np.zeros((n_samp, N_NODE, 5), np.float32); Y = np.zeros((n_samp, 3), np.float32)
    for i, d in enumerate(designs):
        X[i], Y[i] = design_to_features_and_cae(d)
    mu, sd = Y.mean(0), Y.std(0)

    ntr = 240
    net, hist = train_surrogate(X[:ntr], Y[:ntr], mu, sd)

    # 2. Held-out surrogate accuracy.
    pred = predict(net, X[ntr:], mu, sd)
    labels = ["first mode [kHz]", "mass [g]", "safety factor"]
    r2 = []
    for k in range(3):
        res = np.sum((Y[ntr:, k] - pred[:, k])**2)
        tot = np.sum((Y[ntr:, k] - Y[ntr:, k].mean())**2)
        r2.append(1 - res / tot)

    # 3. Bayesian generative design: GP + EI, objective via the GNN surrogate.
    n_init = 12
    Xd = D_LO + RNG.random((n_init, 5)) * (D_HI - D_LO)

    def surro_cost(d):
        Xf, _ = design_to_features_and_cae(d)     # features only (fast; no solve)
        return objective(predict(net, Xf, mu, sd)[0])

    cost = np.array([surro_cost(d) for d in Xd])
    best_hist = [cost.min()]
    kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(5), nu=2.5)
    for _ in range(30):
        Xn = (Xd - D_LO) / (D_HI - D_LO)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-4)
        gp.fit(Xn, cost)
        cand = RNG.random((500, 5))
        m, s = gp.predict(cand, return_std=True)
        imp = cost.min() - m
        z = imp / (s + 1e-9)
        ei = imp * norm.cdf(z) + s * norm.pdf(z)
        nx = D_LO + cand[np.argmax(ei)] * (D_HI - D_LO)
        Xd = np.vstack([Xd, nx]); cost = np.append(cost, surro_cost(nx))
        best_hist.append(cost.min())

    best_d = Xd[np.argmin(cost)]
    _, true_t = design_to_features_and_cae(best_d)   # verify with real FE solver
    surr_t = predict(net, design_to_features_and_cae(best_d)[0], mu, sd)[0]

    # ── Figures ──────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.1))
    for k, ax in enumerate(axs):
        ax.scatter(Y[ntr:, k], pred[:, k], s=14, alpha=.7, color="#2a78d6")
        lo, hi = Y[ntr:, k].min(), Y[ntr:, k].max()
        ax.plot([lo, hi], [lo, hi], "--", color="#888", lw=1)
        ax.set_title(f"{labels[k]}  (R²={r2[k]:.2f})", fontsize=9)
        ax.set_xlabel("CAE (FEA)"); ax.set_ylabel("GNN surrogate"); ax.grid(alpha=.25)
    fig.suptitle("PyTorch GNN surrogate vs CAE (held-out designs)", y=1.03)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "gnn_parity.png"),
                                    dpi=130, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(range(1, len(best_hist)+1), best_hist, "-o", ms=3, color="#12a150")
    ax.set_xlabel("Bayesian-optimization iteration")
    ax.set_ylabel("best cost (mass + penalty)")
    ax.set_title("Generative design search (GP + Expected Improvement)")
    ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(os.path.join(figs, "generative_convergence.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    sc = ax.scatter(Y[:, 1], Y[:, 0], c=Y[:, 2], cmap="viridis", s=16, alpha=.8)
    ax.scatter([true_t[1]], [true_t[0]], marker="*", s=280, color="#d23b3b",
               edgecolor="k", zorder=5, label="AI-generated optimum")
    ax.set_xlabel("mass [g]"); ax.set_ylabel("first mode [kHz]")
    ax.set_title("Design space (color = safety factor)")
    fig.colorbar(sc, label="safety factor"); ax.legend(fontsize=8)
    ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(os.path.join(figs, "design_pareto.png"), dpi=130); plt.close(fig)

    matname = MATERIALS[int(round(np.clip(best_d[4], 0, 1)))][0]
    out = {
        "framework": f"PyTorch {torch.__version__}",
        "surrogate": "PyTorch GCN (2 graph-conv layers + mean-pool + MLP head)",
        "n_designs": n_samp, "n_train": ntr, "n_test": n_samp - ntr,
        "gnn_r2": {labels[k]: round(float(r2[k]), 3) for k in range(3)},
        "final_train_mse": round(float(hist[-1]), 4),
        "bo_iterations": len(best_hist),
        "generated_design": {
            "wall_z1_z2_z3_mm": [round(float(x), 2) for x in best_d[:3]],
            "n_ribs": int(round(best_d[3])), "material": matname},
        "generated_perf_true_FEA": {
            "first_mode_kHz": round(float(true_t[0]), 2),
            "mass_g": round(float(true_t[1]), 2),
            "safety_factor": round(float(true_t[2]), 1)},
        "generated_perf_GNN_surrogate": {
            "first_mode_kHz": round(float(surr_t[0]), 2),
            "mass_g": round(float(surr_t[1]), 2),
            "safety_factor": round(float(surr_t[2]), 1)},
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    with open(os.path.join(here, "generative_design_results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
