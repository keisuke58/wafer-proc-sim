#!/usr/bin/env python3
"""etch_surrogate.py — feature-scale etch profile simulation + a machine-
learning surrogate that replaces it, and the inverse-design payoff.

This is the piece a process-equipment company (TEL-class) actually lives on:
a physics simulation of how a trench evolves under plasma etch, and — because
that simulation is too slow to sit inside a recipe-optimization loop — a
learned surrogate that predicts the outcome in microseconds. Physics AND the
ML that makes physics usable, together.

  FEATURE-SCALE ETCH  a 2-D cross-section trench evolves under three fluxes,
      each aspect-ratio (AR) throttled (RIE lag / ARDE):
        * vertical ion-driven bottom advance, ARDE-limited transport
        * isotropic neutral sidewall attack, suppressed by the passivation
          film the recipe deposits
        * near-top ion-scattering that gouges the classic BOW
      The half-width profile hw(z) and depth D advance in time, giving the
      real knobs an integrator reads: bottom CD, depth, bow, sidewall angle.
      One parameter (passivation) sweeps bowed -> vertical -> tapered.

  ML SURROGATE  the physics model is swept over a 5-D recipe space; a Gaussian-
      process surrogate learns recipe -> (CD, depth, bow, angle). Held-out
      R^2 quantifies fidelity; the wall-clock speedup (sim vs surrogate call)
      quantifies why you build it — the surrogate is what lets Bayesian
      recipe search evaluate thousands of candidates.

  INVERSE DESIGN  the payoff: given a target bottom-CD and depth, search the
      recipe space *on the surrogate* for the recipe that hits it, then verify
      the found recipe against the true physics simulation. Error < a few %
      with a >1000x evaluation speedup is the whole value proposition.

Pure numpy + scipy + scikit-learn + matplotlib.  Run:  python3 telproc/etch_surrogate.py
"""
import json
import os
import time

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "etch_surrogate_results.json")

# ── recipe space: [passivation, ion_frac, flux, mask_w_nm, time_s] ──────────
PARAM_NAMES = ["passivation", "ion_frac", "flux", "mask_w_nm", "time_s"]
P_LO = np.array([0.05, 0.45, 0.7, 30.0, 60.0])
P_HI = np.array([0.85, 0.95, 1.6, 90.0, 260.0])

# physics constants (reduced feature-scale model)
R_ION0 = 4.5             # bare vertical ion etch rate scale [nm/s]
R_NEU0 = 1.1             # bare isotropic neutral rate scale [nm/s]
K_ARDE = 9.0             # ion transport knee vs aspect ratio
K_ARDE_N = 3.0           # neutrals are more conformal (weaker ARDE)
BOW_AMP = 0.55           # ion-scatter lateral rate near the top [nm/s]
BOW_DEPTH_FRAC = 0.18    # bow center as a fraction of current depth
BOW_SIGMA_FRAC = 0.12    # bow gaussian width as a fraction of depth
NZ = 260                 # depth grid rows
DZ = 3.0                 # nm per row


def simulate_profile(params, dt=0.5):
    """Time-stepped feature-scale etch. params = [passivation, ion_frac,
    flux, mask_w_nm, time_s]. Returns dict of profile + scalar metrics."""
    passivation, ion_frac, flux, mask_w, t_total = params
    neu_frac = 1.0 - ion_frac
    hw0 = mask_w / 2.0                       # mask half-opening [nm]

    z = np.arange(NZ) * DZ                    # depth of each row [nm]
    hw = np.zeros(NZ)                         # half-width at each opened row
    opened = np.zeros(NZ, bool)
    depth = 1e-6                              # etched depth [nm]

    steps = max(int(t_total / dt), 1)
    dt = t_total / steps
    for _ in range(steps):
        # local half-width at the advancing bottom (for AR throttling)
        kb = min(int(depth / DZ), NZ - 1)
        hw_bot = max(hw[kb], hw0)
        ar = depth / (2.0 * hw_bot)           # aspect ratio at the floor
        # vertical bottom advance, ARDE-limited
        r_ion = R_ION0 * flux * ion_frac / (1.0 + ar / K_ARDE)
        depth += r_ion * dt
        depth = min(depth, (NZ - 1) * DZ)

        # newly crossed rows start at the mask opening (ions carve straight)
        kb_new = min(int(depth / DZ), NZ - 1)
        for k in range(NZ):
            if not opened[k] and k <= kb_new:
                opened[k] = True
                hw[k] = hw0

        # sidewall (lateral) advance on every opened row
        idx = np.where(opened)[0]
        if idx.size:
            ar_z = z[idx] / (2.0 * np.maximum(hw[idx], hw0))
            r_neu = R_NEU0 * flux * neu_frac * (1.0 - passivation) \
                / (1.0 + ar_z / K_ARDE_N)
            # bow: ion scattering gouges the sidewall near the top; the
            # passivation film that protects the wall also mitigates the bow
            zc = BOW_DEPTH_FRAC * depth
            sig = max(BOW_SIGMA_FRAC * depth, DZ)
            bow = BOW_AMP * flux * ion_frac * (1.0 - passivation) \
                * np.exp(-0.5 * ((z[idx] - zc) / sig) ** 2)
            hw[idx] = hw[idx] + (r_neu + bow) * dt

    # ── metrics an integrator reads ──
    idx = np.where(opened)[0]
    depth_nm = float(depth)
    top_k = idx[:max(len(idx) // 12, 1)]
    bot_k = idx[-max(len(idx) // 12, 1):]
    cd_top = float(2.0 * hw[top_k].mean())
    cd_bot = float(2.0 * hw[bot_k].mean())
    bow_nm = float(2.0 * (hw[idx].max() - hw[top_k].mean()))
    # sidewall angle: 90 deg = vertical; <90 tapered (narrowing with depth)
    dwidth = hw[bot_k].mean() - hw[top_k].mean()
    angle = float(np.degrees(np.arctan2(depth_nm, -dwidth))) if depth_nm > 0 \
        else 90.0
    angle = float(np.clip(angle, 60.0, 120.0))
    return {"z": z[idx], "hw": hw[idx], "depth_nm": depth_nm,
            "cd_top_nm": cd_top, "cd_bot_nm": cd_bot, "bow_nm": bow_nm,
            "sidewall_deg": angle}


TARGETS = ["cd_bot_nm", "depth_nm", "bow_nm", "sidewall_deg"]


def _metrics_vec(params):
    r = simulate_profile(params)
    return np.array([r[k] for k in TARGETS])


def build_dataset(n, seed=0):
    rng = np.random.default_rng(seed)
    X = P_LO + rng.random((n, 5)) * (P_HI - P_LO)
    Y = np.array([_metrics_vec(x) for x in X])
    return X, Y


def train_surrogate(Xtr, Ytr):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (ConstantKernel, Matern,
                                                  WhiteKernel)
    from sklearn.preprocessing import StandardScaler

    xs = StandardScaler().fit(Xtr)
    ys = StandardScaler().fit(Ytr)
    Xn, Yn = xs.transform(Xtr), ys.transform(Ytr)
    models = []
    for j in range(Ytr.shape[1]):
        k = ConstantKernel(1.0) * Matern(length_scale=np.ones(5), nu=2.5) \
            + WhiteKernel(1e-3)
        gp = GaussianProcessRegressor(kernel=k, normalize_y=False,
                                      alpha=1e-6, n_restarts_optimizer=0)
        gp.fit(Xn, Yn[:, j])
        models.append(gp)
    return {"xs": xs, "ys": ys, "models": models}


def predict(sur, X):
    Xn = sur["xs"].transform(np.atleast_2d(X))
    Yn = np.column_stack([m.predict(Xn) for m in sur["models"]])
    return sur["ys"].inverse_transform(Yn)


def r2(y, yhat):
    ss = np.sum((y - yhat) ** 2, axis=0)
    tot = np.sum((y - y.mean(0)) ** 2, axis=0)
    return 1.0 - ss / tot


def inverse_design(sur, target_cd_bot, target_depth, seed=1, n=20000):
    """Search recipe space on the SURROGATE for a recipe hitting the target
    bottom-CD and depth, then verify against the true physics sim."""
    rng = np.random.default_rng(seed)
    cand = P_LO + rng.random((n, 5)) * (P_HI - P_LO)
    pred = predict(sur, cand)                 # [n, 4] cheap
    cost = ((pred[:, 0] - target_cd_bot) / target_cd_bot) ** 2 \
        + ((pred[:, 1] - target_depth) / target_depth) ** 2
    best = cand[np.argmin(cost)]
    true = _metrics_vec(best)                  # verify with real physics
    return best, true


def run():
    n = 260
    X, Y = build_dataset(n, seed=0)
    ntr = 210
    sur = train_surrogate(X[:ntr], Y[:ntr])
    pred = predict(sur, X[ntr:])
    r2s = r2(Y[ntr:], pred)

    # speedup: full sim vs surrogate call
    t0 = time.perf_counter()
    for x in X[ntr:ntr + 20]:
        _metrics_vec(x)
    t_sim = (time.perf_counter() - t0) / 20
    t1 = time.perf_counter()
    predict(sur, X[ntr:ntr + 20])
    t_sur = (time.perf_counter() - t1) / 20
    speedup = t_sim / max(t_sur, 1e-9)

    # inverse design demo
    tgt_cd, tgt_d = 45.0, 520.0
    recipe, true = inverse_design(sur, tgt_cd, tgt_d)
    cd_err = 100 * abs(true[0] - tgt_cd) / tgt_cd
    d_err = 100 * abs(true[1] - tgt_d) / tgt_d

    # passivation sweep: bowed -> vertical -> tapered
    base = (0.5, 0.8, 1.1, 60.0, 200.0)
    sweep = {}
    for p in (0.1, 0.5, 0.8):
        r = simulate_profile((p,) + base[1:])
        sweep[f"passivation_{p}"] = {"bow_nm": round(r["bow_nm"], 1),
                                     "sidewall_deg": round(r["sidewall_deg"],
                                                           1)}

    return {
        "dataset": {"n_samples": n, "n_train": ntr, "n_test": n - ntr},
        "surrogate_r2": {k: round(float(v), 4)
                         for k, v in zip(TARGETS, r2s)},
        "speedup": {"sim_ms": round(t_sim * 1e3, 2),
                    "surrogate_ms": round(t_sur * 1e3, 4),
                    "factor": int(round(speedup))},
        "inverse_design": {
            "target": {"cd_bot_nm": tgt_cd, "depth_nm": tgt_d},
            "found_recipe": {k: round(float(v), 3)
                             for k, v in zip(PARAM_NAMES, recipe)},
            "verified_true": {"cd_bot_nm": round(float(true[0]), 1),
                              "depth_nm": round(float(true[1]), 1)},
            "cd_err_pct": round(cd_err, 2), "depth_err_pct": round(d_err, 2)},
        "passivation_sweep": sweep,
    }


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3))

    # 1: profiles for passivation sweep (bowed -> vertical -> tapered)
    ax = axes[0]
    base = (0.8, 1.1, 60.0, 200.0)
    for p, c, lab in ((0.1, "#c0392b", "low passiv. (bowed)"),
                      (0.5, "#b4740a", "mid (vertical)"),
                      (0.8, "#1a7a4a", "high (tapered)")):
        r = simulate_profile((p,) + base)
        hw, z = r["hw"], r["z"]
        ax.plot(hw, z, color=c, lw=2, label=lab)
        ax.plot(-hw, z, color=c, lw=2)
    ax.invert_yaxis()
    ax.set_xlabel("half-width [nm]")
    ax.set_ylabel("depth [nm]")
    ax.set_title("Feature-scale etch: one knob\n(passivation) sets the profile",
                 fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # 2: surrogate parity
    ax = axes[1]
    n = res["dataset"]["n_samples"]
    ntr = res["dataset"]["n_train"]
    X, Y = build_dataset(n, seed=0)
    sur = train_surrogate(X[:ntr], Y[:ntr])
    pred = predict(sur, X[ntr:])
    j = TARGETS.index("cd_bot_nm")
    ax.scatter(Y[ntr:, j], pred[:, j], s=16, alpha=0.7, color="#2a78d6")
    lo, hi = Y[ntr:, j].min(), Y[ntr:, j].max()
    ax.plot([lo, hi], [lo, hi], "--", color="0.5", lw=1)
    rr = res["surrogate_r2"]["cd_bot_nm"]
    sp = res["speedup"]["factor"]
    ax.set_xlabel("physics sim  bottom CD [nm]")
    ax.set_ylabel("GP surrogate  bottom CD [nm]")
    ax.set_title(f"ML surrogate replaces the sim\nR^2={rr:.2f}, "
                 f"{sp}x faster per eval", fontsize=10)
    ax.grid(alpha=0.3)

    # 3: inverse design result
    ax = axes[2]
    idd = res["inverse_design"]
    rec = [idd["found_recipe"][k] for k in PARAM_NAMES]
    r = simulate_profile(rec)
    ax.plot(r["hw"], r["z"], color="#6a1b9a", lw=2)
    ax.plot(-r["hw"], r["z"], color="#6a1b9a", lw=2)
    ax.invert_yaxis()
    tgt = idd["target"]
    ver = idd["verified_true"]
    ax.set_xlabel("half-width [nm]")
    ax.set_ylabel("depth [nm]")
    ax.set_title(f"Inverse design on surrogate ->\nverify: CD {ver['cd_bot_nm']:.0f}"
                 f"/{tgt['cd_bot_nm']:.0f}nm, depth {ver['depth_nm']:.0f}"
                 f"/{tgt['depth_nm']:.0f}nm", fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot3d(path):
    """Revolve the simulated 2-D profile into the real 3-D etched hole for
    three passivation regimes (cutaway 3/4 turn)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = [(0.10, "#c0392b", "low passiv. -> BOWED"),
             (0.50, "#b4740a", "mid -> VERTICAL"),
             (0.85, "#1a7a4a", "high -> TAPERED")]
    base = (0.8, 1.1, 60.0, 200.0)          # ion_frac, flux, mask_w, time
    th = np.linspace(0.0, 1.5 * np.pi, 90)
    fig = plt.figure(figsize=(13.5, 5.0))
    for i, (p, col, lab) in enumerate(cases):
        r = simulate_profile((p,) + base)
        z = r["z"]
        hw = np.convolve(r["hw"], np.ones(3) / 3, mode="same")
        Z, TH = np.meshgrid(z, th)
        HW = np.meshgrid(hw, th)[0]
        X, Y = HW * np.cos(TH), HW * np.sin(TH)
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.plot_surface(X, Y, -Z, rstride=2, cstride=2, color=col, alpha=0.9,
                        linewidth=0, antialiased=True, shade=True)
        ax.plot(hw[0] * np.cos(th), hw[0] * np.sin(th),
                -z[0] * np.ones_like(th), color="0.25", lw=1.2)
        ax.set_title(f"{lab}\ndepth {r['depth_nm']:.0f} nm, "
                     f"bow {r['bow_nm']:.0f} nm", fontsize=10)
        ax.set_xlabel("x [nm]"); ax.set_ylabel("y [nm]")
        ax.set_zlabel("depth [nm]")
        ax.view_init(elev=22, azim=-60)
        m = max(hw.max(), 1)
        ax.set_xlim(-m, m); ax.set_ylim(-m, m)
        ax.set_box_aspect((1, 1, 2.2))
    fig.suptitle("Feature-scale etched hole — 3-D shape by revolving the "
                 "simulated profile (cutaway)", fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    res = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "tel_etch_surrogate.png"), res)
    _plot3d(os.path.join(ASSETS, "tel_etch_3d.png"))
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
