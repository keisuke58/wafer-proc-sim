#!/usr/bin/env python3
"""etch_ald.py — plasma-etch profile evolution and ALD conformality.

Two reduced-order models of the physics a process-equipment company sells:

  ETCH PROFILE — a high-AR trench evolves under three competing fluxes:
    * vertical ion-driven etch, throttled by aspect-ratio-dependent
      transport (ARDE / RIE lag: dD/dt = R0 / (1 + AR/K))
    * isotropic chemical attack on the sidewall, suppressed by the
      passivation film the recipe deposits
    * near-top ion scattering, which gouges the classic BOW
    One knob (passivation) sweeps the profile from bowed -> vertical ->
    tapered: the entire recipe-tuning conversation in one parameter.

  ALD CONFORMALITY — a self-limiting precursor diffusing down a hole
    (Knudsen transport) while chemisorbing on unreacted sites:
      dc/dt = D d2c/dz2 - k c (1-theta),   dtheta/dt = k c (1-theta)
    Coverage advances as a FRONT; the dose needed to coat the bottom of
    an AR-alpha hole grows ~ AR^2 (transport-limited) — the quantified
    reason ALD holds the high-AR market and why 3D NAND made it king.

Pure numpy + matplotlib.  Run:  python3 telproc/etch_ald.py
"""
import json
import os

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "etch_ald_results.json")

# ── etch profile model ──────────────────────────────────────────────────────
R0 = 5.0                 # open-area vertical etch rate [nm/s]
K_ARDE = 12.0            # transport knee (AR at which rate halves ~ K)
R_CHEM = 0.35            # isotropic sidewall attack, unpassivated [nm/s]
BOW_ION = 0.25           # ion-scattering lateral rate peak [nm/s]
BOW_DEPTH_FRAC = 0.18    # bow center as fraction of current depth


def etch_profile(w0_nm=50.0, t_s=240.0, passivation=0.5, nz=240, dt=0.5):
    """Time-stepped reduced profile model. Returns (z, half_width, depth)."""
    depth = 1e-3
    zmax = R0 * t_s * 1.2
    z = np.linspace(0.0, zmax, nz)
    half = np.full(nz, w0_nm / 2.0)
    for _ in range(int(t_s / dt)):
        ar = depth / w0_nm
        depth += dt * R0 / (1.0 + ar / K_ARDE)
        inside = z < depth
        # sidewall chemical attack (suppressed by passivation)
        half[inside] += dt * R_CHEM * (1.0 - passivation)
        # ion-scattering bow near the top of the feature
        bc = BOW_DEPTH_FRAC * depth
        bow = BOW_ION * (1.0 - passivation) ** 1.5 \
            * np.exp(-0.5 * ((z - bc) / (0.12 * depth + 1.0)) ** 2)
        half[inside] += dt * bow[inside]
        # heavy passivation pinches the feature going down (taper)
        taper = 0.10 * max(passivation - 0.55, 0.0)
        half[inside] -= dt * taper * (z[inside] / max(depth, 1.0)) \
            * R_CHEM * 2.0
    half = np.clip(half, 2.0, None)
    return z, half, float(depth)


def arde_depth(w0_nm, t_s=240.0, dt=0.5):
    depth = 1e-3
    ds = []
    ts = np.arange(0, t_s, dt)
    for _ in ts:
        depth += dt * R0 / (1.0 + (depth / w0_nm) / K_ARDE)
        ds.append(depth)
    return ts, np.asarray(ds)


# ── ALD conformality ────────────────────────────────────────────────────────
def ald_coverage(ar=40.0, dose=1.0, nz=100, n_macro=400):
    """Quasi-static reaction-diffusion down a hole of aspect ratio `ar`
    (the standard ALD treatment: gas transport equilibrates much faster
    than surface sites fill).

    Nondimensional z = depth/L in [0,1]; the Knudsen penalty enters as the
    Thiele modulus: c'' = K2 (1-theta) c with K2 = k0 * AR^2. Each macro
    step solves the steady profile (tridiagonal) for the current theta,
    then advances the coverage; `dose` is total exposure in units where
    dose~1 saturates an AR~10 hole."""
    k0 = 0.35
    K2 = k0 * ar ** 2
    z = np.linspace(0, 1, nz)
    dz = z[1] - z[0]
    theta = np.zeros(nz)
    dtau = dose / n_macro
    for _ in range(n_macro):
        # steady state: (c[i-1] - 2c[i] + c[i+1])/dz^2 = K2 (1-th) c
        a = np.full(nz, 1.0)
        b = -2.0 - K2 * (1.0 - theta) * dz ** 2
        rhs = np.zeros(nz)
        # boundary: c[0]=1 ; closed bottom c[n-1]=c[n-2]
        b[0], rhs[0] = 1.0, 1.0
        A = np.zeros((nz, nz))
        idx = np.arange(1, nz - 1)
        A[idx, idx - 1] = a[idx]
        A[idx, idx] = b[idx]
        A[idx, idx + 1] = a[idx]
        A[0, 0] = 1.0
        A[nz - 1, nz - 1] = 1.0
        A[nz - 1, nz - 2] = -1.0
        c = np.linalg.solve(A, rhs)
        theta = np.clip(theta + dtau * 3.0 * c * (1.0 - theta), 0.0, 1.0)
    return z, theta


def dose_for_bottom_coverage(ar, target=0.95):
    lo, hi = 0.05, 3000.0
    z, th = ald_coverage(ar, hi)
    if th[-1] < target:
        return np.inf
    for _ in range(22):
        mid = np.sqrt(lo * hi)
        _, th = ald_coverage(ar, mid)
        if th[-1] >= target:
            hi = mid
        else:
            lo = mid
    return hi


def run():
    _, _, d_open = etch_profile(w0_nm=400.0)
    _, _, d_20 = etch_profile(w0_nm=20.0)
    ars = np.array([10.0, 20.0, 40.0, 80.0])
    doses = np.array([dose_for_bottom_coverage(a) for a in ars])
    slope = np.polyfit(np.log(ars), np.log(doses), 1)[0]
    return {
        "etch": {
            "depth_400nm_open_nm": round(d_open, 1),
            "depth_20nm_trench_nm": round(d_20, 1),
            "rie_lag_pct": round(100 * (1 - d_20 / d_open), 1),
        },
        "ald": {
            "dose_to_95pct": {f"AR{int(a)}": round(float(d), 2)
                              for a, d in zip(ars, doses)},
            "dose_vs_ar_exponent": round(float(slope), 2),
        },
    }


def _plot(path_etch, path_ald, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    ax = axes[0]
    for p, c, lab in ((0.2, "#c44", "low passivation -> BOW"),
                      (0.5, "#26a", "balanced -> vertical"),
                      (0.85, "#2a7", "heavy passivation -> TAPER")):
        z, half, depth = etch_profile(passivation=p)
        m = z < depth
        ax.plot(half[m], -z[m], color=c, lw=1.8, label=lab)
        ax.plot(-half[m], -z[m], color=c, lw=1.8)
    ax.set_xlabel("half width [nm]")
    ax.set_ylabel("depth [nm]")
    ax.set_title("One knob, three profiles: passivation sweep\n"
                 "(50 nm trench, 240 s)")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    for w, c in ((100.0, "#26a"), (40.0, "#b4740a"), (20.0, "#c44")):
        ts, ds = arde_depth(w)
        ax.plot(ts, ds, color=c, lw=1.8, label=f"trench width {w:.0f} nm")
    ax.set_xlabel("etch time [s]")
    ax.set_ylabel("depth [nm]")
    ax.set_title(f"ARDE / RIE lag: narrow features fall behind "
                 f"({res['etch']['rie_lag_pct']:.0f}% at 240 s)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path_etch, dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax = axes[0]
    for dose, c in ((0.3, "#c86"), (1.0, "#b4740a"), (3.0, "#26a"),
                    (10.0, "#2a7")):
        z, th = ald_coverage(40.0, dose)
        ax.plot(th, -z, color=c, lw=1.8, label=f"dose x{dose:g}")
    ax.set_xlabel("coverage theta")
    ax.set_ylabel("normalized depth")
    ax.set_title("ALD coats as a FRONT (AR 40 hole):\n"
                 "self-limiting -> perfectly conformal, eventually")
    ax.legend(fontsize=9)

    ax = axes[1]
    ars = [10, 20, 40, 80]
    doses = [res["ald"]["dose_to_95pct"][f"AR{a}"] for a in ars]
    ax.loglog(ars, doses, "o-", color="#26a", lw=1.8,
              label="dose for 95% bottom coverage")
    aa = np.array([8.0, 100.0])
    ref = doses[0] * (aa / ars[0]) ** 2
    ax.loglog(aa, ref, "k--", lw=1, label="~ AR^2 reference")
    ax.set_xlabel("hole aspect ratio")
    ax.set_ylabel("required dose [a.u.]")
    ax.set_title(f"The AR^2 tax (fit exponent "
                 f"{res['ald']['dose_vs_ar_exponent']:.2f}):\n"
                 f"why 3D NAND made ALD king")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path_ald, dpi=110)
    plt.close(fig)


def main():
    res = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "tel_etch.png"),
          os.path.join(ASSETS, "tel_ald.png"), res)
    print("figures ->", ASSETS)


if __name__ == "__main__":
    main()
