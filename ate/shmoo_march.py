#!/usr/bin/env python3
"""shmoo_march.py — the tester's two icons: the shmoo plot and March test.

  SHMOO — the Vdd × frequency pass/fail map. The device model is the
  alpha-power critical path: delay ∝ V / (V - Vt)^alpha, plus on-chip
  variation and a metastability band at the boundary. From the map the
  tester extracts what engineering actually consumes: the Fmax(V) curve,
  Vmin at the rated clock, and the guard-banded operating point.

  MARCH — memory test algorithms are PROVABLE. A bit-level fault simulator
  injects stuck-at (SAF), transition (TF), and inversion/idempotent
  coupling faults (CFin/CFid), then runs March C- {⇕(w0); ⇑(r0,w1);
  ⇑(r1,w0); ⇓(r0,w1); ⇓(r1,w0); ⇕(r0)} against a naive
  write-0/read/write-1/read baseline. The coverage table shows WHY the
  industry pays 10N reads/writes: the naive 4N test catches stuck-at
  faults but is blind to most coupling faults; March C- catches every
  injected class.

Pure numpy + matplotlib.  Run:  python3 ate/shmoo_march.py
"""
import json
import os

import numpy as np

ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__),
                            "shmoo_march_results.json")

# ── shmoo: alpha-power critical path ────────────────────────────────────────
VT = 0.35                # threshold voltage [V]
ALPHA = 1.3              # velocity-saturation exponent
K_DELAY = 0.42           # ns scale factor: delay = K V/(V-Vt)^alpha
OCV_SIGMA = 0.015        # on-chip variation of delay (fraction)
GUARD_NS = 0.02          # timing guard band


def path_delay_ns(vdd):
    v = np.asarray(vdd, dtype=float)
    return K_DELAY * v / np.maximum(v - VT, 1e-6) ** ALPHA


def shmoo_map(v_range=(0.6, 1.1), f_range=(0.4, 2.2), n=64, seed=0):
    """Pass/fail over the Vdd x freq grid, with OCV noise per point (the
    speckled boundary every tester engineer recognizes)."""
    rng = np.random.default_rng(seed)
    vs = np.linspace(*v_range, n)
    fs = np.linspace(*f_range, n)
    V, F = np.meshgrid(vs, fs)
    delay = path_delay_ns(V) * (1.0 + OCV_SIGMA * rng.standard_normal(V.shape))
    ok = (1.0 / F) > (delay + GUARD_NS)
    return vs, fs, ok


def fmax_ghz(vdd):
    return 1.0 / (path_delay_ns(vdd) + GUARD_NS)


def vmin_at(f_ghz, v_lo=0.4, v_hi=1.3):
    """Smallest Vdd meeting the clock (bisection on the nominal model)."""
    if fmax_ghz(v_hi) < f_ghz:
        return np.nan
    for _ in range(50):
        mid = 0.5 * (v_lo + v_hi)
        if fmax_ghz(mid) >= f_ghz:
            v_hi = mid
        else:
            v_lo = mid
    return v_hi


# ── memory fault simulator + March C- ──────────────────────────────────────
class FaultyMemory:
    """N-bit memory with one injected fault.

    kinds: SAF0/SAF1 (stuck), TF (can't 0->1), CFin (write on aggressor
    inverts victim), CFid<d> (aggressor 0->1 forces victim to d)."""

    def __init__(self, n, kind, a, v=None):
        self.n = n
        self.kind = kind
        self.a = a                      # faulty / aggressor cell
        self.v = v                      # victim cell (coupling faults)
        self.m = np.zeros(n, dtype=np.int8)

    def write(self, i, val):
        old = self.m[self.a]
        if i == self.a:
            if self.kind == "SAF0":
                self.m[i] = 0
                return
            if self.kind == "SAF1":
                self.m[i] = 1
                return
            if self.kind == "TF" and old == 0 and val == 1:
                return                   # up-transition fails
            self.m[i] = val
            new = self.m[self.a]
            if self.kind == "CFin" and old != new:
                self.m[self.v] ^= 1      # inversion coupling
            if self.kind == "CFid" and old == 0 and new == 1:
                self.m[self.v] = 1       # idempotent coupling (forces 1)
        else:
            self.m[i] = val

    def read(self, i):
        if i == self.a and self.kind == "SAF0":
            return 0
        if i == self.a and self.kind == "SAF1":
            return 1
        return int(self.m[i])


def march_c_minus(mem):
    """Returns True if ANY read mismatches (fault detected)."""
    n = mem.n
    det = False
    for i in range(n):
        mem.write(i, 0)
    for i in range(n):                              # up r0,w1
        det |= mem.read(i) != 0
        mem.write(i, 1)
    for i in range(n):                              # up r1,w0
        det |= mem.read(i) != 1
        mem.write(i, 0)
    for i in reversed(range(n)):                    # down r0,w1
        det |= mem.read(i) != 0
        mem.write(i, 1)
    for i in reversed(range(n)):                    # down r1,w0
        det |= mem.read(i) != 1
        mem.write(i, 0)
    for i in range(n):
        det |= mem.read(i) != 0
    return det


def naive_test(mem):
    """w0 all, r0 all, w1 all, r1 all — the intuitive 4N test."""
    n = mem.n
    det = False
    for i in range(n):
        mem.write(i, 0)
    for i in range(n):
        det |= mem.read(i) != 0
    for i in range(n):
        mem.write(i, 1)
    for i in range(n):
        det |= mem.read(i) != 1
    return det


def coverage(n_mem=64, n_trials=40, seed=2):
    """Detection rate per fault class for both algorithms."""
    rng = np.random.default_rng(seed)
    classes = ("SAF0", "SAF1", "TF", "CFin", "CFid")
    out = {}
    for kind in classes:
        d_march = d_naive = 0
        for _ in range(n_trials):
            a = int(rng.integers(0, n_mem))
            v = int(rng.integers(0, n_mem - 1))
            v = v + 1 if v >= a else v
            d_march += march_c_minus(FaultyMemory(n_mem, kind, a, v))
            d_naive += naive_test(FaultyMemory(n_mem, kind, a, v))
        out[kind] = {"march_c": round(d_march / n_trials, 3),
                     "naive_4n": round(d_naive / n_trials, 3)}
    return out


def run():
    cov = coverage()
    return {
        "shmoo": {
            "vt_V": VT, "alpha": ALPHA,
            "fmax_1v0_ghz": round(float(fmax_ghz(1.0)), 3),
            "vmin_1ghz_V": round(float(vmin_at(1.0)), 3),
            "vmin_1p5ghz_V": round(float(vmin_at(1.5)), 3),
        },
        "march_coverage": cov,
        "march_ops_per_bit": 10, "naive_ops_per_bit": 4,
    }


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))

    ax = axes[0]
    vs, fs, ok = shmoo_map()
    ax.imshow(ok, origin="lower", aspect="auto", cmap="RdYlGn",
              extent=[vs[0], vs[-1], fs[0], fs[-1]], alpha=0.85)
    vv = np.linspace(vs[0], vs[-1], 120)
    ax.plot(vv, fmax_ghz(vv), "k-", lw=1.6, label="Fmax(V) model")
    vmin = res["shmoo"]["vmin_1ghz_V"]
    ax.plot([vmin], [1.0], "b*", ms=14,
            label=f"Vmin @ 1 GHz = {vmin:.3f} V")
    ax.set_xlabel("Vdd [V]")
    ax.set_ylabel("clock frequency [GHz]")
    ax.set_title("Shmoo plot (green=pass) - alpha-power critical path,\n"
                 "OCV speckle at the boundary")
    ax.legend(fontsize=9, loc="upper left")

    ax = axes[1]
    cov = res["march_coverage"]
    ks = list(cov)
    x = np.arange(len(ks))
    ax.bar(x - 0.18, [cov[k]["march_c"] * 100 for k in ks], width=0.36,
           color="#26a", label="March C- (10N ops)")
    ax.bar(x + 0.18, [cov[k]["naive_4n"] * 100 for k in ks], width=0.36,
           color="#c86", label="naive w/r (4N ops)")
    ax.set_xticks(x, ks)
    ax.set_ylabel("fault detection [%]")
    ax.set_ylim(0, 112)
    ax.set_title("Memory-test coverage by fault class\n"
                 "(why the industry pays 10N)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    res = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    _plot(os.path.join(ASSETS, "ate_shmoo.png"), res)
    print("figure ->", ASSETS)


if __name__ == "__main__":
    main()
