#!/usr/bin/env python
"""
benchmark_all_kernels.py
Measures C++ kernel speedup vs pure-Python / NumPy baselines.
Run from repo root:
    python benchmark_all_kernels.py
"""
from __future__ import annotations
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

REPS = 5  # median of REPS runs for each measurement

def timeit(fn, n_repeats=REPS):
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


RESULTS: list[dict] = []

def record(name, py_s, cpp_s):
    speedup = py_s / cpp_s if cpp_s > 0 else float("inf")
    RESULTS.append({"kernel": name, "py_ms": py_s*1e3, "cpp_ms": cpp_s*1e3,
                    "speedup": speedup})


# ─────────────────────────────────────────────────────────────────────────────
# 1. NSGA-II  (nondominated sort + crowding)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from optimization import _nsga2_kernel as _nsga2

    rng = np.random.default_rng(0)
    POP = 300
    F = np.ascontiguousarray(rng.random((POP, 3)))

    def _py_dominates(a, b):
        return np.all(a <= b) and np.any(a < b)

    def py_nondom_sort():
        fronts = [[]]
        n = len(F)
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, int)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if _py_dominates(F[i], F[j]):
                    S[i].append(j)
                elif _py_dominates(F[j], F[i]):
                    n_dom[i] += 1
            if n_dom[i] == 0:
                fronts[0].append(i)
        i_front = 0
        while fronts[i_front]:
            nxt = []
            for p in fronts[i_front]:
                for q in S[p]:
                    n_dom[q] -= 1
                    if n_dom[q] == 0:
                        nxt.append(q)
            fronts.append(nxt)
            i_front += 1
        return fronts

    py_t  = timeit(py_nondom_sort)
    cpp_t = timeit(lambda: _nsga2.nondominated_sort(F))
    record("NSGA-II nondom sort (N=300, obj=3)", py_t, cpp_t)
except Exception as e:
    print(f"[skip] nsga2: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. EnKF blade wear APC  (ensemble analysis step)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from ml import _enkf_kernel as _enkf

    rng = np.random.default_rng(1)
    NE, NX = 200, 3   # blade wear state dim = 3
    X  = np.ascontiguousarray(np.abs(rng.standard_normal((NE, NX))) * 5)
    # enkf_analysis expects 1-D yf (NE,), v (NE,)
    y_f   = np.ascontiguousarray(np.abs(rng.standard_normal(NE)))
    v_f   = np.ascontiguousarray(rng.standard_normal(NE))
    y_obs_val = float(rng.standard_normal())
    R_val = 0.1

    def py_enkf_blade():
        Xf   = X.copy()
        y_pred = np.array(y_f)
        innov  = y_obs_val - y_pred
        Pf  = np.cov(Xf.T)
        # scalar measurement: H = [1,0,0]
        PHt = Pf[:, 0]
        S   = Pf[0, 0] + R_val
        K   = PHt / S
        for i in range(NE):
            Xf[i] += K * innov[i]
        return Xf

    py_t  = timeit(py_enkf_blade)
    cpp_t = timeit(lambda: _enkf.enkf_analysis(X, y_f, y_obs_val, v_f, R_val))
    record("EnKF analysis (NE=200, Nx=3)", py_t, cpp_t)
except Exception as e:
    print(f"[skip] enkf: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. GP RBF kernel matrix  (N=300, D=8)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from ml import _gp_inference_kernel as _gp

    rng = np.random.default_rng(2)
    N, D = 300, 8
    X  = np.ascontiguousarray(rng.standard_normal((N, D)))
    ls = np.ascontiguousarray(np.ones(D))

    def py_rbf_matrix():
        diff = X[:, None, :] - X[None, :, :]   # (N, N, D)
        s2   = np.sum((diff / ls)**2, axis=-1)
        return np.exp(-0.5 * s2)

    py_t  = timeit(py_rbf_matrix)
    cpp_t = timeit(lambda: _gp.rbf_kernel_matrix(X, ls, 1.0))
    record("GP RBF kernel matrix (N=300, D=8)", py_t, cpp_t)
except Exception as e:
    print(f"[skip] gp_matrix: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. GP predictive mean  (Ntest=500 vs Ntrain=200, D=6)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from ml import _gp_inference_kernel as _gp

    rng = np.random.default_rng(3)
    Nte, Ntr, D = 500, 200, 6
    Xte   = np.ascontiguousarray(rng.standard_normal((Nte, D)))
    Xtr   = np.ascontiguousarray(rng.standard_normal((Ntr, D)))
    alpha = np.ascontiguousarray(rng.standard_normal(Ntr))
    ls    = np.ascontiguousarray(np.ones(D))

    def py_gp_mean():
        K = np.exp(-0.5 * np.sum(
            ((Xte[:,None,:] - Xtr[None,:,:]) / ls)**2, axis=-1))
        return K @ alpha

    py_t  = timeit(py_gp_mean)
    cpp_t = timeit(lambda: _gp.gp_predict_mean(Xte, Xtr, alpha, ls, 1.0))
    record("GP predict mean (Mtest=500, N=200, D=6)", py_t, cpp_t)
except Exception as e:
    print(f"[skip] gp_mean: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. SVPWM  (10 000 samples)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from fem import _servo_inverter_kernel as _inv

    rng = np.random.default_rng(4)
    N   = 10_000
    v_a = rng.uniform(-200, 200, N)
    v_b = rng.uniform(-200, 200, N)
    vdc = 600.0

    def py_svpwm_batch():
        out = []
        for va, vb in zip(v_a, v_b):
            v1 = vb; v2 = -0.5*vb + 0.866*va; v3 = -0.5*vb - 0.866*va
            vmax = max(v1, v2, v3); vmin = min(v1, v2, v3)
            v_off = -(vmax + vmin) / 2
            da = np.clip((va + v_off) / vdc + 0.5, 0, 1)
            db = np.clip((-0.5*va + 0.866*vb + v_off) / vdc + 0.5, 0, 1)
            dc = np.clip((-0.5*va - 0.866*vb + v_off) / vdc + 0.5, 0, 1)
            out.append((da, db, dc))
        return out

    def cpp_svpwm_batch():
        return [_inv.svpwm(va, vb, vdc) for va, vb in zip(v_a, v_b)]

    py_t  = timeit(py_svpwm_batch)
    cpp_t = timeit(cpp_svpwm_batch)
    record("SVPWM batch (N=10 000)", py_t, cpp_t)
except Exception as e:
    print(f"[skip] svpwm: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Quadrature encoder decode  (100 000 samples)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from fem import _encoder_kernel as _enc

    rng = np.random.default_rng(5)
    N   = 100_000
    theta_rad = np.linspace(0, 20 * 2 * np.pi, N)
    ppr = 2500
    enc_sim   = _enc.EncoderSimulator(ppr, 0.0)
    sig       = enc_sim.simulate(theta_rad, dt_s=1e-5)
    A_arr     = np.ascontiguousarray(np.array(sig["a_sig"], dtype=np.int32))
    B_arr     = np.ascontiguousarray(np.array(sig["b_sig"], dtype=np.int32))

    def py_decode(A, B):
        lut = {(0,0,0,0):0,(0,0,0,1):-1,(0,0,1,0):1,(0,0,1,1):0,
               (0,1,0,0):1,(0,1,0,1):0,(0,1,1,0):0,(0,1,1,1):-1,
               (1,0,0,0):-1,(1,0,0,1):0,(1,0,1,0):0,(1,0,1,1):1,
               (1,1,0,0):0,(1,1,0,1):1,(1,1,1,0):-1,(1,1,1,1):0}
        cnt = 0; counts = []
        prev_a, prev_b = A[0], B[0]
        for a, b in zip(A, B):
            cnt += lut.get((prev_a, prev_b, a, b), 0)
            counts.append(cnt)
            prev_a, prev_b = a, b
        return np.array(counts)

    py_t  = timeit(lambda: py_decode(A_arr, B_arr))
    cpp_t = timeit(lambda: _enc.decode_quadrature(A_arr, B_arr))
    record("Quadrature decode (N=100 000)", py_t, cpp_t)
except Exception as e:
    print(f"[skip] encoder: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. DiscoMachine unified tick  (10 000 steps)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from machine import _disco_machine as _dm

    N_STEPS = 10_000

    def cpp_machine_sim():
        m = _dm.DiscoMachine(_dm.MachineParams())
        return m.simulate(N_STEPS)

    # Python baseline: same physics as DiscoMachine tick()
    # PMSM (dq-axis), PI FOC, Clarke, SVPWM sectors, 3-axis P-ctrl
    Rs = 0.5; Ld = 1.5e-3; Lq = 1.5e-3; pn = 4; psi_f = 0.08
    J = 1e-4; B_fric = 0.01; dt_s = 10e-6
    Kp_i = 10.0; Ki_i = 50.0; Kp_m = 50.0; v_max = 300.0
    omega_ref = 30000 * 2 * 3.14159 / 60

    def py_full_sim():
        omega = 0.0; i_d = 0.0; i_q = 0.0; int_iq = 0.0; theta = 0.0
        x = 0.0; y = 0.0; z = 0.0
        for _ in range(N_STEPS):
            # FOC: iq ref from speed error
            iq_ref = min(20.0, max(-20.0, (omega_ref - omega) * 5.0))
            # PI current ctrl (q-axis)
            err_q   = iq_ref - i_q
            int_iq += err_q * dt_s
            ff_d    = -pn * omega * Lq * i_q
            ff_q    =  pn * omega * (Ld * i_d + psi_f)
            v_d = Kp_i * (0.0 - i_d) + ff_d
            v_q = Kp_i * err_q + Ki_i * int_iq + ff_q
            # dq current dynamics
            T_e   = 1.5 * pn * (psi_f * i_q + (Ld - Lq) * i_d * i_q)
            omega += (T_e - B_fric * omega) / J * dt_s
            theta += omega * dt_s
            i_d   += (v_d - Rs * i_d + pn * omega * Lq * i_q) / Ld * dt_s
            i_q   += (v_q - Rs * i_q - pn * omega * (Ld * i_d + psi_f)) / Lq * dt_s
            # Clarke → SVPWM sector
            ct = 0.866 * 2 / 3   # approx
            v_al = v_d; v_be = v_q
            v1 = v_be; v2 = -0.5*v_be + 0.866*v_al; v3 = -0.5*v_be - 0.866*v_al
            vmax_ = max(v1, v2, v3); vmin_ = min(v1, v2, v3)
            v_off = -(vmax_ + vmin_) / 2
            # 3-axis P-control + encoder count update
            err_x = 50.0 - x; vx = min(v_max, max(-v_max, Kp_m * err_x)); x += vx * dt_s
            err_y = 0.0  - y; vy = min(v_max, max(-v_max, Kp_m * err_y)); y += vy * dt_s
            err_z = 0.0  - z; vz = min(v_max, max(-v_max, Kp_m * err_z)); z += vz * dt_s
        return omega

    py_t  = timeit(py_full_sim)
    cpp_t = timeit(cpp_machine_sim)
    record("DiscoMachine unified (spindle+SVPWM+3axis, N=10 000)", py_t, cpp_t)
except Exception as e:
    print(f"[skip] machine: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Print table
# ─────────────────────────────────────────────────────────────────────────────
print()
print(f"{'Kernel':<46} {'Python':>9} {'C++':>9} {'Speedup':>9}")
print("-" * 78)
for r in RESULTS:
    print(f"{r['kernel']:<46} {r['py_ms']:8.2f}ms {r['cpp_ms']:8.2f}ms {r['speedup']:7.1f}×")
print()
