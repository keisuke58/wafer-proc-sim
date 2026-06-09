//! wafer-proc-sim Rust kernels
//!
//! Hot-loop reimplementation of the DISCO dicing saw physics in Rust,
//! exposed to Python via PyO3.  Mirrors the C++ pybind11 kernels but
//! demonstrates the same architecture in a memory-safe language.
//!
//! Build:
//!   cd rust && maturin develop --release      # installs into current venv
//!   # or: maturin build --release             # builds .whl

use pyo3::prelude::*;
use pyo3::types::PyDict;

// ── PMSM spindle parameters ──────────────────────────────────────────────────

struct SpindleParams {
    rs: f64,       // stator resistance [Ω]
    ld: f64,       // d-axis inductance [H]
    lq: f64,       // q-axis inductance [H]
    psi_f: f64,    // flux linkage [Wb]
    pn: f64,       // pole pairs
    j: f64,        // rotor inertia [kg·m²]
    b_fric: f64,   // viscous friction coefficient
    kp_i: f64,     // current loop P gain
    ki_i: f64,     // current loop I gain
}

impl Default for SpindleParams {
    fn default() -> Self {
        Self {
            rs: 0.5, ld: 1.5e-3, lq: 1.5e-3, psi_f: 0.08,
            pn: 4.0, j: 1e-4, b_fric: 0.01,
            kp_i: 10.0, ki_i: 50.0,
        }
    }
}

// ── SpindleKernel Python class ────────────────────────────────────────────────

/// PMSM + FOC spindle simulation in Rust.
/// Matches `fem/_spindle_kernel.cpp` (C++ reference implementation).
#[pyclass]
struct SpindleKernel {
    p: SpindleParams,
    omega: f64,
    i_d: f64,
    i_q: f64,
    int_iq: f64,
}

#[pymethods]
impl SpindleKernel {
    #[new]
    fn new() -> Self {
        Self { p: SpindleParams::default(), omega: 0.0, i_d: 0.0, i_q: 0.0, int_iq: 0.0 }
    }

    /// Single integration step.
    /// Returns (omega_rad_s, i_q, T_e_Nm).
    fn tick(&mut self, omega_ref_rpm: f64, dt_us: f64) -> (f64, f64, f64) {
        let dt = dt_us * 1e-6;
        let p = &self.p;

        // Speed controller → Iq reference
        let omega_ref = omega_ref_rpm * std::f64::consts::TAU / 60.0;
        let iq_ref = ((omega_ref - self.omega) * 5.0).clamp(-20.0, 20.0);

        // PI current controller (q-axis)
        let err_q = iq_ref - self.i_q;
        self.int_iq += err_q * dt;
        let ff_q = p.pn * self.omega * (p.ld * self.i_d + p.psi_f);
        let ff_d = -p.pn * self.omega * p.lq * self.i_q;
        let v_q = p.kp_i * err_q + p.ki_i * self.int_iq + ff_q;
        let v_d = p.kp_i * (0.0 - self.i_d) + ff_d;

        // dq current dynamics (Euler)
        self.i_d += (v_d - p.rs * self.i_d + p.pn * self.omega * p.lq * self.i_q)
                    / p.ld * dt;
        self.i_q += (v_q - p.rs * self.i_q - p.pn * self.omega * (p.ld * self.i_d + p.psi_f))
                    / p.lq * dt;

        // Mechanical dynamics
        let t_e = 1.5 * p.pn * (p.psi_f * self.i_q + (p.ld - p.lq) * self.i_d * self.i_q);
        self.omega += (t_e - p.b_fric * self.omega) / p.j * dt;

        (self.omega, self.i_q, t_e)
    }

    /// Simulate n_steps, return telemetry as Python dict.
    fn simulate(&mut self, n_steps: usize, omega_ref_rpm: f64, dt_us: f64, py: Python<'_>)
        -> PyResult<PyObject>
    {
        let mut omega_log = Vec::with_capacity(n_steps);
        let mut iq_log    = Vec::with_capacity(n_steps);
        let mut te_log    = Vec::with_capacity(n_steps);

        for _ in 0..n_steps {
            let (om, iq, te) = self.tick(omega_ref_rpm, dt_us);
            omega_log.push(om);
            iq_log.push(iq);
            te_log.push(te);
        }

        let d = PyDict::new_bound(py);
        d.set_item("omega_rad_s", omega_log)?;
        d.set_item("i_q",        iq_log)?;
        d.set_item("T_e_Nm",     te_log)?;
        d.set_item("speed_rpm",
            omega_log.last().copied().unwrap_or(0.0) * 60.0 / std::f64::consts::TAU)?;
        Ok(d.into())
    }

    fn reset(&mut self) {
        self.omega = 0.0;
        self.i_d   = 0.0;
        self.i_q   = 0.0;
        self.int_iq = 0.0;
    }

    #[getter] fn omega_rad_s(&self) -> f64 { self.omega }
    #[getter] fn speed_rpm(&self)   -> f64 { self.omega * 60.0 / std::f64::consts::TAU }
    #[getter] fn i_q(&self)         -> f64 { self.i_q }
    #[getter] fn i_d(&self)         -> f64 { self.i_d }
}

// ── NSGA-II nondominated sort (Rust hot loop) ─────────────────────────────────

/// Nondominated sort for a (pop, obj) f64 matrix.
/// Returns Vec of front indices: fronts[0] = Pareto front, etc.
#[pyfunction]
fn nondominated_sort(f: Vec<Vec<f64>>) -> Vec<Vec<usize>> {
    let n = f.len();
    if n == 0 { return vec![]; }

    let dominates = |a: &Vec<f64>, b: &Vec<f64>| -> bool {
        a.iter().zip(b).all(|(ai, bi)| ai <= bi) &&
        a.iter().zip(b).any(|(ai, bi)| ai < bi)
    };

    let mut sp: Vec<Vec<usize>> = vec![vec![]; n];
    let mut np: Vec<usize>      = vec![0; n];
    let mut fronts: Vec<Vec<usize>> = vec![vec![]];

    for i in 0..n {
        for j in 0..n {
            if i == j { continue; }
            if dominates(&f[i], &f[j]) {
                sp[i].push(j);
            } else if dominates(&f[j], &f[i]) {
                np[i] += 1;
            }
        }
        if np[i] == 0 { fronts[0].push(i); }
    }

    let mut fi = 0;
    while !fronts[fi].is_empty() {
        let mut next = vec![];
        for &p in &fronts[fi] {
            for &q in &sp[p] {
                np[q] -= 1;
                if np[q] == 0 { next.push(q); }
            }
        }
        fronts.push(next);
        fi += 1;
    }
    fronts.pop(); // remove trailing empty
    fronts
}

// ── RBF kernel matrix (Rust hot loop) ────────────────────────────────────────

/// Compute N×N symmetric RBF kernel matrix.
/// k(x,x') = signal_var * exp(-0.5 * sum((xi-xj)²/li²))
#[pyfunction]
fn rbf_kernel_matrix(
    x: Vec<Vec<f64>>,
    length_scales: Vec<f64>,
    signal_var: f64,
) -> Vec<Vec<f64>> {
    let n = x.len();
    let d = length_scales.len();
    let mut k = vec![vec![0.0f64; n]; n];

    for i in 0..n {
        k[i][i] = signal_var;
        for j in (i + 1)..n {
            let s: f64 = (0..d)
                .map(|dim| { let diff = (x[i][dim] - x[j][dim]) / length_scales[dim]; diff * diff })
                .sum();
            let kij = signal_var * (-0.5 * s).exp();
            k[i][j] = kij;
            k[j][i] = kij;
        }
    }
    k
}

// ── module ────────────────────────────────────────────────────────────────────

#[pymodule]
fn wafer_proc_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SpindleKernel>()?;
    m.add_function(wrap_pyfunction!(nondominated_sort, m)?)?;
    m.add_function(wrap_pyfunction!(rbf_kernel_matrix, m)?)?;
    Ok(())
}
