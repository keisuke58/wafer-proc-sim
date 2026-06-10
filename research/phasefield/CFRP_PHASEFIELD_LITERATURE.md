# Phase-Field Fracture for CFRP Laminates — Formulation Memo

Literature survey (2026-06-11) supporting the extension of `at2_simulator_2d.py`
(SiC single-crystal AT2) toward CFRP laminates, for the Keio M-thesis pipeline:

> NDT Bayesian posterior of a defect (position / layer / size)
> → phase-field prognosis of defect growth in a CFRP laminate
> → probability of failure P(growth | measurement).

Implementation target: `cfrp_phasefield_2d.py` (2D FD, numpy/scipy, no FEniCS).

---

## 0. Baseline we extend: variational AT2

Total energy (Bourdin–Francfort–Marigo regularization of Griffith):

```
E[u, d] = ∫ g(d) ψ_e(ε(u)) dV  +  Gc ∫ γ(d, ∇d) dV
g(d)    = (1-d)² + κ                            (degradation)
γ_AT2   = d²/(2ℓ) + (ℓ/2) |∇d|²                 (crack surface density)
```

Staggered Euler–Lagrange for d with history field H = max_t ψ_e⁺ (Miehe 2010):

```
-Gc ℓ ∇·(∇d) + (Gc/ℓ + 2H) d = 2H,   d̊ ≥ 0 (irreversible)
```

This is exactly what `at2_simulator_2d.py` solves, with the anisotropic
generalization below already in place.

---

## 1. Anisotropic PF for fiber-reinforced plies (intra-ply matrix cracking)

### 1.1 Teichtmeister, Kienle, Aldakheel & Keip (2017), IJNLM 97:1-21
"Phase field modeling of fracture in anisotropic brittle solids."

Anisotropy enters **geometrically** through a second-order structural tensor in
the crack surface density:

```
γ(d, ∇d) = d²/(2ℓ) + (ℓ/2) ∇d · A ∇d
A        = I + β (a ⊗ a)        (transverse isotropy; β > -1)
```

- `a` = unit vector of the penalized crack-normal direction; β > 0 makes
  cracks whose normal ∥ a energetically expensive ⇒ cracks prefer to run
  **parallel** to the plane with normal ⊥ a.
- Effective toughness for a crack with normal n: `Gc_eff(n) ≈ Gc (1 + β (n·a)²)`.
- Higher symmetry classes (cubic) require fourth-order structural tensors and
  second gradients of d (not needed for unidirectional plies).

This is **identical in structure** to the existing `PFParams2D.anisotropy_tensor()`
(`A = I + β c⊗c`, c = cleavage-plane normal). For a unidirectional CFRP ply the
mapping is: **c = fiber direction** (a matrix crack cutting fibers has its
normal along the fibers, which must be penalized), β >> 1.

### 1.2 Quintanas-Corominas, Reinoso, Casoni, Turon & Mayugo (2019),
Composite Structures 220:899-911 — "A phase field approach to simulate
intralaminar and translaminar fracture in long fiber composite materials."

Applies the Teichtmeister-type structural-tensor PF to laminate plies:

- Per-ply structural tensor rotated by the **ply fiber angle θ_ply**:
  `A(θ) = I + β a(θ)⊗a(θ)`, a = fiber direction in the analysis plane.
- β chosen large (order 10–100) so matrix cracks run along fibers;
  translaminar (fiber-breaking) cracks only occur when the driving force
  overcomes the β-amplified toughness.
- Two toughnesses in concept: matrix-dominated `Gc_m` and fiber-dominated
  `Gc_f`, with `Gc_f / Gc_m` of order 10²:
  - transverse intralaminar (matrix) Gc ≈ 0.2–1.0 kJ/m²
  - translaminar (fiber breaking) Gc ≈ 13–133 kJ/m² depending on system
    (e.g. T300/913 tensile fiber failure: 91.6 kJ/m² initiation, 133 kJ/m²
    propagation; plain-weave carbon mode-I translaminar ≈ 13.5 kJ/m²).
- Orthotropic ply elasticity (E1 >> E2). In a 2D FD prototype we keep
  isotropic elasticity and put ALL anisotropy into the fracture term — a
  documented approximation (crack-path anisotropy is dominated by the
  surface-energy anisotropy when β >> 1).

### 1.3 Bleyer & Alessi (2018), CMAME 336:213-236
"Phase-field modeling of anisotropic brittle fracture including several
damage mechanisms."

Instead of a structural tensor, use **several damage variables** d_i, each
with its own toughness and its own portion of the elastic energy:

```
E[u, d_1..d_n] = ∫ ψ(ε, d_1..d_n) dV + Σ_i (Gc_i) ∫ γ_i(d_i, ∇d_i) dV
LTD example:   ψ = g(d_f) ψ_longitudinal(ε) + g(d_m) ψ_transverse+shear(ε)
```

- d_f degrades the fiber-direction (longitudinal) stiffness, d_m the
  transverse/shear stiffness; each has its own Gc_f >> Gc_m and possibly its
  own ℓ. Anisotropic crack paths emerge without any structural tensor.
- Cost: n coupled damage solves per staggered step + an energy split by
  direction (requires orthotropic elasticity to be meaningful).

### 1.4 Tan & Martínez-Pañeda (2021), Comp. Sci. Tech. 202:108539 (+ 2022,
Composite Structures) — micro-scale PF + CZM embedded-cell virtual testing.

- Micro scale: fibers/matrix resolved; PF in matrix and fibers, CZM for
  fiber–matrix debonding. Reproduces experimental R-curves; shows R-curve
  sensitivity to matrix Gc and interface properties.
- Relevant to us only as parameter source (matrix epoxy Gc ≈ 0.1–0.3 kJ/m²
  neat resin; in-situ ply-level transverse Gc higher, 0.2–1 kJ/m²) — our
  simulator works at the **ply/meso scale**, not micro scale.

## 2. Interface / delamination

### 2.1 Paggi & Reinoso (2017), CMAME 321:145-172
"Revisiting the problem of a crack impinging on an interface…"

- PF (AT2) in the bulk + **cohesive zone model on the interface** as a
  separate energy term: `E = ∫_Ω g(d) ψ_e + Gc γ(d,∇d) dV + ∫_Γ G_int(⟦u⟧, d) dS`.
- Key coupling: the interface traction–separation stiffness is degraded by the
  bulk phase field d evaluated at the interface (apparent interface toughness
  and stiffness drop as the surrounding bulk is damaged) — reproduces the
  He–Hutchinson crack deflection-vs-penetration competition.

### 2.2 Carollo, Reinoso & Paggi (2017), Composite Structures 182:636-651
(and 2018 follow-ups) — 3D finite-strain PF + CZM for laminates: intralayer
cracks via PF, interlayer delamination via CZM; captures delamination
migration and matrix-crack-induced delamination.

### 2.3 Quintanas-Corominas et al. (2020), CMAME — "A phase field approach
enhanced with a cohesive zone model for modeling delamination induced by
matrix cracking." Same architecture (PF intra-ply + CZM interlaminar) at the
meso scale, mode-mix-dependent interface toughness.

**Interface parameters (CFRP/epoxy, DCB/ENF data):**
- Mode-I interlaminar GIc ≈ 0.2–0.5 kJ/m² (0.205–0.279 brittle epoxies,
  0.487–0.52 toughened systems; interleaved laminates up to ≈ 0.7).
- Mode-II GIIc typically 2–4× GIc (≈ 0.6–1.8 kJ/m²).
- Interface GIc is comparable to or LOWER than in-situ ply transverse Gc ⇒
  delamination is competitive whenever a matrix crack reaches an interface.

### 2.4 What a pure-PF FD code can honestly do

A true CZM needs displacement-jump DOFs (interface elements) — not available
in our continuous FD grid. The standard "PF-only" surrogate (used e.g. in
heterogeneous-Gc PF studies and block-AMR PF delamination work) is a **thin
interface band of width ~ℓ with strongly reduced Gc** (and optionally its own
structural tensor aligned with the interface). This reproduces deflection vs
penetration competition qualitatively (crack chooses the low-Gc path when
Gc_int/Gc_ply is below a geometry-dependent threshold) but NOT the
traction–separation law, mode-mix dependence, or correct interface stiffness.
We adopt this surrogate and document it as such.

## 3. PF-CZM: Wu's unified phase-field theory (cohesive cracks)

Wu (2017), JMPS 103:72-99 "A unified phase-field theory for the mechanics of
damage and quasi-brittle failure"; Wu & Nguyen (2018), JMPS 119:20-42
(length-scale-insensitive version).

Two generic characteristic functions:

```
γ(d,∇d) = (1/c_α) [ α(d)/ℓ + ℓ |∇d|² ],   α(d) = ξ d + (1-ξ) d²,  c_α = ∫₀¹ 4√α dδ
g(d)    = (1-d)^p / [ (1-d)^p + a₁ d (1 + a₂ d + a₃ d²) ]
a₁      = (4 / (π ℓ)) · l_ch,   l_ch = E Gc / f_t²   (Irwin length)
```

- ξ = 2 (α = 2d − d², c_α = π) gives a finite damage bandwidth and an elastic
  stage; (p, a₂, a₃) select the softening law:
  - linear softening: p = 2, a₂ = −1/2, a₃ = 0
  - **Cornelissen et al. (concrete-like, default for quasi-brittle):
    p = 2.5, a₂ ≈ 1.3868, a₃ ≈ 0.6567**
- Results are insensitive to ℓ (for ℓ small enough vs l_ch) and the model has
  a genuine strength f_t — unlike AT2 whose nominal strength scales as
  ℓ^(-1/2). This is the modern standard for quasi-brittle/cohesive fracture
  and the natural choice when the epoxy matrix / interface softening matters.
- Cost in our FD setting: α'(d) = 2−2d makes the damage problem a bound-
  constrained variational inequality (the AT2 trick "solve linear system,
  then clip" is no longer the exact KKT solution) and g(d) is rational ⇒
  the staggered damage update needs an inner Newton/active-set iteration.

## 4. Review

Bui & Hu (2021), Engineering Fracture Mechanics 248:107705 — "A review of
phase-field models, fundamentals and their applications to composite
laminates." Confirms the taxonomy used above: (i) structural-tensor
anisotropic PF for intra-ply cracking, (ii) PF+CZM hybrids for delamination,
(iii) multi-phase-field (Bleyer–Alessi-type) for distinct mechanisms,
(iv) PF-CZM for quasi-brittle constituents.

---

## 5. What we adopt (decision)

| Ingredient | Choice | Source |
|---|---|---|
| Bulk model | AT2 + history field (reuse existing solver) | Miehe 2010; existing code |
| Ply anisotropy | Per-ply structural tensor A(θ_ply) = I + β a⊗a, a = fiber direction, **spatially varying** | Teichtmeister 2017; QC 2019 |
| Direction-dependent Gc | Optional scalar Gc map per ply + β-amplified Gc_eff(n) | QC 2019 |
| Delamination | Thin interface band (width ~ℓ) with Gc_int << Gc_ply and interface-aligned A | surrogate of Paggi–Reinoso 2017 / Carollo 2017 (§2.4 caveats) |
| Two damage fields | **Deferred (future work)**: requires orthotropic energy split to be meaningful; single-d + spatial Gc/A maps covers the crack-path physics we need for P(growth) | Bleyer–Alessi 2018 |
| PF-CZM (Wu) | **Deferred**: AT2 retained for solver simplicity; noted that predicted strengths are ℓ-dependent ⇒ report P(growth) trends vs load, not absolute strengths | Wu 2017 |
| Defect seeding | Initial d=1 disk/ellipse at (cx, cy, layer, size) = format of FMPE NDT posterior; P(growth) = fraction of posterior draws whose crack grows beyond threshold | thesis pipeline |

**Parameter defaults adopted (normalized units; physical anchors):**
- Gc_ply (matrix-dominated transverse) — anchor 0.2–1.0 kJ/m²
- Gc_interface / Gc_ply ≈ 0.3–0.6 — anchor GIc ≈ 0.2–0.5 kJ/m²
- β_ply = 20–50 (fiber-direction toughening; effective Gc_f/Gc_m = 1+β ≈
  20–50, conservative vs the experimental 10²; β > ~100 degrades FD
  conditioning at nx ≈ 60)
- ℓ ≥ 2 grid spacings (FD resolution), interface band thickness ≈ ℓ

### Numerical implications for the FD solver (spatially varying Gc, A)
The damage operator becomes `-ℓ ∇·(Gc(x) A(x) ∇d) + (Gc(x)/ℓ + 2H) d = 2H`
in conservative (flux) form: face-averaged `Gc·A` coefficients, mirroring how
the elasticity matrix already face-averages g. Cross-derivative terms use
cell-centered Axy averaging. This is a strict generalization of
`_build_2d_laplacian` (constant-A case recovers it exactly).

## 6. Open issues
- AT2 strength ℓ-dependence: P(growth) thresholds are relative, calibrate ℓ
  against in-situ ply strength before quoting absolute failure loads (or
  migrate to Wu PF-CZM).
- Isotropic elasticity per ply: crack DRIVING force ignores E1/E2 ≈ 10–20 of
  real UD plies; structural-tensor surface energy dominates path selection but
  load-displacement curves are not quantitative.
- Interface band is a Gc-surrogate, not a traction–separation law: no
  mode-mix, no interface stiffness; calibration of Gc_int to DCB data is
  only order-of-magnitude.
- Two-field (Bleyer–Alessi) model deferred; would double solver cost and
  requires an orthotropic split — future work once FEniCS version exists.

## 7. References
1. Teichtmeister, Kienle, Aldakheel, Keip (2017). Int. J. Non-Linear Mech. 97:1-21.
2. Quintanas-Corominas, Reinoso, Casoni, Turon, Mayugo (2019). Compos. Struct. 220:899-911.
3. Quintanas-Corominas et al. (2020). CMAME 360:112731.
4. Bleyer, Alessi (2018). CMAME 336:213-236.
5. Paggi, Reinoso (2017). CMAME 321:145-172.
6. Carollo, Reinoso, Paggi (2017). Compos. Struct. 182:636-651.
7. Tan, Martínez-Pañeda (2021). Compos. Sci. Technol. 202:108539.
8. Wu (2017). JMPS 103:72-99; Wu, Nguyen (2018). JMPS 119:20-42.
9. Bui, Hu (2021). Eng. Fract. Mech. 248:107705.
10. Miehe, Hofacker, Welschinger (2010). CMAME 199:2765-2778.
