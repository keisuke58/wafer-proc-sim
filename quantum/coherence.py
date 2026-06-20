"""
quantum.coherence — coherence-limited dicing model for quantum-processor chips.

For a logic die the binding dicing constraint is strength / throughput / chipping.
For a quantum chip it is **qubit coherence (T1)**. Dicing-induced subsurface
damage, debris and wet contamination create two-level-system (TLS) defects at the
chip edge. Their dielectric loss adds a tan δ channel whose participation at the
qubit limits the energy-relaxation time:

        1/T1 ≈ ω · Σ_i  p_i · tanδ_i              (Müller, Cole & Lisenfeld 2019)

where p_i is the electric-field participation ratio of lossy region i and tanδ_i
its loss tangent. The dicing edge is one such region: deeper subsurface damage
raises tanδ_edge, and the qubit's exposure to it falls with the keep-out distance
L (field participation decays ~exp(−L/λ)).

The consequence flips the classical dicing ranking: the wet, high-damage **blade**
route — cheap and perfectly fine for logic dies — is the **worst** choice for a
qubit chip, while dry, low-damage **stealth** dicing wins. The discriminator is no
longer die strength or throughput but *process cleanliness*.

Everything here is a transparent, literature-scaled **surrogate** (orders of
magnitude, monotone trends), not a calibrated predictor. Absolute T1 values depend
on the fab's real materials stack; treat the *rankings* and *relative* keep-out
costs as the message, not the digits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


TWO_PI = 2.0 * math.pi


# ── what the qubit needs ──────────────────────────────────────────────────────
@dataclass
class QubitSpec:
    """Design target for the qubit nearest the dicing edge."""
    f_GHz: float = 5.0               # transmon transition frequency
    T1_target_us: float = 100.0      # coherence budget the edge qubit must keep
    keepout_um: float = 200.0        # qubit ↔ dice-edge distance (a layout knob)
    debris_kill_um: float = 8.0      # a redeposited particle ≥ this shorts/spoils the qubit

    @property
    def omega(self) -> float:
        return TWO_PI * self.f_GHz * 1e9


# ── substrate dielectric quality (set the pristine T1 ceiling) ────────────────
@dataclass
class SubstrateTLS:
    """Intrinsic (damage-free) dielectric loss of the substrate at mK."""
    name: str
    tan_delta_bulk: float            # intrinsic substrate loss tangent
    p_bulk: float                    # bulk field participation
    field_decay_um: float            # edge-participation decay length λ


#: representative mK values — high-resistivity Si and c-plane sapphire are the
#: workhorse transmon substrates; diamond carries colour-centre spin qubits.
QUBIT_SUBSTRATES = {
    "Si":       SubstrateTLS("Si",       tan_delta_bulk=2.0e-6, p_bulk=0.08, field_decay_um=50.0),
    "Sapphire": SubstrateTLS("Sapphire", tan_delta_bulk=1.2e-6, p_bulk=0.08, field_decay_um=55.0),
    "Diamond":  SubstrateTLS("Diamond",  tan_delta_bulk=3.0e-6, p_bulk=0.08, field_decay_um=45.0),
}


# ── how a given dicing method damages the edge ────────────────────────────────
@dataclass
class ProcessSignature:
    """Damage fingerprint of a singulation method at the die sidewall."""
    method: str
    ssd_um: float            # subsurface-damage / modified-layer depth
    debris_um: float         # largest redeposited particle on the active face
    wet: bool                # wet kerf (water/slurry) → hydroxyl + particle contamination


#: canonical fingerprints (literature-typical for thinned dies; surrogate values).
#:   blade   : wide damage, large chips, water-cooled  → worst for coherence
#:   plasma  : near-zero mechanical damage, dry, but a thin chemical/fluoropolymer skin
#:   stealth : laser modified layer buried + clean cleave, dry, debris-free → best
CANONICAL_SIGNATURES = {
    "blade":   ProcessSignature("blade",   ssd_um=10.0, debris_um=12.0, wet=True),
    "plasma":  ProcessSignature("plasma",  ssd_um=0.5,  debris_um=1.0,  wet=False),
    "stealth": ProcessSignature("stealth", ssd_um=1.0,  debris_um=0.5,  wet=False),
}


def signature_from_outcome(outcome, wet: Optional[bool] = None) -> ProcessSignature:
    """Derive a ProcessSignature from a hybrid-stage ``StageOutcome``.

    Uses the deeper of the subsurface crack / HAZ as the damage depth and the
    stage chipping as the debris proxy. ``wet`` defaults from the method name.
    """
    method = getattr(outcome, "method", "unknown")
    if wet is None:
        wet = method in ("blade",)
    ssd = max(getattr(outcome, "subsurface_crack_um", 0.0),
              getattr(outcome, "haz_um", 0.0))
    return ProcessSignature(method, ssd_um=ssd,
                            debris_um=getattr(outcome, "chipping_um", 0.0), wet=wet)


# ── loss-budget constants (surrogate; chosen for plausible mK T1 ~ 80–250 µs) ─
_TAN_EDGE_BASE = 2.0e-3      # loss tangent of a thin damaged sidewall layer
_SSD_REF_UM = 2.0            # damage depth at which the edge loss has ~doubled
_P_EDGE0 = 1.0e-3           # edge-region participation extrapolated to L→0
_TAN_WET = 1.0e-3           # extra loss tangent from wet hydroxyl/particle skin
_P_WET0 = 2.0e-3           # wet-skin participation at L→0


# ── coherence evaluation ──────────────────────────────────────────────────────
@dataclass
class CoherenceReport:
    method: str
    T1_us: float
    tls_loss: float                  # total Σ p·tanδ
    debris_fail: bool                # a killer particle reached the qubit
    feasible: bool                   # meets T1 budget AND survives debris
    min_keepout_um: float            # keep-out needed to just meet T1_target (∞ if impossible)
    violations: list = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.feasible


def _edge_amplitude(sig: ProcessSignature) -> float:
    """L→0 participation-weighted loss of the damaged edge (incl. wet skin)."""
    tan_edge = _TAN_EDGE_BASE * (1.0 + sig.ssd_um / _SSD_REF_UM)
    amp = _P_EDGE0 * tan_edge
    if sig.wet:
        amp += _P_WET0 * _TAN_WET
    return amp


def evaluate_coherence(sig: ProcessSignature, qubit: QubitSpec,
                       substrate: SubstrateTLS) -> CoherenceReport:
    """Estimate edge-qubit T1 for a dicing signature and the keep-out it needs."""
    loss_bulk = substrate.p_bulk * substrate.tan_delta_bulk
    amp = _edge_amplitude(sig)
    decay = math.exp(-qubit.keepout_um / substrate.field_decay_um)
    loss = loss_bulk + amp * decay
    T1_us = 1.0 / (qubit.omega * loss) * 1e6

    debris_fail = sig.debris_um >= qubit.debris_kill_um

    # keep-out that just meets the T1 budget: loss_bulk + amp·e^{−L/λ} = loss_target
    loss_target = 1.0 / (qubit.omega * qubit.T1_target_us * 1e-6)
    if loss_target <= loss_bulk:
        min_keepout = math.inf                       # bulk alone already too lossy
    else:
        ratio = (loss_target - loss_bulk) / amp
        min_keepout = 0.0 if ratio >= 1.0 else -substrate.field_decay_um * math.log(ratio)

    viols = []
    if T1_us < qubit.T1_target_us:
        viols.append(f"T1 {T1_us:.0f} < target {qubit.T1_target_us:.0f} µs "
                     f"(needs keep-out ≥ {min_keepout:.0f} µm at L={qubit.keepout_um:.0f})")
    if debris_fail:
        viols.append(f"debris {sig.debris_um:.1f} ≥ kill {qubit.debris_kill_um:.1f} µm "
                     "(redeposited particle on active face)")

    return CoherenceReport(
        method=sig.method, T1_us=T1_us, tls_loss=loss,
        debris_fail=debris_fail, feasible=(len(viols) == 0),
        min_keepout_um=min_keepout, violations=viols,
    )


def rank_methods(qubit: QubitSpec, substrate: SubstrateTLS,
                 signatures=None) -> list:
    """Evaluate every canonical method and sort best-coherence-first.

    Returns a list of (ProcessSignature, CoherenceReport) sorted by descending T1.
    """
    signatures = signatures or list(CANONICAL_SIGNATURES.values())
    reports = [(s, evaluate_coherence(s, qubit, substrate)) for s in signatures]
    reports.sort(key=lambda sr: sr[1].T1_us, reverse=True)
    return reports
