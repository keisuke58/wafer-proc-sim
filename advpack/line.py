"""
packaging.line — the connected advanced-packaging process line.

Analytic, literature-typical physics (order-of-magnitude prototype, to be
calibrated against real fab/DISCO data — same disclaimer as wafer-proc-sim).

Key coupling (the headline): warpage follows Stoney ~ σ·t_f / (M·t_s²), so the
bow grows QUADRATICALLY as the wafer is thinned. That warp then drives bonding
overlay error and void risk — i.e. the thinning recipe upstream decides whether
the bonded stack is even manufacturable downstream.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from data.materials.material_properties import ALL_MATERIALS
import semiconductor_fracture_methods as sfm
from fem.hybrid_bonding_model import bond_strength


# ── config ───────────────────────────────────────────────────────────────────
@dataclass
class StackConfig:
    material: str = "Si"
    wafer_diameter_mm: float = 300.0
    start_thickness_um: float = 775.0
    target_thickness_um: float = 50.0      # thinned die (HBM/3D-class)
    n_stack: int = 4                        # dies in the 3D stack
    die_size_mm: float = 8.0

    # backgrind
    grit_um: float = 2.0
    grind_feed_um_s: float = 50.0

    # stress drivers (stress×thickness tuned so a full-thickness wafer bows ~tens of µm)
    film_stress_MPa: float = 30.0           # BEOL/film residual stress
    film_thickness_um: float = 2.0          # effective stressed layer
    process_dT_C: float = 200.0             # thermal excursion seen by the stack

    # hybrid bonding
    bond_temp_C: float = 250.0
    bond_pressure_MPa: float = 10.0
    bond_time_s: float = 1.0
    surface_Ra_nm: float = 0.3
    overlay_budget_nm: float = 50.0         # advanced-node overlay spec
    bond_gap_tol_um: float = 1.0            # local-warp tolerance for void-free bond

    # singulation of the bonded stack
    dice_method: str = "laser"              # laser | blade
    street_width_um: float = 40.0

    def copy(self, **kw) -> "StackConfig":
        d = asdict(self); d.update(kw); return StackConfig(**d)


# ── propagated state ──────────────────────────────────────────────────────────
@dataclass
class StackState:
    stage: str
    thickness_um: float
    bow_um: float = 0.0                  # peak-to-valley warp
    residual_stress_MPa: float = 0.0
    ssd_um: float = 0.0                 # subsurface damage depth
    overlay_error_nm: float = 0.0
    void_risk: float = 0.0             # 0..1
    chipping_um: float = 0.0
    delam_risk: float = 0.0           # 0..1
    die_strength_MPa: float = 0.0
    extra: dict = field(default_factory=dict)

    def row(self) -> dict:
        d = asdict(self); d.pop("extra"); return d


# ── shared physics ────────────────────────────────────────────────────────────
def stoney_bow_um(sigma_MPa: float, t_film_um: float, t_sub_um: float,
                  D_mm: float, material: str) -> float:
    """Peak-to-valley wafer bow from Stoney's equation.

    κ = 6·σ·t_f / (M_s·t_s²),  bow = κ·D²/8,  M_s = E/(1−ν).
    """
    m = ALL_MATERIALS.get(material, ALL_MATERIALS["Si"])
    M_s = m["E"] / (1.0 - m["nu"])                 # Pa, biaxial modulus
    sigma = sigma_MPa * 1e6
    t_f = max(t_film_um, 1e-3) * 1e-6
    t_s = max(t_sub_um, 1e-3) * 1e-6
    kappa = 6.0 * sigma * t_f / (M_s * t_s ** 2)   # 1/m
    D = D_mm * 1e-3
    bow_m = kappa * D ** 2 / 8.0
    return bow_m * 1e6                              # µm


def _frac(material: str) -> dict:
    m = ALL_MATERIALS.get(material, ALL_MATERIALS["Si"])
    return {"E": m["E"] / 1e9, "H": m["H_v"] / 1e9, "KIc": m["K_Ic"] / 1e6}


# ── stage contract ────────────────────────────────────────────────────────────
class PackagingStage(ABC):
    name: str = "stage"

    @abstractmethod
    def step(self, cfg: StackConfig, s: StackState) -> StackState:
        raise NotImplementedError


# ── 1. backgrind / thinning ───────────────────────────────────────────────────
class ThinGrindStage(PackagingStage):
    name = "thin_grind"

    def step(self, cfg: StackConfig, s: StackState) -> StackState:
        t = cfg.target_thickness_um
        # subsurface damage ~ grit size, mild feed dependence (Pei 2008).
        ssd = 1.5 * cfg.grit_um * (cfg.grind_feed_um_s / 50.0) ** 0.3
        # grind leaves a compressive damaged layer → a modest stress source.
        grind_stress = 15.0 * (cfg.grit_um / 2.0) * (cfg.grind_feed_um_s / 50.0) ** 0.5
        return StackState(
            stage=self.name, thickness_um=t, ssd_um=ssd,
            residual_stress_MPa=grind_stress,
            extra={"removed_um": cfg.start_thickness_um - t})


# ── 2. warpage (Stoney) ───────────────────────────────────────────────────────
class WarpageStage(PackagingStage):
    name = "warpage"

    #: beyond ~2 mm bow a thinned wafer is un-handleable / fractures; Stoney
    #: (small-deflection) is invalid past here, so we cap and flag.
    BOW_FRACTURE_UM = 2000.0

    def step(self, cfg: StackConfig, s: StackState) -> StackState:
        m = ALL_MATERIALS.get(cfg.material, ALL_MATERIALS["Si"])
        # total driving stress: film + grind damage + thermal mismatch (CTE·E·ΔT).
        thermal_stress = m["E"] * m["alpha_thermal"] * cfg.process_dT_C / 1e6  # MPa
        # warp-driving stress (film-like): drives Stoney bow.
        sigma_warp = cfg.film_stress_MPa + s.residual_stress_MPa + 0.3 * thermal_stress
        # bulk/interface residual that persists into bonding (smaller; the thin
        # film stress is largely surface and does not threaten the bond line).
        bulk_residual = s.residual_stress_MPa + 0.2 * thermal_stress
        bow_raw = stoney_bow_um(sigma_warp, cfg.film_thickness_um, s.thickness_um,
                                cfg.wafer_diameter_mm, cfg.material)
        cracks = bow_raw > self.BOW_FRACTURE_UM
        bow = min(bow_raw, self.BOW_FRACTURE_UM)
        return StackState(
            stage=self.name, thickness_um=s.thickness_um, ssd_um=s.ssd_um,
            residual_stress_MPa=bulk_residual, bow_um=bow,
            extra={"thermal_stress_MPa": thermal_stress, "sigma_warp_MPa": sigma_warp,
                   "bow_raw_um": bow_raw, "wafer_cracks": cracks})


# ── 3. hybrid bonding ─────────────────────────────────────────────────────────
class BondStage(PackagingStage):
    name = "bond"

    def step(self, cfg: StackConfig, s: StackState) -> StackState:
        # warp localised to a die footprint distorts the field → overlay error.
        die_frac = cfg.die_size_mm / cfg.wafer_diameter_mm
        local_bow_um = s.bow_um * die_frac ** 2
        overlay_nm = local_bow_um * 1e3 * 0.5          # warp run-out → in-plane error
        # void risk: if local warp exceeds the bonding gap tolerance.
        void = min(1.0, local_bow_um / max(cfg.bond_gap_tol_um, 1e-3))
        # achievable Cu-Cu bond strength (reuse hybrid_bonding_model).
        bs = bond_strength(cfg.bond_temp_C, cfg.bond_pressure_MPa,
                           cfg.bond_time_s, cfg.surface_Ra_nm)["bond_strength_MPa"]
        # stacking n dies accumulates CTE-mismatch residual stress on cooling.
        stack_stress = s.residual_stress_MPa * (1.0 + 0.10 * (cfg.n_stack - 1))
        return StackState(
            stage=self.name, thickness_um=s.thickness_um, ssd_um=s.ssd_um,
            bow_um=s.bow_um, residual_stress_MPa=stack_stress,
            overlay_error_nm=overlay_nm, void_risk=void,
            extra={"bond_strength_MPa": bs, "local_bow_um": local_bow_um})


# ── 4. singulate the bonded stack ─────────────────────────────────────────────
class SingulateStage(PackagingStage):
    name = "singulate"

    def step(self, cfg: StackConfig, s: StackState) -> StackState:
        fp = _frac(cfg.material)
        # dicing-induced subsurface crack (Lawn–Evans–Marshall), method-dependent.
        load_N = 0.02 if cfg.dice_method == "blade" else 0.005   # laser gentler
        crack_um = float(sfm.median_crack_um(fp, load_N))
        chipping = (0.6 if cfg.dice_method == "blade" else 0.2) * crack_um
        # die strength from governing flaw (Griffith), de-rated by residual stress.
        m = ALL_MATERIALS.get(cfg.material, ALL_MATERIALS["Si"])
        c = max(crack_um, 0.05) * 1e-6
        sigma_f = m["K_Ic"] / (1.12 * math.sqrt(math.pi * c)) / 1e6
        die_strength = min(sigma_f, m["sigma_flex"] / 1e6) - s.residual_stress_MPa
        # delamination at the bond: dicing stress + residual vs bond strength.
        bs = s.extra.get("bond_strength_MPa", 60.0)
        dice_stress = 12.0 if cfg.dice_method == "laser" else 25.0   # blade more violent
        delam = min(1.0, (s.residual_stress_MPa + dice_stress) / max(bs, 1e-3))
        return StackState(
            stage=self.name, thickness_um=s.thickness_um, ssd_um=s.ssd_um,
            bow_um=s.bow_um, residual_stress_MPa=s.residual_stress_MPa,
            overlay_error_nm=s.overlay_error_nm, void_risk=s.void_risk,
            chipping_um=chipping, delam_risk=delam,
            die_strength_MPa=max(die_strength, 0.0),
            extra={"crack_um": crack_um, "dice_stress_MPa": dice_stress})


# ── orchestrator ──────────────────────────────────────────────────────────────
STAGES = [ThinGrindStage, WarpageStage, BondStage, SingulateStage]


@dataclass
class LineResult:
    cfg: StackConfig
    states: List[StackState]
    yielded: bool
    violations: List[str]

    @property
    def final(self) -> StackState:
        return self.states[-1]

    def summary(self) -> dict:
        f = self.final
        return {
            "material": self.cfg.material,
            "target_um": self.cfg.target_thickness_um,
            "bow_um": round(f.bow_um, 1),
            "overlay_nm": round(f.overlay_error_nm, 1),
            "void_risk": round(f.void_risk, 3),
            "delam_risk": round(f.delam_risk, 3),
            "die_strength_MPa": round(f.die_strength_MPa, 1),
            "yield": self.yielded,
            "n_violations": len(self.violations),
        }


class PackagingLine:
    """Runs thin → warp → bond → singulate and checks stack yield."""

    def __init__(self, die_strength_min_MPa: float = 100.0,
                 void_risk_max: float = 0.5, delam_risk_max: float = 0.75):
        self.die_strength_min = die_strength_min_MPa
        self.void_risk_max = void_risk_max
        self.delam_risk_max = delam_risk_max
        self.stages = [S() for S in STAGES]

    def run(self, cfg: StackConfig) -> LineResult:
        states: List[StackState] = []
        s = StackState(stage="start", thickness_um=cfg.start_thickness_um)
        for stage in self.stages:
            s = stage.step(cfg, s)
            states.append(s)
        f = states[-1]

        viols: List[str] = []
        if f.overlay_error_nm > cfg.overlay_budget_nm:
            viols.append(f"overlay {f.overlay_error_nm:.0f} > budget "
                         f"{cfg.overlay_budget_nm:.0f} nm (warp-driven)")
        if f.void_risk > self.void_risk_max:
            viols.append(f"bond void risk {f.void_risk:.2f} > {self.void_risk_max}")
        if f.delam_risk > self.delam_risk_max:
            viols.append(f"interface delamination risk {f.delam_risk:.2f} "
                         f"> {self.delam_risk_max}")
        if f.die_strength_MPa < self.die_strength_min:
            viols.append(f"die strength {f.die_strength_MPa:.0f} < "
                         f"{self.die_strength_min:.0f} MPa")
        return LineResult(cfg=cfg, states=states,
                          yielded=(len(viols) == 0), violations=viols)
