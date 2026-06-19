"""
sic.hybrid.adapters — wrap existing physics solvers behind DicingStage.

NO existing solver is modified; each stage is a thin adapter that calls the
established standalone models and maps their outputs onto StageOutcome:

  LaserGrooveStage  → fem.laser_groove_thermal_2d.analytical_groove
  StealthStage      → fem.laser_groove_thermal_2d.stealth_dicing / bessel_beam_stealth
  BladeStage        → fem.dicing_heat_sim.DicingHeatSim  (+ fracture model)

Subsurface-crack / chipping estimates reuse semiconductor_fracture_methods
(Lawn–Evans–Marshall median crack, Bifano ductile–brittle depth).
"""
from __future__ import annotations

import math
from typing import Optional

from sic.hybrid.base import (
    Recipe, StageOutcome, DicingStage, scan_throughput_wph,
)

# Existing solvers (repo root is on sys.path via sic.hybrid.__init__)
from fem.laser_groove_thermal_2d import (
    analytical_groove, stealth_dicing, bessel_beam_stealth, LASER_REGIMES,
)
from data.materials.material_properties import ALL_MATERIALS
import semiconductor_fracture_methods as sfm


# ── fracture-property bridge ───────────────────────────────────────────────────
def _frac_props(material: str) -> dict:
    """Map data.materials entry → semiconductor_fracture_methods unit convention
    (E [GPa], H [GPa], KIc [MPa·m^0.5])."""
    m = ALL_MATERIALS.get(material, ALL_MATERIALS["SiC"])
    return {
        "E":   m["E"] / 1e9,
        "H":   m["H_v"] / 1e9,
        "KIc": m["K_Ic"] / 1e6,
    }


def _thermal_residual_MPa(material: str, t_peak_C: float, t_amb_C: float = 25.0,
                          relax: float = 0.25) -> float:
    """Crude thermal residual stress σ ≈ relax · E · α · ΔT, capped at flexural
    strength. `relax` lumps constraint factor + viscoelastic relaxation."""
    m = ALL_MATERIALS.get(material, ALL_MATERIALS["SiC"])
    dT = max(0.0, t_peak_C - t_amb_C)
    sigma = relax * m["E"] * m["alpha_thermal"] * dT          # Pa
    return min(sigma, m["sigma_flex"]) / 1e6                  # MPa


# ── Stage A: laser grooving ─────────────────────────────────────────────────────
class LaserGrooveStage(DicingStage):
    """Clears the BEOL low-k/metal/TEG stack from the street (advanced-node prep)."""

    name = "laser_groove"

    def simulate(self, recipe: Recipe,
                 upstream: Optional[StageOutcome] = None) -> StageOutcome:
        p = {
            "laser_power_W":   recipe.groove_power_W,
            "scan_speed_mm_s": recipe.groove_speed_mm_s,
            "pulse_freq_kHz":  recipe.groove_freq_kHz,
            "beam_radius_um":  recipe.groove_beam_radius_um,
            "absorptivity":    recipe.groove_absorptivity,
            "n_passes":        recipe.groove_passes,
            "pulse_regime":    recipe.groove_pulse_regime,
        }
        g = analytical_groove(p)

        depth = g["groove_depth_um"]
        width = g["groove_width_um"]
        # The groove itself is removed material; what threatens the adjacent low-k
        # is the heat penetrating BEYOND the ablated trench (excess HAZ).
        haz_excess = max(0.0, g["HAZ_depth_um"] - depth)

        # Chipping along an ablated groove is regime-dependent: cold (ps/fs)
        # ablation chips far less than thermal (ns). HAZ_factor encodes that.
        haz_factor = LASER_REGIMES.get(recipe.groove_pulse_regime,
                                       LASER_REGIMES["ns"])["HAZ_factor"]
        chipping = width * (0.04 + 0.12 * haz_factor)
        # Thermal peak proxy from excess HAZ → residual stress.
        t_peak = 25.0 + 1800.0 * min(1.0, haz_excess / 10.0)
        residual = _thermal_residual_MPa(recipe.material, t_peak)
        # Thermal micro-crack depth beyond the groove tracks the excess HAZ.
        subsurface = 0.5 * haz_excess

        return StageOutcome(
            stage="A", method=self.name,
            kerf_um=width,
            chipping_um=chipping,
            haz_um=haz_excess,
            subsurface_crack_um=subsurface,
            residual_stress_MPa=residual,
            lowk_cleared_um=depth,
            throughput_wph=scan_throughput_wph(
                recipe, recipe.groove_speed_mm_s, recipe.groove_passes),
            extra=g,
        )


# ── Stage B: stealth dicing ─────────────────────────────────────────────────────
class StealthStage(DicingStage):
    """Non-contact internal-modification singulation (kerfless, low stress)."""

    name = "stealth"

    def simulate(self, recipe: Recipe,
                 upstream: Optional[StageOutcome] = None) -> StageOutcome:
        p = {
            "laser_power_W":      recipe.stealth_power_W,
            "scan_speed_mm_s":    recipe.stealth_speed_mm_s,
            "pulse_freq_kHz":     recipe.stealth_freq_kHz,
            "focal_depth_um":     recipe.stealth_focal_depth_um,
            "numerical_aperture": recipe.stealth_NA,
            "n_layers":           recipe.stealth_layers,
            "n_passes":           1,
            "pulse_regime":       recipe.stealth_regime,
        }
        s = bessel_beam_stealth(p) if recipe.stealth_bessel else stealth_dicing(p)

        kerf = s["modified_layer_w_um"]
        chipping = s["surface_chipping_um"]
        haz = s["HAZ_depth_um"]
        # Modified layer is the controlled "crack source"; latent damage is the
        # part of the modified zone that does not fully cleave (~ small).
        subsurface = 0.2 * s["modified_layer_h_um"]
        # Non-contact, near-zero thermal load at the surface → low residual stress.
        residual = _thermal_residual_MPa(recipe.material, 25.0 + 50.0, relax=0.1)

        return StageOutcome(
            stage="B", method=("bessel_stealth" if recipe.stealth_bessel else self.name),
            kerf_um=kerf,
            chipping_um=chipping,
            haz_um=haz,
            subsurface_crack_um=subsurface,
            residual_stress_MPa=residual,
            throughput_wph=scan_throughput_wph(
                recipe, recipe.stealth_speed_mm_s, recipe.stealth_layers),
            extra=s,
        )


# ── Stage B: blade dicing ───────────────────────────────────────────────────────
class BladeStage(DicingStage):
    """Mechanical blade singulation — thermal via DicingHeatSim, chipping via FM."""

    name = "blade"

    def simulate(self, recipe: Recipe,
                 upstream: Optional[StageOutcome] = None) -> StageOutcome:
        # Thermal peak from the existing DISCO dicing-saw simulator (bounded cost:
        # local quasi-steady dwell on a small window — we only need peak temp).
        from fem.dicing_heat_sim import DicingHeatSim, _numpy_solve_with_source
        m = ALL_MATERIALS.get(recipe.material, ALL_MATERIALS["SiC"])
        alpha = m["k_thermal"] / (m["density"] * m["Cp"])
        rho_cp = m["density"] * m["Cp"]
        sim = DicingHeatSim(nx=48, ny=48, dx=20e-6, alpha=alpha, rho_cp=rho_cp)
        sim.set_ambient(25.0)
        src = sim.blade_source_field(x_m=24 * sim.dx, y_m=24 * sim.dx,
                                     Q=recipe.blade_Q_W_per_m2, sigma_y=24)
        # Dwell time the blade spends over a spot ≈ contact-width / feed, capped.
        dwell_s = (160e-6) / max(recipe.blade_feed_mm_s * 1e-3, 1e-6)
        n_steps = int(min(max(dwell_s / sim.dt, 50), 400))
        T = _numpy_solve_with_source(sim.T.copy(), src, alpha, rho_cp,
                                     sim.dx, sim.dt, n_steps)
        t_peak = float(T.max())

        # Fracture-driven chipping/subsurface (Lawn–Evans–Marshall).
        fp = _frac_props(recipe.material)
        # Effective indentation load ~ feed × blade engagement (heuristic, N).
        p_eff_N = 0.02 * (recipe.blade_feed_mm_s / 80.0) * (recipe.blade_W_um / 23.0)
        subsurface = float(sfm.median_crack_um(fp, max(p_eff_N, 1e-4)))
        # Front-side chipping ≈ lateral-crack fraction of the median crack,
        # amplified by the material chipping index relative to SiC.
        chip_idx_rel = sfm.chipping_index(fp) / sfm.chipping_index(_frac_props("SiC"))
        chipping = 0.6 * subsurface * math.sqrt(chip_idx_rel)

        residual = _thermal_residual_MPa(recipe.material, t_peak)

        return StageOutcome(
            stage="B", method=self.name,
            kerf_um=recipe.blade_W_um,
            chipping_um=chipping,
            haz_um=0.0,
            subsurface_crack_um=subsurface,
            residual_stress_MPa=residual,
            throughput_wph=scan_throughput_wph(
                recipe, recipe.blade_feed_mm_s, 1),
            extra={"t_peak_C": t_peak, "p_eff_N": p_eff_N,
                   "chip_idx_rel": chip_idx_rel,
                   "gpu": DicingHeatSim.gpu_available()},
        )
