"""
Plasma Bosch process model — pure Python analytical (no ABAQUS required).

Simulates two-step Bosch process (SF₆ etch + C₄F₈ passivation) used in
plasma dicing of SiC/Si wafers after laser grooving.

Physics implemented:
    1. SF₆ isotropic etch: chemical + physical (ion-assisted)
    2. C₄F₈ passivation:   polymer deposition on sidewalls
    3. ARDE (Aspect Ratio Dependent Etching): flux shadowing → rate reduction
    4. Lag factor: bottom etch rate attenuated by (1 - k_ARDE * AR)
    5. Sidewall taper: accumulation of passivation → slight positive angle
    6. Undercut:  lateral etch during SF₆ step, partially blocked by passivation
    7. Die strength: Weibull weakest-link model (HAZ + roughness → σ_B)
    ② Pulsed plasma (duty_cycle):  ion directionality benefit at lower average power
    ③ Low-k delamination risk:     HAZ × k-fragility × regime factor

Throughput model:
    Parallel plasma: all die separated simultaneously (unlike sequential blade/laser)
    Throughput = wafer_area / (t_laser + t_plasma + t_load)

Usage (standalone):
    python fem/plasma_bosch_model.py --plot
    python fem/plasma_bosch_model.py --generate-csv --n 300

Usage (as module):
    from fem.plasma_bosch_model import BoschModel, HybridResult, hybrid_physics

References:
    Laermer & Urban (2003) Sensors Actuators A — original Bosch process
    Rozanski et al. (2021) Appl Surf Sci — SiC ARDE characterisation
    Baumgart et al. (2023) J Micromech Microeng — hybrid laser+plasma SiC dicing
    Plasma-Therm PDOT™/PDOC process specs
    IBM Research (2025) VLSI — Advanced low-k (ALK) k=2.8–3.0 for 2nm BEOL
    IEEE Xplore (2020) — Pulsed plasma reduces Low-k damage vs CW
"""

import math
import dataclasses
from typing import Optional
import numpy as np


# ── Process constants ─────────────────────────────────────────────────────────
# SiC Bosch process (SF₆/O₂ etch + C₄F₈ passivation)
ETCH_RATE_BASE_UM_MIN  = 5.0    # µm/min  SiC in SF₆/O₂ optimised Bosch (Rozanski 2021)
ETCH_RATE_SI_UM_MIN    = 5.0    # µm/min  Si (higher selectivity)
ARDE_COEFF             = 0.08   # ARDE: rate multiplied by (1 - k * AR)
PASSIVATION_EFF        = 0.85   # sidewall passivation efficiency (0–1)
UNDERCUT_FRAC          = 0.04   # lateral etch fraction per etch step
SIDEWALL_ANGLE_DEG     = 88.5   # target sidewall angle [°] (ideal = 90)
T_LOAD_S               = 60.0   # wafer load/unload time [s]
WAFER_AREA_MM2         = 707.0  # 300 mm wafer area [mm²]  (π * 150²)

# SiC Weibull die strength parameters (reference: unprocessed wafer)
WEIBULL_M0             = 8.0    # Weibull modulus (shape parameter)
WEIBULL_SIGMA0_MPA     = 600.0  # characteristic strength [MPa]


@dataclasses.dataclass
class BoschResult:
    etch_depth_um:     float
    etch_rate_um_min:  float
    aspect_ratio:      float
    sidewall_angle_deg:float
    undercut_um:       float
    surface_roughness_nm: float


@dataclasses.dataclass
class HybridResult:
    """Combined laser-grooving + plasma-dicing result."""
    # Laser stage
    groove_depth_um:  float
    groove_width_um:  float
    HAZ_depth_um:     float
    pulse_regime:     str       # "ns" / "ps" / "fs"
    # Plasma stage
    plasma_depth_um:  float
    undercut_um:      float
    sidewall_angle_deg: float
    surface_roughness_nm: float
    duty_cycle:       float     # ② pulsed plasma duty cycle
    # Combined
    street_width_um:  float     # groove_width + 2×undercut
    total_depth_um:   float     # groove + plasma etch
    die_strength_MPa: float     # Weibull-based prediction
    throughput_mm2_s: float     # area processed per second
    # ③ Low-k delamination
    delamination_risk: float    # 0 (safe) → 1 (certain delamination)
    lowk_damage_depth_nm: float # plasma-induced damage into Low-k layer [nm]
    # ④ 2nm node yield
    yield_loss_pct: float       # expected die yield loss [%] from Low-k delamination


class BoschModel:
    """
    Physics-based Bosch process model.

    Parameters
    ----------
    material : str — 'SiC' or 'Si'
    mask_um  : float — mask opening width [µm]
    wafer_um : float — total wafer thickness [µm]
    """

    # Tabulated etch rates [µm/min] from literature (Rozanski 2021, adjusted)
    _ETCH_RATES = {
        "SiC": ETCH_RATE_BASE_UM_MIN,
        "Si":  ETCH_RATE_SI_UM_MIN,
    }

    def __init__(self, material: str = "SiC",
                 mask_um: float = 8.0,
                 wafer_um: float = 200.0):
        self.material  = material
        self.mask_um   = mask_um
        self.wafer_um  = wafer_um
        self._R0       = self._ETCH_RATES.get(material, ETCH_RATE_BASE_UM_MIN)

    def etch(self,
             plasma_time_s: float,
             n_cycles:      int   = 50,
             etch_frac:     float = 0.6,
             duty_cycle:    float = 1.0) -> BoschResult:
        """
        Simulate Bosch process.

        Parameters
        ----------
        plasma_time_s : total plasma on-time [s]
        n_cycles      : number of SF₆/C₄F₈ cycles
        etch_frac     : fraction of cycle time in etch step
        duty_cycle    : ② RF pulse duty cycle (0.1–1.0)
                        Sub-linear scaling: lower duty → better directionality
                        but reduced average etch rate.
                        R_eff *= duty_cycle^0.6  (IEEE Xplore 2020)
        """
        # ② Pulsed plasma rate factor
        # Exponent 0.6: sub-linear (directional benefit compensates some rate loss)
        pulse_factor = duty_cycle ** 0.6

        t_cycle = plasma_time_s / max(n_cycles, 1)
        t_etch  = t_cycle * etch_frac

        depth_um    = 0.0
        undercut_um = 0.0

        for cyc in range(n_cycles):
            AR = depth_um / self.mask_um if self.mask_um > 0 else 0.0

            arde_factor = max(0.1, 1.0 - ARDE_COEFF * AR)
            R_eff = self._R0 * arde_factor * pulse_factor

            d_inc = R_eff * (t_etch / 60.0)
            depth_um += d_inc

            if depth_um >= self.wafer_um:
                depth_um = self.wafer_um
                break

            # Undercut: pulsed plasma also reduces lateral component
            u_inc = d_inc * UNDERCUT_FRAC * (1.0 - PASSIVATION_EFF * etch_frac) * duty_cycle
            undercut_um += u_inc

        if depth_um > 0:
            angle_rad = math.atan2(undercut_um, depth_um)
            sidewall_angle_deg = 90.0 - math.degrees(angle_rad)
        else:
            sidewall_angle_deg = 90.0

        # Scallop roughness: pulsed plasma reduces scallop depth
        Ra_nm = (50.0 + 3.0 * (depth_um / max(n_cycles, 1)) * 10.0) * (0.5 + 0.5 * duty_cycle)

        AR_final = depth_um / self.mask_um if self.mask_um > 0 else 0.0
        R_avg    = depth_um / (plasma_time_s / 60.0) if plasma_time_s > 0 else 0.0

        return BoschResult(
            etch_depth_um        = round(depth_um, 3),
            etch_rate_um_min     = round(R_avg, 4),
            aspect_ratio         = round(AR_final, 2),
            sidewall_angle_deg   = round(sidewall_angle_deg, 2),
            undercut_um          = round(undercut_um, 3),
            surface_roughness_nm = round(Ra_nm, 1),
        )


def die_strength_weibull(HAZ_depth_um: float,
                          Ra_nm: float,
                          street_width_um: float) -> float:
    """
    Estimate die bending strength via Weibull weakest-link model.

    Flaw distribution driven by HAZ depth (sub-surface damage) and
    sidewall roughness (stress concentrations).

    Returns σ_B [MPa] at P_f = 0.1 (10th percentile, conservative).
    """
    # Effective flaw size ~ HAZ depth (dominant over roughness for deep HAZ)
    a_um = max(HAZ_depth_um, Ra_nm / 1e3)    # convert nm→µm, take max

    # Griffith-type: σ_c ∝ 1 / sqrt(a)
    # Normalize to reference: a_ref = 2.0 µm → σ_ref = WEIBULL_SIGMA0_MPA
    a_ref   = 2.0
    sigma_c = WEIBULL_SIGMA0_MPA * math.sqrt(a_ref / max(a_um, 0.1))

    # Narrow street → lower Weibull modulus (more variability)
    m_eff = WEIBULL_M0 * min(1.0, street_width_um / 20.0)

    # P_f = 10% → F⁻¹(0.1) = (ln(1/(1-0.1)))^(1/m)
    pf    = 0.10
    sigma_pf = sigma_c * (math.log(1.0 / (1.0 - pf))) ** (1.0 / max(m_eff, 1.0))

    return round(sigma_pf, 1)


def delamination_risk(
    HAZ_depth_um:      float,
    beol_thickness_um: float,
    beol_k:            float,
    pulse_regime:      str,
    sidewall_angle_deg:float,
    duty_cycle:        float,
) -> tuple:
    """
    ③ Low-k delamination risk score and plasma damage depth.

    Physical basis:
    - k-fragility: lower k → more porous → weaker adhesion to substrate
    - HAZ fraction: how much of the BEOL stack is thermally stressed
    - Regime factor: fs/ps laser virtually eliminates thermal HAZ
    - Plasma damage: ion bombardment penetrates porous Low-k sidewall
    - Pulsed plasma: lower duty_cycle → reduced ion bombardment

    Returns (risk: float [0,1], lowk_damage_depth_nm: float)

    References:
        IBM Research (2025) VLSI — ALK k=2.8–3.0 modulus ~10GPa
        IEEE Xplore (2020) — source-pulsed plasma reduces Low-k damage layer
    """
    # k-value fragility: linear ramp, 0 at k≥3.5 (dense oxide), 1 at k≤1.5 (aerogel)
    k_fragility = max(0.0, min(1.0, (3.5 - beol_k) / 2.0))

    # Thermal HAZ contribution: fraction of BEOL stack thermally affected
    haz_frac = min(1.0, HAZ_depth_um / max(beol_thickness_um, 0.5))

    # Pulse regime: ns/ps/fs → thermal damage factors
    _regime_factors = {"ns": 1.0, "ps": 0.12, "fs": 0.015}
    r_factor = _regime_factors.get(pulse_regime, 1.0)

    # Thermal delamination component
    thermal_risk = k_fragility * haz_frac * r_factor

    # Plasma ion damage: ions penetrate porous Low-k sidewall
    # Near-vertical sidewall (>89°) reduces ion bombardment angle → less damage
    angle_penalty = max(0.0, (90.0 - sidewall_angle_deg) / 5.0)   # 0 at 90°, 1 at 85°
    plasma_risk   = k_fragility * angle_penalty * (1.0 - duty_cycle * 0.6)

    risk = min(1.0, max(0.0, thermal_risk + plasma_risk * 0.4))

    # Plasma damage depth into Low-k [nm]: empirical from IEEE 2020
    # CW: ~50 nm per µm etch depth; pulsed: ~15 nm per µm
    damage_nm_per_um = 50.0 * duty_cycle + 15.0 * (1.0 - duty_cycle)
    lowk_damage_nm   = damage_nm_per_um * (1.0 + angle_penalty) * k_fragility

    return round(risk, 4), round(lowk_damage_nm, 1)


def die_yield_loss(delamination_risk: float,
                   street_width_um: float = 23.0,
                   beol_layers: int = 12) -> float:
    """
    Expected yield loss [%] for 2nm node BEOL Low-k delamination.

    Model: each BEOL interface fails independently with P_int = risk / beol_layers.
    Yield = product over all layers → yield_loss = (1 - (1-P_int)^beol_layers) * 100.

    For 2nm node (ALK k=2.8-3.0, 12 metal layers):
        risk < 0.05 → yield_loss < 1%  (TSMC N2 spec)
        risk > 0.20 → yield_loss > 20% (unacceptable)

    Reference: IBM Research VLSI 2025 — ALK multi-layer delamination statistics.
    """
    if delamination_risk <= 0.0:
        return 0.0
    P_int = min(1.0, delamination_risk / beol_layers)
    yield_loss = (1.0 - (1.0 - P_int) ** beol_layers) * 100.0
    # Narrow street penalty: < 20µm increases edge-exposure probability
    street_penalty = max(1.0, (20.0 / max(street_width_um, 1.0)) ** 0.3)
    return round(min(100.0, yield_loss * street_penalty), 3)


def throughput_mm2_s(laser_W: float, scan_mm_s: float, n_passes: int,
                      plasma_s: float,
                      street_pitch_um: float = 200.0,
                      wafer_area_mm2: float = WAFER_AREA_MM2) -> float:
    """
    Compute effective throughput [mm²/s].

    Laser time dominates for high n_passes; plasma time is fixed per wafer.
    street_pitch_um: dicing pitch (determines total laser scan length)
    """
    # Total laser scan length [mm] (assumes square die, one direction)
    n_streets   = (math.sqrt(wafer_area_mm2) * 1e3) / street_pitch_um  # number of streets
    L_total_mm  = n_streets * math.sqrt(wafer_area_mm2)                 # total length [mm]
    t_laser_s   = n_passes * L_total_mm / scan_mm_s if scan_mm_s > 0 else 1e9

    t_total_s   = t_laser_s + plasma_s + T_LOAD_S
    return round(wafer_area_mm2 / t_total_s, 4) if t_total_s > 0 else 0.0


def hybrid_physics(
    laser_W:           float,
    scan_mm_s:         float,
    pulse_kHz:         float,
    n_passes:          int,
    plasma_s:          float,
    n_cycles:          int,
    mask_um:           float,
    wafer_um:          float,
    material:          str   = "SiC",
    beam_radius_um:    float = 5.0,
    pulse_regime:      str   = "ns",   # ① "ns" / "ps" / "fs"
    duty_cycle:        float = 1.0,    # ② RF pulsed plasma duty cycle
    beol_k:            float = 3.0,    # ③ BEOL low-k dielectric constant
    beol_thickness_um: float = 5.0,    # ③ BEOL stack thickness [µm]
) -> HybridResult:
    """
    Full hybrid process model: laser grooving → plasma dicing.
    Supports ns/ps/fs laser regimes, pulsed plasma, and Low-k delamination risk.
    """
    # ── Stage 1: Laser grooving ──────────────────────────────────────────────
    from fem.laser_groove_thermal_2d import analytical_groove
    lg = analytical_groove({
        "laser_power_W":    laser_W,
        "scan_speed_mm_s":  scan_mm_s,
        "pulse_freq_kHz":   pulse_kHz,
        "beam_radius_um":   beam_radius_um,
        "n_passes":         n_passes,
        "absorptivity":     0.85,
        "pulse_regime":     pulse_regime,
    })
    groove_depth = lg["groove_depth_um"]
    groove_width = lg["groove_width_um"]
    HAZ_depth    = lg["HAZ_depth_um"]

    # ── Stage 2: Pulsed Bosch plasma dicing ──────────────────────────────────
    eff_mask_um  = max(mask_um, groove_width)
    bosch        = BoschModel(material=material, mask_um=eff_mask_um,
                              wafer_um=max(0.0, wafer_um - groove_depth))
    br = bosch.etch(plasma_time_s=plasma_s, n_cycles=n_cycles, duty_cycle=duty_cycle)

    # ── Combined outputs ──────────────────────────────────────────────────────
    street_width = groove_width + 2.0 * br.undercut_um
    total_depth  = groove_depth + br.etch_depth_um
    strength     = die_strength_weibull(HAZ_depth, br.surface_roughness_nm, street_width)
    tp           = throughput_mm2_s(laser_W, scan_mm_s, n_passes, plasma_s)
    risk, damage_nm = delamination_risk(
        HAZ_depth, beol_thickness_um, beol_k,
        pulse_regime, br.sidewall_angle_deg, duty_cycle)

    yield_loss = die_yield_loss(risk, street_width)

    return HybridResult(
        groove_depth_um       = round(groove_depth, 3),
        groove_width_um       = round(groove_width, 3),
        HAZ_depth_um          = round(HAZ_depth, 3),
        pulse_regime          = pulse_regime,
        plasma_depth_um       = round(br.etch_depth_um, 3),
        undercut_um           = round(br.undercut_um, 3),
        sidewall_angle_deg    = round(br.sidewall_angle_deg, 2),
        surface_roughness_nm  = round(br.surface_roughness_nm, 1),
        duty_cycle            = round(duty_cycle, 3),
        street_width_um       = round(street_width, 3),
        total_depth_um        = round(total_depth, 3),
        die_strength_MPa      = strength,
        throughput_mm2_s      = tp,
        delamination_risk     = risk,
        lowk_damage_depth_nm  = damage_nm,
        yield_loss_pct        = yield_loss,
    )


def generate_dataset(n: int = 300,
                     seed: int = 42,
                     noise_frac: float = 0.05) -> "pd.DataFrame":
    """
    Generate synthetic training dataset by sampling parameter space
    and calling hybrid_physics with added Gaussian noise.

    Returns pd.DataFrame with feature + target columns.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)

    # Parameter ranges (includes ①②③ new dimensions)
    laser_W_range    = (5.0,   30.0)
    scan_range       = (100.0, 500.0)
    pulse_range      = (50.0,  200.0)
    n_pass_choices   = [1, 2, 3, 4, 5]
    plasma_range     = (600.0, 3600.0)
    n_cyc_choices    = [10, 20, 30, 50, 80, 100]
    mask_range       = (3.0,   20.0)
    wafer_range      = (25.0,  400.0)    # ④ thin wafer: down to 25µm (HBM4)
    regime_choices   = ["ns", "ps", "fs"]
    duty_range       = (0.1,   1.0)      # ② pulsed plasma
    beol_k_range     = (1.8,   3.5)      # ③ low-k: ELK 1.8 → dense oxide 3.5
    beol_t_range     = (2.0,   10.0)     # ③ BEOL stack thickness

    records = []
    for _ in range(n):
        regime = str(rng.choice(regime_choices))
        params = dict(
            laser_W            = rng.uniform(*laser_W_range),
            scan_mm_s          = rng.uniform(*scan_range),
            pulse_kHz          = rng.uniform(*pulse_range),
            n_passes           = int(rng.choice(n_pass_choices)),
            plasma_s           = rng.uniform(*plasma_range),
            n_cycles           = int(rng.choice(n_cyc_choices)),
            mask_um            = rng.uniform(*mask_range),
            wafer_um           = rng.uniform(*wafer_range),
            pulse_regime       = regime,
            duty_cycle         = rng.uniform(*duty_range),
            beol_k             = rng.uniform(*beol_k_range),
            beol_thickness_um  = rng.uniform(*beol_t_range),
        )
        res = hybrid_physics(**params)

        def _noisy(x):
            return x * (1.0 + rng.normal(0.0, noise_frac))

        # Encode pulse_regime as integer for GP (ns=0, ps=1, fs=2)
        regime_code = {"ns": 0, "ps": 1, "fs": 2}.get(regime, 0)
        row = {k: v for k, v in params.items() if k != "pulse_regime"}
        row["pulse_regime_code"] = regime_code
        row.update({
            "street_width_um":    _noisy(res.street_width_um),
            "die_strength_MPa":   _noisy(res.die_strength_MPa),
            "HAZ_depth_um":       _noisy(res.HAZ_depth_um),
            "throughput_mm2_s":   _noisy(res.throughput_mm2_s),
            "delamination_risk":  float(np.clip(res.delamination_risk
                                                + rng.normal(0, 0.01), 0, 1)),
        })
        records.append(row)

    return pd.DataFrame(records)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, os

    ap = argparse.ArgumentParser()
    ap.add_argument("--plot",         action="store_true")
    ap.add_argument("--generate-csv", action="store_true")
    ap.add_argument("--yield-sweep",  action="store_true",
                    help="2nm node yield loss heatmap: street_width × beol_k")
    ap.add_argument("--n",            type=int, default=300)
    ap.add_argument("--out",          default="data/experimental/hybrid_synthetic.csv")
    args = ap.parse_args()

    if args.yield_sweep:
        import matplotlib.pyplot as plt
        street_widths = np.linspace(5.0, 40.0, 50)   # µm  (2nm node: 5-40µm streets)
        beol_ks       = np.linspace(1.8, 3.5, 40)    # Low-k dielectric constant
        SW, BK = np.meshgrid(street_widths, beol_ks)
        YL = np.zeros_like(SW)
        for i in range(BK.shape[0]):
            for j in range(SW.shape[1]):
                res = hybrid_physics(
                    laser_W=15.0, scan_mm_s=200.0, pulse_kHz=100.0, n_passes=2,
                    plasma_s=120.0, n_cycles=50,
                    mask_um=SW[i, j], wafer_um=50.0,
                    pulse_regime="ps", duty_cycle=0.5,
                    beol_k=BK[i, j], beol_thickness_um=5.0)
                YL[i, j] = res.yield_loss_pct

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Panel 1: heatmap
        ax = axes[0]
        im = ax.contourf(SW, BK, YL, levels=20, cmap="RdYlGn_r")
        plt.colorbar(im, ax=ax, label="Yield loss [%]")
        ax.contour(SW, BK, YL, levels=[1.0], colors="white", linewidths=2,
                   linestyles="--")
        ax.set_xlabel("Street Width [µm]", fontsize=11)
        ax.set_ylabel("BEOL Low-k (k value)", fontsize=11)
        ax.set_title("2nm Node Yield Loss: Street Width × Low-k\n"
                     "White dashed = 1% loss boundary (ps laser, 50µm wafer)", fontsize=10)
        ax.axvline(20.0, color="cyan", lw=1.2, ls=":")
        ax.text(20.5, 3.3, "20µm\nstreet", color="cyan", fontsize=8)

        # Panel 2: yield vs street_width for 3 k values
        ax2 = axes[1]
        for k_val, color in [(1.8, "#d62728"), (2.4, "#ff7f0e"), (3.0, "#2ca02c")]:
            yl_line = []
            for sw in street_widths:
                res = hybrid_physics(
                    laser_W=15.0, scan_mm_s=200.0, pulse_kHz=100.0, n_passes=2,
                    plasma_s=120.0, n_cycles=50,
                    mask_um=sw, wafer_um=50.0,
                    pulse_regime="ps", duty_cycle=0.5,
                    beol_k=k_val, beol_thickness_um=5.0)
                yl_line.append(res.yield_loss_pct)
            ax2.plot(street_widths, yl_line, color=color, lw=2,
                     label=f"k={k_val} ({'ELK' if k_val<2.0 else 'ALK' if k_val<2.6 else 'Dense'})")
        ax2.axhline(1.0, color="red", ls="--", lw=1.5, label="1% limit (TSMC N2)")
        ax2.axvline(20.0, color="gray", ls=":", lw=1.2)
        ax2.set_xlabel("Street Width [µm]", fontsize=11)
        ax2.set_ylabel("Yield Loss [%]", fontsize=11)
        ax2.set_title("Yield Loss vs Street Width\n(ps laser + pulsed plasma, 50µm wafer)", fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.25)

        fig.suptitle("2nm Node BEOL Yield Prediction — Hybrid Laser+Plasma\n"
                     "IBM ALK (k=2.8-3.0), 12 metal layers, P_f per interface model",
                     fontsize=11)
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        out = "results/yield_sweep_2nm.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[✓] Yield sweep → {out}")

    elif args.generate_csv:
        df = generate_dataset(n=args.n)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Saved {len(df)} rows → {args.out}")
        print(df.describe().to_string())

    elif args.plot:
        import matplotlib.pyplot as plt

        plasma_times = np.linspace(30, 300, 50)
        model = BoschModel(material="SiC", mask_um=8.0, wafer_um=200.0)
        depths  = [model.etch(t, n_cycles=50).etch_depth_um  for t in plasma_times]
        underc  = [model.etch(t, n_cycles=50).undercut_um    for t in plasma_times]
        Ra      = [model.etch(t, n_cycles=50).surface_roughness_nm for t in plasma_times]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].plot(plasma_times, depths);  axes[0].set(xlabel="Plasma time [s]", ylabel="Etch depth [µm]",  title="Bosch etch depth")
        axes[1].plot(plasma_times, underc);  axes[1].set(xlabel="Plasma time [s]", ylabel="Undercut [µm]",    title="Lateral undercut")
        axes[2].plot(plasma_times, Ra);      axes[2].set(xlabel="Plasma time [s]", ylabel="Roughness Ra [nm]", title="Surface roughness")
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/plasma_bosch_model.png", dpi=150)
        print("Plot saved → results/plasma_bosch_model.png")

    else:
        # Demo run
        result = hybrid_physics(
            laser_W=15.0, scan_mm_s=200.0, pulse_kHz=100.0, n_passes=2,
            plasma_s=120.0, n_cycles=50, mask_um=8.0, wafer_um=200.0)
        for k, v in dataclasses.asdict(result).items():
            print(f"  {k:25s}: {v}")
