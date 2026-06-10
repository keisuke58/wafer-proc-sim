"""
Keyence LJ-X8080 Laser Line Profiler — SiC Wafer Dicing Metrology
==================================================================
Models the Keyence LJ-X8080 2D laser line profiler (laser triangulation)
for inline SiC wafer dicing metrology.  This is a completely different
measurement principle from the VK-X confocal microscope (keyence_metrology_model.py):

  LJ-X8080 (this module) — laser triangulation
  -----------------------------------------------
  • 405 nm line laser projected at ~30° triangulation angle
  • CMOS area sensor images the reflected/scattered laser stripe
  • z-resolution  : 0.5 µm  (limited by pixel projection geometry)
  • lateral (Y)   : 5 µm    (along the laser line, 24 mm FOV)
  • scan rate     : 64 000 profiles/s  → suitable for INLINE dicing
  • Field of view : 24 mm × 3 mm      → full kerf + die surface in one pass

  VK-X3100 (sibling module) — confocal microscopy
  ------------------------------------------------
  • Same 405 nm laser, pinhole optics, piezo z-scan
  • z-resolution : 0.22 µm, lateral: 0.26 µm, throughput: 12 wph offline

Primary applications modelled here:
  1.  Kerf width & depth measurement  (inline, blade cut groove 40–200 µm)
  2.  Wafer surface warp / bow before dicing  (150 mm 4H-SiC: 20–80 µm bow)
  3.  Blade wear estimation from kerf depth trend over many wafers
  4.  Scan-speed vs. measurement noise trade-off

Physics implemented:
  triangulation_z_resolution  — theoretical δz from geometry
  kerf_profile_scan           — simulate a single kerf cross-section measurement
  wafer_warp_map              — 2-D bow map (Zernike-like paraboloid + noise)
  blade_wear_from_kerf        — fit kerf-width trend → wear rate [µm/wafer]
  sigma_vs_scan_speed         — σ_kerf degradation at high scan speeds

References:
  [1] Keyence LJ-X8080 Specification Sheet (Keyence Corp., 2024).
      https://www.keyence.com/products/measure/laser-2d/lj-x8000/
  [2] Gåsvik K.J. (2002) Optical Metrology, 3rd ed., Wiley. §11 — triangulation.
  [3] Wang Y. et al. (2026) Micromachines 17(2):187 — SiC dicing dataset.

Usage:
    python fem/keyence_lj_profiler.py          # run + plot
    python fem/keyence_lj_profiler.py --no-plot
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import scipy.ndimage as ndi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow, Arc
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Output directory (same convention as sibling modules)
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ============================================================================
# 0.  Instrument specification (LJ-X8080 public datasheet, 2024)
# ============================================================================

LJX_SPEC = {
    # Optical / illumination
    "wavelength_nm":           405.0,   # violet line laser
    "triangulation_angle_deg":  30.0,   # nominal projection angle [°]
    "standoff_mm":              35.0,   # reference standoff distance [mm]
    "pixel_pitch_um":            3.45,  # CMOS pixel pitch [µm]
    "magnification":             0.5,   # optical magnification onto CMOS
    # Measurement resolution
    "z_resolution_um":           0.5,   # height resolution [µm]
    "lateral_y_um":              5.0,   # lateral resolution along laser line [µm]
    "scan_pitch_min_um":         0.5,   # minimum scan pitch in X direction [µm]
    # Field of view
    "fov_y_mm":                 24.0,   # field width along laser line [mm]
    "fov_z_mm":                  3.0,   # measurement depth range [mm]
    # Speed
    "max_scan_rate_hz":      64_000,    # profiles per second
    # Noise (empirical, SiC surface)
    "sigma_kerf_um":             0.8,   # 1-σ kerf-width repeatability [µm]
    "sigma_z_nm":              500.0,   # surface height noise (=0.5 µm spec)
    # Kerf geometry (4H-SiC blade dicing, typical)
    "kerf_width_min_um":        40.0,
    "kerf_width_max_um":       200.0,
    "kerf_depth_typical_um":   350.0,   # typical blade dicing depth
}

# ============================================================================
# 1.  Triangulation geometry → theoretical z-resolution
# ============================================================================

def triangulation_z_resolution(
    standoff_mm: float = 35.0,
    angle_deg: float = 30.0,
    pixel_pitch_um: float = 3.45,
    magnification: float = 0.5,
    subpixel_factor: float = 27.6,
) -> float:
    """Return theoretical z-resolution (µm) for a laser triangulation sensor.

    The CMOS images the projected laser stripe at angle θ.  A surface height
    change δz displaces the stripe centroid by

        Δy_sensor = M · δz · sin(θ)

    on the sensor plane.  With pure single-pixel resolution:

        δz_pixel = pixel_pitch / (M · sin(θ))

    For 3.45 µm pixel, M = 0.5×, θ = 30° this gives ~13.8 µm — far coarser
    than the 0.5 µm LJ-X8080 specification.  The sensor achieves 0.5 µm by
    Gaussian centroid fitting of the bright laser stripe across many pixels
    (sub-pixel interpolation), which improves effective resolution by a factor
    `subpixel_factor` ≈ 27.6 (= 13.8 / 0.5):

        δz = pixel_pitch / (M · sin(θ) · subpixel_factor)

    The default `subpixel_factor` is calibrated so that the nominal LJ-X8080
    inputs (standoff=35 mm, angle=30°, p=3.45 µm, M=0.5×) return exactly
    0.5 µm, matching the public datasheet.

    Parameters
    ----------
    standoff_mm : float
        Nominal working distance [mm].  Documented for context; the formula
        is standoff-independent for a telecentric design.
    angle_deg : float
        Triangulation angle between laser beam and sensor optical axis [°].
    pixel_pitch_um : float
        CMOS pixel pitch [µm].
    magnification : float
        Optical magnification onto the sensor (dimensionless).
    subpixel_factor : float
        Sub-pixel centroid interpolation gain.  Default 27.6 reproduces the
        LJ-X8080 datasheet value of 0.5 µm.

    Returns
    -------
    float
        Effective z-resolution δz [µm].

    References
    ----------
    Gåsvik (2002) §11.2, eq. 11.4 (single-pixel limit).
    Keyence LJ-X8080 Spec Sheet (2024) — 0.5 µm height resolution.
    """
    theta = math.radians(angle_deg)
    dz_pixel = pixel_pitch_um / (magnification * math.sin(theta))
    dz = dz_pixel / subpixel_factor
    return dz


# ============================================================================
# 2.  KerfProfile dataclass + scan simulation
# ============================================================================

@dataclass
class KerfProfile:
    """Simulated LJ-X8080 measurement of a single dicing kerf cross-section.

    Attributes
    ----------
    y_um : ndarray
        Lateral positions along the laser line [µm], shape (N,).
    z_um : ndarray
        Measured surface height [µm], shape (N).  Zero = nominal wafer surface.
        Negative values = inside the kerf (groove).
    kerf_width_measured_um : float
        Extracted kerf width from the profile [µm].
    kerf_depth_measured_um : float
        Extracted kerf depth (max depression) [µm].
    sidewall_angle_deg : float
        Average sidewall angle from vertical [°].
    wafer_index : int
        Sequential wafer number (used for wear trend analysis).
    true_kerf_um : float
        Ground-truth kerf width used when generating this profile [µm].
    Ra_nm : float
        Surface roughness Ra of kerf sidewalls [nm].
    """
    y_um: np.ndarray
    z_um: np.ndarray
    kerf_width_measured_um: float
    kerf_depth_measured_um: float
    sidewall_angle_deg: float
    wafer_index: int
    true_kerf_um: float
    Ra_nm: float


def kerf_profile_scan(
    true_kerf_um: float = 100.0,
    Ra_nm: float = 80.0,
    n_profiles: int = 32,
    seed: Optional[int] = 42,
    wafer_index: int = 0,
    depth_um: float = 350.0,
    sidewall_deg: float = 2.5,
) -> KerfProfile:
    """Simulate an LJ-X8080 kerf cross-section measurement.

    The sensor averages `n_profiles` consecutive line scans to reduce noise,
    which is the typical operating mode for kerf metrology.

    Physical model
    --------------
    * Kerf shape: flat-bottom trapezoidal groove with sidewall angle
      `sidewall_deg` (positive = walls slant outward → wider at top).
    * Surface roughness Ra on kerf floor and sidewalls modelled as
      band-limited Gaussian noise filtered to lateral correlation length
      ~15 µm (abrasive grain size).
    * Sensor noise: σ_z = 0.5 µm (triangulation z-resolution), reduced by
      √n_profiles averaging.
    * Kerf-width extraction: threshold at z = -depth_um/2; distance between
      first/last crossing gives measured width.

    Parameters
    ----------
    true_kerf_um : float
        True kerf width at the wafer surface [µm].
    Ra_nm : float
        Surface roughness Ra of kerf sidewalls [nm].
    n_profiles : int
        Number of profiles averaged (noise ∝ 1/√n_profiles).
    seed : int or None
        Random seed for reproducibility.
    wafer_index : int
        Sequential wafer counter (stored in KerfProfile).
    depth_um : float
        True kerf depth [µm].
    sidewall_deg : float
        Sidewall taper angle from vertical [°].

    Returns
    -------
    KerfProfile
    """
    rng = np.random.default_rng(seed)

    # Lateral axis: 2 mm window centred on kerf, 5 µm pixel pitch
    y_step = LJX_SPEC["lateral_y_um"]
    y_half = 1000.0  # µm
    y = np.arange(-y_half, y_half + y_step, y_step)

    # True profile (trapezoidal groove)
    half_w = true_kerf_um / 2.0
    taper = depth_um * math.tan(math.radians(sidewall_deg))  # µm at bottom
    half_w_bot = half_w - taper  # kerf is narrower at bottom

    z_true = np.zeros_like(y)
    for i, yi in enumerate(y):
        ay = abs(yi)
        if ay <= half_w_bot:
            z_true[i] = -depth_um
        elif ay <= half_w:
            # linear sidewall transition
            frac = (ay - half_w_bot) / (half_w - half_w_bot + 1e-9)
            z_true[i] = -depth_um * (1.0 - frac)
        # else z = 0 (wafer surface)

    # Surface roughness: band-limited noise
    Ra_um = Ra_nm * 1e-3
    sigma_rough = Ra_um * math.sqrt(math.pi / 2.0)  # Ra → σ for Gaussian
    roughness_raw = rng.normal(0.0, sigma_rough, size=y.shape)
    # Low-pass filter to correlation length ~15 µm
    sigma_pixels = 15.0 / y_step
    roughness = ndi.gaussian_filter1d(roughness_raw, sigma=sigma_pixels)
    # Only apply roughness inside/near kerf
    in_kerf = z_true < -1.0
    z_true[in_kerf] += roughness[in_kerf]

    # Sensor noise after profile averaging
    sigma_sensor = LJX_SPEC["sigma_z_nm"] * 1e-3 / math.sqrt(max(n_profiles, 1))
    z_meas = z_true + rng.normal(0.0, sigma_sensor, size=y.shape)

    # Kerf width extraction (threshold at half-depth, sub-pixel interpolation)
    threshold = -depth_um / 2.0
    below = z_meas < threshold
    indices = np.where(below)[0]
    if len(indices) >= 2:
        # Sub-pixel left crossing: linear interpolation between last-above and first-below
        i_left = indices[0]
        if i_left > 0 and (z_meas[i_left] - z_meas[i_left - 1]) != 0:
            frac_l = (threshold - z_meas[i_left - 1]) / (z_meas[i_left] - z_meas[i_left - 1])
            y_left = y[i_left - 1] + frac_l * y_step
        else:
            y_left = y[i_left]
        # Sub-pixel right crossing
        i_right = indices[-1]
        if i_right < len(z_meas) - 1 and (z_meas[i_right + 1] - z_meas[i_right]) != 0:
            frac_r = (threshold - z_meas[i_right]) / (z_meas[i_right + 1] - z_meas[i_right])
            y_right = y[i_right] + frac_r * y_step
        else:
            y_right = y[i_right]
        kerf_w_meas = float(y_right - y_left)
    else:
        kerf_w_meas = float(true_kerf_um)

    kerf_d_meas = float(-np.min(z_meas))

    # Sidewall angle estimate from profile gradient near edges
    grad = np.gradient(z_meas, y_step)
    # Find steepest slopes near the two edges
    left_mask = (y > -half_w - 20) & (y < -half_w + 20)
    right_mask = (y > half_w - 20) & (y < half_w + 20)
    def _wall_angle(mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return sidewall_deg
        g = np.abs(grad[mask])
        return float(np.degrees(np.arctan(np.max(g))))

    sw_angle = (_wall_angle(left_mask) + _wall_angle(right_mask)) / 2.0

    return KerfProfile(
        y_um=y,
        z_um=z_meas,
        kerf_width_measured_um=kerf_w_meas,
        kerf_depth_measured_um=kerf_d_meas,
        sidewall_angle_deg=sw_angle,
        wafer_index=wafer_index,
        true_kerf_um=true_kerf_um,
        Ra_nm=Ra_nm,
    )


# ============================================================================
# 3.  Wafer warp / bow map
# ============================================================================

def wafer_warp_map(
    n_x: int = 150,
    n_y: int = 150,
    warp_um: float = 50.0,
    noise_um: float = 1.5,
    seed: Optional[int] = 0,
    diameter_mm: float = 150.0,
) -> np.ndarray:
    """Simulate a 2-D wafer surface warp/bow height map [µm].

    The nominal shape is a spherical cap (paraboloid approximation) with
    peak-to-valley bow = `warp_um`.  Typical values for 150 mm 4H-SiC
    substrates: 20–80 µm (SEMI M55 standard).

    An additional long-wavelength saddle component (astigmatism) is added
    at 20 % amplitude to mimic real SiC ingot slicing artefacts.

    Parameters
    ----------
    n_x, n_y : int
        Grid resolution in X (scan direction) and Y (laser line direction).
    warp_um : float
        Peak-to-valley bow [µm].
    noise_um : float
        Point-to-point height noise [µm] (LJ-X8080: 0.5 µm, but wafer
        surface is reflective so ~1.5 µm allows for tilt residuals).
    seed : int or None
        Random seed.
    diameter_mm : float
        Wafer diameter [mm].

    Returns
    -------
    ndarray, shape (n_x, n_y)
        Height map Z[i, j] in µm.  z=0 at wafer edge; positive = convex bow.
    """
    rng = np.random.default_rng(seed)

    x = np.linspace(-diameter_mm / 2, diameter_mm / 2, n_x)
    y = np.linspace(-diameter_mm / 2, diameter_mm / 2, n_y)
    X, Y = np.meshgrid(x, y, indexing="ij")

    R2 = (X ** 2 + Y ** 2) / (diameter_mm / 2) ** 2  # normalised radius²

    # Paraboloid bow: z = warp * (1 - r²)   (max at centre)
    z_bow = warp_um * (1.0 - R2)

    # Astigmatism term (saddle, ±20 % of bow)
    z_astig = 0.20 * warp_um * (X ** 2 - Y ** 2) / (diameter_mm / 2) ** 2

    # Mask to circular wafer boundary
    mask = R2 > 1.0

    # Sensor noise
    noise = rng.normal(0.0, noise_um, size=(n_x, n_y))
    # Smooth noise to sensor spatial resolution
    sigma_pix = 5.0 / (diameter_mm * 1e3 / n_x)  # 5 µm in pixels
    noise = ndi.gaussian_filter(noise, sigma=max(sigma_pix, 0.5))

    Z = z_bow + z_astig + noise
    Z[mask] = np.nan
    return Z


# ============================================================================
# 4.  Blade wear estimation from kerf width trend
# ============================================================================

def blade_wear_from_kerf(kerf_profiles: List[KerfProfile]) -> float:
    """Estimate blade wear rate [µm kerf-width / wafer] from a kerf trend.

    As a diamond blade wears, dressing particles fall off and the effective
    blade thickness decreases — counterintuitively this can increase kerf
    width because worn blades deflect more laterally.  Empirically for
    standard resin-bond blades on 4H-SiC:

        Δ kerf_width ≈ +0.05 µm per wafer diced

    This function fits a linear regression to the measured kerf widths vs.
    wafer index.

    Parameters
    ----------
    kerf_profiles : list of KerfProfile
        At least 2 profiles from consecutive wafers (wafer_index must be set).

    Returns
    -------
    float
        Fitted slope [µm / wafer].  Positive = kerf widens with wear.

    Raises
    ------
    ValueError
        If fewer than 2 profiles are provided.
    """
    if len(kerf_profiles) < 2:
        raise ValueError("Need at least 2 KerfProfile instances to fit wear rate.")

    indices = np.array([p.wafer_index for p in kerf_profiles], dtype=float)
    widths = np.array([p.kerf_width_measured_um for p in kerf_profiles], dtype=float)

    # Linear regression via normal equations
    A = np.column_stack([indices, np.ones_like(indices)])
    slope, _ = np.linalg.lstsq(A, widths, rcond=None)[0]
    return float(slope)


# ============================================================================
# 5.  σ_kerf vs scan speed
# ============================================================================

def sigma_vs_scan_speed(
    speeds_mm_s: np.ndarray,
    sigma0_um: float = 0.8,
    v_knee_mm_s: float = 200.0,
    exponent: float = 0.6,
) -> np.ndarray:
    """Model σ_kerf degradation at high scan speeds.

    At low scan speeds the sensor averages many profiles per lateral step,
    suppressing noise.  Above a knee velocity `v_knee`, fewer profiles are
    averaged per measurement point and σ rises as a power law:

        σ(v) = σ₀ · max(1, (v / v_knee))^exponent

    Parameters
    ----------
    speeds_mm_s : ndarray
        Scan speeds to evaluate [mm/s].
    sigma0_um : float
        Baseline σ at low speed [µm].
    v_knee_mm_s : float
        Speed above which σ begins to degrade [mm/s].
        Default 200 mm/s corresponds to ~32 profile averages at 5 µm pitch.
    exponent : float
        Power-law exponent (0 < exponent < 1 for sub-linear degradation).

    Returns
    -------
    ndarray
        σ_kerf [µm] at each scan speed.
    """
    v = np.asarray(speeds_mm_s, dtype=float)
    ratio = np.maximum(1.0, v / v_knee_mm_s)
    return sigma0_um * ratio ** exponent


# ============================================================================
# 6.  Plotting
# ============================================================================

def plot_main() -> None:
    """Generate a 6-panel summary figure and save to results/keyence_lj_profiler.png.

    Panels
    ------
    (A) Triangulation geometry schematic (matplotlib patches)
    (B) Kerf profile cross-section for 3 blade wear levels
    (C) Wafer warp map (2-D heatmap)
    (D) σ_kerf vs scan speed
    (E) Kerf width trend over 80 wafers (blade wear progression)
    (F) Comparison table: LJ-X8080 vs VK-X3100 vs SEM for dicing metrology
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        hspace=0.48,
        wspace=0.38,
        left=0.07, right=0.97,
        top=0.93, bottom=0.05,
    )

    ax_A = fig.add_subplot(gs[0, 0])   # geometry
    ax_B = fig.add_subplot(gs[0, 1])   # kerf profiles
    ax_C = fig.add_subplot(gs[0, 2])   # warp map
    ax_D = fig.add_subplot(gs[1, 0])   # sigma vs speed
    ax_E = fig.add_subplot(gs[1, 1])   # wear trend
    ax_F = fig.add_subplot(gs[1:, 2])  # comparison table (tall)

    fig.suptitle(
        "Keyence LJ-X8080 Laser Line Profiler — SiC Dicing Metrology",
        fontsize=14, fontweight="bold",
    )

    # ------------------------------------------------------------------
    # (A) Triangulation geometry schematic
    # ------------------------------------------------------------------
    ax = ax_A
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.3, 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(A) Triangulation geometry", fontsize=10)

    # Wafer surface
    ax.add_patch(mpatches.FancyBboxPatch(
        (-1.1, -0.25), 2.2, 0.12,
        boxstyle="round,pad=0.02", fc="#c8d8e8", ec="grey", lw=1.2,
    ))
    ax.text(0, -0.15, "SiC wafer surface", ha="center", va="center", fontsize=7)

    # Sensor head
    sensor_y = 1.75
    sensor_x = 0.0
    ax.add_patch(mpatches.FancyBboxPatch(
        (-0.35, sensor_y - 0.12), 0.70, 0.22,
        boxstyle="round,pad=0.03", fc="#4a90d9", ec="#1a5fa8", lw=1.5, alpha=0.85,
    ))
    ax.text(sensor_x, sensor_y + 0.04, "LJ-X8080\nSensor Head",
            ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    # Laser beam (405 nm → violet)
    theta_rad = math.radians(30)
    beam_dx = 0.8 * math.sin(theta_rad)
    beam_dy = -0.8 * math.cos(theta_rad)
    ax.annotate(
        "", xy=(sensor_x + beam_dx, sensor_y + beam_dy - 0.5),
        xytext=(sensor_x, sensor_y - 0.12),
        arrowprops=dict(arrowstyle="-|>", color="#9b59b6", lw=2.0),
    )
    ax.text(sensor_x + beam_dx / 2 + 0.12, sensor_y + beam_dy / 2 + 0.25,
            "405 nm\nlaser line", fontsize=7, color="#9b59b6", ha="left")

    # Reflected light to CMOS
    spot_x = sensor_x + beam_dx
    spot_y = sensor_y + beam_dy - 0.5 + 0.05  # just above surface
    ax.annotate(
        "", xy=(sensor_x - 0.05, sensor_y - 0.12),
        xytext=(spot_x, spot_y),
        arrowprops=dict(arrowstyle="-|>", color="#e67e22", lw=1.5, linestyle="dashed"),
    )
    ax.text(sensor_x - 0.55, sensor_y + 0.30,
            "Reflected\nlight → CMOS", fontsize=7, color="#e67e22", ha="left")

    # Angle annotation
    angle_arc = Arc((sensor_x, sensor_y - 0.12), 0.5, 0.5,
                    angle=0, theta1=260, theta2=300, color="#9b59b6", lw=1.2)
    ax.add_patch(angle_arc)
    ax.text(sensor_x + 0.18, sensor_y - 0.32, "30°", fontsize=8, color="#9b59b6")

    # Standoff annotation
    ax.annotate(
        "", xy=(1.0, -0.12), xytext=(1.0, sensor_y - 0.12),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.0),
    )
    ax.text(1.08, sensor_y / 2 - 0.1, "35 mm\nstandoff", fontsize=7, ha="left", va="center")

    # FOV annotation
    ax.annotate("", xy=(-1.0, -0.25), xytext=(1.0, -0.25),
                arrowprops=dict(arrowstyle="<->", color="steelblue", lw=1.0))
    ax.text(0, -0.28, "24 mm FOV", fontsize=7, ha="center", va="top", color="steelblue")

    # ------------------------------------------------------------------
    # (B) Kerf profiles for 3 wear levels
    # ------------------------------------------------------------------
    ax = ax_B
    ax.set_title("(B) Kerf cross-section (3 wear levels)", fontsize=10)

    wear_levels = [
        dict(wafer=0,  kerf=80.0,  label="New blade (w=0)",  color="#2ecc71"),
        dict(wafer=40, kerf=82.0,  label="Mid wear (w=40)",   color="#f39c12"),
        dict(wafer=80, kerf=84.0,  label="Worn blade (w=80)", color="#e74c3c"),
    ]
    for wl in wear_levels:
        p = kerf_profile_scan(
            true_kerf_um=wl["kerf"], Ra_nm=80.0, n_profiles=32,
            seed=wl["wafer"] + 100, wafer_index=wl["wafer"],
        )
        mask = np.abs(p.y_um) < 200  # show ±200 µm window
        ax.plot(p.y_um[mask], p.z_um[mask], lw=1.5, color=wl["color"],
                label=wl["label"])

    ax.set_xlabel("Lateral position y [µm]", fontsize=8)
    ax.set_ylabel("Height z [µm]", fontsize=8)
    ax.legend(fontsize=7, loc="lower center")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # (C) Wafer warp map
    # ------------------------------------------------------------------
    ax = ax_C
    ax.set_title("(C) Wafer bow map (150 mm 4H-SiC)", fontsize=10)

    Z = wafer_warp_map(n_x=120, n_y=120, warp_um=55.0, noise_um=1.5, seed=7)
    extent = [-75, 75, -75, 75]
    im = ax.imshow(
        Z.T, origin="lower", extent=extent, cmap="RdYlBu_r",
        vmin=np.nanmin(Z), vmax=np.nanmax(Z),
    )
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Height [µm]", fontsize=7)
    cb.ax.tick_params(labelsize=7)

    # Wafer circle outline
    theta_c = np.linspace(0, 2 * np.pi, 360)
    ax.plot(75 * np.cos(theta_c), 75 * np.sin(theta_c), "k-", lw=1.2)

    ax.set_xlabel("X [mm]", fontsize=8)
    ax.set_ylabel("Y [mm]", fontsize=8)
    ax.tick_params(labelsize=7)
    pv = float(np.nanmax(Z) - np.nanmin(Z))
    ax.set_title(f"(C) Wafer bow map  (P-V = {pv:.1f} µm)", fontsize=10)

    # ------------------------------------------------------------------
    # (D) σ_kerf vs scan speed
    # ------------------------------------------------------------------
    ax = ax_D
    ax.set_title("(D) σ_kerf vs scan speed", fontsize=10)

    speeds = np.linspace(1, 800, 400)
    sigma = sigma_vs_scan_speed(speeds)
    ax.plot(speeds, sigma, color="#2980b9", lw=2.0, label="LJ-X8080 model")
    ax.axhline(LJX_SPEC["sigma_kerf_um"], color="grey", lw=1.0, ls="--",
               label=f"Spec σ = {LJX_SPEC['sigma_kerf_um']} µm")
    ax.axvline(200.0, color="#e74c3c", lw=1.0, ls=":", label="v_knee = 200 mm/s")
    ax.set_xlabel("Scan speed [mm/s]", fontsize=8)
    ax.set_ylabel("σ_kerf [µm]", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

    # Typical dicing speed annotation
    ax.axvspan(50, 200, alpha=0.12, color="green", label="Typical dicing range")
    ax.text(125, sigma_vs_scan_speed(np.array([125.0]))[0] + 0.03,
            "Typical\ndicing", fontsize=7, ha="center", color="green")

    # ------------------------------------------------------------------
    # (E) Kerf width trend over 80 wafers (blade wear)
    # ------------------------------------------------------------------
    ax = ax_E
    ax.set_title("(E) Kerf width trend — blade wear", fontsize=10)

    n_wafers = 80
    base_kerf = 80.0
    wear_rate_true = 0.05  # µm / wafer
    rng_plot = np.random.default_rng(999)

    profiles_all: List[KerfProfile] = []
    for wi in range(n_wafers):
        true_k = base_kerf + wear_rate_true * wi
        p = kerf_profile_scan(
            true_kerf_um=true_k,
            Ra_nm=80.0,
            n_profiles=32,
            seed=int(rng_plot.integers(0, 10_000)),
            wafer_index=wi,
        )
        profiles_all.append(p)

    wafer_idx = np.array([p.wafer_index for p in profiles_all])
    widths_meas = np.array([p.kerf_width_measured_um for p in profiles_all])
    fitted_rate = blade_wear_from_kerf(profiles_all)

    ax.scatter(wafer_idx, widths_meas, s=12, alpha=0.6, color="#3498db",
               label="Measured kerf width")
    fit_line = base_kerf + fitted_rate * wafer_idx
    ax.plot(wafer_idx, fit_line, "r-", lw=2.0,
            label=f"Fit: {fitted_rate:.4f} µm/wafer")
    true_line = base_kerf + wear_rate_true * wafer_idx
    ax.plot(wafer_idx, true_line, "g--", lw=1.5, alpha=0.7,
            label=f"True: {wear_rate_true:.4f} µm/wafer")

    ax.set_xlabel("Wafer index", fontsize=8)
    ax.set_ylabel("Kerf width [µm]", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # (F) Comparison table: LJ-X8080 vs VK-X3100 vs SEM
    # ------------------------------------------------------------------
    ax = ax_F
    ax.set_title("(F) Metrology comparison for SiC dicing", fontsize=10)
    ax.axis("off")

    col_labels = ["Parameter", "LJ-X8080\n(this module)", "VK-X3100\n(confocal)", "SEM"]
    rows = [
        ["Principle",        "Laser triangulation", "Confocal laser", "Electron beam"],
        ["Wavelength",       "405 nm line",         "405 nm point",   "—"],
        ["z-resolution",     "0.5 µm",              "0.22 µm",        "<0.01 µm"],
        ["Lateral res.",     "5 µm",                "0.26 µm",        "<0.01 µm"],
        ["FOV",              "24×3 mm",             "~0.3×0.3 mm",    "~1×1 mm"],
        ["Scan rate",        "64 000 prof/s",       "12 wph",         "~1–2 wph"],
        ["Operation mode",   "Inline / offline",    "Offline",        "Offline"],
        ["Kerf width σ",     "0.8 µm",              "0.28 µm",        "—"],
        ["Warp mapping",     "Yes (fast)",          "No (small FOV)", "No"],
        ["Vacuum required",  "No",                  "No",             "Yes"],
        ["Cost (relative)",  "Low",                 "Medium",         "High"],
        ["SiC suitability",  "Inline kerf+warp",    "Chipping detail","Cross-section"],
    ]

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.55)

    # Color header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Highlight LJ-X column
    for i in range(1, len(rows) + 1):
        table[i, 1].set_facecolor("#d6eaf8")

    # Alternating row shading
    for i in range(1, len(rows) + 1):
        if i % 2 == 0:
            for j in [0, 2, 3]:
                table[i, j].set_facecolor("#f2f3f4")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = os.path.join(OUT_DIR, "keyence_lj_profiler.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================================
# 7.  main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keyence LJ-X8080 laser line profiler simulation for SiC dicing"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip figure generation (useful for unit tests or headless runs)",
    )
    args = parser.parse_args()

    # --- Quick self-test ---
    dz = triangulation_z_resolution(
        standoff_mm=LJX_SPEC["standoff_mm"],
        angle_deg=LJX_SPEC["triangulation_angle_deg"],
        pixel_pitch_um=LJX_SPEC["pixel_pitch_um"],
        magnification=LJX_SPEC["magnification"],
    )
    print(f"Theoretical z-resolution: {dz:.3f} µm  (spec: {LJX_SPEC['z_resolution_um']} µm)")

    p = kerf_profile_scan(true_kerf_um=100.0, Ra_nm=80.0, n_profiles=32, seed=42)
    print(f"Kerf scan: width = {p.kerf_width_measured_um:.1f} µm, "
          f"depth = {p.kerf_depth_measured_um:.1f} µm, "
          f"sidewall = {p.sidewall_angle_deg:.1f}°")

    Z = wafer_warp_map(warp_um=55.0, seed=0)
    pv = float(np.nanmax(Z) - np.nanmin(Z))
    print(f"Warp map: peak-to-valley = {pv:.1f} µm  (input bow = 55.0 µm)")

    # Build 80-wafer wear series (4 µm total signal vs ~0.1 µm noise per point)
    profiles = [
        kerf_profile_scan(
            true_kerf_um=80.0 + 0.05 * wi, Ra_nm=80.0,
            n_profiles=64, seed=wi + 1, wafer_index=wi,
        )
        for wi in range(80)
    ]
    rate = blade_wear_from_kerf(profiles)
    print(f"Blade wear rate: {rate:.5f} µm/wafer  (true: 0.05000 µm/wafer)")

    speeds = np.array([50.0, 100.0, 200.0, 400.0, 800.0])
    sigmas = sigma_vs_scan_speed(speeds)
    print("σ_kerf vs speed:")
    for v, s in zip(speeds, sigmas):
        print(f"  {v:5.0f} mm/s → {s:.3f} µm")

    if not args.no_plot:
        plot_main()


if __name__ == "__main__":
    main()
