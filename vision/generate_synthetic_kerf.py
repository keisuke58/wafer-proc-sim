"""
Parametric synthetic kerf image generator for dicing saw process simulation.

Physical model
--------------
- Silicon substrate : uniform mid-gray with Gaussian texture noise
- Kerf void         : dark channel (saw cut removes Si material)
- Kerf walls        : Gaussian-profile bright ridge (specular reflection from vertical walls)
- Chipping          : dark patches in ±chipping_extent_px band outside kerf walls
- Blade tip         : bright elliptical spot at top-centre of kerf (optional)

Typical DISCO DFL7160 specs (for reference):
  blade thickness : 15–200 μm → kerf width ≈ blade + lateral runout
  camera pixel    : 5–20 μm/pixel
  image size      : 512×512–2048×2048 px

For demo / portfolio use, kerf_width_mm=2.0 pixel_per_mm=10.0 is convenient.
Real process: kerf_width_mm=0.03–0.20, pixel_per_mm=100–200.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class KerfImageConfig:
    H: int = 128
    W: int = 256
    kerf_width_mm: float = 2.0         # visible kerf width [mm]
    pixel_per_mm: float = 10.0         # camera resolution [px/mm]
    kerf_center_frac: float = 0.5      # kerf centre as fraction of W
    wall_brightness: float = 160.0     # peak specular wall brightness ABOVE substrate [0-255]
    wall_sigma_px: float = 2.0         # Gaussian wall-edge spread [px]
    substrate_mean: float = 60.0       # silicon substrate reflectance [0-255]
    # Design constraint: substrate_mean + wall_brightness + 5×substrate_noise < 255
    # to avoid saturation clipping that shifts the signed-Sobel zero-crossing.
    # 60 + 160 + 5×6 = 250 < 255 ✓  (>99.99 % of pixels are unclipped)
    # Physical note: real DISCO cameras are set to ~70 % of full-scale so walls
    # are bright (~200 DN) but not saturated.
    substrate_noise: float = 6.0       # surface texture σ [0-255]
    kerf_void_mean: float = 15.0       # kerf interior brightness [0-255]
    kerf_void_noise: float = 3.0       # kerf interior noise σ
    sensor_noise: float = 5.0          # camera Gaussian noise σ
    chipping_severity: float = 0.0     # 0 = none, 1 = heavy chipping
    chipping_extent_px: float = 6.0    # max lateral chip extent [px]
    blade_wear: float = 0.0            # 0 = sharp, 1 = worn (broadens wall σ)
    include_blade_tip: bool = False    # add blade-tip bright ellipse at top
    seed: Optional[int] = 42


def generate_kerf_image(cfg: KerfImageConfig) -> Tuple[np.ndarray, dict]:
    """
    Generate one synthetic float32 H×W kerf image.

    Returns
    -------
    img : np.ndarray, shape (H, W), dtype float32, range [0, 255]
    ground_truth : dict
        kerf_left_px, kerf_right_px, kerf_width_mm, kerf_width_px,
        chipping_severity, has_tip
    """
    rng = np.random.default_rng(cfg.seed)
    H, W = cfg.H, cfg.W

    kerf_width_px = cfg.kerf_width_mm * cfg.pixel_per_mm
    cx = cfg.kerf_center_frac * W
    left_wall  = cx - kerf_width_px / 2.0
    right_wall = cx + kerf_width_px / 2.0

    # Silicon substrate
    img = rng.normal(cfg.substrate_mean, cfg.substrate_noise, (H, W)).astype(np.float32)

    # Kerf void (dark channel)
    l_idx = int(np.clip(np.round(left_wall),  1, W - 2))
    r_idx = int(np.clip(np.round(right_wall), 1, W - 2))
    for c in range(l_idx + 1, r_idx):
        img[:, c] = rng.normal(cfg.kerf_void_mean, cfg.kerf_void_noise, H).astype(np.float32)

    # Kerf walls: Gaussian brightness ridge
    # blade_wear widens σ → diffuse, ragged walls
    sigma = cfg.wall_sigma_px * (1.0 + cfg.blade_wear * 2.0)
    cols = np.arange(W, dtype=np.float32)
    wall_profile = (
        cfg.wall_brightness * np.exp(-0.5 * ((cols - left_wall)  / sigma) ** 2) +
        cfg.wall_brightness * np.exp(-0.5 * ((cols - right_wall) / sigma) ** 2)
    ).astype(np.float32)
    img += wall_profile[np.newaxis, :]

    # Chipping: dark irregular patches outside kerf walls
    if cfg.chipping_severity > 0.0:
        n_chips = max(1, int(cfg.chipping_severity * H * 2))
        for _ in range(n_chips):
            side_offset = float(rng.uniform(1.0, max(2.0, cfg.chipping_extent_px)))
            chip_c = int(left_wall - side_offset) if rng.random() < 0.5 else int(right_wall + side_offset)
            chip_c = int(np.clip(chip_c, 0, W - 1))
            chip_r = int(rng.uniform(0, H - 1))
            chip_h = max(1, int(rng.uniform(1, max(2, H // 8))))
            chip_w = max(1, int(rng.uniform(1, max(2, cfg.chipping_extent_px))))
            r0, r1 = int(np.clip(chip_r,          0, H)), int(np.clip(chip_r + chip_h, 0, H))
            c0, c1 = int(np.clip(chip_c,          0, W)), int(np.clip(chip_c + chip_w, 0, W))
            img[r0:r1, c0:c1] *= float(rng.uniform(0.05, 0.25))

    # Blade tip: bright ellipse at top of kerf
    if cfg.include_blade_tip:
        tip_row = H // 8
        tip_cx  = int(cx)
        tip_rx  = max(2, int(kerf_width_px * 0.4))
        tip_ry  = max(2, H // 12)
        rr = np.arange(max(0, tip_row - tip_ry), min(H, tip_row + tip_ry))
        cc = np.arange(max(0, tip_cx  - tip_rx), min(W, tip_cx  + tip_rx))
        R, C = np.meshgrid(rr, cc, indexing="ij")
        dist = ((R - tip_row) / tip_ry) ** 2 + ((C - tip_cx) / tip_rx) ** 2
        mask = dist <= 1.0
        img[rr[0]:rr[-1]+1, cc[0]:cc[-1]+1][mask] = np.minimum(
            255.0, img[rr[0]:rr[-1]+1, cc[0]:cc[-1]+1][mask] + 220.0 * (1.0 - dist[mask])
        )

    # Sensor noise
    img += rng.normal(0.0, cfg.sensor_noise, (H, W)).astype(np.float32)
    img = np.clip(img, 0.0, 255.0)

    return img, {
        "kerf_left_px":      float(left_wall),
        "kerf_right_px":     float(right_wall),
        "kerf_width_mm":     cfg.kerf_width_mm,
        "kerf_width_px":     kerf_width_px,
        "chipping_severity": cfg.chipping_severity,
        "has_tip":           cfg.include_blade_tip,
    }


def make_dataset(
    n_images: int = 100,
    base_cfg: Optional[KerfImageConfig] = None,
    vary_width: bool = True,
    vary_chipping: bool = True,
    vary_wear: bool = True,
    seed: int = 0,
) -> list:
    """
    Generate a labeled dataset of synthetic kerf images.

    Returns list of dicts: {image, ground_truth, config}
    """
    rng = np.random.default_rng(seed)
    base = base_cfg or KerfImageConfig()
    dataset = []
    for _ in range(n_images):
        cfg = KerfImageConfig(
            H=base.H, W=base.W,
            pixel_per_mm=base.pixel_per_mm,
            kerf_width_mm=float(rng.uniform(0.5, 4.0)) if vary_width else base.kerf_width_mm,
            kerf_center_frac=float(rng.uniform(0.35, 0.65)),
            chipping_severity=float(rng.uniform(0.0, 0.8)) if vary_chipping else base.chipping_severity,
            blade_wear=float(rng.uniform(0.0, 1.0)) if vary_wear else base.blade_wear,
            include_blade_tip=bool(rng.random() < 0.3),
            seed=int(rng.integers(0, 2**31)),
        )
        img, gt = generate_kerf_image(cfg)
        dataset.append({"image": img, "ground_truth": gt, "config": cfg})
    return dataset
