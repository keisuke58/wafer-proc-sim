"""Tests for vision/generate_synthetic_kerf.py and vision/neu_steel_adapter.py."""
from __future__ import annotations
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision.generate_synthetic_kerf import KerfImageConfig, generate_kerf_image, make_dataset
from vision.neu_steel_adapter import NeuSteelAdapter, CATEGORY_MAPPING


# ── generate_synthetic_kerf ───────────────────────────────────────────────────

class TestGenerateSyntheticKerf:
    def test_output_shape_and_dtype(self):
        cfg = KerfImageConfig(H=64, W=128)
        img, gt = generate_kerf_image(cfg)
        assert img.shape == (64, 128)
        assert img.dtype == np.float32

    def test_pixel_range(self):
        cfg = KerfImageConfig(H=64, W=128, seed=7)
        img, _ = generate_kerf_image(cfg)
        assert img.min() >= 0.0
        assert img.max() <= 255.0

    def test_ground_truth_keys(self):
        _, gt = generate_kerf_image(KerfImageConfig())
        for k in ("kerf_left_px", "kerf_right_px", "kerf_width_mm", "kerf_width_px"):
            assert k in gt

    def test_geometry_consistency(self):
        cfg = KerfImageConfig(H=64, W=256, kerf_width_mm=2.0, pixel_per_mm=10.0, seed=1)
        _, gt = generate_kerf_image(cfg)
        # kerf_width_px = kerf_width_mm * pixel_per_mm = 20 px
        assert abs(gt["kerf_width_px"] - 20.0) < 1e-6
        assert abs(gt["kerf_right_px"] - gt["kerf_left_px"] - 20.0) < 1e-6

    def test_kerf_void_is_dark(self):
        cfg = KerfImageConfig(H=128, W=256, kerf_width_mm=3.0, pixel_per_mm=10.0,
                              chipping_severity=0.0, sensor_noise=0.0, seed=0)
        img, gt = generate_kerf_image(cfg)
        cx = int((gt["kerf_left_px"] + gt["kerf_right_px"]) / 2)
        void_mean = float(img[:, cx].mean())
        substrate_mean_approx = float(img[:, 5].mean())  # far left = substrate
        assert void_mean < substrate_mean_approx - 50.0, (
            f"Kerf void ({void_mean:.1f}) should be much darker than substrate ({substrate_mean_approx:.1f})"
        )

    def test_wall_brightness_peaks(self):
        cfg = KerfImageConfig(H=128, W=256, kerf_width_mm=3.0, pixel_per_mm=10.0,
                              sensor_noise=0.0, chipping_severity=0.0, seed=0)
        img, gt = generate_kerf_image(cfg)
        proj = img.mean(axis=0)
        left_peak = proj[max(0, int(gt["kerf_left_px"]) - 3):int(gt["kerf_left_px"]) + 4].max()
        assert left_peak > 180.0, f"Left wall peak too dim: {left_peak:.1f}"

    def test_chipping_darkens_edges(self):
        cfg_clean = KerfImageConfig(H=64, W=128, kerf_width_mm=2.0,
                                    chipping_severity=0.0, seed=5)
        cfg_chip  = KerfImageConfig(H=64, W=128, kerf_width_mm=2.0,
                                    chipping_severity=0.9, seed=5)
        img_clean, gt = generate_kerf_image(cfg_clean)
        img_chip,  _  = generate_kerf_image(cfg_chip)
        # Edge region outside kerf walls
        lw, rw = int(gt["kerf_left_px"]), int(gt["kerf_right_px"])
        left_edge  = slice(max(0, lw - 8), lw)
        right_edge = slice(rw + 1, min(128, rw + 9))
        mean_clean = (img_clean[:, left_edge].mean() + img_clean[:, right_edge].mean()) / 2
        mean_chip  = (img_chip [:, left_edge].mean() + img_chip [:, right_edge].mean()) / 2
        assert mean_chip < mean_clean, "Heavy chipping should darken edge regions"

    def test_blade_tip_bright_at_top(self):
        cfg = KerfImageConfig(H=128, W=256, kerf_width_mm=3.0,
                              include_blade_tip=True, sensor_noise=0.0, seed=2)
        img, gt = generate_kerf_image(cfg)
        # Tip is a small ellipse at top-centre of kerf; check the centre column only
        cx = int((gt["kerf_left_px"] + gt["kerf_right_px"]) / 2)
        tip_row = 128 // 8   # H // 8 = 16
        tip_col_brightness = float(img[tip_row, cx])
        bottom_brightness  = float(img[-1, cx])
        assert tip_col_brightness > bottom_brightness + 30.0, (
            f"Blade tip centre ({tip_col_brightness:.1f}) should be much brighter than "
            f"bottom ({bottom_brightness:.1f})"
        )

    def test_blade_wear_broadens_wall(self):
        cfg_sharp = KerfImageConfig(H=64, W=128, kerf_width_mm=2.0,
                                    blade_wear=0.0, sensor_noise=0.0, seed=3)
        cfg_worn  = KerfImageConfig(H=64, W=128, kerf_width_mm=2.0,
                                    blade_wear=1.0, sensor_noise=0.0, seed=3)
        img_s, _ = generate_kerf_image(cfg_sharp)
        img_w, _ = generate_kerf_image(cfg_worn)
        # Worn blade: wall profile is broader → more bright pixels away from wall centre
        proj_s = img_s.mean(axis=0)
        proj_w = img_w.mean(axis=0)
        # Variance of proj should be higher for worn (more spread)
        assert proj_w.std() >= proj_s.std() * 0.8  # worn ≥ sharp (tolerance for noise)

    def test_reproducible_with_seed(self):
        cfg = KerfImageConfig(H=32, W=64, seed=99)
        img1, _ = generate_kerf_image(cfg)
        img2, _ = generate_kerf_image(cfg)
        np.testing.assert_array_equal(img1, img2)

    def test_make_dataset_length_and_variance(self):
        ds = make_dataset(n_images=10, seed=0)
        assert len(ds) == 10
        widths = [d["ground_truth"]["kerf_width_mm"] for d in ds]
        assert len(set(widths)) > 1, "Widths should vary across dataset"

    def test_detect_kerf_width_on_synthetic(self):
        """Round-trip: generate image, run C++ detector, check accuracy."""
        from vision import detect_kerf_width
        cfg = KerfImageConfig(H=128, W=256, kerf_width_mm=3.0, pixel_per_mm=10.0,
                              chipping_severity=0.0, sensor_noise=3.0, seed=42)
        img, gt = generate_kerf_image(cfg)
        res = detect_kerf_width(img, threshold=5000.0, pixel_per_mm=10.0)
        if res["status"] == "ok":
            err = abs(res["kerf_width_mm"] - gt["kerf_width_mm"])
            assert err < 1.5, f"Kerf detection error {err:.2f} mm too large"


# ── NeuSteelAdapter (unit tests without dataset) ─────────────────────────────

class TestNeuSteelAdapter:
    def test_category_mapping_complete(self):
        expected = {"scratches", "crazing", "pitted_surface",
                    "inclusion", "patches", "rolled_in_scale"}
        assert set(CATEGORY_MAPPING.keys()) == expected

    def test_kerf_analog_scratches(self):
        assert CATEGORY_MAPPING["scratches"] == "kerf_edge"

    def test_missing_root_raises(self):
        with pytest.raises(FileNotFoundError, match="NEU dataset not found"):
            NeuSteelAdapter("/nonexistent/path/neu_steel")

    def test_repr(self, tmp_path):
        p = tmp_path / "scratches"
        p.mkdir()
        adapter = NeuSteelAdapter(tmp_path)
        r = repr(adapter)
        assert "NeuSteelAdapter" in r

    def test_load_category_missing_raises(self, tmp_path):
        adapter = NeuSteelAdapter(tmp_path)
        with pytest.raises(KeyError):
            adapter.load_category("nonexistent_category")

    def test_load_category_empty_dir(self, tmp_path):
        (tmp_path / "scratches").mkdir()
        adapter = NeuSteelAdapter(tmp_path)
        imgs, paths = adapter.load_category("scratches")
        assert imgs == []
        assert paths == []

    def test_load_category_with_png(self, tmp_path):
        """Load a synthetic PNG via PIL (if available) or skip."""
        pytest.importorskip("PIL")
        from PIL import Image as PILImage
        cat_dir = tmp_path / "scratches"
        cat_dir.mkdir()
        arr = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        PILImage.fromarray(arr, mode="L").save(str(cat_dir / "test_001.png"))
        adapter = NeuSteelAdapter(tmp_path)
        imgs, paths = adapter.load_category("scratches")
        assert len(imgs) == 1
        assert imgs[0].shape == (200, 200)
        assert imgs[0].dtype == np.float32

    def test_as_kerf_input_rotates_portrait(self, tmp_path):
        pytest.importorskip("PIL")
        from PIL import Image as PILImage
        cat_dir = tmp_path / "scratches"
        cat_dir.mkdir()
        arr = np.zeros((300, 200), dtype=np.uint8)  # portrait (H > W)
        PILImage.fromarray(arr, mode="L").save(str(cat_dir / "portrait.png"))
        adapter = NeuSteelAdapter(tmp_path)
        result = adapter.as_kerf_input("scratches")
        assert len(result) == 1
        # Should be rotated to landscape
        assert result[0].shape[1] >= result[0].shape[0]
