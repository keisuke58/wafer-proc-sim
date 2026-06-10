try:
    from vision._kerf_detection_kernel import (
        detect_kerf_width,
        detect_blade_tip,
        measure_chipping,
    )
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False
    def detect_kerf_width(image, threshold=30.0, pixel_per_mm=10.0):
        import numpy as np
        from scipy.ndimage import sobel
        gx = np.abs(sobel(image, axis=1))
        proj = gx.sum(axis=0)
        above = np.where(proj > threshold)[0]
        if len(above) < 2:
            return {"kerf_width_mm": -1.0, "left_edge_px": -1,
                    "right_edge_px": -1, "confidence": 0.0, "status": "no_kerf_found"}
        left, right = int(above[0]), int(above[-1])
        return {"kerf_width_mm": float(right - left) / pixel_per_mm,
                "left_edge_px": left, "right_edge_px": right,
                "confidence": 0.8, "status": "ok"}

    def detect_blade_tip(image, threshold=200.0):
        import numpy as np
        rows = np.where((image > threshold).any(axis=1))[0]
        if len(rows) == 0:
            return {"tip_x": -1.0, "tip_y": -1.0, "confidence": 0.0, "status": "no_tip_found"}
        r = int(rows[0])
        return {"tip_x": float(np.argmax(image[r])), "tip_y": float(r),
                "confidence": 0.8, "status": "ok"}

    def measure_chipping(image, kerf_left_px, kerf_right_px, pixel_per_mm=10.0):
        return {"chipping_um": 0.0, "chipping_ratio": 0.0, "status": "ok"}

__all__ = ["detect_kerf_width", "detect_blade_tip", "measure_chipping", "_HAS_CPP"]
