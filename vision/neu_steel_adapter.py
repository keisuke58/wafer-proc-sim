"""
Adapter for the NEU Surface Defect Database for kerf inspection demos.

NEU dataset
-----------
  URL     : http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
  Kaggle  : https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
  Images  : 300 per category, 200×200 px, 8-bit grayscale BMP
  License : Academic use with citation

Category → kerf inspection analog
----------------------------------
  scratches      → kerf edge (linear bright/dark features, closest analog)
  crazing        → distributed chipping (irregular surface cracks)
  pitted_surface → point chipping defects
  inclusion      → hard-particle inclusions in Si substrate
  patches        → surface stain / contamination
  rolled-in_scale → scale/debris defect

Usage
-----
    from vision.neu_steel_adapter import NeuSteelAdapter
    adapter = NeuSteelAdapter("/path/to/NEU-CLS")
    imgs, paths = adapter.load_category("scratches", max_images=20)
    # imgs: list of float32 (200, 200) arrays, range [0, 255]

    # Preprocess for detect_kerf_width
    kerf_inputs = adapter.as_kerf_input("scratches")
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


CATEGORY_MAPPING: dict = {
    "scratches":        "kerf_edge",
    "crazing":          "distributed_chipping",
    "pitted_surface":   "point_chipping",
    "inclusion":        "hard_particle",
    "patches":          "surface_stain",
    "rolled_in_scale":  "scale_defect",
}

_CANDIDATE_NAMES = [
    "scratches",       "Scratches",
    "crazing",         "Crazing",
    "pitted_surface",  "Pitted_surface",  "pitted surface",
    "inclusion",       "Inclusion",
    "patches",         "Patches",
    "rolled-in_scale", "Rolled-in_Scale", "rolled_in_scale",
]


def _load_image_grayscale(path: str) -> np.ndarray:
    """Load image as float32 grayscale [0, 255]. Tries PIL, then cv2, then raw BMP."""
    try:
        from PIL import Image
        return np.array(Image.open(path).convert("L"), dtype=np.float32)
    except ImportError:
        pass
    try:
        import cv2  # type: ignore
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img.astype(np.float32)
    except ImportError:
        pass
    if path.lower().endswith(".bmp"):
        return _parse_bmp_grayscale(path)
    raise ImportError(
        "PIL (Pillow) or OpenCV required for non-BMP images. "
        "Install with: pip install Pillow"
    )


def _parse_bmp_grayscale(path: str) -> np.ndarray:
    """Pure-Python BMP reader (8-bit or 24-bit → float32 grayscale)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] != b"BM":
        raise ValueError(f"Not a BMP: {path}")
    px_off = struct.unpack_from("<I", data, 10)[0]
    w      = struct.unpack_from("<i", data, 18)[0]
    h      = struct.unpack_from("<i", data, 22)[0]
    bpp    = struct.unpack_from("<H", data, 28)[0]
    flip   = h > 0
    h      = abs(h)
    px     = data[px_off:]
    if bpp == 8:
        row_sz = (w + 3) & ~3
        arr = np.frombuffer(px[:row_sz * h], dtype=np.uint8).reshape(h, row_sz)[:, :w]
    elif bpp == 24:
        row_sz = (w * 3 + 3) & ~3
        raw = np.frombuffer(px[:row_sz * h], dtype=np.uint8).reshape(h, row_sz)
        bgr = raw[:, :w * 3].reshape(h, w, 3)
        arr = (0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported BMP bpp={bpp} in {path}")
    return (arr[::-1] if flip else arr).astype(np.float32)


class NeuSteelAdapter:
    """
    Load and preprocess NEU Surface Defect Dataset images for kerf inspection demos.

    Parameters
    ----------
    root : str | Path
        Root directory containing subdirectories like 'scratches/', 'crazing/', etc.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"NEU dataset not found at '{root}'.\n"
                "Download from:\n"
                "  http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html\n"
                "  https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database\n"
                "Then pass the extracted directory path to NeuSteelAdapter(root=...)."
            )
        self._cat_dirs: dict = self._scan()

    def _scan(self) -> dict:
        found: dict = {}
        for name in _CANDIDATE_NAMES:
            p = self.root / name
            if p.is_dir():
                key = name.lower().replace("-", "_").replace(" ", "_")
                found[key] = p
        return found

    @property
    def available_categories(self) -> List[str]:
        return list(self._cat_dirs.keys())

    def load_category(
        self,
        category: str,
        max_images: Optional[int] = None,
        normalize: bool = False,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load images from one NEU category.

        Parameters
        ----------
        category    : e.g. "scratches", "crazing", "pitted_surface"
        max_images  : cap on number of images (default: all 300)
        normalize   : if True, scale to [0, 1]; otherwise keep [0, 255]

        Returns
        -------
        images : list of float32 (H, W) arrays
        paths  : list of source file paths
        """
        key = category.lower().replace("-", "_").replace(" ", "_")
        if key not in self._cat_dirs:
            raise KeyError(
                f"Category '{category}' not found. "
                f"Available: {self.available_categories}"
            )
        cat_dir = self._cat_dirs[key]
        exts = {".bmp", ".png", ".jpg", ".jpeg"}
        paths = sorted(p for p in cat_dir.iterdir() if p.suffix.lower() in exts)
        if max_images is not None:
            paths = paths[:max_images]

        images, valid_paths = [], []
        for p in paths:
            try:
                img = _load_image_grayscale(str(p))
                if normalize:
                    img = img / 255.0
                images.append(img)
                valid_paths.append(str(p))
            except Exception:
                continue
        return images, valid_paths

    def as_kerf_input(
        self,
        category: str = "scratches",
        max_images: Optional[int] = 20,
    ) -> List[np.ndarray]:
        """
        Return images ready for detect_kerf_width / measure_chipping.
        Rotates portrait images to landscape (scratches are often vertical).
        """
        images, _ = self.load_category(category, max_images=max_images)
        result = []
        for img in images:
            if img.shape[0] > img.shape[1]:
                img = np.rot90(img)
            result.append(img)
        return result

    def kerf_analog(self, category: str) -> str:
        """Return the kerf inspection analog label for a NEU category."""
        key = category.lower().replace("-", "_").replace(" ", "_")
        return CATEGORY_MAPPING.get(key, "unknown")

    def __repr__(self) -> str:
        return f"NeuSteelAdapter(root={self.root}, categories={self.available_categories})"
