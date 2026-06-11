"""
SECOM loader — UCI semiconductor fab process sensor data (CC BY 4.0).

SECOM is the real-world analogue of our synthetic APC sensor stream:
590 process-tool sensor signals per production lot, with a pass(-1)/fail(+1)
yield label and a timestamp.  It is heavily imbalanced (104 fails / 1463 pass)
and full of missing values — exactly the regime our fault-detection layers
(ml/anomaly_detection.py: Isolation Forest + Shewhart chart) must survive.

Files (data/external/SECOM/):
    secom.data         1567 × 590 floats, space-separated, 'NaN' for missing
    secom_labels.data  '<label> <timestamp>' per row (label in {-1, +1})

Usage
-----
    from data.load_secom import load_secom
    X, y, ts = load_secom()        # X:(1567,590) y:(1567,) {0=pass,1=fail}
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd

SECOM_DIR = os.path.join(os.path.dirname(__file__), "external", "SECOM")


def load_secom(secom_dir: str = SECOM_DIR, impute: bool = True
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X  : (1567, 590) float32 sensor matrix (NaNs column-mean imputed if impute)
    y  : (1567,) int8 — 0 = pass (qualified), 1 = fail (the rare/defect class)
    ts : (1567,) datetime64 timestamps (lot order for time-series monitoring)
    """
    data_path = os.path.join(secom_dir, "secom.data")
    label_path = os.path.join(secom_dir, "secom_labels.data")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"SECOM not found at {secom_dir}\n"
            "Download: https://archive.ics.uci.edu/static/public/179/secom.zip"
        )

    X = pd.read_csv(data_path, sep=r"\s+", header=None, na_values="NaN").to_numpy(
        dtype=np.float32)
    if impute:
        col_mean = np.nanmean(X, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)  # all-NaN cols → 0
        idx = np.where(np.isnan(X))
        X[idx] = np.take(col_mean, idx[1])

    # labels file: '<label> "<dd/mm/YYYY HH:MM:SS>"' — datetime is one quoted field
    lab = pd.read_csv(label_path, sep=r"\s+", header=None, quotechar='"',
                      names=["label", "datetime"])
    y = (lab["label"].to_numpy() == 1).astype(np.int8)  # +1 fail → 1
    ts = pd.to_datetime(lab["datetime"], format="%d/%m/%Y %H:%M:%S",
                        errors="coerce").to_numpy()

    # chronological order — required for honest drift/Shewhart monitoring
    order = np.argsort(ts)
    return X[order], y[order], ts[order]


if __name__ == "__main__":
    X, y, ts = load_secom()
    print(f"SECOM: X={X.shape}, fails={int(y.sum())}/{len(y)} "
          f"({y.mean():.1%}), span {ts.min()} → {ts.max()}")
