"""
Sensor Fusion — Multi-Modal Anomaly Detection for SiC Dicing.

In a real dicing saw, multiple sensors are available simultaneously:
    1. Spindle current [A]    → blade load / wear indicator
    2. Vibration RMS [g]      → chatter / blade fracture indicator
    3. Coolant temperature [°C] → thermal overload
    4. Acoustic emission [dB] → crack propagation indicator
    5. Camera chipping [µm]   → direct quality measurement (intermittent)

Fusion strategies:
    A. Weighted Average: fixed weights per sensor reliability
    B. Kalman Filter: dynamic state estimation (best for drift)
    C. GP-based: predict chipping from fused sensor readings
    D. Majority Vote: anomaly if K/N sensors agree

This module implements strategies A, B, D and demonstrates how fusing
multiple sensors reduces false-positive rate vs. any single sensor.

Usage:
    python ml/sensor_fusion.py --demo
    python ml/sensor_fusion.py --demo --plot
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

# Sensor noise model (realistic parameters for DISCO DAD3350)
SENSOR_CONFIG = {
    "spindle_current_A":    {"std": 0.05, "weight": 0.25, "drift_per_wafer": 0.002},
    "vibration_rms_g":      {"std": 0.03, "weight": 0.25, "drift_per_wafer": 0.003},
    "coolant_temp_C":       {"std": 0.5,  "weight": 0.15, "drift_per_wafer": 0.01},
    "acoustic_emission_dB": {"std": 1.0,  "weight": 0.20, "drift_per_wafer": 0.005},
    "camera_chip_um":       {"std": 0.8,  "weight": 0.15, "drift_per_wafer": 0.0,
                              "available_prob": 0.3},  # intermittent (sampling)
}

# Normal operating points (fresh blade, nominal recipe)
NOMINAL = {
    "spindle_current_A":    2.5,
    "vibration_rms_g":      0.15,
    "coolant_temp_C":       22.0,
    "acoustic_emission_dB": 45.0,
    "camera_chip_um":       4.5,
}

# Alarm thresholds (3σ from nominal)
THRESHOLDS = {k: NOMINAL[k] + 3 * v["std"] * 10 for k, v in SENSOR_CONFIG.items()}


# ── Sensor simulation ─────────────────────────────────────────────────────────

def simulate_sensors(n_wafers=100, n_anomaly=15, blade_wear_alpha=2.0,
                      n_life=200, seed=42) -> pd.DataFrame:
    """
    Simulate multi-sensor stream with blade wear drift + random anomaly spikes.
    Returns DataFrame with columns: wafer_id, sensor readings, true_chip_um.
    """
    rng = np.random.default_rng(seed)
    rows = []

    # True chipping from blade wear model
    from optimization.blade_wear import BladeWearModel
    wear = BladeWearModel(chip_0=NOMINAL["camera_chip_um"], n_life=n_life,
                           alpha=blade_wear_alpha, usl=15.0)

    # Random anomaly wafers
    anom_wafers = set(rng.choice(range(20, n_wafers), size=n_anomaly, replace=False))

    for n in range(n_wafers):
        true_chip = wear.predict_chip(n)
        # Sensor values scale with true_chip (physical relationship)
        scale = true_chip / NOMINAL["camera_chip_um"]
        is_anom = n in anom_wafers
        anom_scale = rng.uniform(1.5, 2.5) if is_anom else 1.0

        row = {"wafer_id": n, "true_chip_um": true_chip, "is_anomaly": is_anom}
        for sensor, cfg in SENSOR_CONFIG.items():
            base  = NOMINAL[sensor] * (1 + (scale - 1) * 0.5) * anom_scale
            noise = rng.normal(0, cfg["std"])
            available = True
            if "available_prob" in cfg:
                available = rng.random() < cfg["available_prob"]
            row[sensor]         = float(base + noise) if available else np.nan
            row[f"{sensor}_ok"] = available
        rows.append(row)

    return pd.DataFrame(rows)


# ── Fusion strategies ─────────────────────────────────────────────────────────

def strategy_A_weighted(row: pd.Series, z_thresh=3.0) -> tuple[bool, float]:
    """Weighted average of per-sensor z-scores."""
    total_w = 0.0
    z_avg   = 0.0
    for sensor, cfg in SENSOR_CONFIG.items():
        val = row.get(sensor, np.nan)
        if np.isnan(val):
            continue
        z = abs(val - NOMINAL[sensor]) / max(cfg["std"] * 3, 1e-9)
        z_avg   += cfg["weight"] * z
        total_w += cfg["weight"]
    if total_w == 0:
        return False, 0.0
    z_avg /= total_w
    return z_avg > z_thresh, float(z_avg)


def strategy_B_kalman(data: pd.DataFrame, Q=0.01, R=0.5) -> pd.Series:
    """
    1D Kalman filter on spindle_current_A as primary chip proxy.
    Returns filtered values.
    """
    col = "spindle_current_A"
    x_hat = NOMINAL[col]
    P     = 1.0
    filtered = []
    for val in data[col].fillna(method="ffill").values:
        # Predict
        x_hat_minus = x_hat
        P_minus     = P + Q
        # Update
        K     = P_minus / (P_minus + R)
        x_hat = x_hat_minus + K * (val - x_hat_minus)
        P     = (1 - K) * P_minus
        filtered.append(x_hat)
    return pd.Series(filtered, index=data.index)


def strategy_D_majority(row: pd.Series, z_thresh=2.5, k=2) -> tuple[bool, int]:
    """Anomaly if k or more sensors individually exceed threshold."""
    votes = 0
    for sensor, cfg in SENSOR_CONFIG.items():
        val = row.get(sensor, np.nan)
        if np.isnan(val):
            continue
        z = abs(val - NOMINAL[sensor]) / max(cfg["std"] * 3, 1e-9)
        if z > z_thresh:
            votes += 1
    return votes >= k, votes


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_strategy(preds: np.ndarray, truth: np.ndarray) -> dict:
    tp = ((preds == 1) & (truth == 1)).sum()
    fp = ((preds == 1) & (truth == 0)).sum()
    fn = ((preds == 0) & (truth == 1)).sum()
    tn = ((preds == 0) & (truth == 0)).sum()
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    fpr       = fp / (fp + tn + 1e-9)
    return {"precision": precision, "recall": recall, "F1": f1, "FPR": fpr,
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_fusion(data: pd.DataFrame, results: dict, out_dir=RESULTS):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Sensor streams
    sensors_to_plot = ["spindle_current_A", "vibration_rms_g", "acoustic_emission_dB"]
    colors = ["#2166ac", "#d62728", "#2ca02c"]
    for i, (sensor, color) in enumerate(zip(sensors_to_plot, colors)):
        ax = axes[0, 0] if i == 0 else (axes[0, 1] if i == 1 else axes[1, 0])
        vals = data[sensor].values
        ax.plot(vals, color=color, lw=1.5, alpha=0.85, label=sensor.replace("_", " "))
        ax.axhline(THRESHOLDS[sensor], color=color, ls="--", lw=1.0, alpha=0.6)
        anom_idx = data.index[data["is_anomaly"]].tolist()
        ax.scatter(anom_idx, vals[anom_idx], color="red", s=60, zorder=5,
                   marker="x", lw=2, label="True anomaly")
        ax.set_title(sensor.replace("_", " ").title(), fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.2)

    # Strategy comparison
    ax4 = axes[1, 1]
    strategies = list(results.keys())
    f1_vals = [results[s]["F1"] for s in strategies]
    fpr_vals = [results[s]["FPR"] for s in strategies]
    bar_colors = ["#2166ac", "#d62728", "#2ca02c", "#f97316"]
    bars = ax4.bar(strategies, f1_vals, color=bar_colors[:len(strategies)], alpha=0.8,
                   edgecolor="k", lw=0.6)
    for bar, fpr in zip(bars, fpr_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"FPR={fpr:.0%}", ha="center", fontsize=8)
    ax4.set_ylim(0, 1.15)
    ax4.set_ylabel("F1 Score"); ax4.set_title("Strategy Comparison (F1, FPR)")
    ax4.axhline(0.9, color="gray", ls=":", lw=1.0, label="F1=0.9 target")
    ax4.legend(fontsize=8); ax4.grid(alpha=0.2, axis="y")

    fig.suptitle(
        "Sensor Fusion — Multi-Modal Anomaly Detection\n"
        "Spindle current / Vibration / Acoustic emission / Camera chip",
        fontsize=10,
    )
    plt.tight_layout()
    out = os.path.join(out_dir, "sensor_fusion.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Sensor fusion plot → {out}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",    action="store_true")
    parser.add_argument("--plot",    action="store_true")
    parser.add_argument("--n",       type=int, default=100)
    parser.add_argument("--anomaly", type=int, default=15)
    args = parser.parse_args()

    if not args.demo and not args.plot:
        args.demo = True

    print(f"[*] Simulating {args.n} wafers ({args.anomaly} anomalous)...")
    data = simulate_sensors(n_wafers=args.n, n_anomaly=args.anomaly)
    truth = data["is_anomaly"].values.astype(int)

    # Single-sensor baselines
    results = {}
    for sensor in ["spindle_current_A", "vibration_rms_g", "acoustic_emission_dB"]:
        preds = np.array([
            abs(row[sensor] - NOMINAL[sensor]) / max(SENSOR_CONFIG[sensor]["std"]*3, 1e-9) > 3.0
            for _, row in data.iterrows()
        ]).astype(int)
        results[sensor.split("_")[0]] = evaluate_strategy(preds, truth)

    # Strategy A: weighted fusion
    preds_A = np.array([strategy_A_weighted(row)[0] for _, row in data.iterrows()]).astype(int)
    results["Weighted\nFusion"] = evaluate_strategy(preds_A, truth)

    # Strategy D: majority vote
    preds_D = np.array([strategy_D_majority(row)[0] for _, row in data.iterrows()]).astype(int)
    results["Majority\nVote"] = evaluate_strategy(preds_D, truth)

    print("\n[*] Strategy comparison:")
    print(f"  {'Strategy':>15}  {'F1':>6}  {'Precision':>10}  {'Recall':>8}  FPR")
    for name, m in results.items():
        print(f"  {name.replace(chr(10),'_'):>15}  {m['F1']:.3f}  "
              f"{m['precision']:.3f}       {m['recall']:.3f}   {m['FPR']:.3f}")

    best = max(results, key=lambda k: results[k]["F1"])
    print(f"\n  Best strategy: {best.replace(chr(10),' ')}  "
          f"(F1={results[best]['F1']:.3f})")
    print(f"  Single-sensor false positive rate: "
          f"{results['spindle']['FPR']:.1%} → Fusion: {results['Weighted'+chr(10)+'Fusion']['FPR']:.1%}")

    if args.plot:
        plot_fusion(data, results)


if __name__ == "__main__":
    main()
