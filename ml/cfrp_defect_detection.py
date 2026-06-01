"""
CFRP Defect Detection — ML Models
===================================
GP-based anomaly detection + multi-class defect classifier
using features extracted from multi-modal NDT signals.

Pipeline:
    NDT signals (UT + Thermo + ECT + AE + X-ray)
        ↓ feature extraction
    Feature vector (12 dims)
        ↓
    Stage 1: GP anomaly score  (healthy vs defective)
    Stage 2: Defect classifier (delamination / void / fiber_break / impact / matrix_crack)
    Stage 3: Sizing estimator  (depth + diameter regression)

Usage:
    python ml/cfrp_defect_detection.py
    python ml/cfrp_defect_detection.py --n-samples 200
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fem.cfrp_defect_model import (
    multimodal_ndt, defect_params,
    ut_ascan, thermography_response,
    ect_impedance_scan, ae_event_simulation,
    xray_transmission,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)

DEFECT_TYPES = ["delamination", "void", "fiber_break", "impact", "matrix_crack"]


# ════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ════════════════════════════════════════════════════════════════════════════

def extract_features(defect_type: str, depth_mm: float,
                     diameter_mm: float = None,
                     noise_seed: int = None) -> np.ndarray:
    """
    Extract 12-dim feature vector from all 5 NDT methods.

    Features:
        [0]  UT peak amplitude at defect echo
        [1]  UT echo ToF [µs]
        [2]  UT backwall attenuation ratio
        [3]  Thermo max ΔT [°C]  (log-scaled)
        [4]  Thermo peak time [s]  (log-scaled)
        [5]  ECT |ΔZ| peak magnitude
        [6]  ECT skin depth [mm]
        [7]  AE hit rate (hits/100 cycles)
        [8]  AE b-value
        [9]  X-ray contrast peak [%]
        [10] Defect diameter (from UT C-scan FWHM) [mm]
        [11] Estimated depth from UT ToF [mm]
    """
    rng = np.random.default_rng(noise_seed)

    defect = defect_params(defect_type, depth_mm)
    if diameter_mm is not None:
        defect["diameter_mm"] = diameter_mm

    # Measurement noise (σ fraction)
    noise_frac = 0.05

    def noisy(val, frac=noise_frac):
        return float(val) * (1 + rng.normal(0, frac))

    # UT
    ut = ut_ascan(depth_mm, defect)
    idx_def  = int(ut["t_defect_us"] / (ut["t_us"][1] - ut["t_us"][0]))
    idx_def  = min(idx_def, len(ut["amplitude"]) - 1)
    amp_def  = noisy(abs(ut["amplitude"][idx_def]))
    amp_bw   = noisy(abs(ut["amplitude"][
        min(int(ut["t_backwall_us"] / (ut["t_us"][1] - ut["t_us"][0])),
            len(ut["amplitude"]) - 1)
    ]))
    tof_def  = noisy(ut["t_defect_us"])

    # Thermography
    thm = thermography_response(defect)
    dT_max = noisy(max(thm["max_delta_T_C"], 1e-6))
    t_peak = noisy(max(thm["t_peak_s"], 1e-6))

    # ECT
    ect = ect_impedance_scan(defect)
    dz_peak = noisy(float(ect["dZ_mag"].max()))
    skin_d  = noisy(ect["skin_depth_mm"])

    # AE
    ae = ae_event_simulation(defect, n_cycles=500, seed=int(noise_seed or 0) + 1)
    hit_rate = ae["n_hits"] / 5.0  # per 100 cycles
    b_val    = ae["b_value"] if not np.isnan(ae["b_value"]) else 1.5

    # X-ray
    xr = xray_transmission(defect)
    xr_contrast = noisy(float(xr["contrast"].max()) * 100)

    # Derived features
    v_mm_us = 2.9   # mm/µs
    depth_est = tof_def * v_mm_us / 2.0

    # C-scan FWHM (proxy for defect size)
    from fem.cfrp_defect_model import ut_cscan
    cs = ut_cscan(defect)
    row_mid = cs["amplitude"][cs["amplitude"].shape[0] // 2, :]
    peak    = row_mid.max()
    fwhm_px = max(np.sum(row_mid > peak / 2), 1)
    scan_range = cs["X"].max() - cs["X"].min()
    fwhm_mm = fwhm_px / cs["X"].shape[0] * scan_range

    features = np.array([
        amp_def,
        tof_def,
        amp_bw / (amp_def + 1e-9),
        np.log(dT_max + 1e-9),
        np.log(t_peak + 1e-9),
        dz_peak,
        skin_d,
        hit_rate,
        b_val,
        xr_contrast,
        fwhm_mm,
        depth_est,
    ])
    return features


def build_dataset(n_per_class: int = 30,
                  include_healthy: bool = True,
                  seed: int = 0) -> tuple:
    """
    Generate synthetic labelled dataset for all defect types.

    Returns X (N, 12), y_label (N,), y_depth (N,), y_diam (N,)
    """
    rng = np.random.default_rng(seed)
    X_list, labels, depths, diams = [], [], [], []

    # Healthy samples (no defect → simulated with tiny depth/reflectivity)
    if include_healthy:
        for i in range(n_per_class):
            d     = rng.uniform(3.5, 4.5)    # "defect" at back wall
            diam  = rng.uniform(0.3, 0.8)    # tiny
            feat  = extract_features("void", d, diam, noise_seed=seed + i)
            feat[0] *= 0.1    # attenuate defect echo to near zero
            feat[5] *= 0.05   # ECT near zero
            X_list.append(feat)
            labels.append("healthy")
            depths.append(0.0)
            diams.append(0.0)

    for defect_type in DEFECT_TYPES:
        for i in range(n_per_class):
            d    = rng.uniform(0.5, 3.0)
            base_diam = defect_params(defect_type, d)["diameter_mm"]
            diam = base_diam * rng.uniform(0.6, 1.5)
            feat = extract_features(defect_type, d, diam,
                                    noise_seed=seed + i * 100 + DEFECT_TYPES.index(defect_type))
            X_list.append(feat)
            labels.append(defect_type)
            depths.append(d)
            diams.append(diam)

    X = np.array(X_list)
    return X, np.array(labels), np.array(depths), np.array(diams)


# ════════════════════════════════════════════════════════════════════════════
# Stage 1: GP anomaly detection
# ════════════════════════════════════════════════════════════════════════════

def train_anomaly_detector(X_healthy: np.ndarray) -> tuple:
    """
    GP one-class anomaly detector trained on healthy samples only.
    Anomaly score = negative GP log-likelihood under healthy distribution.

    Returns (gp, scaler) for inference.
    """
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X_healthy)

    kernel = ConstantKernel(1.0) * Matern(nu=1.5) + WhiteKernel(noise_level=0.1)
    gp     = GaussianProcessRegressor(kernel=kernel,
                                      n_restarts_optimizer=5,
                                      normalize_y=True)
    # Fit GP to reconstruct each feature from others (leave-one-out proxy)
    # Simplified: fit to predict feature[0] from features[1:] as anomaly proxy
    gp.fit(Xs[:, 1:], Xs[:, 0])
    return gp, scaler


def anomaly_score(gp, scaler, X_new: np.ndarray) -> np.ndarray:
    """
    Return anomaly scores for new samples.
    Higher score → more anomalous.
    """
    Xs   = scaler.transform(X_new)
    mu, sigma = gp.predict(Xs[:, 1:], return_std=True)
    resid = np.abs(Xs[:, 0] - mu) / (sigma + 1e-9)
    return resid


# ════════════════════════════════════════════════════════════════════════════
# Stage 2: Multi-class defect classifier
# ════════════════════════════════════════════════════════════════════════════

def train_classifier(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Random Forest classifier for defect type.
    Returns (clf, scaler, label_encoder).
    """
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    le     = LabelEncoder()
    y_enc  = le.fit_transform(y)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
    )
    clf.fit(Xs, y_enc)
    return clf, scaler, le


# ════════════════════════════════════════════════════════════════════════════
# Stage 3: Defect sizing (depth + diameter regression)
# ════════════════════════════════════════════════════════════════════════════

def train_sizing_gp(X: np.ndarray,
                    y_depth: np.ndarray,
                    y_diam: np.ndarray) -> tuple:
    """
    Two GP regressors for depth and diameter estimation.
    Returns (gp_depth, gp_diam, scaler).
    """
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=0.01)

    gp_d = GaussianProcessRegressor(kernel=kernel.clone_with_theta(kernel.theta),
                                     n_restarts_optimizer=5, normalize_y=True)
    gp_w = GaussianProcessRegressor(kernel=kernel.clone_with_theta(kernel.theta),
                                     n_restarts_optimizer=5, normalize_y=True)
    gp_d.fit(Xs, y_depth)
    gp_w.fit(Xs, y_diam)
    return gp_d, gp_w, scaler


# ════════════════════════════════════════════════════════════════════════════
# Evaluation + visualisation
# ════════════════════════════════════════════════════════════════════════════

def evaluate_pipeline(n_per_class: int = 40, seed: int = 0) -> dict:
    print(f"[*] Generating dataset ({n_per_class} samples/class) …")
    X, y_label, y_depth, y_diam = build_dataset(n_per_class, seed=seed)

    # Split healthy vs defective
    healthy_mask  = (y_label == "healthy")
    defective_mask = ~healthy_mask

    X_h = X[healthy_mask]
    X_d = X[defective_mask]
    y_d = y_label[defective_mask]
    d_d = y_depth[defective_mask]
    w_d = y_diam[defective_mask]

    print(f"  Healthy: {X_h.shape[0]}  Defective: {X_d.shape[0]}")
    print(f"  Defect classes: {np.unique(y_d)}")

    # ── Stage 1: anomaly ─────────────────────────────────────────────────
    print("[*] Training anomaly detector …")
    gp_anom, sc_anom = train_anomaly_detector(X_h)
    score_h = anomaly_score(gp_anom, sc_anom, X_h)
    score_d = anomaly_score(gp_anom, sc_anom, X_d)
    threshold = np.percentile(score_h, 95)
    tp = np.mean(score_d > threshold)
    fp = np.mean(score_h > threshold)
    print(f"  Threshold={threshold:.2f}  Detection rate={tp*100:.0f}%  "
          f"False-alarm={fp*100:.0f}%")

    # ── Stage 2: classifier ───────────────────────────────────────────────
    print("[*] Training defect classifier (RF) …")
    clf, sc_clf, le = train_classifier(X_d, y_d)
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    Xds = sc_clf.transform(X_d)
    y_enc = le.transform(y_d)
    cv_scores = cross_val_score(clf, Xds, y_enc, cv=cv, scoring="accuracy")
    print(f"  5-fold CV accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    # ── Stage 3: sizing ───────────────────────────────────────────────────
    print("[*] Training sizing GP …")
    gp_dep, gp_dia, sc_sz = train_sizing_gp(X_d, d_d, w_d)
    Xds2 = sc_sz.transform(X_d)
    dep_pred, _ = gp_dep.predict(Xds2, return_std=True)
    dia_pred, _ = gp_dia.predict(Xds2, return_std=True)
    rmse_dep = float(np.sqrt(mean_squared_error(d_d, dep_pred)))
    rmse_dia = float(np.sqrt(mean_squared_error(w_d, dia_pred)))
    print(f"  Depth RMSE = {rmse_dep:.3f} mm   Diameter RMSE = {rmse_dia:.3f} mm")

    return {
        "score_healthy":  score_h,
        "score_defective": score_d,
        "threshold": threshold,
        "detection_rate": tp,
        "false_alarm": fp,
        "cv_accuracy": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "clf": clf, "sc_clf": sc_clf, "le": le,
        "gp_dep": gp_dep, "gp_dia": gp_dia, "sc_sz": sc_sz,
        "X_d": X_d, "y_d": y_d,
        "d_d": d_d, "w_d": w_d,
        "dep_pred": dep_pred, "dia_pred": dia_pred,
        "rmse_dep": rmse_dep, "rmse_dia": rmse_dia,
        "feature_importance": clf.feature_importances_,
    }


FEATURE_NAMES = [
    "UT amp", "UT ToF", "BW ratio",
    "ln(ΔT)", "ln(t_peak)",
    "ECT |ΔZ|", "skin depth",
    "AE rate", "AE b-val",
    "X-ray C%",
    "UT FWHM", "depth est",
]


def plot_results(res: dict, save: bool = True) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    palette = {
        "healthy":      "#2166ac",
        "delamination": "#d62728",
        "void":         "#ff7f0e",
        "fiber_break":  "#2ca02c",
        "impact":       "#9467bd",
        "matrix_crack": "#8c564b",
    }

    # ── Panel 1: Anomaly score distribution ──────────────────────────────────
    ax = axes[0, 0]
    ax.hist(res["score_healthy"],   bins=20, alpha=0.7, color="#2166ac",
            label=f"Healthy (N={len(res['score_healthy'])})")
    ax.hist(res["score_defective"], bins=20, alpha=0.7, color="#d62728",
            label=f"Defective (N={len(res['score_defective'])})")
    ax.axvline(res["threshold"], color="k", ls="--", lw=1.5,
               label=f"Threshold (95th %ile)")
    ax.set_xlabel("Anomaly score"); ax.set_ylabel("Count")
    ax.set_title(f"Stage 1: Anomaly Detection\n"
                 f"DR={res['detection_rate']*100:.0f}%  "
                 f"FA={res['false_alarm']*100:.0f}%")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    # ── Panel 2: Confusion matrix ─────────────────────────────────────────────
    ax = axes[0, 1]
    le  = res["le"]
    clf = res["clf"]
    Xds = res["sc_clf"].transform(res["X_d"])
    y_pred_enc = clf.predict(Xds)
    y_true_enc = le.transform(res["y_d"])
    cm  = confusion_matrix(y_true_enc, y_pred_enc)
    im  = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(le.classes_)))
    ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels([c[:7] for c in le.classes_], rotation=35, fontsize=7)
    ax.set_yticklabels([c[:7] for c in le.classes_], fontsize=7)
    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)
    ax.set_title(f"Stage 2: Classifier Confusion Matrix\n"
                 f"CV acc={res['cv_accuracy']*100:.1f}%±{res['cv_std']*100:.1f}%")
    plt.colorbar(im, ax=ax)

    # ── Panel 3: Feature importance ───────────────────────────────────────────
    ax = axes[0, 2]
    imp   = res["feature_importance"]
    order = np.argsort(imp)[::-1]
    ax.barh(range(12), imp[order], color="#1f77b4", alpha=0.8)
    ax.set_yticks(range(12))
    ax.set_yticklabels([FEATURE_NAMES[i] for i in order], fontsize=8)
    ax.set_xlabel("Importance"); ax.set_title("Feature Importance (RF)")
    ax.grid(alpha=0.2, axis="x")

    # ── Panel 4: Depth estimation ─────────────────────────────────────────────
    ax = axes[1, 0]
    for dt in DEFECT_TYPES:
        mask = res["y_d"] == dt
        ax.scatter(res["d_d"][mask], res["dep_pred"][mask],
                   color=palette.get(dt, "gray"), s=30, alpha=0.7, label=dt[:7])
    lim = [0, max(res["d_d"].max(), res["dep_pred"].max()) * 1.1]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("True depth [mm]"); ax.set_ylabel("Predicted depth [mm]")
    ax.set_title(f"Stage 3: Depth Sizing\nRMSE={res['rmse_dep']:.3f} mm")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.2)

    # ── Panel 5: Diameter estimation ─────────────────────────────────────────
    ax = axes[1, 1]
    for dt in DEFECT_TYPES:
        mask = res["y_d"] == dt
        ax.scatter(res["w_d"][mask], res["dia_pred"][mask],
                   color=palette.get(dt, "gray"), s=30, alpha=0.7, label=dt[:7])
    lim2 = [0, max(res["w_d"].max(), res["dia_pred"].max()) * 1.1]
    ax.plot(lim2, lim2, "k--", lw=1)
    ax.set_xlabel("True diameter [mm]"); ax.set_ylabel("Predicted diameter [mm]")
    ax.set_title(f"Stage 3: Diameter Sizing\nRMSE={res['rmse_dia']:.3f} mm")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.2)

    # ── Panel 6: Summary table ────────────────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    table_data = [
        ["Stage", "Method", "Score"],
        ["1 Anomaly", "GP 1-class", f"DR={res['detection_rate']*100:.0f}%"],
        ["2 Class", "Random Forest", f"CV={res['cv_accuracy']*100:.1f}%"],
        ["3 Depth", "GP regression", f"RMSE={res['rmse_dep']:.3f}mm"],
        ["3 Diam.", "GP regression", f"RMSE={res['rmse_dia']:.3f}mm"],
    ]
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.3, 2.0)
    ax.set_title("Pipeline Summary", fontsize=11)

    fig.suptitle("CFRP Defect Detection — Multi-modal NDT + ML Pipeline",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save:
        path = os.path.join(OUT_DIR, "cfrp_defect_detection.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [✓] Saved: {path}")
        plt.close()
        return path
    plt.show()
    return ""


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="CFRP Defect Detection ML Pipeline")
    ap.add_argument("--n-samples", type=int, default=40,
                    help="Training samples per defect class (default: 40)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    res = evaluate_pipeline(n_per_class=args.n_samples, seed=args.seed)

    if not args.no_plot:
        plot_results(res, save=True)

    print(f"\n{'═'*55}")
    print(f"  CFRP Defect Detection Summary")
    print(f"{'═'*55}")
    print(f"  Detection rate  : {res['detection_rate']*100:.0f}%")
    print(f"  False alarm     : {res['false_alarm']*100:.0f}%")
    print(f"  Classifier acc  : {res['cv_accuracy']*100:.1f}% ± {res['cv_std']*100:.1f}%")
    print(f"  Depth RMSE      : {res['rmse_dep']:.3f} mm")
    print(f"  Diameter RMSE   : {res['rmse_dia']:.3f} mm")
    print(f"\n  Top features:")
    order = np.argsort(res["feature_importance"])[::-1]
    for i in order[:5]:
        print(f"    {FEATURE_NAMES[i]:<15} {res['feature_importance'][i]:.3f}")


if __name__ == "__main__":
    main()
