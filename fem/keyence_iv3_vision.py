"""
keyence_iv3_vision.py
---------------------
Physics-based model of the Keyence IV3 series machine vision system
for inline SiC dicing inspection.

Specifications modelled:
  - 5 Mpx CMOS sensor, ~0.5 µm/px (2× telecentric lens)
  - Up to 1,000 judgements/sec (real-time inline)
  - Binary pass/fail per die
  - Detects: cracks, missing dies, chipping (>5 µm), contamination

Usage:
    python keyence_iv3_vision.py          # run simulation + save plot
    python keyence_iv3_vision.py --no-plot
"""

import argparse
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


# ---------------------------------------------------------------------------
# 1.  IV3Spec dataclass
# ---------------------------------------------------------------------------
@dataclass
class IV3Spec:
    """Hardware specification of the Keyence IV3 machine vision system."""
    pixel_um: float = 0.5          # µm per pixel (2× telecentric lens)
    fps: int = 200                  # frames per second (inspection rate)
    fov_mm: tuple = (10.0, 10.0)   # field of view (width, height) in mm
    snr_min: float = 8.0            # minimum signal-to-noise ratio for reliable detection

    # Derived attributes
    @property
    def throughput_per_sec(self) -> int:
        """Maximum inspection judgements per second (hardware ceiling: 1000)."""
        return min(self.fps, 1000)

    @property
    def resolution_px(self) -> tuple:
        """Sensor resolution in pixels derived from fov and pixel pitch."""
        w = int(self.fov_mm[0] * 1e3 / self.pixel_um)
        h = int(self.fov_mm[1] * 1e3 / self.pixel_um)
        return (w, h)


# Default global spec instance
IV3_DEFAULT = IV3Spec()


# ---------------------------------------------------------------------------
# 2.  defect_detectability
# ---------------------------------------------------------------------------
def defect_detectability(
    defect_size_um: np.ndarray,
    defect_contrast: float = 0.8,
    tool: str = 'iv3'
) -> np.ndarray:
    """
    Probability of detecting a defect as a function of its physical size.

    Model: P(detect) = 0.5 * [1 + erf((size - threshold) / (sqrt(2) * sigma_psf))]
    This is the CDF of a Gaussian centred on the detection threshold.

    Parameters
    ----------
    defect_size_um : array-like
        Physical size of defects in micrometres.
    defect_contrast : float
        Optical contrast of the defect relative to background (0–1).
        Lower contrast reduces effective SNR and broadens the transition.
    tool : str
        'iv3'   — Keyence IV3 camera vision  (threshold 3 µm = 3× pixel pitch)
        'vkx'   — Keyence VK-X confocal       (threshold 0.5 µm, sub-pixel)
        'human' — Human visual inspector      (threshold 20 µm, high variability)

    Returns
    -------
    np.ndarray
        P(detect) for each defect size, shaped like defect_size_um.
    """
    defect_size_um = np.asarray(defect_size_um, dtype=float)

    tool_params = {
        'iv3':   {'threshold_um': 3.0,  'sigma_psf_um': 1.5},
        'vkx':   {'threshold_um': 0.5,  'sigma_psf_um': 0.3},
        'human': {'threshold_um': 20.0, 'sigma_psf_um': 8.0},
    }
    if tool not in tool_params:
        raise ValueError(f"Unknown tool '{tool}'. Choose from {list(tool_params.keys())}.")

    p = tool_params[tool]
    threshold = p['threshold_um']
    # Scale sigma by inverse contrast: lower contrast → harder to detect → wider transition
    sigma_eff = p['sigma_psf_um'] / max(defect_contrast, 0.05)

    p_detect = 0.5 * (1.0 + erf((defect_size_um - threshold) / (np.sqrt(2.0) * sigma_eff)))
    return p_detect


# ---------------------------------------------------------------------------
# 3.  InspectionResult dataclass
# ---------------------------------------------------------------------------
@dataclass
class InspectionResult:
    """Result of a single die inspection judgement."""
    die_id: int
    pass_fail: bool           # True = PASS, False = FAIL
    defect_type: Optional[str] = None   # 'chipping', 'crack', 'contamination', None
    defect_size_um: float = 0.0
    confidence: float = 1.0   # [0, 1] confidence of the judgement


# ---------------------------------------------------------------------------
# 4.  simulate_inline_inspection
# ---------------------------------------------------------------------------
def simulate_inline_inspection(
    chip_widths_um: np.ndarray,
    crack_lengths_um: np.ndarray,
    n_dies: int = 100,
    seed: Optional[int] = 42,
    alpha: float = 0.02,   # false-positive rate (good die → fail)
    beta: float = 0.05,    # false-negative rate  (bad  die → pass)
) -> List[InspectionResult]:
    """
    Simulate inline IV3 inspection for a batch of dies.

    Each die has an associated chip width and crack length drawn from the
    provided distributions.  Ground-truth failure rule:
        FAIL if chip_width > 5 µm  OR  crack_length > 3 µm

    Noise is injected via:
        alpha (false-positive): probability that a truly GOOD die is judged FAIL
        beta  (false-negative): probability that a truly BAD  die is judged PASS

    Parameters
    ----------
    chip_widths_um    : 1-D array of chipping widths, one per die
    crack_lengths_um  : 1-D array of crack lengths,   one per die
    n_dies            : number of dies to simulate
    seed              : random seed for reproducibility
    alpha             : false-positive rate
    beta              : false-negative rate

    Returns
    -------
    list[InspectionResult]
    """
    rng = np.random.default_rng(seed)

    chip_widths_um = np.asarray(chip_widths_um, dtype=float)
    crack_lengths_um = np.asarray(crack_lengths_um, dtype=float)

    # Broadcast / repeat arrays if shorter than n_dies
    if len(chip_widths_um) < n_dies:
        idx = rng.integers(0, len(chip_widths_um), size=n_dies)
        chip_widths_um = chip_widths_um[idx]
    else:
        chip_widths_um = chip_widths_um[:n_dies]

    if len(crack_lengths_um) < n_dies:
        idx = rng.integers(0, len(crack_lengths_um), size=n_dies)
        crack_lengths_um = crack_lengths_um[idx]
    else:
        crack_lengths_um = crack_lengths_um[:n_dies]

    results: List[InspectionResult] = []

    for i in range(n_dies):
        chip = chip_widths_um[i]
        crack = crack_lengths_um[i]

        # Ground-truth judgement
        chipping_fail = chip > 5.0
        crack_fail = crack > 3.0
        ground_truth_fail = chipping_fail or crack_fail

        # Dominant defect type for reporting
        if chipping_fail and crack_fail:
            defect_type = 'chipping+crack'
            defect_size = max(chip, crack)
        elif chipping_fail:
            defect_type = 'chipping'
            defect_size = float(chip)
        elif crack_fail:
            defect_type = 'crack'
            defect_size = float(crack)
        else:
            defect_type = None
            defect_size = max(chip, crack)  # largest feature even for passing die

        # Apply noise
        rand_val = rng.random()
        if ground_truth_fail:
            # True defective die
            observed_fail = rand_val >= beta  # beta = P(miss → pass)
        else:
            # True good die
            observed_fail = rand_val < alpha  # alpha = P(false alarm → fail)

        # Confidence: higher when defect is clearly above/below threshold
        if ground_truth_fail:
            margin = max(chip - 5.0, crack - 3.0, 0.0)
            confidence = min(0.5 + 0.1 * margin, 0.99)
        else:
            margin = min(5.0 - chip, 3.0 - crack)
            confidence = min(0.5 + 0.05 * max(margin, 0.0), 0.99)

        results.append(InspectionResult(
            die_id=i,
            pass_fail=not observed_fail,
            defect_type=defect_type if observed_fail else None,
            defect_size_um=defect_size,
            confidence=confidence,
        ))

    return results


# ---------------------------------------------------------------------------
# 5.  yield_from_inspection
# ---------------------------------------------------------------------------
def yield_from_inspection(results: List[InspectionResult]) -> dict:
    """
    Aggregate inspection results into yield statistics.

    Returns
    -------
    dict with keys:
        n_total    : total number of dies inspected
        n_pass     : dies judged PASS
        n_fail     : dies judged FAIL
        yield_pct  : n_pass / n_total × 100
        false_pos  : estimated false positives (good dies incorrectly failed)
        false_neg  : estimated false negatives (bad dies incorrectly passed)
    """
    n_total = len(results)
    n_pass = sum(1 for r in results if r.pass_fail)
    n_fail = n_total - n_pass

    # Estimate FP/FN from confidence proxy
    # FP: passed with low confidence (die near threshold, flagged as fail by error)
    # We tag as FP if pass=True but confidence < 0.55 (uncertain region)
    false_pos = sum(1 for r in results if not r.pass_fail and r.confidence < 0.55)
    false_neg = sum(1 for r in results if r.pass_fail and r.confidence < 0.55)

    return {
        'n_total':   n_total,
        'n_pass':    n_pass,
        'n_fail':    n_fail,
        'yield_pct': 100.0 * n_pass / n_total if n_total > 0 else 0.0,
        'false_pos': false_pos,
        'false_neg': false_neg,
    }


# ---------------------------------------------------------------------------
# 6.  iv3_vs_human_inspection
# ---------------------------------------------------------------------------
def iv3_vs_human_inspection(
    n_dies: int = 1000,
    true_defective_frac: float = 0.08,
    seed: int = 0,
) -> dict:
    """
    Compare IV3 automated vision vs human visual inspection.

    Parameters
    ----------
    n_dies               : number of dies in the lot
    true_defective_frac  : true fraction of defective dies in the lot
    seed                 : RNG seed

    Returns
    -------
    dict with sub-dicts 'iv3' and 'human', each containing:
        beta            : false-negative rate used
        throughput_per_s: judgements per second
        time_s          : total inspection time in seconds
        yield_accuracy  : fraction of judgements matching ground truth
        n_missed_defects: number of defective dies incorrectly passed
        n_false_alarms  : number of good dies incorrectly failed
    """
    rng = np.random.default_rng(seed)

    n_defective = int(n_dies * true_defective_frac)
    n_good = n_dies - n_defective

    def _run(beta: float, alpha: float, throughput: float) -> dict:
        correct = 0
        missed = 0
        false_alarm = 0

        # Defective dies
        for _ in range(n_defective):
            judged_fail = rng.random() >= beta
            if judged_fail:
                correct += 1
            else:
                missed += 1

        # Good dies
        for _ in range(n_good):
            judged_pass = rng.random() >= alpha
            if judged_pass:
                correct += 1
            else:
                false_alarm += 1

        accuracy = correct / n_dies
        time_s = n_dies / throughput

        return {
            'beta':             beta,
            'throughput_per_s': throughput,
            'time_s':           time_s,
            'yield_accuracy':   accuracy,
            'n_missed_defects': missed,
            'n_false_alarms':   false_alarm,
        }

    iv3_result = _run(beta=0.05, alpha=0.02, throughput=200.0)
    human_result = _run(beta=0.20, alpha=0.05, throughput=3.0)

    return {'iv3': iv3_result, 'human': human_result}


# ---------------------------------------------------------------------------
# 7.  plot_main — 4-panel figure
# ---------------------------------------------------------------------------
def plot_main(save_path: Optional[str] = None) -> str:
    """
    Generate a 4-panel figure:
      (A) Defect detectability curves: IV3 vs VK-X vs Human
      (B) Simulated wafer map: 10×10 die grid (pass/fail colour-coded)
      (C) Throughput vs false-negative rate tradeoff (ROC-like)
      (D) IV3 vs Human: yield accuracy bar chart

    Parameters
    ----------
    save_path : str or None
        File path to save the figure.  Defaults to OUT_DIR/keyence_iv3_vision.png.

    Returns
    -------
    str : absolute path of the saved figure
    """
    if save_path is None:
        os.makedirs(OUT_DIR, exist_ok=True)
        save_path = os.path.join(OUT_DIR, 'keyence_iv3_vision.png')

    rng = np.random.default_rng(7)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        'Keyence IV3 Machine Vision — SiC Dicing Inline Inspection Model',
        fontsize=14, fontweight='bold', y=0.98
    )

    # ------------------------------------------------------------------
    # (A) Defect detectability curves
    # ------------------------------------------------------------------
    ax_a = axes[0, 0]
    sizes = np.linspace(0, 60, 400)

    for tool, label, color, ls in [
        ('iv3',   'IV3 Camera (0.5 µm/px)',   '#1f77b4', '-'),
        ('vkx',   'VK-X Confocal (0.05 µm)',  '#2ca02c', '--'),
        ('human', 'Human Visual (~20 µm)',     '#d62728', ':'),
    ]:
        p = defect_detectability(sizes, defect_contrast=0.8, tool=tool)
        ax_a.plot(sizes, p, color=color, ls=ls, lw=2, label=label)

    ax_a.axvline(3.0,  color='#1f77b4', lw=0.8, alpha=0.5, ls='-.')
    ax_a.axvline(0.5,  color='#2ca02c', lw=0.8, alpha=0.5, ls='-.')
    ax_a.axvline(20.0, color='#d62728', lw=0.8, alpha=0.5, ls='-.')
    ax_a.axhline(0.5,  color='gray', lw=0.5, ls='--', alpha=0.4)

    ax_a.set_xlabel('Defect Size (µm)')
    ax_a.set_ylabel('P(detect)')
    ax_a.set_title('(A) Defect Detectability Curves')
    ax_a.legend(fontsize=8)
    ax_a.set_xlim(0, 60)
    ax_a.set_ylim(-0.02, 1.05)
    ax_a.grid(True, alpha=0.3)

    # Low-contrast overlay for IV3
    p_lc = defect_detectability(sizes, defect_contrast=0.3, tool='iv3')
    ax_a.fill_between(sizes, p_lc, defect_detectability(sizes, 0.8, 'iv3'),
                      alpha=0.15, color='#1f77b4', label='IV3 contrast 0.3–0.8')
    ax_a.legend(fontsize=7.5)

    # ------------------------------------------------------------------
    # (B) Simulated wafer map (10×10 die grid)
    # ------------------------------------------------------------------
    ax_b = axes[0, 1]
    grid_n = 10
    n_dies = grid_n * grid_n

    chip_w = rng.exponential(scale=3.5, size=n_dies)
    crack_l = rng.exponential(scale=1.8, size=n_dies)

    results = simulate_inline_inspection(chip_w, crack_l, n_dies=n_dies, seed=99)

    # Build colour matrix: green=pass, red=fail, orange=false positive
    color_map = np.zeros((grid_n, grid_n, 3))
    for r in results:
        row = r.die_id // grid_n
        col = r.die_id % grid_n
        if r.pass_fail:
            color_map[row, col] = [0.18, 0.65, 0.18]   # green
        else:
            if r.confidence < 0.55:
                color_map[row, col] = [1.0, 0.55, 0.0]  # orange = uncertain / false pos
            else:
                color_map[row, col] = [0.85, 0.12, 0.12]  # red = clear fail

    ax_b.imshow(color_map, interpolation='nearest', aspect='equal')

    # Annotate defect type on fail dies
    for r in results:
        if not r.pass_fail and r.defect_type:
            row = r.die_id // grid_n
            col = r.die_id % grid_n
            short = 'C' if 'crack' in r.defect_type else 'K'
            ax_b.text(col, row, short, ha='center', va='center',
                      fontsize=6, color='white', fontweight='bold')

    ax_b.set_xticks(range(grid_n))
    ax_b.set_yticks(range(grid_n))
    ax_b.set_xticklabels([str(i) for i in range(grid_n)], fontsize=7)
    ax_b.set_yticklabels([str(i) for i in range(grid_n)], fontsize=7)
    ax_b.set_title('(B) Simulated Wafer Map (10×10 Dies)', fontsize=10)
    ax_b.set_xlabel('Column')
    ax_b.set_ylabel('Row')

    # Legend
    legend_patches = [
        mpatches.Patch(color=[0.18, 0.65, 0.18], label='PASS'),
        mpatches.Patch(color=[0.85, 0.12, 0.12], label='FAIL (clear)'),
        mpatches.Patch(color=[1.0, 0.55, 0.0],   label='FAIL (uncertain/FP)'),
    ]
    ax_b.legend(handles=legend_patches, loc='lower right', fontsize=7,
                framealpha=0.8)

    stats = yield_from_inspection(results)
    ax_b.set_xlabel(
        f"Column  |  Yield = {stats['yield_pct']:.1f}%  "
        f"({stats['n_pass']} pass / {stats['n_fail']} fail)",
        fontsize=8
    )

    # ------------------------------------------------------------------
    # (C) Throughput vs false-negative rate tradeoff (ROC-like)
    # ------------------------------------------------------------------
    ax_c = axes[1, 0]

    beta_vals = np.linspace(0.01, 0.50, 80)
    # Throughput model: as beta increases, inspection is faster (less re-checks)
    # We use a simple inverse model: throughput = base / (1 + k*(1-beta))
    base_throughput_iv3   = 200.0
    base_throughput_human = 3.0
    k = 0.5  # re-check overhead factor

    tp_iv3   = base_throughput_iv3   / (1.0 + k * (1.0 - beta_vals))
    tp_human = base_throughput_human / (1.0 + k * (1.0 - beta_vals))

    ax_c.plot(beta_vals * 100, tp_iv3,   color='#1f77b4', lw=2, label='IV3 (camera)')
    ax_c.plot(beta_vals * 100, tp_human, color='#d62728', lw=2, ls='--', label='Human visual')

    # Mark operating points
    ax_c.scatter([5], [base_throughput_iv3 / (1.0 + k * 0.95)],
                 color='#1f77b4', s=80, zorder=5, label='IV3 operating point (β=5%)')
    ax_c.scatter([20], [base_throughput_human / (1.0 + k * 0.80)],
                 color='#d62728', s=80, marker='s', zorder=5,
                 label='Human operating point (β=20%)')

    ax_c.set_xlabel('False-Negative Rate β (%)')
    ax_c.set_ylabel('Throughput (dies/sec)')
    ax_c.set_title('(C) Throughput vs False-Negative Rate')
    ax_c.legend(fontsize=8)
    ax_c.grid(True, alpha=0.3)
    ax_c.set_xlim(0, 52)

    # ------------------------------------------------------------------
    # (D) IV3 vs Human: yield accuracy bar chart
    # ------------------------------------------------------------------
    ax_d = axes[1, 1]

    comparison = iv3_vs_human_inspection(n_dies=1000, true_defective_frac=0.08, seed=42)
    iv3_data   = comparison['iv3']
    human_data = comparison['human']

    labels = ['Yield Accuracy (%)', 'Missed Defects (%)', 'False Alarms (%)']
    n_dies_total = 1000

    iv3_vals = [
        iv3_data['yield_accuracy'] * 100,
        iv3_data['n_missed_defects'] / n_dies_total * 100,
        iv3_data['n_false_alarms']  / n_dies_total * 100,
    ]
    human_vals = [
        human_data['yield_accuracy'] * 100,
        human_data['n_missed_defects'] / n_dies_total * 100,
        human_data['n_false_alarms']  / n_dies_total * 100,
    ]

    x = np.arange(len(labels))
    width = 0.32

    bars_iv3   = ax_d.bar(x - width / 2, iv3_vals,   width, label='IV3 Camera',
                          color='#1f77b4', alpha=0.85)
    bars_human = ax_d.bar(x + width / 2, human_vals, width, label='Human Visual',
                          color='#d62728', alpha=0.85)

    # Value labels on bars
    for bar in list(bars_iv3) + list(bars_human):
        h = bar.get_height()
        ax_d.text(bar.get_x() + bar.get_width() / 2.0, h + 0.3,
                  f'{h:.1f}', ha='center', va='bottom', fontsize=8)

    throughput_ratio = iv3_data['throughput_per_s'] / human_data['throughput_per_s']
    ax_d.set_title(
        f'(D) IV3 vs Human Yield Accuracy\n'
        f'(Throughput ratio: IV3 = {throughput_ratio:.0f}× faster)',
        fontsize=9
    )
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(labels, fontsize=8)
    ax_d.set_ylabel('Percentage (%)')
    ax_d.legend(fontsize=9)
    ax_d.set_ylim(0, max(max(iv3_vals), max(human_vals)) * 1.25)
    ax_d.grid(axis='y', alpha=0.3)

    # ------------------------------------------------------------------
    # Finalise and save
    # ------------------------------------------------------------------
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return os.path.abspath(save_path)


# ---------------------------------------------------------------------------
# 8.  main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Keyence IV3 inline SiC dicing inspection model'
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Skip plot generation (run simulation only)'
    )
    parser.add_argument(
        '--n-dies', type=int, default=200,
        help='Number of dies to simulate (default: 200)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    args = parser.parse_args()

    print('=== Keyence IV3 Machine Vision — SiC Dicing Inspection ===')

    # Show spec
    spec = IV3Spec()
    print(f'\nIV3Spec:')
    print(f'  pixel pitch  : {spec.pixel_um} µm/px')
    print(f'  FPS          : {spec.fps}')
    print(f'  FOV          : {spec.fov_mm[0]} × {spec.fov_mm[1]} mm')
    print(f'  Sensor res   : {spec.resolution_px[0]} × {spec.resolution_px[1]} px')
    print(f'  Min SNR      : {spec.snr_min} dB')
    print(f'  Max throughput: {spec.throughput_per_sec} judgements/s')

    # Simulate inspection
    rng = np.random.default_rng(args.seed)
    chip_w  = rng.exponential(scale=4.0, size=args.n_dies)
    crack_l = rng.exponential(scale=2.0, size=args.n_dies)

    results = simulate_inline_inspection(
        chip_widths_um=chip_w,
        crack_lengths_um=crack_l,
        n_dies=args.n_dies,
        seed=args.seed,
    )

    stats = yield_from_inspection(results)
    print(f'\nInspection results ({args.n_dies} dies):')
    print(f'  PASS        : {stats["n_pass"]}')
    print(f'  FAIL        : {stats["n_fail"]}')
    print(f'  Yield       : {stats["yield_pct"]:.1f}%')
    print(f'  Est. FP     : {stats["false_pos"]}')
    print(f'  Est. FN     : {stats["false_neg"]}')

    # Compare IV3 vs human
    comp = iv3_vs_human_inspection(n_dies=1000, seed=args.seed)
    print('\nIV3 vs Human inspection (1000 dies, 8% defective):')
    for name, d in comp.items():
        print(f'  [{name.upper():5s}]  accuracy={d["yield_accuracy"]*100:.1f}%  '
              f'missed={d["n_missed_defects"]}  FP={d["n_false_alarms"]}  '
              f'throughput={d["throughput_per_s"]:.0f}/s  '
              f'time={d["time_s"]:.1f}s')

    if not args.no_plot:
        path = plot_main()
        print(f'\nFigure saved to: {path}')
    else:
        print('\n[--no-plot] Skipping figure generation.')


if __name__ == '__main__':
    main()
