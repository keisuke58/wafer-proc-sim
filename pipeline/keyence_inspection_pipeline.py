"""
keyence_inspection_pipeline.py
End-to-end Keyence inspection pipeline for SiC wafer dicing.

Instruments integrated:
  Step 1: LJ-X8080 — pre-dicing wafer warp check + post-dicing kerf width measurement
  Step 2: IV3      — inline pass/fail inspection at dicing speed
  Step 3: VK-X    — offline chipping width measurement (Grade A reference)
"""

import sys
import os
import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, '/home/nishioka/git/wafer-proc-sim')

OUT_DIR = '/home/nishioka/git/wafer-proc-sim/results'

# ---------------------------------------------------------------------------
# Instrument constants (inline — no circular imports)
# ---------------------------------------------------------------------------

# LJ-X8080 laser profiler
LJX_WARP_THRESHOLD_UM = 50.0       # µm — quarantine if wafer warp exceeds this
LJX_KERF_SIGMA_UM = 0.3            # µm measurement noise for kerf width

# IV3 vision sensor (inline pass/fail)
IV3_CHIP_THRESHOLD_UM = 5.0        # µm
IV3_CRACK_THRESHOLD_UM = 3.0       # µm
IV3_FALSE_NEGATIVE_RATE = 0.05     # 5% of true-fail dice pass IV3

# VK-X laser confocal (Grade A reference)
VKX_SIGMA_UM = 0.05                # µm — high-precision measurement noise
VKX_SAMPLE_EVERY = 10              # sample 1 in 10 wafers

# Dicing process parameters
DICE_CHIP_MEAN_UM = 5.0
DICE_CHIP_STD_UM = 1.5
DICE_CRACK_MEAN_UM = 1.0
DICE_CRACK_STD_UM = 0.5
DICE_WARP_MEAN_UM = 30.0
DICE_WARP_STD_UM = 15.0

DIES_PER_WAFER = 200               # nominal die count per 150 mm SiC wafer

# Cost model (JPY)
COST_DICING_PER_WAFER_JPY = 8_000
COST_IV3_INSPECT_PER_DIE_JPY = 5
COST_VKX_INSPECT_PER_WAFER_JPY = 3_000
COST_QUARANTINE_PER_WAFER_JPY = 1_000

# Throughput (wafers per hour)
TPH_VKXONLY = 2.0    # slow: 100% VK-X sampling
TPH_IV3ONLY = 12.0   # fast: inline only
TPH_COMBINED = 10.0  # IV3 + 10% VK-X sampling


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WaferResult:
    wafer_id: int
    quarantined: bool
    warp_um: float
    # Per-die arrays (empty if quarantined)
    chip_widths_um: np.ndarray = field(default_factory=lambda: np.array([]))
    crack_widths_um: np.ndarray = field(default_factory=lambda: np.array([]))
    iv3_pass: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    vkx_measured: bool = False
    vkx_chip_um: Optional[float] = None       # mean chip from VK-X sample
    true_yield: float = 0.0                    # fraction of truly good dice
    iv3_yield: float = 0.0                     # fraction passing IV3


@dataclass
class PipelineResult:
    yield_pct: float                           # % of dice passing all gates
    throughput_wph: float                      # wafers per hour
    quarantine_count: int
    vkx_rmse: float                            # RMSE of VK-X vs ground truth (µm)
    cost_per_wafer_jpy: float
    wafer_results: List[WaferResult] = field(default_factory=list)
    config_name: str = "IV3+VKX_10pct"


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class KeyenceInspectionPipeline:
    """
    Simulated end-to-end inspection pipeline for SiC wafer dicing.

    Instruments
    -----------
    LJ-X8080 : laser displacement profiler
        - Pre-dicing: measure wafer warp; quarantine if warp > 50 µm
        - Post-dicing: kerf width measurement (noise ~0.3 µm)
    IV3      : inline vision sensor at dicing speed
        - Pass/fail per die; 5% false-negative rate for borderline chips
    VK-X     : confocal laser scanning microscope
        - Offline; sampled every 10th wafer for Grade A reference measurement
    """

    def __init__(self, n_wafers: int = 100, seed: int = 0):
        self.n_wafers = n_wafers
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Step 1: LJ-X8080 warp check
    # ------------------------------------------------------------------
    def _ljx_warp_check(self, wafer_id: int) -> tuple[float, bool]:
        """Return (warp_um, quarantined)."""
        warp = self.rng.normal(DICE_WARP_MEAN_UM, DICE_WARP_STD_UM)
        warp = max(warp, 0.0)
        quarantined = warp > LJX_WARP_THRESHOLD_UM
        return warp, quarantined

    # ------------------------------------------------------------------
    # Step 2: Dicing simulation
    # ------------------------------------------------------------------
    def _dice(self, n_dies: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (chip_widths_um, crack_widths_um) for all dies."""
        chips = self.rng.normal(DICE_CHIP_MEAN_UM, DICE_CHIP_STD_UM, n_dies)
        cracks = self.rng.normal(DICE_CRACK_MEAN_UM, DICE_CRACK_STD_UM, n_dies)
        chips = np.maximum(chips, 0.0)
        cracks = np.maximum(cracks, 0.0)
        return chips, cracks

    # ------------------------------------------------------------------
    # Step 3: IV3 inline inspection
    # ------------------------------------------------------------------
    def _iv3_inspect(
        self,
        chip_widths: np.ndarray,
        crack_widths: np.ndarray,
    ) -> np.ndarray:
        """
        Return boolean pass array.
        True failures: chip > threshold OR crack > threshold.
        False-negative: IV3 misses IV3_FALSE_NEGATIVE_RATE fraction of true failures.
        """
        true_fail = (chip_widths > IV3_CHIP_THRESHOLD_UM) | (crack_widths > IV3_CRACK_THRESHOLD_UM)
        # Some true failures slip through (false negative)
        fn_mask = self.rng.random(len(true_fail)) < IV3_FALSE_NEGATIVE_RATE
        # IV3 pass = not (true_fail AND NOT false_negative)
        iv3_fail = true_fail & ~fn_mask
        return ~iv3_fail  # True = pass

    # ------------------------------------------------------------------
    # Step 4: VK-X offline measurement
    # ------------------------------------------------------------------
    def _vkx_measure(self, chip_widths: np.ndarray) -> float:
        """Return precise mean chipping measurement (µm) with small noise."""
        true_mean = float(np.mean(chip_widths))
        measured = true_mean + self.rng.normal(0.0, VKX_SIGMA_UM)
        return measured

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------
    def run(self) -> PipelineResult:
        """
        Run the full pipeline for self.n_wafers wafers.

        Returns PipelineResult with aggregate metrics.
        """
        wafer_results: List[WaferResult] = []
        quarantine_count = 0
        all_true_yield_fracs: List[float] = []
        all_iv3_yield_fracs: List[float] = []
        vkx_errors: List[float] = []

        for wid in range(self.n_wafers):
            # --- LJ-X8080: warp check ---
            warp, quarantined = self._ljx_warp_check(wid)

            if quarantined:
                quarantine_count += 1
                wr = WaferResult(
                    wafer_id=wid,
                    quarantined=True,
                    warp_um=warp,
                )
                wafer_results.append(wr)
                continue

            # --- Dicing simulation ---
            chips, cracks = self._dice(DIES_PER_WAFER)

            # --- IV3 inline inspection ---
            iv3_pass = self._iv3_inspect(chips, cracks)

            true_good = ~((chips > IV3_CHIP_THRESHOLD_UM) | (cracks > IV3_CRACK_THRESHOLD_UM))
            true_yield = float(np.mean(true_good))
            iv3_yield = float(np.mean(iv3_pass))

            all_true_yield_fracs.append(true_yield)
            all_iv3_yield_fracs.append(iv3_yield)

            # --- VK-X: sample 1 in 10 wafers ---
            vkx_measured = False
            vkx_chip = None
            if wid % VKX_SAMPLE_EVERY == 0:
                vkx_chip = self._vkx_measure(chips)
                vkx_measured = True
                # Ground truth mean chip among sampled dies
                true_chip_mean = float(np.mean(chips))
                vkx_errors.append(vkx_chip - true_chip_mean)

            wr = WaferResult(
                wafer_id=wid,
                quarantined=False,
                warp_um=warp,
                chip_widths_um=chips,
                crack_widths_um=cracks,
                iv3_pass=iv3_pass,
                vkx_measured=vkx_measured,
                vkx_chip_um=vkx_chip,
                true_yield=true_yield,
                iv3_yield=iv3_yield,
            )
            wafer_results.append(wr)

        # --- Aggregate metrics ---
        diced_wafers = self.n_wafers - quarantine_count
        yield_pct = 100.0 * (
            float(np.mean(all_iv3_yield_fracs)) if all_iv3_yield_fracs else 0.0
        )

        vkx_rmse = (
            float(np.sqrt(np.mean(np.array(vkx_errors) ** 2))) if vkx_errors else 0.0
        )

        # Cost model
        vkx_sample_wafers = max(1, diced_wafers // VKX_SAMPLE_EVERY)
        total_cost = (
            self.n_wafers * COST_DICING_PER_WAFER_JPY
            + diced_wafers * DIES_PER_WAFER * COST_IV3_INSPECT_PER_DIE_JPY
            + vkx_sample_wafers * COST_VKX_INSPECT_PER_WAFER_JPY
            + quarantine_count * COST_QUARANTINE_PER_WAFER_JPY
        )
        cost_per_wafer = total_cost / self.n_wafers if self.n_wafers > 0 else 0.0

        return PipelineResult(
            yield_pct=yield_pct,
            throughput_wph=TPH_COMBINED,
            quarantine_count=quarantine_count,
            vkx_rmse=vkx_rmse,
            cost_per_wafer_jpy=cost_per_wafer,
            wafer_results=wafer_results,
            config_name="IV3+VKX_10pct",
        )

    # ------------------------------------------------------------------
    # Configuration comparison
    # ------------------------------------------------------------------
    def compare_configurations(self) -> dict:
        """
        Compare 3 inspection configurations.

        A. VK-X only  : 100% sampling, slow, precise
        B. IV3 only   : 100% inline, fast, 5% false-negative
        C. IV3 + VK-X 10% sampling (default pipeline)

        Returns dict with per-config metrics.
        """
        rng = self.rng  # reuse for reproducibility

        results = {}

        # Re-simulate a common wafer population
        n = self.n_wafers
        warps = np.maximum(rng.normal(DICE_WARP_MEAN_UM, DICE_WARP_STD_UM, n), 0.0)
        quarantine_mask = warps > LJX_WARP_THRESHOLD_UM
        n_diced = int(np.sum(~quarantine_mask))

        # Generate die-level data for diced wafers
        all_chips = np.maximum(
            rng.normal(DICE_CHIP_MEAN_UM, DICE_CHIP_STD_UM, (n_diced, DIES_PER_WAFER)), 0.0
        )
        all_cracks = np.maximum(
            rng.normal(DICE_CRACK_MEAN_UM, DICE_CRACK_STD_UM, (n_diced, DIES_PER_WAFER)), 0.0
        )
        true_fail = (all_chips > IV3_CHIP_THRESHOLD_UM) | (all_cracks > IV3_CRACK_THRESHOLD_UM)
        true_yield = float(np.mean(~true_fail))

        # --- Config A: VK-X only (100% sampling, treats VK-X as ground truth) ---
        vkx_noise = rng.normal(0.0, VKX_SIGMA_UM, (n_diced, DIES_PER_WAFER))
        vkx_chips = all_chips + vkx_noise
        vkx_fail_a = (vkx_chips > IV3_CHIP_THRESHOLD_UM)
        yield_a = float(np.mean(~vkx_fail_a))
        yield_accuracy_a = 1.0 - abs(yield_a - true_yield)

        cost_a = (
            n * COST_DICING_PER_WAFER_JPY
            + n_diced * COST_VKX_INSPECT_PER_WAFER_JPY * 10  # 100% sampling is ~10× more expensive
            + int(np.sum(quarantine_mask)) * COST_QUARANTINE_PER_WAFER_JPY
        ) / n

        results["A_VKX_only"] = {
            "config": "VK-X only (100% sampling)",
            "throughput_wph": TPH_VKXONLY,
            "yield_accuracy": round(yield_accuracy_a, 4),
            "cost_per_wafer_jpy": round(cost_a, 0),
            "yield_pct": round(100 * yield_a, 2),
        }

        # --- Config B: IV3 only (100% inline, 5% false-negative) ---
        fn_mask_b = rng.random((n_diced, DIES_PER_WAFER)) < IV3_FALSE_NEGATIVE_RATE
        iv3_fail_b = true_fail & ~fn_mask_b
        yield_b = float(np.mean(~iv3_fail_b))
        yield_accuracy_b = 1.0 - abs(yield_b - true_yield)

        cost_b = (
            n * COST_DICING_PER_WAFER_JPY
            + n_diced * DIES_PER_WAFER * COST_IV3_INSPECT_PER_DIE_JPY
            + int(np.sum(quarantine_mask)) * COST_QUARANTINE_PER_WAFER_JPY
        ) / n

        results["B_IV3_only"] = {
            "config": "IV3 only (100% inline)",
            "throughput_wph": TPH_IV3ONLY,
            "yield_accuracy": round(yield_accuracy_b, 4),
            "cost_per_wafer_jpy": round(cost_b, 0),
            "yield_pct": round(100 * yield_b, 2),
        }

        # --- Config C: IV3 + VK-X 10% sampling ---
        fn_mask_c = rng.random((n_diced, DIES_PER_WAFER)) < IV3_FALSE_NEGATIVE_RATE
        iv3_fail_c = true_fail & ~fn_mask_c
        yield_c = float(np.mean(~iv3_fail_c))
        yield_accuracy_c = 1.0 - abs(yield_c - true_yield)

        vkx_sample_n = max(1, n_diced // VKX_SAMPLE_EVERY)
        cost_c = (
            n * COST_DICING_PER_WAFER_JPY
            + n_diced * DIES_PER_WAFER * COST_IV3_INSPECT_PER_DIE_JPY
            + vkx_sample_n * COST_VKX_INSPECT_PER_WAFER_JPY
            + int(np.sum(quarantine_mask)) * COST_QUARANTINE_PER_WAFER_JPY
        ) / n

        results["C_IV3_VKX_10pct"] = {
            "config": "IV3 + VK-X 10% sampling",
            "throughput_wph": TPH_COMBINED,
            "yield_accuracy": round(yield_accuracy_c, 4),
            "cost_per_wafer_jpy": round(cost_c, 0),
            "yield_pct": round(100 * yield_c, 2),
        }

        return results

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot(self, result: PipelineResult, save_dir: str = OUT_DIR) -> str:
        """
        3-panel figure:
          Panel 1: Pipeline flow diagram (LJ-X8080 -> IV3 -> VK-X)
          Panel 2: Yield comparison across configurations
          Panel 3: Cost vs throughput scatter
        Saves to save_dir/keyence_inspection_pipeline.png.
        Returns the saved file path.
        """
        os.makedirs(save_dir, exist_ok=True)

        configs = self.compare_configurations()
        config_names = [v["config"] for v in configs.values()]
        yields = [v["yield_pct"] for v in configs.values()]
        costs = [v["cost_per_wafer_jpy"] for v in configs.values()]
        tphs = [v["throughput_wph"] for v in configs.values()]
        colors = ["#2196F3", "#FF9800", "#4CAF50"]

        fig = plt.figure(figsize=(17, 6))
        fig.suptitle(
            "Keyence SiC Wafer Dicing Inspection Pipeline",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )

        # ---- Panel 1: Flow diagram ----------------------------------------
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.axis("off")
        ax1.set_title("Pipeline Flow", fontsize=11, fontweight="bold")

        boxes = [
            (5, 9.0, "Wafer\nIn", "#BBDEFB"),
            (5, 7.3, "LJ-X8080\nWarp Check\n(<50µm OK)", "#E3F2FD"),
            (5, 5.5, "Dicing\nProcess", "#FFF9C4"),
            (5, 3.7, "IV3 Inline\nPass/Fail", "#E8F5E9"),
            (5, 1.9, "VK-X\n(1-in-10 wafers)\nGrade A Ref.", "#FCE4EC"),
            (5, 0.3, "Good Die\nOutput", "#C8E6C9"),
        ]

        quarantine_box = (8.5, 7.3, "Quarantine\n(warp>50µm)", "#FFCDD2")

        def draw_box(ax, cx, cy, label, facecolor, width=3.0, height=0.9):
            x0 = cx - width / 2
            y0 = cy - height / 2
            rect = plt.Rectangle(
                (x0, y0), width, height,
                facecolor=facecolor, edgecolor="#555", linewidth=1.2, zorder=3,
            )
            ax.add_patch(rect)
            ax.text(
                cx, cy, label,
                ha="center", va="center", fontsize=7.5, zorder=4,
            )

        for (cx, cy, lbl, fc) in boxes:
            draw_box(ax1, cx, cy, lbl, fc)

        draw_box(ax1, *quarantine_box, width=2.6, height=0.7)

        # Arrows between boxes
        arrow_kw = dict(arrowstyle="-|>", color="#333", lw=1.2)
        for i in range(len(boxes) - 1):
            _, y_from, _, _ = boxes[i]
            _, y_to, _, _ = boxes[i + 1]
            ax1.annotate(
                "", xy=(5, y_to + 0.45), xytext=(5, y_from - 0.45),
                arrowprops=arrow_kw, zorder=5,
            )

        # Quarantine branch from LJ-X8080
        ax1.annotate(
            "", xy=(7.2, 7.3), xytext=(6.5, 7.3),
            arrowprops=dict(arrowstyle="-|>", color="#F44336", lw=1.2),
        )
        ax1.text(6.9, 7.55, "fail", color="#F44336", fontsize=7, ha="center")

        # Stats annotation
        n_q = result.quarantine_count
        n_total = result.n_wafers if hasattr(result, "n_wafers") else self.n_wafers
        ax1.text(
            5, -0.4,
            f"n={n_total} wafers | quarantine={n_q} | yield={result.yield_pct:.1f}%\n"
            f"VK-X RMSE={result.vkx_rmse:.3f}µm | cost=¥{result.cost_per_wafer_jpy:,.0f}/wafer",
            ha="center", va="top", fontsize=7, color="#444",
            transform=ax1.transData,
        )

        # ---- Panel 2: Yield comparison ------------------------------------
        ax2 = fig.add_subplot(1, 3, 2)
        x = np.arange(len(config_names))
        bars = ax2.bar(x, yields, color=colors, edgecolor="#555", linewidth=0.8, width=0.55)
        ax2.set_xticks(x)
        ax2.set_xticklabels(
            ["A\nVK-X\nonly", "B\nIV3\nonly", "C\nIV3+VK-X\n10%"],
            fontsize=8,
        )
        ax2.set_ylabel("Yield (%)", fontsize=10)
        ax2.set_title("Yield by Configuration", fontsize=11, fontweight="bold")
        ax2.set_ylim(0, 100)
        ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax2.set_axisbelow(True)
        for bar, val in zip(bars, yields):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

        # Accuracy annotation
        accuracies = [v["yield_accuracy"] for v in configs.values()]
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                2,
                f"acc={acc:.3f}",
                ha="center", va="bottom", fontsize=7, color="white", fontweight="bold",
            )

        # ---- Panel 3: Cost vs Throughput scatter -------------------------
        ax3 = fig.add_subplot(1, 3, 3)
        for i, (cfg_key, cfg_val) in enumerate(configs.items()):
            ax3.scatter(
                cfg_val["throughput_wph"],
                cfg_val["cost_per_wafer_jpy"],
                s=120,
                color=colors[i],
                edgecolors="#333",
                linewidth=0.8,
                zorder=5,
                label=f"{chr(65+i)}: {cfg_val['config']}",
            )
            ax3.annotate(
                chr(65 + i),
                xy=(cfg_val["throughput_wph"], cfg_val["cost_per_wafer_jpy"]),
                xytext=(0.25, 150),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color=colors[i],
            )

        ax3.set_xlabel("Throughput (wafers/hr)", fontsize=10)
        ax3.set_ylabel("Cost per Wafer (JPY)", fontsize=10)
        ax3.set_title("Cost vs Throughput", fontsize=11, fontweight="bold")
        ax3.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax3.xaxis.grid(True, linestyle="--", alpha=0.5)
        ax3.set_axisbelow(True)
        legend_patches = [
            mpatches.Patch(color=colors[i], label=f"{chr(65+i)}: {v['config']}")
            for i, v in enumerate(configs.values())
        ]
        ax3.legend(handles=legend_patches, fontsize=7, loc="upper right")

        plt.tight_layout()
        out_path = os.path.join(save_dir, "keyence_inspection_pipeline.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Keyence end-to-end SiC wafer dicing inspection pipeline"
    )
    parser.add_argument("--n-wafers", type=int, default=100, help="Number of wafers to simulate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--save-dir", type=str, default=OUT_DIR, help="Directory to save plot"
    )
    parser.add_argument("--compare", action="store_true", help="Print configuration comparison")
    parser.add_argument("--no-plot", action="store_true", help="Skip figure generation")
    args = parser.parse_args()

    pipeline = KeyenceInspectionPipeline(n_wafers=args.n_wafers, seed=args.seed)

    print(f"Running Keyence inspection pipeline: {args.n_wafers} wafers, seed={args.seed}")
    result = pipeline.run()

    print("\n=== Pipeline Result (Config C: IV3 + VK-X 10%) ===")
    print(f"  Yield:             {result.yield_pct:.2f}%")
    print(f"  Throughput:        {result.throughput_wph:.1f} wph")
    print(f"  Quarantine count:  {result.quarantine_count}")
    print(f"  VK-X RMSE:         {result.vkx_rmse:.4f} µm")
    print(f"  Cost per wafer:    ¥{result.cost_per_wafer_jpy:,.0f}")

    if args.compare:
        configs = pipeline.compare_configurations()
        print("\n=== Configuration Comparison ===")
        for key, val in configs.items():
            print(
                f"  {key}: throughput={val['throughput_wph']} wph, "
                f"yield={val['yield_pct']:.2f}%, "
                f"accuracy={val['yield_accuracy']:.4f}, "
                f"cost=¥{val['cost_per_wafer_jpy']:,.0f}"
            )

    if not args.no_plot:
        out_path = pipeline.plot(result, save_dir=args.save_dir)
        print(f"\nPlot saved to: {out_path}")
    else:
        print("\n[--no-plot] Skipping figure generation.")


if __name__ == "__main__":
    main()
