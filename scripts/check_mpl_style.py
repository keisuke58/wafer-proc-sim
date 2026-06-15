#!/usr/bin/env python3
"""check_mpl_style.py — matplotlib figure-style compliance gate (source-level).

Detects *.py files that USE matplotlib/pyplot but do NOT reference `thesis_style`
(i.e. never call thesis_style.use()).  Uses a committed baseline allowlist so that
*existing* non-compliant files are exempt — only NEW violations fail the gate
(ratchet pattern).  Pairs with a pre-commit hook.

File list comes from `git ls-files '*.py'` (tracked + staged), so __pycache__,
results/, data/ etc. are auto-excluded.

Usage:
    python scripts/check_mpl_style.py                 # gate: exit 1 on NEW violations
    python scripts/check_mpl_style.py --update-baseline  # snapshot current violations
    python scripts/check_mpl_style.py --list          # show all current violations
"""
import re
import subprocess
import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(
    subprocess.run(["git", "rev-parse", "--show-toplevel"],
                   capture_output=True, text=True, check=True).stdout.strip()
)
BASELINE = REPO_ROOT / ".mpl_style_baseline.txt"

# Compliance marker: file must reference this to count as styled.
COMPLIANT_MARKER = "thesis_style"

# A file "uses matplotlib for plotting" if it imports it.
USES_MPL = re.compile(
    r"^\s*(?:import\s+matplotlib"
    r"|from\s+matplotlib\b"
    r"|import\s+matplotlib\.pyplot"
    r"|import\s+\S+\s+as\s+plt)\b",
    re.MULTILINE,
)


def tracked_py() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "*.py"],
                         cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return [REPO_ROOT / p for p in out.stdout.split("\n") if p.strip()]


def uses_mpl(text: str) -> bool:
    return bool(USES_MPL.search(text))


def is_compliant(text: str) -> bool:
    return COMPLIANT_MARKER in text


def current_violations() -> list[str]:
    """Relative paths of files that use matplotlib but aren't compliant."""
    viol = []
    for p in tracked_py():
        if not p.exists():
            continue
        text = p.read_text(errors="replace")
        if uses_mpl(text) and not is_compliant(text):
            viol.append(str(p.relative_to(REPO_ROOT)))
    return sorted(viol)


def load_baseline() -> set[str]:
    if not BASELINE.exists():
        return set()
    return {ln.strip() for ln in BASELINE.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-baseline", action="store_true",
                    help="Snapshot current violations into the baseline (exempt them)")
    ap.add_argument("--list", action="store_true",
                    help="List all current violations (baseline-exempt or not)")
    args = ap.parse_args()

    viol = current_violations()

    if args.update_baseline:
        header = ("# matplotlib-style baseline — files using matplotlib WITHOUT "
                  f"{COMPLIANT_MARKER}.\n# These are exempt from the gate (ratchet). "
                  "Migrate + remove lines over time.\n")
        BASELINE.write_text(header + "\n".join(viol) + ("\n" if viol else ""))
        print(f"[baseline] wrote {len(viol)} exempt file(s) → "
              f"{BASELINE.relative_to(REPO_ROOT)}")
        return

    if args.list:
        print(f"matplotlib-using files WITHOUT {COMPLIANT_MARKER} "
              f"({len(viol)} total):")
        for v in viol:
            print(f"  {v}")
        return

    baseline = load_baseline()
    new_viol = [v for v in viol if v not in baseline]
    exempt = len(viol) - len(new_viol)

    print(f"matplotlib-style gate  (marker: {COMPLIANT_MARKER}.use())")
    print(f"  baseline-exempt : {exempt}")
    print(f"  NEW violations  : {len(new_viol)}")

    if new_viol:
        print(f"\nFAIL — new matplotlib file(s) without {COMPLIANT_MARKER}:")
        for v in new_viol:
            print(f"  ✗  {v}")
        print(f"\n  Fix: call `{COMPLIANT_MARKER}.use()` before plt.subplots(),")
        print(f"       or `python scripts/check_mpl_style.py --update-baseline` "
              f"to intentionally exempt it.")
        sys.exit(1)

    print(f"[OK] no new {COMPLIANT_MARKER} violations.")


if __name__ == "__main__":
    main()
