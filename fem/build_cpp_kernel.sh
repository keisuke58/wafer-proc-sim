#!/usr/bin/env bash
# Build the pybind11 C++ stealth-dicing kernel in place.
#
# The compiled .so is platform-specific and git-ignored; if it is absent the
# Python wrapper transparently falls back to the numba / NumPy kernels, so this
# build is purely an opt-in speed-up.
#
# Requires: g++ (C++14), pybind11 (pip install pybind11).
set -euo pipefail

# Use the *running* interpreter's real extension suffix (python3-config can be
# stale / point at a different minor version than `python`).
PY="${PYTHON:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUFFIX="$("${PY}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
OUT="${HERE}/_stealth_kernel${SUFFIX}"

echo "[*] Compiling _stealth_kernel.cpp -> $(basename "$OUT")"
g++ -O3 -Wall -shared -std=c++14 -fPIC \
    $("${PY}" -m pybind11 --includes) \
    "${HERE}/_stealth_kernel.cpp" \
    -o "${OUT}"
echo "[✓] Built ${OUT}"
