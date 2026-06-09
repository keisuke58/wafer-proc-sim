#!/usr/bin/env bash
# Build all pybind11 C++ kernels in one shot.
#
# Each compiled .so is platform-specific and git-ignored. If a kernel .so is
# absent its Python wrapper transparently falls back to the NumPy / pure-Python
# implementation, so this build is purely an opt-in speed-up.
#
# Requires: g++ (C++14), pybind11 (pip install pybind11).
#
# Usage:
#   bash build_all_kernels.sh          # build all
#   bash build_all_kernels.sh stealth  # rebuild only the stealth kernel
set -euo pipefail

PY="${PYTHON:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUFFIX="$("${PY}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
PY_INC="$("${PY}" -m pybind11 --includes)"

build_kernel() {
    local src="$1"          # path to .cpp relative to repo root
    local out_stem="$2"     # output filename stem (no extension)
    local out_dir="$3"      # output directory

    local out="${out_dir}/${out_stem}${SUFFIX}"
    echo "[*] Compiling $(basename "${src}") -> $(basename "${out}")"
    g++ -O3 -Wall -shared -std=c++14 -fPIC \
        ${PY_INC} \
        "${HERE}/${src}" \
        -o "${out}"
    echo "[✓] Built ${out}"
}

TARGET="${1:-all}"

build_stealth() {
    build_kernel "fem/_stealth_kernel.cpp"        "_stealth_kernel"  "${HERE}/fem"
}
build_arde() {
    build_kernel "fem/_arde_kernel.cpp"           "_arde_kernel"     "${HERE}/fem"
}
build_nsga2() {
    build_kernel "optimization/_nsga2_kernel.cpp" "_nsga2_kernel"    "${HERE}/optimization"
}
build_motion() {
    build_kernel "pipeline/_motion_kernel.cpp"    "_motion_kernel"   "${HERE}/pipeline"
}
build_enkf() {
    build_kernel "ml/_enkf_kernel.cpp"            "_enkf_kernel"     "${HERE}/ml"
}
build_statemachine() {
    build_kernel "pipeline/_state_machine.cpp"    "_state_machine"   "${HERE}/pipeline"
}
build_spindle() {
    build_kernel "fem/_spindle_kernel.cpp"        "_spindle_kernel"  "${HERE}/fem"
}

case "${TARGET}" in
    all)
        build_stealth
        build_arde
        build_nsga2
        build_motion
        build_enkf
        build_statemachine
        build_spindle
        ;;
    stealth)       build_stealth       ;;
    arde)          build_arde          ;;
    nsga2)         build_nsga2         ;;
    motion)        build_motion        ;;
    enkf)          build_enkf          ;;
    statemachine)  build_statemachine  ;;
    spindle)       build_spindle       ;;
    *)
        echo "Unknown target '${TARGET}'. Valid: all stealth arde nsga2 motion enkf statemachine spindle"
        exit 1
        ;;
esac

echo ""
echo "All requested kernels built. Run 'python -c \"from fem import _arde_kernel\"' etc. to verify."
