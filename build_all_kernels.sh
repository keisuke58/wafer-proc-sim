#!/usr/bin/env bash
# Build all pybind11 C++ / CUDA kernels in one shot.
#
# Each compiled .so is platform-specific and git-ignored. If a kernel .so is
# absent its Python wrapper transparently falls back to the NumPy / pure-Python
# implementation, so this build is purely an opt-in speed-up.
#
# Requires (C++ kernels):  g++ (C++14), pybind11 (pip install pybind11)
# Requires (CUDA kernels): nvcc (CUDA ≥ 11.8), same pybind11
#
# Usage:
#   bash build_all_kernels.sh                  # build all (skips CUDA if no nvcc)
#   bash build_all_kernels.sh stealth          # rebuild only the stealth kernel
#   bash build_all_kernels.sh heat_diffusion   # rebuild CUDA heat kernel
set -euo pipefail

PY="${PYTHON:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUFFIX="$("${PY}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
PY_INC="$("${PY}" -m pybind11 --includes)"

# ── C++ kernel builder ────────────────────────────────────────────────────────
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

# ── CUDA kernel builder ───────────────────────────────────────────────────────
# Target arch: sm_89 = RTX 4090 (Ada Lovelace).  Override with CUDA_ARCH env var.
#   e.g.  CUDA_ARCH=sm_86 bash build_all_kernels.sh heat_diffusion  (RTX 3090)
CUDA_ARCH="${CUDA_ARCH:-sm_89}"

build_cuda_kernel() {
    local src="$1"       # path to .cu relative to repo root
    local out_stem="$2"  # output filename stem
    local out_dir="$3"   # output directory

    if ! command -v nvcc &>/dev/null; then
        echo "[!] nvcc not found — skipping CUDA kernel $(basename "${src}")"
        echo "    Install CUDA Toolkit or run on Vancouver (RTX 4090 nodes)."
        return 0
    fi

    local out="${out_dir}/${out_stem}${SUFFIX}"
    echo "[*] Compiling (CUDA) $(basename "${src}") -> $(basename "${out}")"
    nvcc -O3 -shared \
        --std=c++14 \
        -arch "${CUDA_ARCH}" \
        --compiler-options "-fPIC -Wall" \
        ${PY_INC} \
        "${HERE}/${src}" \
        -o "${out}"
    echo "[✓] Built ${out}  (arch=${CUDA_ARCH})"
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
build_frame() {
    build_kernel "fem/_frame_stiffness_kernel.cpp" "_frame_stiffness_kernel" "${HERE}/fem"
}
build_5axis() {
    build_kernel "pipeline/_5axis_interpolation.cpp" "_5axis_interpolation" "${HERE}/pipeline"
}
build_inverter() {
    build_kernel "fem/_servo_inverter_kernel.cpp"  "_servo_inverter_kernel" "${HERE}/fem"
}
build_encoder() {
    build_kernel "fem/_encoder_kernel.cpp"         "_encoder_kernel"        "${HERE}/fem"
}
build_recipe() {
    build_kernel "pipeline/_recipe_sequencer.cpp"  "_recipe_sequencer"      "${HERE}/pipeline"
}
build_interlock() {
    build_kernel "pipeline/_interlock_monitor.cpp" "_interlock_monitor"     "${HERE}/pipeline"
}
build_machine() {
    build_kernel "machine/_disco_machine.cpp"      "_disco_machine"         "${HERE}/machine"
}
build_gp() {
    build_kernel "ml/_gp_inference_kernel.cpp"     "_gp_inference_kernel"   "${HERE}/ml"
}
build_mpc() {
    build_kernel "pipeline/_mpc_kernel.cpp"        "_mpc_kernel"            "${HERE}/pipeline"
}

# ── CUDA kernels ──────────────────────────────────────────────────────────────
build_heat_diffusion() {
    build_cuda_kernel "fem/_heat_diffusion_kernel.cu" "_heat_diffusion_kernel" "${HERE}/fem"
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
        build_frame
        build_5axis
        build_inverter
        build_encoder
        build_recipe
        build_interlock
        build_machine
        build_gp
        build_mpc
        build_heat_diffusion   # CUDA — skipped silently if nvcc absent
        ;;
    stealth)          build_stealth          ;;
    arde)             build_arde             ;;
    nsga2)            build_nsga2            ;;
    motion)           build_motion           ;;
    enkf)             build_enkf             ;;
    statemachine)     build_statemachine     ;;
    spindle)          build_spindle          ;;
    frame)            build_frame            ;;
    5axis)            build_5axis            ;;
    inverter)         build_inverter         ;;
    encoder)          build_encoder          ;;
    recipe)           build_recipe           ;;
    interlock)        build_interlock        ;;
    machine)          build_machine          ;;
    gp)               build_gp               ;;
    mpc)              build_mpc              ;;
    heat_diffusion)   build_heat_diffusion   ;;
    *)
        echo "Unknown target '${TARGET}'."
        echo "Valid: all stealth arde nsga2 motion enkf statemachine spindle frame"
        echo "       5axis inverter encoder recipe interlock heat_diffusion"
        exit 1
        ;;
esac

echo ""
echo "All requested kernels built. Run 'python -c \"from fem import _arde_kernel\"' etc. to verify."
echo "CUDA kernels: 'python -c \"from fem._heat_diffusion_kernel import gpu_info; print(gpu_info())\"'"
