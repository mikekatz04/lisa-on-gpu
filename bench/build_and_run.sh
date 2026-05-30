#!/usr/bin/env bash
#
# build_and_run.sh -- compile + run the standalone chunked-het GPU timing
# harness against the lisa-on-gpu sources, with NO Python install required.
#
# GPU-only: nvcc + sm_80 (A100). The CPU path is covered by the regular
# `pip install -e .[cpu]` build + gb_chunked_test_script.py, so we don't
# duplicate it here.
#
# Usage:
#   ./build_and_run.sh                              # defaults
#   ./build_and_run.sh --num_bin=500 --grid_dim=128 # forward harness args
#   NVCC=/path/to/nvcc ./build_and_run.sh           # override compiler
#   SM_ARCH=sm_90 ./build_and_run.sh                # override arch
#   MATHDX_DIR=/opt/nvidia/mathdx ./build_and_run.sh # enable cufftdx
#                                                   # (must contain include/cufftdx.hpp)

set -euo pipefail

# --- Locate the sprint source tree ------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LISA_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
SPRINT_ROOT="$( cd "${LISA_ROOT}/.." && pwd )"

GBT_INC="${SPRINT_ROOT}/GPUBackendTools/src/gpubackendtools/cutils"
LAT_INC="${SPRINT_ROOT}/LISAanalysistools/src/lisatools/cutils"
LRG_INC="${LISA_ROOT}/src/fastlisaresponse/cutils"

TDI_SRC="${LRG_INC}/TDIonTheFly.cu"
DET_SRC="${LAT_INC}/Detector.cu"
LRP_SRC="${LRG_INC}/LISAResponse.cu"
INT_SRC="${GBT_INC}/Interpolate.cu"

OUT_DIR="${SCRIPT_DIR}/build"
mkdir -p "${OUT_DIR}"
BIN="${OUT_DIR}/chunked_het_bench"

LAPACK_CFLAGS=$(pkg-config --cflags lapacke 2>/dev/null || true)
LAPACK_LDLIBS=$(pkg-config --libs   lapacke 2>/dev/null || true)

NVCC="${NVCC:-nvcc}"
SM_ARCH="${SM_ARCH:-sm_80}"

# Optional cufftdx (NVIDIA MathDx). Set MATHDX_DIR to the install root
# (containing include/cufftdx.hpp) to enable. Empty -> radix-2 fallback.
CUFFTDX_FLAGS=()
if [[ -n "${MATHDX_DIR:-}" ]]; then
    if [[ -f "${MATHDX_DIR}/include/cufftdx.hpp" ]]; then
        echo "[build_and_run] cufftdx enabled from ${MATHDX_DIR}"
        CUFFTDX_FLAGS=(-I "${MATHDX_DIR}/include" -DLISA_USE_CUFFTDX)
    else
        echo "[build_and_run] WARNING: MATHDX_DIR='${MATHDX_DIR}' has no \
include/cufftdx.hpp; cufftdx NOT enabled."
    fi
fi

"${NVCC}" --version >/dev/null
echo "[build_and_run] CUDA build via ${NVCC} (-arch=${SM_ARCH})"

"${NVCC}" \
    -x cu \
    -arch="${SM_ARCH}" \
    --extended-lambda \
    --expt-relaxed-constexpr \
    -rdc=true \
    -DBENCH_GPU=1 \
    -std=c++17 \
    -O3 \
    -Xcompiler -Wall \
    -Xcompiler -Wno-unused-variable,-Wno-unused-parameter,-Wno-unused-but-set-variable,-Wno-deprecated-declarations \
    -I "${GBT_INC}" \
    -I "${LAT_INC}" \
    -I "${LAT_INC}/.." \
    -I "${LRG_INC}" \
    -I "${LRG_INC}/.." \
    "${CUFFTDX_FLAGS[@]}" \
    ${LAPACK_CFLAGS} \
    "${SCRIPT_DIR}/chunked_het_bench.cxx" \
    "${TDI_SRC}" \
    "${DET_SRC}" \
    "${LRP_SRC}" \
    "${INT_SRC}" \
    -lcusparse -lcublas -lcusolver -lcufft \
    ${LAPACK_LDLIBS} \
    -o "${BIN}"

echo "[build_and_run] built ${BIN}"
echo "[build_and_run] running with args: $@"
"${BIN}" "$@"
