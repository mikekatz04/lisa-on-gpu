#!/usr/bin/env bash
#
# build_and_run.sh -- compile + run the standalone chunked-het timing harness
# against the lisa-on-gpu sources, with NO Python install required.
#
# Defaults to a CPU build using brew llvm@20 (Apple clang 17 fails on the
# CUDA_SHARED anonymous union default-ctor in TDIonTheFly.cu; see the sprint
# memory). Set BENCH_GPU=1 to use nvcc for a CUDA build instead.
#
# Usage:
#   ./build_and_run.sh                # CPU build, defaults
#   ./build_and_run.sh --num_bin=500  # forward harness args
#   BENCH_GPU=1 ./build_and_run.sh    # CUDA build (Linux, A100/H100)

set -euo pipefail

# --- Locate the sprint source tree ------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LISA_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
SPRINT_ROOT="$( cd "${LISA_ROOT}/.." && pwd )"

GBT_INC="${SPRINT_ROOT}/GPUBackendTools/src/gpubackendtools/cutils"
LAT_INC="${SPRINT_ROOT}/LISAanalysistools/src/lisatools/cutils"
LRG_INC="${LISA_ROOT}/src/fastlisaresponse/cutils"

# --- Source files we need to compile (the .cu files compile cleanly as .cxx
# on CPU because the CUDA_SHARED / THREAD_START_X macros collapse to host
# expressions). On GPU we pass the same files to nvcc as .cu.
TDI_SRC="${LRG_INC}/TDIonTheFly.cu"
DET_SRC="${LAT_INC}/Detector.cu"
LRP_SRC="${LRG_INC}/LISAResponse.cu"
INT_SRC="${GBT_INC}/Interpolate.cu"

OUT_DIR="${SCRIPT_DIR}/build"
mkdir -p "${OUT_DIR}"
BIN="${OUT_DIR}/chunked_het_bench"

LAPACK_CFLAGS=$(PKG_CONFIG_PATH=/usr/local/opt/lapack/lib/pkgconfig:${PKG_CONFIG_PATH:-} pkg-config --cflags lapacke 2>/dev/null || true)
LAPACK_LDLIBS=$(PKG_CONFIG_PATH=/usr/local/opt/lapack/lib/pkgconfig:${PKG_CONFIG_PATH:-} pkg-config --libs   lapacke 2>/dev/null || true)

CXXFLAGS_COMMON=(
    -std=c++17
    -O3
    -Wall -Wno-unused-variable -Wno-unused-parameter -Wno-unused-but-set-variable
    -Wno-deprecated-declarations -Wno-nan-infinity-disabled
    -I "${GBT_INC}"
    -I "${LAT_INC}"      # for Detector.hpp / Detector.cu cross-refs
    -I "${LAT_INC}/.."   # lisatools/cutils/Detector.hpp via "lisatools/cutils/..."
    -I "${LRG_INC}"      # for LISAResponse.hh / TDIonTheFly.hh cross-refs
    -I "${LRG_INC}/.."   # fastlisaresponse/cutils/TDIonTheFly.hh via "fastlisaresponse/cutils/..."
    ${LAPACK_CFLAGS}
)

if [[ "${BENCH_GPU:-0}" == "1" ]]; then
    # ------------ CUDA build (Linux nvcc) -----------------------------------
    CXX="${NVCC:-nvcc}"
    "${CXX}" --version >/dev/null
    echo "[build_and_run] CUDA build via ${CXX}"
    "${CXX}" \
        -x cu \
        -arch=sm_80 \
        --extended-lambda \
        --expt-relaxed-constexpr \
        -DBENCH_GPU=1 \
        "${CXXFLAGS_COMMON[@]}" \
        "${SCRIPT_DIR}/chunked_het_bench.cxx" \
        "${TDI_SRC}" \
        "${DET_SRC}" \
        "${LRP_SRC}" \
        "${INT_SRC}" \
        -o "${BIN}"
else
    # ------------ CPU build (clang++) ---------------------------------------
    CXX="${CXX:-/usr/local/opt/llvm@20/bin/clang++}"
    if [[ ! -x "${CXX}" ]]; then
        # Fallback to /usr/local/bin/g++-N or system clang++; user can override CXX=
        CXX=clang++
    fi
    echo "[build_and_run] CPU build via ${CXX}"
    # Copy .cu -> .cxx (clang++ needs the .cxx extension to compile as C++).
    cp "${TDI_SRC}" "${OUT_DIR}/TDIonTheFly.cxx"
    cp "${DET_SRC}" "${OUT_DIR}/Detector.cxx"
    cp "${LRP_SRC}" "${OUT_DIR}/LISAResponse.cxx"
    cp "${INT_SRC}" "${OUT_DIR}/Interpolate.cxx"
    "${CXX}" \
        "${CXXFLAGS_COMMON[@]}" \
        "${SCRIPT_DIR}/chunked_het_bench.cxx" \
        "${OUT_DIR}/TDIonTheFly.cxx" \
        "${OUT_DIR}/Detector.cxx" \
        "${OUT_DIR}/LISAResponse.cxx" \
        "${OUT_DIR}/Interpolate.cxx" \
        -lpthread \
        ${LAPACK_LDLIBS} \
        -o "${BIN}"
fi

echo "[build_and_run] built ${BIN}"
echo "[build_and_run] running with args: $@"
"${BIN}" "$@"
