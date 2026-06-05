# =============================================================================
# Phase 3L.7n (2026-06-04): fastlisaresponse is now a pure-Python wheel.
#
# All native C++/CUDA compilation has been carved out:
#   * Generic LISA-response  -> LISAanalysistools (Phase 3E + 3L.1-3L.6)
#   * GB-specific kernels    -> GBGPU             (Phase 3L.7a-h)
#   * SOBBH-specific kernels -> BBHx              (Phase 3L.8)
#   * Python shim chain      -> deleted           (Phase 3L.7m)
#   * Tdionthefly binding    -> deleted           (Phase 3L.7n, this commit)
#
# Nothing remains to compile. The wheel just packs __init__.py +
# _version.py + _editable.py + cutils/__init__.py (retirement-notice
# docstring). LANGUAGES NONE skips the C/CXX/Fortran/CUDA toolchain
# probes entirely; scikit-build-core still picks up the Python sources
# via `wheel.packages = ["src/fastlisaresponse"]` in pyproject.toml.
#
# Phase 3L.7o will retire the package outright once consumers have had
# one release cycle to switch off `import fastlisaresponse`.
# =============================================================================

cmake_minimum_required(VERSION 3.23...3.31)

project(fastlisaresponse VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES NONE)

# Phony interface target -- kept so the git-version block in
# src/fastlisaresponse/CMakeLists.txt can still query WITH_CPU/WITH_GPU
# properties (both forced to OFF since there's nothing to compile).
add_library(fastlisaresponse INTERFACE)
set_target_properties(fastlisaresponse PROPERTIES WITH_CPU OFF WITH_GPU OFF)

add_subdirectory(src)
