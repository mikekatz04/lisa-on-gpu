# =============================
# ==== BACKEND COMPILATION ====
# =============================

# In the project root CMakeLists.txt, we defined a "fastlisaresponse" interface
# target with properties WITH_CPU and WITH_GPU defining whether the CPU and a
# GPU backend need to be compiled. Let's retrieve these information here:
get_target_property(FASTLISA_WITH_CPU fastlisaresponse WITH_CPU)
get_target_property(FASTLISA_WITH_GPU fastlisaresponse WITH_GPU)

# Adapter to let inplace editable install work: the compiled backend will be
# placed into the source-tree in the 'src' directory:
# TODO: is this needed ?
if(SKBUILD_STATE STREQUAL "editable")
  set(BACKEND_BASE_OUTPUT_DIRECTORY "${fastlisaresponse_BINARY_DIR}/src")
else()
  set(BACKEND_BASE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif()

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    import gpubackendtools
    print(gpubackendtools.__file__.split('__init__.py')[0] + 'cutils')"
  OUTPUT_VARIABLE GBT_CUTILS
  OUTPUT_STRIP_TRAILING_WHITESPACE)

set(GBT_CMAKE_TOOLS "${GBT_CUTILS}/cmake_functions.cmake")

# Check if the file exists
if(EXISTS "${GBT_CMAKE_TOOLS}")
    message("INFO: ${GBT_CMAKE_TOOLS} exists.")
else()
    message(FATAL_ERROR "${GBT_CMAKE_TOOLS} not found.")
    # Handle the case where the file is missing
endif()

include(${GBT_CMAKE_TOOLS})
set(HERE_CUDA_ARCH ${FASTLISA_CUDA_ARCH})
set(GBT_LAPACK_DETECT_WITH ${FASTLISA_LAPACK_DETECT_WITH})

# * * * * * * * * * * * * * * * * * * * *
# * * Definition of compiled backends * *
# * * * * * * * * * * * * * * * * * * * *

# ----------------
# --- responselisa ---
# ----------------


execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    import lisatools
    _tmp = lisatools.__file__
    print(_tmp.split('/__init__.py')[0])"
  OUTPUT_VARIABLE LISATOOLS_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# Phase 3L.7c (2026-06-04): pull GBGPU's cutils directory so the
# GBTDIonTheFly class declaration (sourced from gb_tdi_on_the_fly.hh
# via `#include` in TDIonTheFly.hh) resolves. Transitional dep --
# lisa-on-gpu retires once Phase 3L.7/3L.8 finish carving out the
# GB and SOBBH source classes.
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "import gbgpu; print(gbgpu.get_include())"
  OUTPUT_VARIABLE GBGPU_CUTILS
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "GBGPU location is ${GBGPU_CUTILS}.")

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    import gpubackendtools
    _tmp = gpubackendtools.__file__
    print(_tmp.split('/__init__.py')[0])"
  OUTPUT_VARIABLE GBT_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    import lisatools
    print(lisatools._version.version)"
  OUTPUT_VARIABLE LISATOOLS_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE)

message(STATUS "LISATOOLS location is ${LISATOOLS_DIR}.")
file(GLOB MY_FILES "${LISATOOLS_DIR}/cutils/*")
message(STATUS "LISATOOLS Source files: ${MY_FILES}")
message(STATUS "LISATOOLS version: ${LISATOOLS_VERSION}")

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    import gpubackendtools
    print(gpubackendtools._version.version)"
  OUTPUT_VARIABLE GBT_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE)


execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    from lisatools_backend_cpu import pycppdetector
    print(pycppdetector.__file__)"
  OUTPUT_VARIABLE LISATOOLS_DETECTOR_CPU
  OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    from gbt_backend_cpu import interp
    print(interp.__file__)"
  OUTPUT_VARIABLE GBT_INTERP_CPU
  OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "TEST CHECK 1: ${GBT_INTERP_CPU}")

message(STATUS "GBT version: ${GBT_VERSION}")

message(STATUS "LISATOOLS location is ${LISATOOLS_DIR}.")
message(STATUS "GBT location is ${GBT_DIR}.")
file(GLOB MY_FILES "${LISATOOLS_DIR}/cutils/*" "${GBT_DIR}/cutils/*")
message(STATUS "Source files: ${MY_FILES}")

if(NOT EXISTS "${LISATOOLS_DIR}/cutils/binding.hpp")
  message(FATAL_ERROR "${LISATOOLS_DIR}/cutils/binding.hpp does not exist.")
endif()

if(NOT EXISTS "${GBT_DIR}/cutils/gbt_binding.hpp")
  message(FATAL_ERROR "${GBT_DIR}/cutils/gbt_binding.hpp does not exist.")
endif()

get_filename_component(LISATOOLS_DETECTOR_CPU_DIR ${LISATOOLS_DETECTOR_CPU} DIRECTORY)
get_filename_component(LISATOOLS_DETECTOR_CPU_LIB_NAME ${LISATOOLS_DETECTOR_CPU} NAME)
message(STATUS "dir: ${LISATOOLS_DETECTOR_CPU_DIR}")

find_library(LISATOOLS_DETECTOR_STATIC_LIB_CPU
    NAMES detector # The name of the library (without 'lib' prefix or '.a' extension)
    PATHS ${LISATOOLS_DETECTOR_CPU_DIR} # The specific folder to check
    NO_DEFAULT_PATH_LOGIC # Optional: Restricts search to only the provided PATHS
)

if(LISATOOLS_DETECTOR_STATIC_LIB_CPU)
    message("Found static library at: ${LISATOOLS_DETECTOR_STATIC_LIB_CPU}")
    # You can now use the library, e.g., add_library(MyImportedLib STATIC IMPORTED)
    # and set its location property. Or link directly in target_link_libraries
else()
    message(FATAL_ERROR "LISA Analysis Tools detector static library not found in the specified folder!")
endif()

message(STATUS "TEST CHECK: ${GBT_INTERP_CPU}")
get_filename_component(GBT_INTERP_CPU_DIR ${GBT_INTERP_CPU} DIRECTORY)
get_filename_component(GBT_INTERP_CPU_LIB_NAME ${GBT_INTERP_CPU} NAME)
message(STATUS "dir: ${GBT_INTERP_CPU_DIR}")

find_library(GBT_INTERP_STATIC_LIB_CPU
    NAMES interp # The name of the library (without 'lib' prefix or '.a' extension)
    PATHS ${GBT_INTERP_CPU_DIR} # The specific folder to check
    NO_DEFAULT_PATH_LOGIC # Optional: Restricts search to only the provided PATHS
)

if(GBT_INTERP_STATIC_LIB_CPU)
    message("Found static library at: ${GBT_INTERP_STATIC_LIB_CPU}")
    # You can now use the library, e.g., add_library(MyImportedLib STATIC IMPORTED)
    # and set its location property. Or link directly in target_link_libraries
else()
    message(FATAL_ERROR "GBT interp static library not found in the specified folder!")
endif()

# interp needs to link against LAPACKE. USe the following method to detect
# it and obtain ${GBT_LAPACKE_LIBS}. In case of failure, it will stop the
# processing execution.
enable_language(Fortran)
get_lapacke()

# Phase 3E (2026-06-02): the responselisa pybind11 module was retired.
# LISAResponse.cu / .hh and binding_flr.cxx / .hpp now live in
# LISAanalysistools/src/lisatools/cutils/ and are compiled into LAT's
# `pycppdetector` pybind11 module. The lisa-on-gpu `responselisa` module
# is no longer built; lisa-on-gpu/src/fastlisaresponse/cutils/__init__.py
# now sources LISAResponseWrap / TDIConfigWrap / OrbitsWrap_responselisa
# / CubicSplineWrap_responselisa from `lisatools_backend_*.pycppdetector`.
#
# The `tdionthefly` module below still needs the LISAResponse C++ symbols
# and binding_flr.hpp inlined wrapper classes at compile time (binding_tof.cxx
# uses them extensively); it points at LAT's copies via ${LISATOOLS_DIR}.

#####################
## tdi on the fly  ##
#####################


# II. Declare the CPU backend
if(FASTLISA_WITH_CPU)
  add_custom_command(
    OUTPUT "TDIonTheFly.cxx"
    COMMENT "Copy TDIonTheFly.cu to TDIonTheFly.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/TDIonTheFly.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/TDIonTheFly.cxx"
    DEPENDS "TDIonTheFly.cu"
    VERBATIM)

  # Phase 3E: LISAResponse.cu now lives in LAT; copy from there into our build dir.
  add_custom_command(
    OUTPUT "LISAResponse.cxx"
    COMMENT "Copy LISAResponse.cu (LAT) to LISAResponse.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${LISATOOLS_DIR}/cutils/LISAResponse.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/LISAResponse.cxx"
    DEPENDS "${LISATOOLS_DIR}/cutils/LISAResponse.cu"
    VERBATIM)

  # Phase 3L.5: LISATDIonTheFly base class moved to LAT; copy-compile.
  add_custom_command(
    OUTPUT "lat_tdi_on_the_fly.cxx"
    COMMENT "Copy lat_tdi_on_the_fly.cu (LAT) to lat_tdi_on_the_fly.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${LISATOOLS_DIR}/cutils/lat_tdi_on_the_fly.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/lat_tdi_on_the_fly.cxx"
    DEPENDS "${LISATOOLS_DIR}/cutils/lat_tdi_on_the_fly.cu"
    VERBATIM)

  # Phase 3L.6: FDSpline + TDSpline TDIWaveform moved to LAT; copy-compile.
  add_custom_command(
    OUTPUT "lat_spline_tdi_waveform.cxx"
    COMMENT "Copy lat_spline_tdi_waveform.cu (LAT) to lat_spline_tdi_waveform.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${LISATOOLS_DIR}/cutils/lat_spline_tdi_waveform.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/lat_spline_tdi_waveform.cxx"
    DEPENDS "${LISATOOLS_DIR}/cutils/lat_spline_tdi_waveform.cu"
    VERBATIM)

  # Phase 3L.7d: GBTDIonTheFly method bodies + gb_run_wave_tdi
  # kernel/wrap moved to GBGPU; copy-compile here so the still-resident
  # gb_run_fd_wave_tdi + GBComputationGroup code in TDIonTheFly.cu /
  # binding_tof.cxx can still link against them.
  add_custom_command(
    OUTPUT "gb_tdi_on_the_fly.cxx"
    COMMENT "Copy gb_tdi_on_the_fly.cu (GBGPU) to gb_tdi_on_the_fly.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${GBGPU_CUTILS}/gb_tdi_on_the_fly.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/gb_tdi_on_the_fly.cxx"
    DEPENDS "${GBGPU_CUTILS}/gb_tdi_on_the_fly.cu"
    VERBATIM)

    add_custom_command(
    OUTPUT "Detector.cxx"
    COMMENT "Copy Detector.cu to Detector.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${LISATOOLS_DIR}/cutils/Detector.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/Detector.cxx"
    DEPENDS "${LISATOOLS_DIR}/cutils/Detector.cu"
    VERBATIM)

  add_custom_command(
    OUTPUT "Interpolate.cxx"
    COMMENT "Copy Interpolate.cu to Interpolate.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${GBT_DIR}/cutils/Interpolate.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/Interpolate.cxx"
    DEPENDS "${GBT_DIR}/cutils/Interpolate.cu"
    VERBATIM)

  add_library(fastlisaresponse_cpu_tdionthefly_static STATIC Interpolate.cxx Detector.cxx LISAResponse.cxx lat_tdi_on_the_fly.cxx lat_spline_tdi_waveform.cxx gb_tdi_on_the_fly.cxx TDIonTheFly.cxx)
  apply_cpu_backend_common_options(tdionthefly fastlisaresponse fastlisaresponse true )
  message(STATUS "GBT_LAPACKE_INCLUDE_DIR: ${GBT_LAPACKE_INCLUDE_DIR}")
  target_include_directories(fastlisaresponse_cpu_tdionthefly_static PUBLIC "${LISATOOLS_DIR}/cutils/" ${GBT_CUTILS} ${GBGPU_CUTILS} ${GBT_LAPACKE_INCLUDE_DIR})  
  # target_link_libraries(fastlisaresponse_cpu_tdionthefly_static PUBLIC ${LISATOOLS_DETECTOR_STATIC_LIB_CPU})
  # target_link_directories(fastlisaresponse_cpu_tdionthefly_static PUBLIC ${LISATOOLS_DETECTOR_CPU_DIR})
  target_sources(fastlisaresponse_cpu_tdionthefly_static PUBLIC FILE_SET HEADERS FILES
                                        TDIonTheFly.hh)
  target_link_libraries(fastlisaresponse_cpu_tdionthefly_static LINK_PUBLIC ${GBT_LAPACKE_LIBS}
                        ${GBT_LAPACKE_EXTRA_LIBS})

  # Phase 3E: binding_flr.cxx now lives in LAT.
  pybind11_add_module(fastlisaresponse_cpu_tdionthefly "${LISATOOLS_DIR}/cutils/binding.cxx" "${GBT_DIR}/cutils/gbt_binding.cxx" "${LISATOOLS_DIR}/cutils/binding_flr.cxx" binding_tof.cxx)
  apply_cpu_backend_common_options(tdionthefly fastlisaresponse fastlisaresponse false)
  
  # Link your target to the imported target name
  target_link_libraries(fastlisaresponse_cpu_tdionthefly PUBLIC fastlisaresponse_cpu_tdionthefly_static)  #  ${LISATOOLS_DETECTOR_STATIC_LIB_CPU})
  # target_link_directories(fastlisaresponse_cpu_tdionthefly PUBLIC ${LISATOOLS_DETECTOR_CPU_DIR})  
  target_include_directories(fastlisaresponse_cpu_tdionthefly PUBLIC "${LISATOOLS_DIR}/cutils/" ${GBT_CUTILS} ${GBGPU_CUTILS} ${GBT_LAPACKE_INCLUDE_DIR})  
  target_sources(fastlisaresponse_cpu_tdionthefly PUBLIC FILE_SET HEADERS FILES
                                          binding_tof.hpp)
  target_link_libraries(fastlisaresponse_cpu_tdionthefly LINK_PUBLIC ${GBT_LAPACKE_LIBS}
                        ${GBT_LAPACKE_EXTRA_LIBS})



endif()


# II. Declare the CPU backend
if(FASTLISA_WITH_GPU)
  # Phase 3E: LISAResponse.cu moved to LAT.
  # Phase 3L.5: lat_tdi_on_the_fly.cu (LISATDIonTheFly base class) also from LAT.
  # Phase 3L.6: lat_spline_tdi_waveform.cu (FDSpline/TDSpline subclasses) also from LAT.
  add_library(fastlisaresponse_gpu_tdionthefly_gpu_static STATIC ${GBT_DIR}/cutils/Interpolate.cu ${LISATOOLS_DIR}/cutils/Detector.cu ${LISATOOLS_DIR}/cutils/LISAResponse.cu ${LISATOOLS_DIR}/cutils/lat_tdi_on_the_fly.cu ${LISATOOLS_DIR}/cutils/lat_spline_tdi_waveform.cu ${GBGPU_CUTILS}/gb_tdi_on_the_fly.cu TDIonTheFly.cu)
  apply_gpu_backend_common_options(tdionthefly_gpu fastlisaresponse fastlisaresponse true )
  target_include_directories(fastlisaresponse_gpu_tdionthefly_gpu_static PUBLIC "${LISATOOLS_DIR}/cutils/" ${GBT_CUTILS} ${GBGPU_CUTILS})
  target_sources(fastlisaresponse_gpu_tdionthefly_gpu_static PUBLIC FILE_SET HEADERS FILES
                                        TDIonTheFly.hh)

  # ---- Optional cufftdx (NVIDIA MathDx Block FFT) ----
  # When FASTLISA_MATHDX_DIR is set, swap the hand-rolled radix-2 in
  # WDMSplineHelpers.hh for cufftdx's cooperative block FFT.
  if(FASTLISA_MATHDX_DIR)
    if(NOT EXISTS "${FASTLISA_MATHDX_DIR}/include/cufftdx.hpp")
      message(WARNING "FASTLISA_MATHDX_DIR='${FASTLISA_MATHDX_DIR}' does not \
contain include/cufftdx.hpp -- cufftdx will NOT be enabled.")
    else()
      message(STATUS "cufftdx enabled from ${FASTLISA_MATHDX_DIR}")
      target_include_directories(fastlisaresponse_gpu_tdionthefly_gpu_static
        PUBLIC "${FASTLISA_MATHDX_DIR}/include")
      target_compile_definitions(fastlisaresponse_gpu_tdionthefly_gpu_static
        PUBLIC LISA_USE_CUFFTDX)
    endif()
  endif()

  # Phase 3E: binding_flr.cxx now lives in LAT.
  pybind11_add_module(fastlisaresponse_gpu_tdionthefly "${LISATOOLS_DIR}/cutils/binding.cxx" "${GBT_DIR}/cutils/gbt_binding.cxx" "${LISATOOLS_DIR}/cutils/binding_flr.cxx" binding_tof.cxx)
  apply_gpu_backend_common_options(tdionthefly fastlisaresponse fastlisaresponse false)

  # Link your target to the imported target name
  target_link_libraries(fastlisaresponse_gpu_tdionthefly PUBLIC fastlisaresponse_gpu_tdionthefly_gpu_static)  #  ${LISATOOLS_DETECTOR_STATIC_LIB_CPU})
  target_include_directories(fastlisaresponse_gpu_tdionthefly PUBLIC "${LISATOOLS_DIR}/cutils/" ${GBT_CUTILS} ${GBGPU_CUTILS})
  target_sources(fastlisaresponse_gpu_tdionthefly PUBLIC FILE_SET HEADERS FILES
                                          binding_tof.hpp)

endif()



