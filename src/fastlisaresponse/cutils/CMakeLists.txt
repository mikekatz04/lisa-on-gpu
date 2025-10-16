# =============================
# ==== BACKEND COMPILATION ====
# =============================

# * * * * * * * * * * * * * * * * * * * * *
# * * Helper functions to detect LAPACK * *
# * * * * * * * * * * * * * * * * * * * * *

# fastlisaresponse_try_get_lapacke_with_cmake
# ------------------------------
#
# This method will try to locate LAPACKE using CMake find_package() mechanism.
#
# This functions sets the following variables in the parent scope:
#
# * In case of success:
#
#   * LAPACKE_WITH_CMAKE_SUCCESS to ON
#   * LAPACKE_WITH_CMAKE_LIBS to targets to link against lapack
#
# * In case of failure:
#
#   * LAPACKE_WITH_CMAKE_SUCCESS to OFF
function(fastlisaresponse_try_get_lapacke_with_cmake)

  message(CHECK_START "Trying with find_package()")
  find_package(LAPACKE)

  if(LAPACKE_FOUND)
    set(LAPACKE_WITH_CMAKE_SUCCESS ON PARENT_SCOPE)
    set(LAPACKE_WITH_CMAKE_LIBS lapacke PARENT_SCOPE)
    message(CHECK_PASS "success! Found LAPACKE ${LAPACKE_VERSION}")
  else()
    set(LAPACKE_WITH_CMAKE_SUCCESS OFF PARENT_SCOPE)
    message(CHECK_FAIL "not found")
  endif()
endfunction()

# fastlisaresponse_try_get_lapacke_with_pkgconfig
# ----------------------------------
#
# This method will try to locate LAPACKE using PkgConfig.
#
# This functions sets the following variables in the parent scope:
#
# * In case of success:
#
#   * LAPACKE_WITH_PKGCONFIG_SUCCESS to ON
#   * LAPACKE_WITH_PKGCONFIG_LIBS to targets to link against lapack
#
# * In case of failure:
#
#   * LAPACKE_WITH_PKGCONFIG_SUCCESS to OFF
#   * LAPACKE_WITH_PKGCONFIG_REASON to
#
#     * "MISSING_PKGCONFIG" if PKGCONFIG is not available
#     * "MISSING_LAPACKE" if lapacke.pc is not found
function(fastlisaresponse_try_get_lapacke_with_pkgconfig)
  message(CHECK_START "Trying with PkgConfig")
  find_package(PkgConfig)
  if(NOT PkgConfig_FOUND)
    message(CHECK_FAIL "PkgConfig not available")
    set(LAPACKE_WITH_PKGCONFIG_SUCCESS OFF PARENT_SCOPE)
    set(LAPACKE_WITH_PKGCONFIG_REASON "MISSING_PKGCONFIG" PARENT_SCOPE)
    return()
  endif()

  pkg_check_modules(lapacke IMPORTED_TARGET lapacke lapack blas)
  if(NOT lapacke_FOUND)
    message(CHECK_FAIL "not found")
    set(LAPACKE_WITH_PKGCONFIG_SUCCESS OFF PARENT_SCOPE)
    set(LAPACKE_WITH_PKGCONFIG_REASON "MISSING_LAPACKE" PARENT_SCOPE)
    return()
  endif()

  message(CHECK_PASS
          "success! Found LAPACKE ${lapacke_VERSION} in ${lapacke_LIBDIR}")
  set(LAPACKE_WITH_PKGCONFIG_SUCCESS ON PARENT_SCOPE)
  set(LAPACKE_WITH_PKGCONFIG_LIBS PkgConfig::lapacke PARENT_SCOPE)
endfunction()

# fastlisaresponse_try_get_lapacke_with_cpm
# ----------------------------
#
# This method will use CPM (CMake Package Manager) to fetch LAPACK sources and
# add them to the build tree with enabled LAPACKE support.
#
# This function sets to following variables in the parent scope:
#
# * LAPACKE_WITH_CPM_SUCCESS to ON
# * LAPACKE_WITH_CPM_LIBS to the link targets
function(fastlisaresponse_try_get_lapacke_with_cpm)
  include(CPM)
  message(CHECK_START "Trying with automatic fetching of Reference LAPACK")
  enable_language(Fortran)
  CPMAddPackage(
    NAME lapack
    GITHUB_REPOSITORY Reference-LAPACK/lapack
    GIT_TAG 6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca # v3.12.1
    OPTIONS "LAPACKE"
            "ON"
            "CMAKE_POSITION_INDEPENDENT_CODE"
            "ON"
            "CMAKE_UNITY_BUILD"
            "ON"
            "CMAKE_UNITY_BUILD_BATCH_SIZE"
            64)
  set(LAPACKE_WITH_CPM_SUCCESS ON PARENT_SCOPE)
  set(LAPACKE_WITH_CPM_LIBS lapacke lapack PARENT_SCOPE)
  message(CHECK_PASS "done")
endfunction()

# fastlisaresponse_get_lapacke
# ---------------
#
# This method will try to make LAPACKE available using the following strategies:
#
# * PKGCONFIG: use the function "fastlisaresponse_try_get_lapacke_with_pkgconfig"
# * CMAKE: use the function "fastlisaresponse_try_get_lapacke_with_cmake"
# * FETCH: use the function "fastlisaresponse_try_get_lapacke_with_cpm"
#
# The following rules are applied to select the attempted strategies:
#
# * The PKGCONFIG strategy is only attempted if FASTLISA_LAPACKE_DETECT_WITH is
#   "AUTO" or "PKGCONFIG"
# * The CMAKE strategy is only attempted if FASTLISA_LAPACKE_DETECT_WITH is "AUTO" or
#   "CMAKE".
# * The FETCH strategy is only attempted if FASTLISA_LAPACKE_FETCH is "AUTO" or "ON".
# * The CMAKE and PKGCONFIG strategies are forcefully disabled if
#   FASTLISA_LAPACKE_FETCH is "ON"
#
# If no strategy is attempted, the function raises a warning and does not set
# any variable in parent scope. In this case, it is the user responsibility to
# set FASTLISA_LAPACKE_LIBS to any relevant value.
#
# If at least one strategy is attempted and if no strategy succeeds, the method
# fails with a fatal error explaining how to help attempted strategies to work.
#
# If one strategy succeeds, the following variables are set in parent scope:
#
# * FASTLISA_LAPACKE_LIBS: list of libraries to link against to use LAPACKE
# * FASTLISA_LAPACKE_GET_SUCCESS: ON
# * FASTLISA_LAPACKE_GET_STRATEGY: strategy that succeeded (PKGCONFIG|CMAKE|FETCH )
function(fastlisaresponse_get_lapacke)
  # cmake-lint: disable=R0912,R0915
  message(CHECK_START "Locating LAPACKE")

  # I. Detect enabled strategies
  if(FASTLISA_LAPACKE_DETECT_WITH STREQUAL "AUTO" OR FASTLISA_LAPACKE_DETECT_WITH
                                                STREQUAL "PKGCONFIG")
    set(pkgconfig_strategy_enabled ON)
  else()
    set(pkgconfig_strategy_enabled OFF)
  endif()
  if(FASTLISA_LAPACKE_DETECT_WITH STREQUAL "AUTO" OR FASTLISA_LAPACKE_DETECT_WITH
                                                STREQUAL "CMAKE")
    set(cmake_strategy_enabled ON)
  else()
    set(cmake_strategy_enabled OFF)
  endif()
  if(FASTLISA_LAPACKE_FETCH STREQUAL "AUTO" OR FASTLISA_LAPACKE_FETCH STREQUAL "ON")
    set(fetch_strategy_enabled ON)
  else()
    set(fetch_strategy_enabled OFF)
  endif()
  if(FASTLISA_LAPACKE_FETCH STREQUAL "ON")
    set(pkgconfig_strategy_enabled OFF)
    set(cmake_strategy_enabled OFF)
  endif()

  if(pkgconfig_strategy_enabled OR cmake_strategy_enabled
     OR fetch_strategy_enabled)
    set(any_strategy_enabled ON)
  else()
    set(any_strategy_enabled OFF)
  endif()

  # II. Apply the PkgConfig strategy if enabled
  if(pkgconfig_strategy_enabled)
    fastlisaresponse_try_get_lapacke_with_pkgconfig()
    if(LAPACKE_WITH_PKGCONFIG_SUCCESS)
      set(FASTLISA_LAPACKE_LIBS "${LAPACKE_WITH_PKGCONFIG_LIBS}" PARENT_SCOPE)
      set(FASTLISA_LAPACKE_GET_SUCCESS ON PARENT_SCOPE)
      set(FASTLISA_LAPACKE_GET_STRATEGY "PKGCONFIG" PARENT_SCOPE)
      message(CHECK_PASS "found with pkgconfig")
      return()
    endif()
  endif()

  # III. Apply the CMake strategy if enabled
  if(cmake_strategy_enabled)
    fastlisaresponse_try_get_lapacke_with_cmake()
    if(LAPACKE_WITH_CMAKE_SUCCESS)
      set(FASTLISA_LAPACKE_LIBS "${LAPACKE_WITH_CMAKE_LIBS}" PARENT_SCOPE)
      set(FASTLISA_LAPACKE_GET_SUCCESS ON PARENT_SCOPE)
      set(FASTLISA_LAPACKE_GET_STRATEGY "CMAKE" PARENT_SCOPE)
      message(CHECK_PASS "found with find_package()")
      return()
    endif()
  endif()

  # IV. Apply the Fetch strategy if enabled
  if(fetch_strategy_enabled)
    fastlisaresponse_try_get_lapacke_with_cpm()
    if(LAPACKE_WITH_CPM_SUCCESS)
      set(FASTLISA_LAPACKE_LIBS "${LAPACKE_WITH_CPM_LIBS}" PARENT_SCOPE)
      set(FASTLISA_LAPACKE_GET_SUCCESS ON PARENT_SCOPE)
      set(FASTLISA_LAPACKE_GET_STRATEGY "FETCH" PARENT_SCOPE)
      message(CHECK_PASS "added to build tree with automatic fetching")
      return()
    endif()
  endif()

  # V. Fail if any strategy was applied
  if(any_strategy_enabled)
    message(CHECK_FAIL "not found")
    if(pkgconfig_strategy_enabled)
      message(WARNING "LAPACKE could not be located with PKGCONFIG.")
      if(LAPACKE_WITH_PKGCONFIG_REASON STREQUAL "MISSING_PKGCONFIG")
        message(
          WARNING "Make sure that pkg-config executable is installed in your \
          environment.\n"
                  "On Ubuntu, it can be installed with:\n"
                  "  $ sudo apt install pkg-config\n"
                  "On mac OS, it can be installed with Homebrew with:\n"
                  "  $ brew install pkgconf\n"
                  "In conda environment, it can be installed with:\n"
                  "  $ conda install pkgconfig")
      elseif(LAPACKE_WITH_PKGCONFIG_REASON STREQUAL "MISSING_LAPACKE")
        message(
          WARNING
            "PkgConfig could not locate the file 'lapacke.pc'.\n"
            "If your LAPACK installation provides it, add its directory to \
            the PKG_CONFIG_PATH environment variable.\n"
            "It is usually located in the library install path, in the \
            'lib/pkgconfig' subdirectory.")
      endif()
    endif()
    if(cmake_strategy_enabled)
      message(
        WARNING
          "LAPACKE could not be located with CMake find_package() mechanism.\n"
          "If your LAPACK installation provides a 'lapacke-config.cmake' file \
          (or similar installed target file), add its path to the \
          CMAKE_PREFIX_PATH environment variable.\n"
          "It is usually located in the library install path, in the \
          'lib/cmake' subdirectory.")
    endif()
    if(fetch_strategy_enabled)
      message(
        WARNING "LAPACKE automatic fetching was enabled but somehow failed.\n"
                "CMake processing should have stop much sooner with a detailed \
                explanation of the failure.\nSee previous error messages.")
    endif()
    message(
      FATAL_ERROR
        "LAPACKE support is required but could not be satisfied.\n"
        "READ CAREFULLY PREVIOUS WARNINGS - \
        THEY SHOULD HELP YOU TO FIX THE ISSUE.")
  endif()

  # VI. Add message if lapacke detection is ignored
  message(CHECK_PASS "ignored \
   (FASTLISA_LAPACKE_DETECT_WITH=${FASTLISA_LAPACKE_DETECT_WITH} \
   and FASTLISA_LAPACKE_FETCH=${FASTLISA_LAPACKE_FETCH})")
  message(
    WARNING
      "LAPACKE detection strategies were disabled.\n"
      "Manually define FASTLISA_LAPACKE_LIBS using pip "
      "--config-settings=cmake.define.FASTLISA_LAPACKE_LIBS=value\n\n"
      "Make sure that the compiler can locate lapacke libraries (usually by "
      "adding their directory to "
      "the LIBRARY_PATH environment variable) and the header 'lapacke.h' "
      "(usually done by adding its directory to the CPATH environment "
      "variable).")
endfunction()


# TODO: how can we avoid moving source and header files in this directory ?
#include_directories(BEFORE "include")
#include_directories(BEFORE "src")

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

# apply_gpu_backend_common_options
# --------------------------------
#
# This method applies some common directive to GPU backend targets. It:
#
# * Expects a single "libname" argument
# * Expects the target to be named "fastlisaresponse_gpu_${libname}"
# * Defines the LIBRARY_OUTPUT_DIRECTORY property
# * Defines the OUTPUT_NAME property
# * Installs the target in the GPU backend directory (e.g.
#   'fastlisaresponse_backend_cuda12x')
# * Ensures the target includes the NumPy header directory
# * Disable NumPy deprecated API
# * Ensures the target links against CUDA libraries (cuBLAS, cuSPARSE, ...)
# * Defines the CUDA_ARCHITECTURE property
#
# Usage example: apply_gpu_backend_common_options(fastlisaresponse_utils_wrap)
function(apply_gpu_backend_common_options libname)
  set(target_name "fastlisaresponse_gpu_${libname}")
  set(backend_name "fastlisaresponse_backend_cuda${CUDAToolkit_VERSION_MAJOR}x")
  set_property(
    TARGET ${target_name}
    PROPERTY LIBRARY_OUTPUT_DIRECTORY
             "${BACKEND_BASE_OUTPUT_DIRECTORY}/${backend_name}")
  set_property(TARGET ${target_name} PROPERTY OUTPUT_NAME ${libname})

  install(TARGETS ${target_name} DESTINATION ${backend_name})

  target_include_directories(${target_name} PRIVATE ${Python_NumPy_INCLUDE_DIR})
  target_compile_definitions(${target_name}
                             PRIVATE NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION)
  target_link_libraries(${target_name} PUBLIC CUDA::cudart CUDA::cublas
                                              CUDA::cusparse)
  set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES
                                              ${FASTLISA_CUDA_ARCH})
  message(STATUS "This is the cuda archs to be run: ${FASTLISA_CUDA_ARCH}")
endfunction()

function(apply_cpu_backend_common_options libname)
  set(target_name "fastlisaresponse_cpu_${libname}")
  set_property(
    TARGET ${target_name}
    PROPERTY LIBRARY_OUTPUT_DIRECTORY
             "${BACKEND_BASE_OUTPUT_DIRECTORY}/fastlisaresponse_backend_cpu")
  set_property(TARGET ${target_name} PROPERTY OUTPUT_NAME ${libname})

  install(TARGETS ${target_name} DESTINATION fastlisaresponse_backend_cpu)

  get_target_property(FASTLISA_CXX_MARCH_OPT fastlisaresponse CXX_MARCH)
  if(FASTLISA_CXX_MARCH_OPT)
    target_compile_options(${target_name} PRIVATE "${FASTLISA_CXX_MARCH_OPT}")
  endif()

  target_include_directories(${target_name} PRIVATE ${Python_NumPy_INCLUDE_DIR})
  target_compile_definitions(${target_name}
                             PRIVATE NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION)
endfunction()

# * * * * * * * * * * * * * * * * * * * *
# * * Definition of compiled backends * *
# * * * * * * * * * * * * * * * * * * * *

# * * * * * * * * * * * * * * * * * * * *
# * * Definition of compiled backends * *
# * * * * * * * * * * * * * * * * * * * *

# ----------------
# --- responselisa ---
# ----------------


# 0. Download lisa analysis tools c files.
# file(
#   DOWNLOAD 
#   https://raw.githubusercontent.com/mikekatz04/LISAanalysistools/refs/heads/main/src/lisatools/cutils/Detector.cu 
#   "./src/fastlisaresponse/cutils/Detector.cu"
# )
# file(
#   DOWNLOAD 
#   https://raw.githubusercontent.com/mikekatz04/LISAanalysistools/refs/heads/main/src/lisatools/cutils/Detector.hpp 
#   "./src/fastlisaresponse/cutils/Detector.hpp"
# )
# file(
#   DOWNLOAD 
#   https://raw.githubusercontent.com/mikekatz04/LISAanalysistools/refs/heads/main/src/lisatools/cutils/global.hpp 
#   "./src/fastlisaresponse/cutils/global.hpp"
# )
# file(
#   DOWNLOAD 
#   https://raw.githubusercontent.com/mikekatz04/GPUBackendTools/refs/heads/main/src/gpubackendtools/cutils/Interpolate.cu 
#   "./src/fastlisaresponse/cutils/Interpolate.cu"
# )
# file(
#   DOWNLOAD 
#   https://raw.githubusercontent.com/mikekatz04/GPUBackendTools/refs/heads/main/src/gpubackendtools/cutils/Interpolate.hh 
#   "./src/fastlisaresponse/cutils/Interpolate.hh"
# )

# I. Process fastlisaresponse_utils_wrap.pyx into a C++ file
add_custom_command(
  OUTPUT "responselisa.cxx"
  COMMENT "Cythonize responselisa.pyx into responselisa.cxx"
  COMMAND
    Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/responselisa.pyx"
    --output-file "${CMAKE_CURRENT_BINARY_DIR}/responselisa.cxx" -3 -+ --module-name
    "responselisa" -I "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS "responselisa.pyx"
  VERBATIM)

# II. Declare the CPU backend
if(FASTLISA_WITH_CPU)
  add_custom_command(
    OUTPUT "LISAResponse.cxx"
    COMMENT "Copy LISAResponse.cu to LISAResponse.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/LISAResponse.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/LISAResponse.cxx"
    DEPENDS "LISAResponse.cu"
    VERBATIM)

  add_custom_command(
    OUTPUT "Detector.cxx"
    COMMENT "Copy Detector.cu to Detector.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/Detector.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/Detector.cxx"
    DEPENDS "Detector.cu"
    VERBATIM)

  python_add_library(fastlisaresponse_cpu_responselisa MODULE WITH_SOABI Detector.cxx responselisa.cxx LISAResponse.cxx)
  apply_cpu_backend_common_options(responselisa)

  target_sources(fastlisaresponse_cpu_responselisa PUBLIC FILE_SET HEADERS FILES
                                         global.hpp Detector.hpp cuda_complex.hpp LISAResponse.hh)
endif()

# III. Declare the GPU backend
if(FASTLISA_WITH_GPU)
  message(STATUS "This is the cuda archs to be run: ${FASTLISA_CUDA_ARCH}")
  
  python_add_library(fastlisaresponse_gpu_responselisa MODULE WITH_SOABI responselisa.cxx Detector.cu LISAResponse.cu)
  apply_gpu_backend_common_options(responselisa)
  target_sources(fastlisaresponse_gpu_responselisa PUBLIC FILE_SET HEADERS FILES
                                         global.hpp Detector.hpp cuda_complex.hpp LISAResponse.hh
                                         )
  target_link_libraries(fastlisaresponse_gpu_responselisa PUBLIC CUDA::cudart CUDA::cublas
                                              CUDA::cusparse)
  #set_target_properties(fastlisaresponse_gpu_responselisa PROPERTIES
  #                    CUDA_ARCHITECTURES ${FASTLISA_CUDA_ARCH}
  #                    CUDA_RESOLVE_DEVICE_SYMBOLS ON # <=================== Magic happens here <|
  #                    POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET fastlisaresponse_gpu_responselisa PROPERTY CUDA_ARCHITECTURES ${FASTLISA_CUDA_ARCH})
  set_property(TARGET fastlisaresponse_gpu_responselisa PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET fastlisaresponse_gpu_responselisa PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_property(TARGET fastlisaresponse_gpu_responselisa PROPERTY POSITION_INDEPENDENT_CODE ON)
  message(STATUS "CHECK: ${CMAKE_INSTALL_LIBDIR}/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/fastlisaresponse")

  install(TARGETS fastlisaresponse_gpu_responselisa
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/fastlisaresponse)
endif()

#####################
## tdi on the fly  ##
#####################

# I. Process fastlisaresponse_utils_wrap.pyx into a C++ file
add_custom_command(
  OUTPUT "lisatdionthefly.cxx"
  COMMENT "Cythonize lisatdionthefly.pyx into lisatdionthefly.cxx"
  COMMAND
    Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/lisatdionthefly.pyx"
    --output-file "${CMAKE_CURRENT_BINARY_DIR}/lisatdionthefly.cxx" -3 -+ --module-name
    "tdionthefly" -I "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS "lisatdionthefly.pyx"
  VERBATIM)


# II. Declare the CPU backend
if(FASTLISA_WITH_CPU)
  add_custom_command(
    OUTPUT "TDIonTheFly.cxx"
    COMMENT "Copy TDIonTheFly.cu to TDIonTheFly.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/TDIonTheFly.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/TDIonTheFly.cxx"
    DEPENDS "TDIonTheFly.cu"
    VERBATIM)

  add_custom_command(
    OUTPUT "Interpolate.cxx"
    COMMENT "Copy Interpolate.cu to Interpolate.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/Interpolate.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/Interpolate.cxx"
    DEPENDS "Interpolate.cu"
    VERBATIM)
  
  python_add_library(fastlisaresponse_cpu_tdionthefly MODULE WITH_SOABI Interpolate.cxx Detector.cxx lisatdionthefly.cxx TDIonTheFly.cxx)
  apply_cpu_backend_common_options(tdionthefly)

  target_sources(fastlisaresponse_cpu_tdionthefly PUBLIC FILE_SET HEADERS FILES
                                         global.hpp Detector.hpp cuda_complex.hpp LISAResponse.hh TDIonTheFly.hh Interpolate.hh)
  
  # interp needs to link against LAPACKE. USe the following method to detect
  # it and obtain ${FASTLISA_LAPACKE_LIBS}. In case of failure, it will stop the
  # processing execution.
  enable_language(Fortran)
  fastlisaresponse_get_lapacke()
  # Now, link against ${FASTLISA_LAPACKE_LIBS} to link against liblapacke.so and
  # include lapacke.h
  target_link_libraries(fastlisaresponse_cpu_tdionthefly LINK_PUBLIC ${FASTLISA_LAPACKE_LIBS}
                        ${FASTLISA_LAPACKE_EXTRA_LIBS} -lgfortran)

endif()

# IV. Remove downloaded files. 
# file(REMOVE "./src/fastlisaresponse/cutils/Detector.cu")
# file(REMOVE "./src/fastlisaresponse/cutils/Detector.hpp")
# file(REMOVE "./src/fastlisaresponse/cutils/global.hpp")
# file(REMOVE "./src/fastlisaresponse/cutils/Interpolate.cu")
# file(REMOVE "./src/fastlisaresponse/cutils/interpolate.hh")



