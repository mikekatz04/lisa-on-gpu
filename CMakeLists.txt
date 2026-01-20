# ======================================
# == Fast LISA Response project base configuration ==
# ======================================

# TODO:
#  - check cmake_minimum_required range
#  - check if project-specific CMake function are needed

# ---- CMake related definitions ----
cmake_minimum_required(VERSION 3.23...3.31)


set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- Main project definition ----
project(fastlisaresponse VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX Fortran)

# ---- Find required dependencies ----
find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)
find_package(pybind11 CONFIG)

# This line defines the install directory variables
include(GNUInstallDirs)

# ---- Import project-specific CMake functions ----
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---- Define project specific options ----

# FASTLISA_WITH_GPU controls the activation of both GPU and CPU backend compilations.
# Its allowed values are:
#
# * AUTO: enable CPU backend, and enable GPU backend if CUDA toolchain is found
#   in environment, otherwise disable it
# * ON: enable CPU backend, and enable GPU backend, fail if CUDA toolchain is
#   not available
# * OFF: enable CPU backend and disable GPU backend
# * ONLY: disable CPU backend and enable GPU backend (used in plugin wheel build
#   process)
# * BARE: disable both CPU and GPU backends
set(
  FASTLISA_WITH_GPU "AUTO" CACHE
  STRING "Whether to compile GPU backend"
)
set_property(
  CACHE FASTLISA_WITH_GPU PROPERTY
  STRINGS "AUTO" "ON" "OFF" "ONLY" "BARE"
)

# FASTLISA_CUDA_ARCH will be passed as the CUDA_ARCHITECTURES property for the CUDA
# backend if it is compiled. See the documentation of CUDA_ARCHITECTURES:
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set(
  FASTLISA_CUDA_ARCH "native"
  CACHE STRING "CUDA Architecture targetted for Fast LISA Response compilation (see doc of \
          CMAKE_CUDA_ARCHITECTURES)."
)


# GBT_LAPACKE_DETECT_WITH sets the tool used to try to detect LAPACKE locally.
# Possible values are:
#
# * AUTO: attempts PKGCONFIG and, in case of failure, CMAKE
# * CMAKE: use CMake find_package() mechanism (try to find
#   'lapacke-config.cmake' in CMAKE_PREFIX_PATH or in standard system locations)
# * PKGCONFIG: use pkgconfig (which must be installed), try to find 'lapacke.pc'
#   and its dependencies in standard system locations or in PKG_CONFIG_PATH
# * DISABLE (or any other value): disable detection of local lapacke
#   installation
set(GBT_LAPACKE_DETECT_WITH "AUTO" CACHE STRING "Tool used to locate LAPACKE.")
set_property(CACHE GBT_LAPACKE_DETECT_WITH PROPERTY STRINGS "AUTO" "CMAKE"
                                                    "PKGCONFIG" "DISABLE")

set(GBT_LAPACKE_DETECT_WITH ${GBT_LAPACKE_DETECT_WITH})                                
set_property(GLOBAL APPEND PROPERTY FASTLISA_OPTION_NAMES GBT_LAPACKE_DETECT_WITH)

# GBT_LAPACKE_FETCH controls whether LAPACKE sources will be downloaded and
# compiled with fast LISA response backends. Possible values are:
#
# * AUTO: will try to locate LAPACKE locally and, in case of failure, will
#   enable the fetch mechanism
# * ON: will always fetch LAPACKE sources and compile them, even if already
#   present locally
# * OFF: will disable LAPACKE fetching mechanism
set(GBT_LAPACKE_FETCH "AUTO" CACHE STRING
                                   "Whether to download and compile LAPACK(E)")
set_property(CACHE GBT_LAPACKE_FETCH PROPERTY STRINGS "AUTO" "ON" "OFF")
set_property(GLOBAL APPEND PROPERTY FASTLISA_OPTION_NAMES GBT_LAPACKE_FETCH)

# In CMakeLists.txt
# TODO: Generalize!!!
# if(NOT CMAKE_Fortran_COMPILER)
#     set(CMAKE_Fortran_COMPILER "/opt/homebrew/bin/gfortran" CACHE FILEPATH "Fortran compiler" FORCE)
# endif()

# GBT_LAPACKE_EXTRA_LIBS sets extra libraries that must be linked when linking
# LAPACKE. For example, if importing the CPU backend fails and complains about
# missing "gfortran" symbols, set GBT_LAPACKE_EXTRA_LIBS=gfortran. Note that if
# this option is unset, it will be automatically set to "gfortran" if that
# library in linkable.
include(CheckLinkerFlag)
check_linker_flag(CXX "-lgfortran" GFORTRAN_AVAILABLE)
if(GFORTRAN_AVAILABLE)
  message(STATUS "gfortran available.")
  set(DEFAULT_EXTRA_LIBS "gfortran")
else()
  set(DEFAULT_EXTRA_LIBS "")
  message(STATUS "gfortran NOT available.")
endif()
set(GBT_LAPACKE_EXTRA_LIBS "${DEFAULT_EXTRA_LIBS}"
    CACHE STRING "Extra libs to link to when\
linking against LAPACKE.")
set_property(GLOBAL APPEND PROPERTY FASTLISA_OPTION_NAMES GBT_LAPACKE_EXTRA_LIBS)
unset(GFORTRAN_AVAILABLE)
unset(DEFAULT_EXTRA_LIBS)


set(
  FASTLISA_MARCH "native"
  CACHE STRING "Value of the -march compiler option if supported by compiler"
)

# ---- Phony target for project specific properties ----
add_library(fastlisaresponse INTERFACE)

# ---- Enable building the CPU version of backends by default ----
set_target_properties(fastlisaresponse PROPERTIES WITH_CPU ON)

# ---- Test whether the FASTLISA_MARCH option is supported by CXX compiler ----
include(CheckCXXCompilerFlag)
set(FASTLISA_MARCH_CXX_OPT "-march=${FASTLISA_MARCH}")
check_cxx_compiler_flag("${FASTLISA_MARCH_CXX_OPT}" CXX_COMPILER_SUPPORTS_FASTLISA_MARCH)
if(CXX_COMPILER_SUPPORTS_FASTLISA_MARCH)
  set_property(TARGET fastlisaresponse PROPERTY CXX_MARCH
    "${FASTLISA_MARCH_CXX_OPT}")
  message(STATUS "The CXX compiler supports option '${FASTLISA_MARCH_CXX_OPT}'.")
else()
  message(
    WARNING "The CXX compiler does not support option '${FASTLISA_MARCH_CXX_OPT}'. \
      It will be ignored.")
endif()

# ---- Optionnally check if GPU is supported ----
if(FASTLISA_WITH_GPU STREQUAL "AUTO")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit)
  endif()
  if(CMAKE_CUDA_COMPILER AND CUDAToolkit_FOUND)
    message(
      STATUS
      "Fast LISA Response GPU support was set to AUTO and will be turned ON as CUDA and \
CUDA Toolkit are available with CUDA version \
${CUDAToolkit_VERSION_MAJOR}.")
    set_target_properties(fastlisaresponse PROPERTIES WITH_GPU ON)
  else()
    message(
      STATUS
      "Fast LISA Response GPU support was set to AUTO and will be turned OFF as CUDA and \
CUDA Toolkit are not found (CMAKE_CUDA_COMPILER:${CMAKE_CUDA_COMPILER} \
CUDAToolkit_FOUND:${CUDAToolkit_FOUND}).")
    set_target_properties(fastlisaresponse PROPERTIES WITH_GPU OFF)
  endif()
elseif(FASTLISA_WITH_GPU STREQUAL "ONLY")
  message(
    STATUS
      "Fast LISA Response GPU support is set to ON and CPU support is set to OFF (use only to \
      build a standalone plugin).")
  set_target_properties(fastlisaresponse PROPERTIES WITH_GPU ON)
  set_target_properties(fastlisaresponse PROPERTIES WITH_CPU OFF)
elseif(FASTLISA_WITH_GPU STREQUAL "BARE")
  message(STATUS "Fast LISA Response GPU and CPU support are disabled (use only to build a \
      non-functional pure-python release).")
  set_target_properties(fastlisaresponse PROPERTIES WITH_GPU OFF)
  set_target_properties(fastlisaresponse PROPERTIES WITH_CPU OFF)
else()
  message(STATUS "Fast LISA Response GPU support is set to ${FASTLISA_WITH_GPU}.")
  set_target_properties(fastlisaresponse PROPERTIES WITH_GPU ${FASTLISA_WITH_GPU})
endif()

# ---- Handle GPU support ----
get_target_property(_FASTLISA_WITH_GPU fastlisaresponse WITH_GPU)
if(_FASTLISA_WITH_GPU)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# ---- Include sources ----
add_subdirectory(src)
